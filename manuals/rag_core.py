import hashlib
import json
import logging
import math
import os
import random
import re
import urllib.request
from collections import Counter

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


LOGGER = logging.getLogger("rag")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def sanitize_source_name(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def source_id_from_path(path):
    base = sanitize_source_name(os.path.basename(path))
    digest = hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


def normalize_text(text):
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def load_pdf(path):
    pages = []
    try:
        reader = PdfReader(path)
    except Exception as exc:
        LOGGER.error("Failed to open PDF %s: %s", path, exc)
        return pages

    desc = f"Reading {os.path.basename(path)}"
    for i, page in enumerate(tqdm(reader.pages, desc=desc, unit="page")):
        try:
            text = page.extract_text()
        except Exception as exc:
            LOGGER.warning("Failed to extract page %d from %s: %s", i + 1, path, exc)
            continue
        text = normalize_text(text)
        if not text:
            continue
        pages.append({"page_index": i + 1, "text": text})
    return pages


def load_pdf_with_outline(path):
    pages = []
    outline_entries = []
    try:
        reader = PdfReader(path)
    except Exception as exc:
        LOGGER.error("Failed to open PDF %s: %s", path, exc)
        return pages, outline_entries

    desc = f"Reading {os.path.basename(path)}"
    for i, page in enumerate(tqdm(reader.pages, desc=desc, unit="page")):
        try:
            text = page.extract_text()
        except Exception as exc:
            LOGGER.warning("Failed to extract page %d from %s: %s", i + 1, path, exc)
            continue
        text = normalize_text(text)
        if not text:
            continue
        pages.append({"page_index": i + 1, "text": text})

    outline_entries = extract_outline_entries(reader)
    return pages, outline_entries


def load_text_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
    except Exception as exc:
        LOGGER.error("Failed to read text file %s: %s", path, exc)
        return []
    text = normalize_text(text)
    if not text:
        return []
    return [{"page_index": None, "text": text}]


def load_manual(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    return load_text_file(path)


def tokenize(text):
    return re.findall(r"\S+", text)


def bm25_tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def is_heading_line(line):
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}\b", stripped):
        return False
    if re.match(r"^\d{1,2}[./]\d{1,2}[./]\d{2,4}\b", stripped):
        return False
    if re.match(r"^(figure|fig\.|table|tab\.)\b", lower):
        return False
    if re.match(r"^\d+(\.\d+)*\s+\S+", stripped):
        return True
    if re.match(r"^(element|lesson|tab\.?|table|figure|fig\.?)\b", lower):
        return True
    if re.match(r"^[A-Z0-9][A-Z0-9 \-:/]{5,}$", stripped):
        return True
    return False


def alpha_ratio(text):
    letters = len(re.findall(r"[A-Za-z]", text))
    alnum = len(re.findall(r"[A-Za-z0-9]", text))
    return letters / float(alnum) if alnum else 0.0


def filter_pages(pages, min_page_tokens):
    if min_page_tokens <= 0:
        return pages
    filtered = []
    for page in pages:
        token_count = len(tokenize(page.get("text", "")))
        if token_count >= min_page_tokens:
            filtered.append(page)
    return filtered


def classify_chunk_kind(section, text):
    section = (section or "").lower()
    text = (text or "").lower()
    combined = f"{section} {text}"
    if re.search(r"\b(terms|definitions|abbreviations|glossary)\b", combined):
        return "definition"
    if re.search(r"\b(change log|revision history|document control|version history)\b", combined):
        return "changelog"
    if re.search(r"\b(training|course|overview|introduction)\b", combined):
        return "training_admin"
    if re.search(r"\b(safety|warning|caution|hazard|ppe)\b", combined):
        return "safety"
    if re.search(r"\b(checklist|check list|inspection checklist|qa checklist)\b", combined):
        return "checklist"
    if re.search(r"\b(troubleshooting|diagnostic|fault|failure mode)\b", combined):
        return "troubleshooting"
    if re.search(r"\b(procedure|work instruction|steps|step-by-step|step by step)\b", combined):
        return "procedure"
    return "other"


def extract_outline_entries(reader):
    entries = []
    try:
        outline = reader.outline
    except Exception:
        outline = None
    if not outline:
        return entries

    def walk(items):
        for item in items:
            if isinstance(item, list):
                walk(item)
            else:
                try:
                    title = getattr(item, "title", None) or str(item)
                    page_number = reader.get_destination_page_number(item) + 1
                except Exception:
                    continue
                entries.append({"title": title.strip(), "page_index": page_number})

    walk(outline)
    return entries


def build_outline_blocks(pages, outline_entries):
    if not outline_entries:
        return []
    page_map = {page["page_index"]: page.get("text", "") for page in pages}
    max_page = max(page_map.keys()) if page_map else 0
    ordered = sorted(
        {entry["page_index"]: entry for entry in outline_entries}.values(),
        key=lambda item: item["page_index"],
    )
    blocks = []
    for idx, entry in enumerate(ordered):
        start = entry["page_index"]
        end = ordered[idx + 1]["page_index"] - 1 if idx + 1 < len(ordered) else max_page
        if start <= 0 or end <= 0 or end < start:
            continue
        texts = []
        for page_number in range(start, end + 1):
            text = page_map.get(page_number)
            if text:
                texts.append(text)
        if not texts:
            continue
        blocks.append(
            {
                "page_index": start,
                "page_end": end,
                "heading": entry.get("title"),
                "text": " ".join(texts),
            }
        )
    return blocks


def split_into_blocks(pages):
    blocks = []
    current_lines = []
    current_page = None
    current_heading = None
    for page in pages:
        page_index = page.get("page_index")
        lines = [line.strip() for line in page.get("text", "").splitlines()]
        for line in lines:
            if not line:
                continue
            if is_heading_line(line):
                if current_lines:
                    blocks.append(
                        {
                            "page_index": current_page,
                            "text": " ".join(current_lines),
                            "heading": current_heading,
                            "page_end": current_page,
                        }
                    )
                current_lines = [line]
                current_page = page_index
                current_heading = line
            else:
                if current_page is None:
                    current_page = page_index
                current_lines.append(line)
    if current_lines:
        blocks.append(
            {
                "page_index": current_page,
                "text": " ".join(current_lines),
                "heading": current_heading,
                "page_end": current_page,
            }
        )
    return blocks


def merge_small_blocks(blocks, min_tokens, min_alpha_ratio):
    if not blocks:
        return blocks
    merged = []
    carry = None
    for block in blocks:
        text = block.get("text", "")
        tokens = tokenize(text)
        content_ok = len(tokens) >= min_tokens and alpha_ratio(text) >= min_alpha_ratio
        if carry is None:
            carry = {
                "page_index": block.get("page_index"),
                "text": text,
                "heading": block.get("heading"),
                "page_end": block.get("page_end"),
            }
        else:
            carry["text"] = f"{carry['text']} {text}".strip()
            if not carry.get("heading") and block.get("heading"):
                carry["heading"] = block.get("heading")
            if block.get("page_end"):
                carry["page_end"] = block.get("page_end")
        carry_tokens = tokenize(carry["text"])
        carry_ok = len(carry_tokens) >= min_tokens and alpha_ratio(carry["text"]) >= min_alpha_ratio
        if content_ok and carry_ok:
            merged.append(carry)
            carry = None
    if carry:
        merged.append(carry)
    return merged


def chunk_blocks(
    blocks,
    source_id,
    source_name,
    chunk_size,
    chunk_overlap,
    add_section_chunks,
    max_section_tokens,
    min_section_tokens,
):
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []
    chunk_idx = 0
    for block in blocks:
        block_tokens = tokenize(block.get("text", ""))
        if not block_tokens:
            continue
        chunk_kind = classify_chunk_kind(block.get("heading"), block.get("text", ""))
        if (
            add_section_chunks
            and len(block_tokens) <= max_section_tokens
            and len(block_tokens) >= min_section_tokens
        ):
            if len(block_tokens) <= chunk_size:
                chunk_id = f"manuals::{source_id}::chunk_{chunk_idx:06d}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source": source_name,
                        "page": block.get("page_index"),
                        "page_end": block.get("page_end", block.get("page_index")),
                        "section": block.get("heading"),
                        "chunk_kind": chunk_kind,
                        "text": " ".join(block_tokens),
                    }
                )
                chunk_idx += 1
                continue
            chunk_id = f"manuals::{source_id}::chunk_{chunk_idx:06d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source_name,
                    "page": block.get("page_index"),
                    "page_end": block.get("page_end", block.get("page_index")),
                    "section": block.get("heading"),
                    "chunk_kind": chunk_kind,
                    "text": " ".join(block_tokens),
                }
            )
            chunk_idx += 1
        start = 0
        while start < len(block_tokens):
            end = min(start + chunk_size, len(block_tokens))
            chunk_tokens = block_tokens[start:end]
            if not chunk_tokens:
                break
            chunk_id = f"manuals::{source_id}::chunk_{chunk_idx:06d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source_name,
                    "page": block.get("page_index"),
                    "page_end": block.get("page_end", block.get("page_index")),
                    "section": block.get("heading"),
                    "chunk_kind": chunk_kind,
                    "text": " ".join(chunk_tokens),
                }
            )
            chunk_idx += 1
            if end == len(block_tokens):
                break
            start = end - chunk_overlap
    return chunks


def embed_texts(texts, model_name, batch_size):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avgdl = 0.0
        self.N = len(documents)
        self.idf = {}
        self.inverted = {}

        if not documents:
            return

        doc_freq = Counter()
        for doc_id, text in enumerate(documents):
            tokens = bm25_tokenize(text)
            self.doc_len.append(len(tokens))
            counts = Counter(tokens)
            for term, tf in counts.items():
                doc_freq[term] += 1
                self.inverted.setdefault(term, []).append((doc_id, tf))

        self.avgdl = sum(self.doc_len) / float(self.N) if self.N else 0.0
        for term, df in doc_freq.items():
            self.idf[term] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_text):
        if not self.N:
            return np.zeros(0, dtype="float32")
        scores = np.zeros(self.N, dtype="float32")
        tokens = bm25_tokenize(query_text)
        for term in tokens:
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_id, tf in postings:
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * (self.doc_len[doc_id] / self.avgdl)
                )
                scores[doc_id] += idf * ((tf * (self.k1 + 1.0)) / denom)
        return scores


def save_index(index_dir, index, metadata, config):
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)
    with open(os.path.join(index_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=True, indent=2)


def load_index(index_dir):
    index_path = os.path.join(index_dir, "index.faiss")
    meta_path = os.path.join(index_dir, "metadata.json")
    config_path = os.path.join(index_dir, "config.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index file: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return index, metadata, config


def build_index(
    manual_paths,
    index_dir,
    model_name,
    chunk_size,
    chunk_overlap,
    batch_size,
    seed,
    add_section_chunks,
    max_section_tokens,
    section_manuals,
    skip_first_pages,
    skip_page_manuals,
    min_page_tokens,
    min_alpha_ratio,
):
    setup_logging()
    set_deterministic(seed)
    if chunk_size < 250 or chunk_size > 500:
        LOGGER.warning("Chunk size %d is outside the recommended 250-500 range.", chunk_size)
    if chunk_overlap < 50 or chunk_overlap > 80:
        LOGGER.warning(
            "Chunk overlap %d is outside the recommended 50-80 range.", chunk_overlap
        )

    all_chunks = []
    for path in manual_paths:
        source_name = os.path.basename(path)
        source_id = source_id_from_path(path)
        pages = None
        outline_entries = []
        use_sections = False
        if section_manuals:
            use_sections = "*" in section_manuals or source_name in section_manuals
        if os.path.splitext(path)[1].lower() == ".pdf" and use_sections:
            pages, outline_entries = load_pdf_with_outline(path)
        else:
            pages = load_manual(path)
        if not pages:
            LOGGER.warning("No text extracted from %s", path)
            continue
        if source_name in skip_page_manuals and skip_first_pages > 0:
            pages = [page for page in pages if page.get("page_index", 0) > skip_first_pages]
            if not pages:
                LOGGER.warning("All pages skipped for %s", path)
                continue
        pages = filter_pages(pages, min_page_tokens=min_page_tokens)
        if not pages:
            LOGGER.warning("All pages filtered due to low text for %s", path)
            continue
        if outline_entries:
            blocks = build_outline_blocks(pages, outline_entries)
            if not blocks:
                blocks = split_into_blocks(pages)
        else:
            blocks = split_into_blocks(pages)
        blocks = merge_small_blocks(blocks, min_tokens=120, min_alpha_ratio=min_alpha_ratio)
        if not blocks:
            LOGGER.warning("No text blocks produced for %s", path)
            continue
        chunks = chunk_blocks(
            blocks,
            source_id=source_id,
            source_name=source_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_section_chunks=add_section_chunks,
            max_section_tokens=max_section_tokens,
            min_section_tokens=120,
        )
        if not chunks:
            LOGGER.warning("No chunks produced for %s", path)
            continue
        all_chunks.extend(chunks)
        LOGGER.info("Loaded %d chunks from %s", len(chunks), source_name)

    if not all_chunks:
        raise ValueError("No chunks created. Check your manual paths or file types.")

    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embed_texts(texts, model_name=model_name, batch_size=batch_size)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    embeddings = embeddings.astype("float32", copy=False)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    config = {
        "model_name": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "seed": seed,
        "add_section_chunks": add_section_chunks,
        "max_section_tokens": max_section_tokens,
        "section_manuals": section_manuals,
        "skip_first_pages": skip_first_pages,
        "skip_page_manuals": skip_page_manuals,
        "min_page_tokens": min_page_tokens,
        "min_alpha_ratio": min_alpha_ratio,
        "manual_paths": manual_paths,
        "num_chunks": len(all_chunks),
    }
    save_index(index_dir, index, all_chunks, config)
    LOGGER.info("Saved index with %d chunks to %s", len(all_chunks), index_dir)


class Retriever:
    def __init__(
        self,
        index_dir,
        model_name=None,
        alpha=0.7,
        candidate_multiplier=5,
        synthetic_weight=0.6,
        kind_boost=None,
        allowed_kinds=None,
    ):
        self.index, self.metadata, self.config = load_index(index_dir)
        self.model_name = model_name or self.config.get("model_name")
        if not self.model_name:
            raise ValueError("Model name is missing from config and not provided.")
        self.model = SentenceTransformer(self.model_name)
        self.alpha = alpha
        self.candidate_multiplier = candidate_multiplier
        self.synthetic_weight = synthetic_weight
        self.kind_boost = kind_boost or {}
        self.allowed_kinds = set(allowed_kinds) if allowed_kinds else None
        self.bm25 = BM25Index([item.get("text", "") for item in self.metadata])

    def source_weight(self, source_name):
        if not source_name:
            return 1.0
        if "synthetic" in source_name.lower():
            return self.synthetic_weight
        return 1.0

    def search(self, query_text, top_k):
        if not self.metadata:
            return []
        top_k = min(top_k, len(self.metadata))
        candidate_k = min(len(self.metadata), max(top_k * self.candidate_multiplier, top_k))
        query_emb = self.model.encode(
            [query_text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32", copy=False)
        scores, indices = self.index.search(query_emb, candidate_k)
        emb_scores = {}
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue
            emb_scores[idx] = float(score)

        bm25_scores = self.bm25.score(query_text)
        if bm25_scores.size:
            bm25_top = np.argpartition(-bm25_scores, min(candidate_k - 1, bm25_scores.size - 1))[
                :candidate_k
            ]
            candidates = set(emb_scores.keys()) | set(bm25_top.tolist())
        else:
            candidates = set(emb_scores.keys())

        max_emb = max(emb_scores.values()) if emb_scores else 0.0
        max_bm25 = float(bm25_scores.max()) if bm25_scores.size else 0.0

        combined = []
        for idx in candidates:
            if idx < 0 or idx >= len(self.metadata):
                continue
            emb_norm = emb_scores.get(idx, 0.0) / max_emb if max_emb > 0 else 0.0
            bm25_norm = (
                float(bm25_scores[idx]) / max_bm25 if max_bm25 > 0 else 0.0
            )
            kind = self.metadata[idx].get("chunk_kind")
            if self.allowed_kinds and kind not in self.allowed_kinds:
                continue
            base_score = (self.alpha * emb_norm) + ((1.0 - self.alpha) * bm25_norm)
            source = self.metadata[idx].get("source")
            score = base_score * self.source_weight(source)
            score *= self.kind_boost.get(kind, 1.0)
            combined.append((score, idx))

        combined.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, idx in combined[:top_k]:
            meta = self.metadata[idx]
            results.append(
                {
                    "chunk_id": meta.get("chunk_id"),
                    "score": float(score),
                    "source": meta.get("source"),
                    "text": meta.get("text"),
                    "page": meta.get("page"),
                    "page_end": meta.get("page_end"),
                    "section": meta.get("section"),
                    "chunk_kind": meta.get("chunk_kind"),
                }
            )
        return results


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / float(rank)
    return 0.0


def load_gold_jsonl(gold_path):
    rows = []
    with open(gold_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_retrieval(index_dir, gold_path, top_k):
    retriever = Retriever(index_dir)
    rows = load_gold_jsonl(gold_path)
    if not rows:
        raise ValueError("Gold file is empty.")

    precision_sum = 0.0
    mrr_sum = 0.0
    for row in rows:
        results = retriever.search(row["query_text"], top_k=top_k)
        retrieved_ids = [item["chunk_id"] for item in results]
        relevant_ids = set(row.get("relevant_chunk_ids", []))
        hits = sum(1 for chunk_id in retrieved_ids if chunk_id in relevant_ids)
        precision_sum += hits / float(top_k)
        mrr_sum += reciprocal_rank(retrieved_ids, relevant_ids)

    num_queries = len(rows)
    return {
        "precision_at_k": precision_sum / float(num_queries),
        "mrr": mrr_sum / float(num_queries),
        "num_queries": num_queries,
    }


def call_openai_chat(api_key, model_name, messages, timeout):
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_body = response.read().decode("utf-8")
    return json.loads(response_body)


def extract_json_from_text(text):
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def generate_grounded_report(query_text, retrieved_chunks, model_name, api_key, timeout=60):
    if not api_key:
        return None

    system_prompt = (
        "You are a wind-turbine blade maintenance assistant. "
        "You will receive retrieved manual excerpts (\"chunks\"). These excerpts may be "
        "generic, training-oriented, or partially synthetic and are NOT guaranteed to be "
        "OEM-authoritative. Treat them as guidance, not unquestionable truth. "
        "Rules: Use ONLY the provided chunks. Do not use outside knowledge. "
        "Do not overclaim. If the chunks do not clearly support a procedure or threshold, "
        "say so and recommend safer next steps (inspection, measurements, documentation, "
        "consult OEM documentation). "
        "Every concrete claim or recommendation must cite at least one chunk id in square "
        "brackets, e.g. [manuals::...::chunk_000123]. If you cannot cite it, do not say it. "
        "If chunks conflict, mention the conflict and choose the safer action. "
        "Prefer actionable guidance from chunks tagged \"procedure\", \"checklist\", or "
        "\"troubleshooting\", but do not ignore safety or inspection content. "
        "If any safety or stop-criteria appear, include them. "
        "Do not quote more than 1-2 sentences from any chunk. "
        "Return strict JSON with keys summary, recommended_action, evidence."
    )
    user_payload = {
        "query": query_text,
        "chunks": retrieved_chunks,
        "output_schema": {
            "summary": "string",
            "recommended_action": "string",
            "evidence": [{"chunk_id": "string", "quote_or_reason": "string"}],
        },
    }
    user_prompt = json.dumps(user_payload, ensure_ascii=True, indent=2)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = call_openai_chat(api_key, model_name, messages, timeout=timeout)
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError("Model response was empty.")
    return extract_json_from_text(content)
