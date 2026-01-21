import argparse
import colorsys
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("bbox_viz")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_color_for_category(category_id: int) -> Tuple[int, int, int]:
    hue = (category_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def clamp_bbox(bbox: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    return x1, y1, x2, y2


def build_image_index(images_dir: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    rel_map: Dict[str, Path] = {}
    basename_map: Dict[str, List[Path]] = defaultdict(list)
    for p in images_dir.rglob("*"):
        if not p.is_file():
            continue
        rel_key = p.relative_to(images_dir).as_posix().lower()
        rel_map[rel_key] = p
        basename_map[p.name.lower()].append(p)
    return rel_map, basename_map


def resolve_image_path(
    file_name: str,
    rel_map: Dict[str, Path],
    basename_map: Dict[str, List[Path]],
) -> Tuple[Optional[Path], Optional[str]]:
    rel_key = Path(file_name).as_posix().lower()
    p = rel_map.get(rel_key)
    if p:
        return p, None

    base_key = Path(file_name).name.lower()
    candidates = basename_map.get(base_key, [])
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        return None, "ambiguous_basename"
    return None, "file_not_found"


def load_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def most_common_category(categories: Iterable[int]) -> Optional[int]:
    counts = Counter(categories)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def largest_remainder_allocation(total: int, counts: Dict[int, int]) -> Dict[int, int]:
    if total <= 0:
        return {k: 0 for k in counts}
    overall = sum(counts.values())
    raw = {k: (v / overall) * total for k, v in counts.items()}
    base = {k: int(v) for k, v in raw.items()}
    remainder = sorted(((raw[k] - base[k], k) for k in raw), reverse=True)
    assigned = sum(base.values())
    for i in range(total - assigned):
        base[remainder[i][1]] += 1
    return base


def build_stratified_sample(
    valid_images: List[Tuple[int, str, Path]],
    ann_by_image: Dict[int, List[dict]],
    sample_count: int,
    seed: int,
    logger: logging.Logger,
) -> List[Tuple[int, str, Path]]:
    if sample_count <= 0:
        return []

    cat_to_images: Dict[int, List[Tuple[int, str, Path]]] = defaultdict(list)
    for img_id, file_name, img_path in valid_images:
        cats = [
            a.get("category_id")
            for a in ann_by_image.get(img_id, [])
            if a.get("category_id") is not None
        ]
        primary = most_common_category(cats)
        if primary is None:
            continue
        cat_to_images[int(primary)].append((img_id, file_name, img_path))

    if not cat_to_images:
        logger.warning("Keine Kategorien fuer Stratified Sampling gefunden.")
        return []

    cat_counts = {k: len(v) for k, v in cat_to_images.items()}
    allocation = largest_remainder_allocation(sample_count, cat_counts)

    rng = __import__("random")
    rng.seed(seed)
    selected: List[Tuple[int, str, Path]] = []
    for cat_id, items in sorted(cat_to_images.items()):
        k = allocation.get(cat_id, 0)
        if k <= 0:
            continue
        pool = list(items)
        rng.shuffle(pool)
        selected.extend(pool[:k])

    if len(selected) > sample_count:
        selected = selected[:sample_count]

    selected_counts = Counter(
        most_common_category(
            [
                a.get("category_id")
                for a in ann_by_image.get(img_id, [])
                if a.get("category_id") is not None
            ]
        )
        for img_id, _, _ in selected
    )
    logger.info(f"Stratified Sample Counts: {dict(selected_counts)}")
    return selected


def draw_annotations(
    annotations_path: Path,
    images_dir: Path,
    output_dir: Path,
    sample_count: int,
    sample_output: Path,
    sample_seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "viz.log")

    if not annotations_path.exists():
        logger.error(f"Annotation-Datei nicht gefunden: {annotations_path}")
        sys.exit(1)
    if not images_dir.exists():
        logger.error(f"Bilderordner nicht gefunden: {images_dir}")
        sys.exit(1)

    data = load_json(annotations_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    logger.info(f"Gefundene Bilder (im JSON): {len(images)}")
    logger.info(f"Gefundene Annotationen: {len(annotations)}")
    logger.info(f"Bilder werden geladen aus: {images_dir}")
    logger.info(f"Output-Ordner: {output_dir}")

    category_name_by_id = {c["id"]: c.get("name", "unknown") for c in categories}
    color_by_category_id = {
        c["id"]: build_color_for_category(int(c["id"])) for c in categories
    }

    ann_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        ann_by_image[image_id].append(ann)

    rel_map, basename_map = build_image_index(images_dir)
    font = load_font(size=18)

    processed = 0
    skipped = 0
    invalid_bbox = 0
    error_counts: Counter = Counter()
    valid_images: List[Tuple[int, str, Path]] = []

    for img in images:
        image_id = img.get("id")
        file_name = img.get("file_name", "")
        if image_id is None or not file_name:
            error_counts["missing_file_name"] += 1
            skipped += 1
            continue

        image_path, reason = resolve_image_path(file_name, rel_map, basename_map)
        if not image_path:
            error_counts[reason or "file_not_found"] += 1
            skipped += 1
            logger.warning(
                f"[SKIP] image_id={image_id} file_name={file_name} reason={reason}"
            )
            continue

        try:
            with Image.open(image_path) as im:
                im = im.convert("RGB")
        except OSError:
            error_counts["cannot_open"] += 1
            skipped += 1
            logger.warning(
                f"[SKIP] image_id={image_id} file_name={file_name} reason=cannot_open"
            )
            continue

        valid_images.append((image_id, file_name, image_path))
        draw = ImageDraw.Draw(im)
        width, height = im.size

        for ann in ann_by_image.get(image_id, []):
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                invalid_bbox += 1
                continue

            x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
            if x2 <= x1 or y2 <= y1:
                invalid_bbox += 1
                continue

            category_id = ann.get("category_id")
            category_name = category_name_by_id.get(category_id, "unknown")
            color = color_by_category_id.get(category_id, (255, 0, 0))

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            label = f"{category_id}:{category_name}"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            pad = 2
            text_x1 = x1
            text_y1 = max(0, y1 - text_h - pad * 2)
            text_x2 = min(width, x1 + text_w + pad * 2)
            text_y2 = text_y1 + text_h + pad * 2
            draw.rectangle([text_x1, text_y1, text_x2, text_y2], fill=color)
            draw.text((text_x1 + pad, text_y1 + pad), label, fill=(0, 0, 0), font=font)

        out_rel = image_path.relative_to(images_dir)
        out_path = output_dir / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ext = out_path.suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png"}:
            out_path = out_path.with_suffix(".png")

        try:
            im.save(out_path)
        except OSError:
            error_counts["save_failed"] += 1
            skipped += 1
            logger.warning(
                f"[SKIP] image_id={image_id} file_name={file_name} reason=save_failed"
            )
            continue

        processed += 1
        if processed % 25 == 0:
            logger.info(f"Fortschritt: {processed}/{len(images)} gespeichert...")

    logger.info(f"Gespeichert: {processed}")
    logger.info(f"Uebersprungen: {skipped}")
    if invalid_bbox:
        logger.info(f"Ungueltige BBoxen: {invalid_bbox}")
        error_counts["invalid_bbox"] += invalid_bbox

    if error_counts:
        logger.info(f"Fehlerklassen: {dict(error_counts)}")
    else:
        logger.info("Fehlerklassen: {}")

    if sample_count > 0:
        sample = build_stratified_sample(
            valid_images, ann_by_image, sample_count, sample_seed, logger
        )
        if sample:
            sample_output.parent.mkdir(parents=True, exist_ok=True)
            rel_paths = [
                str(p.relative_to(images_dir)).replace("\\", "/")
                for _, _, p in sample
            ]
            sample_output.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")
            logger.info(f"Sample-Liste geschrieben: {sample_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw COCO-style annotations on images.")
    parser.add_argument(
        "--annotations",
        default="annotations.json",
        help="Path to COCO-like annotations.json",
    )
    parser.add_argument(
        "--images-dir",
        default="test_images",
        help="Directory containing images",
    )
    parser.add_argument(
        "--output-dir",
        default="viz_output",
        help="Directory to save output visualizations",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=40,
        help="Number of stratified sample images to list (0 to disable)",
    )
    parser.add_argument(
        "--sample-output",
        default="sub_dataset_40.txt",
        help="Output path for sample list (relative to script dir by default)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    annotations_path = (script_dir / args.annotations).resolve()
    images_dir = (script_dir / args.images_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    sample_output = (script_dir / args.sample_output).resolve()

    draw_annotations(
        annotations_path,
        images_dir,
        output_dir,
        sample_count=args.sample_count,
        sample_output=sample_output,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
