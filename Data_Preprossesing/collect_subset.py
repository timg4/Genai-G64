import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def read_subset_list(path: Path) -> List[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def build_image_index(dirs: Iterable[Path]) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    for root in dirs:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                index[p.name.lower()].append(p)
    return index


def build_annotation_index(
    annotations_files: Iterable[Path],
) -> Tuple[Dict[str, dict], Dict[int, List[dict]], Dict[int, str], List[str]]:
    image_by_name: Dict[str, dict] = {}
    ann_by_image_id: Dict[int, List[dict]] = defaultdict(list)
    category_names: Dict[int, str] = {}
    source_order: List[str] = []

    for ann_path in annotations_files:
        data = load_json(ann_path)
        source_order.append(ann_path.name)

        if not category_names:
            for c in data.get("categories", []):
                if "id" in c:
                    category_names[int(c["id"])] = c.get("name", "unknown")

        for img in data.get("images", []):
            file_name = img.get("file_name")
            if not file_name:
                continue
            key = Path(file_name).name.lower()
            if key not in image_by_name:
                image_by_name[key] = img

        for ann in data.get("annotations", []):
            image_id = ann.get("image_id")
            if image_id is None:
                continue
            ann_by_image_id[int(image_id)].append(ann)

    return image_by_name, ann_by_image_id, category_names, source_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect subset images and matching annotations into a single folder."
    )
    parser.add_argument(
        "--subset-list",
        required=True,
        help="Path to a text file with image file names (one per line).",
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        required=True,
        help="One or more COCO-style annotation files.",
    )
    parser.add_argument(
        "--images-dirs",
        nargs="+",
        required=True,
        help="One or more directories containing images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for subset images and annotations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset_list = Path(args.subset_list).expanduser().resolve()
    annotations_files = [Path(p).expanduser().resolve() for p in args.annotations]
    images_dirs = [Path(p).expanduser().resolve() for p in args.images_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    requested_names = read_subset_list(subset_list)
    index = build_image_index(images_dirs)
    image_by_name, ann_by_image_id, category_names, sources = build_annotation_index(
        annotations_files
    )

    new_images = []
    new_annotations = []
    next_image_id = 0
    next_ann_id = 0

    missing_images = []
    missing_annotations = []
    duplicate_sources = []
    copied = 0
    category_counts: Counter = Counter()

    for name in requested_names:
        key = Path(name).name.lower()

        img_entry = image_by_name.get(key)
        if not img_entry:
            missing_annotations.append(name)
        else:
            image_id = img_entry.get("id")
            img_anns = ann_by_image_id.get(int(image_id), [])

            new_images.append(
                {
                    "id": next_image_id,
                    "file_name": img_entry.get("file_name", name),
                    "width": img_entry.get("width"),
                    "height": img_entry.get("height"),
                    "depth": img_entry.get("depth"),
                    "folder": None,
                    "path": None,
                }
            )
            for ann in img_anns:
                cat_id = ann.get("category_id")
                if cat_id is not None:
                    category_counts[int(cat_id)] += 1
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_image_id
                new_annotations.append(new_ann)
                next_ann_id += 1
            next_image_id += 1

        candidates = index.get(key, [])
        if not candidates:
            missing_images.append(name)
            continue
        if len(candidates) > 1:
            duplicate_sources.append(name)

        src = candidates[0]
        dst = images_out / src.name
        shutil.copy2(src, dst)
        copied += 1

    out = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": [
            {"id": k, "name": v} for k, v in sorted(category_names.items())
        ],
    }
    save_json(output_dir / "annotations_subset.json", out)

    summary_lines = [
        "Subset collection summary",
        f"- Requested files: {len(requested_names)}",
        f"- Images copied: {copied}",
        f"- Images with annotations: {len(new_images)}",
        f"- Annotations kept: {len(new_annotations)}",
        f"- Annotation sources: {', '.join(sources)}",
        f"- Missing image files: {len(missing_images)}",
        f"- Missing annotations: {len(missing_annotations)}",
        f"- Duplicate image candidates: {len(duplicate_sources)}",
        "",
        "Category distribution:",
    ]
    if category_counts:
        for cat_id, count in sorted(category_counts.items()):
            name = category_names.get(cat_id, "unknown")
            summary_lines.append(f"- {cat_id}:{name} -> {count}")
    else:
        summary_lines.append("- (no categories found)")

    if missing_images:
        summary_lines.append("")
        summary_lines.append("Missing image files:")
        summary_lines.extend(f"- {n}" for n in missing_images)

    if missing_annotations:
        summary_lines.append("")
        summary_lines.append("Missing annotations:")
        summary_lines.extend(f"- {n}" for n in missing_annotations)

    if duplicate_sources:
        summary_lines.append("")
        summary_lines.append("Duplicate image candidates (first used):")
        summary_lines.extend(f"- {n}" for n in duplicate_sources)

    (output_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
