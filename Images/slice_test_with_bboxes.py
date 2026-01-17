#!/usr/bin/env python3
"""
Create 1024x1024 slices from test HR images and draw HR annotations onto each slice.

Uses A-stride slicing (1024 x 1024 grid stride) confirmed in test_slicing.py.
Outputs annotated slices in a flat directory with names like DJI_XXXX_row_col.JPG.
"""

import json
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# Configuration (A-stride)
SLICE_SIZE = 1024
STRIDE_X = 1024
STRIDE_Y = 1024

# Category colors (RGB)
CATEGORY_COLORS = {
    0: (255, 0, 0),      # VG;MT - Red
    1: (0, 255, 0),      # LE;ER - Green
    2: (0, 0, 255),      # LR;DA - Blue
    3: (255, 255, 0),    # LE;CR - Yellow
    4: (255, 0, 255),    # SF;PO - Magenta
}

CATEGORY_NAMES = {
    0: "VG;MT",
    1: "LE;ER",
    2: "LR;DA",
    3: "LE;CR",
    4: "SF;PO",
}


def load_annotations(annotation_file):
    with open(annotation_file, "r") as f:
        return json.load(f)


def find_original_image(original_name, image_dirs):
    for dir_path in image_dirs:
        for ext in [".JPG", ".jpg", ".jpeg", ".JPEG"]:
            path = Path(dir_path) / f"{original_name}{ext}"
            if path.exists():
                return path
    return None


def get_font(font_size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except Exception:
            return ImageFont.load_default()


def annotations_for_slice(annotations, x0, y0, slice_size):
    sliced = []
    x1_slice = x0 + slice_size
    y1_slice = y0 + slice_size

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x1 = max(x, x0)
        y1 = max(y, y0)
        x2 = min(x + w, x1_slice)
        y2 = min(y + h, y1_slice)

        if x2 <= x1 or y2 <= y1:
            continue

        sliced.append(
            {
                "category_id": ann["category_id"],
                "bbox": [x1 - x0, y1 - y0, x2 - x1, y2 - y1],
            }
        )

    return sliced


def draw_annotations_on_slice(img, annotations, line_width=3, font_size=20):
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        category_id = ann["category_id"]
        color = CATEGORY_COLORS.get(category_id, (255, 255, 255))

        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = CATEGORY_NAMES.get(category_id, f"Cat {category_id}")
        text_bbox = draw.textbbox((x1, y1 - font_size - 4), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - font_size - 4), label, fill=(255, 255, 255), font=font)

    return img


def slice_image_with_annotations(img, annotations, output_dir, original_name):
    cols = (img.width - SLICE_SIZE) // STRIDE_X + 1
    rows = (img.height - SLICE_SIZE) // STRIDE_Y + 1

    saved = 0
    drawn_boxes = 0

    for row in range(rows):
        for col in range(cols):
            x0 = col * STRIDE_X
            y0 = row * STRIDE_Y
            crop = img.crop((x0, y0, x0 + SLICE_SIZE, y0 + SLICE_SIZE))

            slice_anns = annotations_for_slice(annotations, x0, y0, SLICE_SIZE)
            drawn_boxes += len(slice_anns)

            if not slice_anns:
                continue

            # Save raw slice (no boxes)
            raw_name = f"{original_name}_{row}_{col}_raw.JPG"
            raw_path = output_dir / raw_name
            crop.save(raw_path, quality=95)
            saved += 1

            # Save annotated slice (with boxes)
            annotated = draw_annotations_on_slice(crop.copy(), slice_anns)
            ann_name = f"{original_name}_{row}_{col}_ann.JPG"
            ann_path = output_dir / ann_name
            annotated.save(ann_path, quality=95)
            saved += 1

    return saved, drawn_boxes


def main():
    project_root = Path(__file__).parent
    annotation_file = project_root / "DTU-annotations" / "annotations" / "test-HR.json"
    image_base = (
        project_root
        / "DTU - Drone inspection images of wind turbine"
        / "DTU - Drone inspection images of wind turbine"
    )
    output_dir = project_root / "annotated_slices_test"
    output_dir.mkdir(exist_ok=True)

    image_dirs = [
        image_base / "Nordtank 2017",
        image_base / "Nordtank 2018",
    ]

    data = load_annotations(annotation_file)
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    grouped = defaultdict(list)
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        filename = id_to_filename.get(image_id)
        if not filename:
            continue
        original_name = Path(filename).stem
        grouped[original_name].append(ann)

    processed = 0
    saved_slices = 0
    drawn_boxes = 0
    not_found = []

    for original_name, annotations in grouped.items():
        image_path = find_original_image(original_name, image_dirs)
        if not image_path:
            not_found.append(original_name)
            continue

        try:
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        saved, drawn = slice_image_with_annotations(
            img, annotations, output_dir, original_name
        )
        processed += 1
        saved_slices += saved
        drawn_boxes += drawn

    print("=== Summary ===")
    print(f"Images processed: {processed}")
    print(f"Slices saved: {saved_slices}")
    print(f"Boxes drawn: {drawn_boxes}")
    if not_found:
        print(f"Images not found: {len(not_found)}")


if __name__ == "__main__":
    main()
