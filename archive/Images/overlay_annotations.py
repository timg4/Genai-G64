#!/usr/bin/env python3
"""
Script to overlay bounding box annotations from sliced images onto original high-resolution images.

DTU Wind Turbine Blade Inspection Dataset (from Gohar et al. paper):
- Original images: 5280 x 2890 pixels
- Sliced images: 1024 x 1024 (non-overlapping patches)
- Grid: 2 rows x 5 columns
- Naming convention: DJI_XXXX_row_col.JPG (sliced), DJI_XXXX.JPG (original)

Categories:
  0: VG;MT (Missing Teeth on Vortex Generator)
  1: LE;ER (Leading Edge Erosion)
  2: LR;DA (Lightning Receptor Damage)
  3: LE;CR (Leading Edge Crack)
  4: SF;PO (Surface Paint-Off)
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# Configuration (confirmed by testing)
ORIGINAL_WIDTH = 5280
ORIGINAL_HEIGHT = 2970
SLICE_SIZE = 1024

# Simple non-overlapping grid with 1024 stride
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

CATEGORY_FULL_NAMES = {
    0: "VG;MT (Missing Teeth)",
    1: "LE;ER (Erosion)",
    2: "LR;DA (Lightning Damage)",
    3: "LE;CR (Crack)",
    4: "SF;PO (Paint-Off)",
}


def parse_slice_filename(filename):
    """
    Parse sliced image filename to extract original name, row, and column.
    Format: DJI_XXXX_row_col.JPG
    Returns: (original_name, row, col) or None if not a slice
    """
    match = re.match(r'^(DJI_\d+)_(\d+)_(\d+)\.JPG$', filename, re.IGNORECASE)
    if match:
        original_name = match.group(1)
        row = int(match.group(2))
        col = int(match.group(3))
        return (original_name, row, col)
    return None


def slice_to_original_coords(bbox, row, col):
    """
    Convert bounding box coordinates from slice space to original image space.

    Args:
        bbox: [x, y, width, height] in slice coordinates
        row: slice row index (0 or 1)
        col: slice column index (0-4)

    Returns:
        [x, y, width, height] in original image coordinates
    """
    x, y, w, h = bbox

    # Calculate the top-left corner of this slice in the original image
    x_offset = col * STRIDE_X
    y_offset = row * STRIDE_Y

    # Transform coordinates
    orig_x = x + x_offset
    orig_y = y + y_offset

    return [orig_x, orig_y, w, h]


def load_annotations(annotation_file):
    """Load annotations from COCO-format JSON file."""
    with open(annotation_file, 'r') as f:
        return json.load(f)


def group_annotations_by_original(data):
    """
    Group sliced annotations by their original image.
    Returns: dict: {original_name: [annotation_dict, ...]}
    """
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    grouped = defaultdict(list)

    for ann in data['annotations']:
        image_id = ann['image_id']
        filename = id_to_filename.get(image_id)

        if not filename:
            continue

        parsed = parse_slice_filename(filename)
        if not parsed:
            continue

        original_name, row, col = parsed
        bbox = ann['bbox']
        category_id = ann['category_id']

        orig_bbox = slice_to_original_coords(bbox, row, col)

        grouped[original_name].append({
            'category_id': category_id,
            'bbox': orig_bbox,
            'slice_name': filename,
            'slice_row': row,
            'slice_col': col,
            'original_bbox': bbox,
        })

    return grouped


def draw_annotations_on_image(img, annotations, line_width=3, font_size=24):
    """
    Draw bounding boxes on an image.
    Returns: modified image
    """
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

    for ann in annotations:
        x, y, w, h = ann['bbox']
        category_id = ann['category_id']
        color = CATEGORY_COLORS.get(category_id, (255, 255, 255))

        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)

        # Draw label
        label = CATEGORY_NAMES.get(category_id, f"Cat {category_id}")
        text_bbox = draw.textbbox((x, y - font_size - 5), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x, y - font_size - 5), label, fill=(255, 255, 255), font=font)

    return img


def create_fault_crop(img, annotation, padding=100, target_size=512):
    """
    Create a cropped region around the fault with padding.
    Returns: cropped and scaled image
    """
    x, y, w, h = annotation['bbox']
    category_id = annotation['category_id']

    # Add padding around the bounding box
    x1 = max(0, int(x - padding))
    y1 = max(0, int(y - padding))
    x2 = min(img.width, int(x + w + padding))
    y2 = min(img.height, int(y + h + padding))

    # Crop the region
    crop = img.crop((x1, y1, x2, y2))

    # Draw the bounding box on the crop (adjusted coordinates)
    draw = ImageDraw.Draw(crop)
    color = CATEGORY_COLORS.get(category_id, (255, 255, 255))
    box_x = x - x1
    box_y = y - y1
    draw.rectangle([box_x, box_y, box_x + w, box_y + h], outline=color, width=4)

    # Add label
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()

    label = CATEGORY_FULL_NAMES.get(category_id, f"Category {category_id}")
    draw.text((10, 10), label, fill=color, font=font)

    # Scale to target size while maintaining aspect ratio
    ratio = min(target_size / crop.width, target_size / crop.height)
    new_size = (int(crop.width * ratio), int(crop.height * ratio))
    crop = crop.resize(new_size, Image.Resampling.LANCZOS)

    return crop


def find_original_image(original_name, image_dirs):
    """Find the original image file in the given directories."""
    for dir_path in image_dirs:
        for ext in ['.JPG', '.jpg', '.jpeg', '.JPEG']:
            path = Path(dir_path) / f"{original_name}{ext}"
            if path.exists():
                return path
    return None


def process_image(image_path, annotations, output_dir, original_name):
    """
    Process a single image:
    1. Draw annotations on full image
    2. Create cropped views of each fault
    """
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return False

    # 1. Save full annotated image
    annotated = img.copy()
    annotated = draw_annotations_on_image(annotated, annotations, line_width=5, font_size=36)
    full_output = output_dir / f"{original_name}_annotated.jpg"
    annotated.save(full_output, quality=95)
    print(f"Saved full image: {full_output.name}")

    # 2. Create cropped views for each annotation
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    for i, ann in enumerate(annotations):
        category_id = ann['category_id']
        cat_name = CATEGORY_NAMES.get(category_id, f"cat{category_id}")

        crop = create_fault_crop(img, ann, padding=150, target_size=600)
        crop_output = crops_dir / f"{original_name}_{cat_name}_{i}.jpg"
        crop.save(crop_output, quality=95)
        print(f"  Saved crop: {crop_output.name}")

    return True


def main():
    project_root = Path(__file__).parent
    annotation_dir = project_root / "DTU-annotations" / "annotations"
    image_base = project_root / "DTU - Drone inspection images of wind turbine" / "DTU - Drone inspection images of wind turbine"
    output_dir = project_root / "annotated_images"

    image_dirs = [
        image_base / "Nordtank 2017",
        image_base / "Nordtank 2018",
    ]

    output_dir.mkdir(exist_ok=True)

    # Load sliced annotations from all files
    annotation_files = list(annotation_dir.glob("*1024-s.json"))

    all_grouped = defaultdict(list)
    for ann_file in annotation_files:
        print(f"Loading annotations from: {ann_file.name}")
        data = load_annotations(ann_file)
        grouped = group_annotations_by_original(data)
        for orig_name, anns in grouped.items():
            all_grouped[orig_name].extend(anns)

    print(f"\nFound annotations for {len(all_grouped)} original images")

    # Statistics
    category_counts = defaultdict(int)
    for anns in all_grouped.values():
        for ann in anns:
            category_counts[ann['category_id']] += 1

    print("\nAnnotation counts by category:")
    for cat_id, count in sorted(category_counts.items()):
        print(f"  {CATEGORY_FULL_NAMES.get(cat_id)}: {count}")

    # Process images
    processed = 0
    failed = []
    not_found = []

    for original_name, annotations in all_grouped.items():
        image_path = find_original_image(original_name, image_dirs)

        if image_path:
            success = process_image(image_path, annotations, output_dir, original_name)
            if success:
                processed += 1
            else:
                failed.append(original_name)
        else:
            not_found.append(original_name)

    print(f"\n=== Summary ===")
    print(f"Processed: {processed} images")
    if failed:
        print(f"Failed (corrupted): {len(failed)} images")
    if not_found:
        print(f"Not found: {len(not_found)} images")


def process_by_category(category_id, max_images=6):
    """
    Process images for a single category, up to max_images.
    Generates both full annotated images and cropped fault views.
    """
    project_root = Path(__file__).parent
    annotation_dir = project_root / "DTU-annotations" / "annotations"
    image_base = project_root / "DTU - Drone inspection images of wind turbine" / "DTU - Drone inspection images of wind turbine"

    cat_name = CATEGORY_NAMES.get(category_id, f"cat{category_id}")
    output_dir = project_root / "annotated_images" / f"category_{cat_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dirs = [
        image_base / "Nordtank 2017",
        image_base / "Nordtank 2018",
    ]

    # Load all annotations
    all_grouped = defaultdict(list)
    for ann_file in annotation_dir.glob("*1024-s.json"):
        data = load_annotations(ann_file)
        grouped = group_annotations_by_original(data)
        for orig_name, anns in grouped.items():
            all_grouped[orig_name].extend(anns)

    # Filter images that have the target category
    images_with_category = []
    for orig_name, anns in all_grouped.items():
        category_anns = [a for a in anns if a['category_id'] == category_id]
        if category_anns:
            images_with_category.append((orig_name, anns))

    print(f"Found {len(images_with_category)} images with {CATEGORY_FULL_NAMES.get(category_id)}")

    # Process images
    processed = 0
    idx = 0
    while processed < max_images and idx < len(images_with_category):
        orig_name, anns = images_with_category[idx]
        idx += 1

        image_path = find_original_image(orig_name, image_dirs)
        if image_path:
            success = process_image(image_path, anns, output_dir, orig_name)
            if success:
                processed += 1

    print(f"\nGenerated outputs for {processed} images with category {cat_name}")
    return processed


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cat_id = int(sys.argv[1])
        max_imgs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        process_by_category(cat_id, max_imgs)
    else:
        main()
