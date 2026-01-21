#!/usr/bin/env python3
"""
Test different slicing mechanisms to find the correct one.
Generates comparison images with different stride configurations.
"""

import json
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import re

# Slice configurations to test
# Actual image size: 5280 x 2970 (confirmed)
# Grid: 2 rows x 5 cols
# Testing A vs D to determine correct Y stride for row=1
SLICE_CONFIGS = {
    "A_stride_1024x1024": {
        "stride_x": 1024,
        "stride_y": 1024,
        "description": "row1 starts at y=1024"
    },
    "D_stride_1024x1946": {
        "stride_x": 1024,
        "stride_y": 1946,  # row 1 at 2970-1024=1946
        "description": "row1 starts at y=1946 (edge)"
    },
}

CATEGORY_COLORS = {
    0: (255, 0, 0),      # VG;MT - Red
    1: (0, 255, 0),      # LE;ER - Green
    2: (0, 0, 255),      # LR;DA - Blue
    3: (255, 255, 0),    # LE;CR - Yellow
    4: (255, 0, 255),    # SF;PO - Magenta
}

CATEGORY_NAMES = {
    0: "VG;MT", 1: "LE;ER", 2: "LR;DA", 3: "LE;CR", 4: "SF;PO",
}


def parse_slice_filename(filename):
    match = re.match(r'^(DJI_\d+)_(\d+)_(\d+)\.JPG$', filename, re.IGNORECASE)
    if match:
        return (match.group(1), int(match.group(2)), int(match.group(3)))
    return None


def slice_to_original_coords_with_config(bbox, row, col, config_name):
    """Convert slice coords to original coords using given config."""
    config = SLICE_CONFIGS[config_name]
    stride_x = config["stride_x"]
    stride_y = config["stride_y"]

    x, y, w, h = bbox
    x_offset = col * stride_x
    y_offset = row * stride_y

    return [x + x_offset, y + y_offset, w, h]


def load_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        return json.load(f)


def get_specific_annotations(annotation_dir, target_slices):
    """Get specific annotations by slice filename."""
    samples = []

    for ann_file in annotation_dir.glob("*1024-s.json"):
        data = load_annotations(ann_file)
        id_to_filename = {img['id']: img['file_name'] for img in data['images']}

        for ann in data['annotations']:
            image_id = ann['image_id']
            filename = id_to_filename.get(image_id)
            if not filename or filename not in target_slices:
                continue

            parsed = parse_slice_filename(filename)
            if not parsed:
                continue

            original_name, row, col = parsed
            samples.append({
                'original_name': original_name,
                'row': row,
                'col': col,
                'bbox': ann['bbox'],
                'category_id': ann['category_id'],
                'slice_name': filename,
            })

    return samples


def get_sample_annotations(annotation_dir, num_samples=3, require_row1=False):
    """Get sample annotations from different original images."""
    samples = []
    seen_originals = set()

    for ann_file in annotation_dir.glob("*1024-s.json"):
        data = load_annotations(ann_file)
        id_to_filename = {img['id']: img['file_name'] for img in data['images']}

        for ann in data['annotations']:
            image_id = ann['image_id']
            filename = id_to_filename.get(image_id)
            if not filename:
                continue

            parsed = parse_slice_filename(filename)
            if not parsed:
                continue

            original_name, row, col = parsed

            # If we need row=1 samples, skip row=0
            if require_row1 and row != 1:
                continue

            # Skip if we already have a sample from this original image
            if original_name in seen_originals:
                continue

            seen_originals.add(original_name)
            samples.append({
                'original_name': original_name,
                'row': row,
                'col': col,
                'bbox': ann['bbox'],
                'category_id': ann['category_id'],
                'slice_name': filename,
            })

            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    return samples


def find_original_image(original_name, image_dirs):
    for dir_path in image_dirs:
        for ext in ['.JPG', '.jpg']:
            path = Path(dir_path) / f"{original_name}{ext}"
            if path.exists():
                return path
    return None


def create_comparison_image(image_path, sample, output_dir):
    """Create comparison image showing bbox position with different configs."""
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Print image dimensions
    print(f"Image: {sample['original_name']}, Size: {img.size}")
    print(f"Slice: {sample['slice_name']} (row={sample['row']}, col={sample['col']})")
    print(f"Original bbox in slice: {sample['bbox']}")
    print(f"Category: {CATEGORY_NAMES.get(sample['category_id'])}")

    for config_name in SLICE_CONFIGS:
        config = SLICE_CONFIGS[config_name]

        # Create copy
        test_img = img.copy()
        draw = ImageDraw.Draw(test_img)

        # Calculate bbox position with this config
        bbox = slice_to_original_coords_with_config(
            sample['bbox'], sample['row'], sample['col'], config_name
        )
        x, y, w, h = bbox

        color = CATEGORY_COLORS.get(sample['category_id'], (255, 255, 255))

        # Draw bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=8)

        # Draw config label
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        except:
            font = ImageFont.load_default()

        label = f"{config_name}\n{config['description']}\nBBox: ({int(x)}, {int(y)})"
        draw.text((50, 50), label, fill=(255, 255, 0), font=font)

        # Also draw slice grid for reference
        stride_x = config["stride_x"]
        stride_y = config["stride_y"]

        # Draw grid lines (lighter)
        for c in range(5):
            gx = c * stride_x
            draw.line([(gx, 0), (gx, img.height)], fill=(128, 128, 128), width=2)
            draw.line([(gx + 1024, 0), (gx + 1024, img.height)], fill=(200, 200, 200), width=1)
        for r in range(2):
            gy = r * stride_y
            draw.line([(0, gy), (img.width, gy)], fill=(128, 128, 128), width=2)
            draw.line([(0, gy + 1024), (img.width, gy + 1024)], fill=(200, 200, 200), width=1)

        # Save (include row/col in filename to distinguish)
        output_path = output_dir / f"{sample['original_name']}_r{sample['row']}_c{sample['col']}_{config_name}.jpg"
        # Resize to make file smaller
        test_img = test_img.resize((1320, 742), Image.Resampling.LANCZOS)
        test_img.save(output_path, quality=90)
        print(f"  Saved: {output_path.name}")


def main():
    project_root = Path(__file__).parent
    annotation_dir = project_root / "DTU-annotations" / "annotations"
    image_base = project_root / "DTU - Drone inspection images of wind turbine" / "DTU - Drone inspection images of wind turbine"
    output_dir = project_root / "slicing_test"
    output_dir.mkdir(exist_ok=True)

    image_dirs = [
        image_base / "Nordtank 2017",
        image_base / "Nordtank 2018",
    ]

    # Get specific large annotations that should be clearly visible
    # DJI_0752_1_2: LE;ER erosion, bbox=[277,85,378,419], area=159179
    # DJI_0697_1_2: SF;PO paint-off, bbox=[266,623,405,400], area=162000
    target_slices = ['DJI_0752_1_2.JPG', 'DJI_0697_1_2.JPG']
    samples = get_specific_annotations(annotation_dir, target_slices)

    print(f"Testing {len(samples)} sample annotations with {len(SLICE_CONFIGS)} slicing configs\n")

    for sample in samples:
        image_path = find_original_image(sample['original_name'], image_dirs)
        if image_path:
            print(f"\n{'='*60}")
            create_comparison_image(image_path, sample, output_dir)

    print(f"\n\nGenerated {len(samples) * len(SLICE_CONFIGS)} test images in {output_dir}")
    print("Please check which configuration has the bounding box correctly on the fault.")


if __name__ == "__main__":
    main()
