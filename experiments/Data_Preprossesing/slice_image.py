import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import json
from PIL import Image


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def iter_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def compute_grid(width: int, height: int, tile: int, stride: int) -> List[Tuple[int, int]]:
    xs = list(range(0, max(1, width - tile + 1), stride))
    ys = list(range(0, max(1, height - tile + 1), stride))
    if xs and xs[-1] + tile < width:
        xs.append(width - tile)
    if ys and ys[-1] + tile < height:
        ys.append(height - tile)
    return [(x, y) for y in ys for x in xs]


def compute_non_overlapping_positions(width: int, height: int, tile: int) -> Tuple[List[int], List[int]]:
    if tile <= 0:
        return [], []
    xs = list(range(0, max(1, width - tile + 1), tile))
    ys = list(range(0, max(1, height - tile + 1), tile))
    return xs, ys


def compute_fixed_positions(
    width: int,
    height: int,
    tile: int,
    rows: int,
    cols: int,
    mode: str,
) -> Tuple[List[int], List[int]]:
    if rows <= 0 or cols <= 0:
        return [], []
    if mode == "top_left":
        xs = [i * tile for i in range(cols)]
        ys = [i * tile for i in range(rows)]
        return xs, ys
    if cols == 1:
        xs = [0]
    else:
        step_x = (width - tile) / (cols - 1)
        xs = [int(step_x * i) for i in range(cols)]
        xs[-1] = max(0, width - tile)
    if rows == 1:
        ys = [0]
    else:
        step_y = (height - tile) / (rows - 1)
        ys = [int(step_y * i) for i in range(rows)]
        ys[-1] = max(0, height - tile)
    return xs, ys


def build_output_path(
    image_path: Path,
    output_dir: Path,
    input_root: Optional[Path],
    index: int,
    name_style: str,
    pad_width: int,
    preserve_dirs: bool,
    row: Optional[int] = None,
    col: Optional[int] = None,
    grid_order: str = "col_row",
) -> Path:
    if name_style == "index_zero_pad":
        suffix = f"_{index:0{pad_width}d}"
    elif name_style == "grid_rc":
        if row is None or col is None:
            raise ValueError("grid_rc requires row and col indices")
        if grid_order == "row_col":
            suffix = f"_{row}_{col}"
        else:
            suffix = f"_{col}_{row}"
    else:
        suffix = f"_{index}"

    out_name = f"{image_path.stem}{suffix}{image_path.suffix}"
    if preserve_dirs and input_root is not None:
        rel_parent = image_path.parent.relative_to(input_root)
        return output_dir / rel_parent / out_name
    return output_dir / out_name


def build_image_index(images_dir: Path) -> Tuple[dict, dict]:
    rel_map = {}
    basename_map = {}
    for p in images_dir.rglob("*"):
        if not p.is_file():
            continue
        rel_key = p.relative_to(images_dir).as_posix().lower()
        rel_map[rel_key] = p
        basename_map.setdefault(p.name.lower(), []).append(p)
    return rel_map, basename_map


def resolve_image_path(file_name: str, rel_map: dict, basename_map: dict) -> Optional[Path]:
    rel_key = Path(file_name).as_posix().lower()
    p = rel_map.get(rel_key)
    if p:
        return p
    base_key = Path(file_name).name.lower()
    candidates = basename_map.get(base_key, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def slice_image(
    image_path: Path,
    output_dir: Path,
    tile: int,
    stride: int,
    pad: bool,
    input_root: Optional[Path],
    name_style: str,
    pad_width: int,
    preserve_dirs: bool,
    grid_rows: Optional[int],
    grid_cols: Optional[int],
    grid_order: str,
    grid_index_base: int,
    grid_mode: str,
) -> int:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        width, height = im.size

        fixed_grid = bool(grid_rows and grid_cols)
        if fixed_grid:
            xs, ys = compute_fixed_positions(
                width, height, tile, grid_rows, grid_cols, grid_mode
            )
            grid = [(x, y) for y in ys for x in xs]
        else:
            grid = compute_grid(width, height, tile, stride)
            xs = sorted({x for x, _ in grid})
            ys = sorted({y for _, y in grid})
        base = image_path.stem
        count = 0

        if fixed_grid:
            idx = 0
            for row_idx, y in enumerate(ys):
                for col_idx, x in enumerate(xs):
                    idx += 1
                    x2 = x + tile
                    y2 = y + tile
                    if x2 <= width and y2 <= height:
                        crop = im.crop((x, y, x2, y2))
                    elif pad:
                        crop = Image.new("RGB", (tile, tile), (0, 0, 0))
                        part = im.crop((x, y, min(x2, width), min(y2, height)))
                        crop.paste(part, (0, 0))
                    else:
                        continue

                    out_path = build_output_path(
                        image_path=image_path,
                        output_dir=output_dir,
                        input_root=input_root,
                        index=idx,
                        name_style=name_style,
                        pad_width=pad_width,
                        preserve_dirs=preserve_dirs,
                        row=row_idx + grid_index_base,
                        col=col_idx + grid_index_base,
                        grid_order=grid_order,
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(out_path)
                    count += 1
        else:
            for idx, (x, y) in enumerate(grid, start=1):
                x2 = x + tile
                y2 = y + tile
                if x2 <= width and y2 <= height:
                    crop = im.crop((x, y, x2, y2))
                elif pad:
                    crop = Image.new("RGB", (tile, tile), (0, 0, 0))
                    part = im.crop((x, y, min(x2, width), min(y2, height)))
                    crop.paste(part, (0, 0))
                else:
                    continue

                out_path = build_output_path(
                    image_path=image_path,
                    output_dir=output_dir,
                    input_root=input_root,
                    index=idx,
                    name_style=name_style,
                    pad_width=pad_width,
                    preserve_dirs=preserve_dirs,
                    row=ys.index(y) + grid_index_base,
                    col=xs.index(x) + grid_index_base,
                    grid_order=grid_order,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(out_path)
                count += 1

        return count


def slice_with_annotations(
    annotations_in: Path,
    annotations_out: Path,
    images_dir: Path,
    output_dir: Path,
    tile: int,
    grid_rows: Optional[int],
    grid_cols: Optional[int],
    grid_order: str,
    grid_index_base: int,
    grid_mode: str,
    pad: bool,
    preserve_dirs: bool,
) -> None:
    data = load_json(annotations_in)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    ann_by_image = {}
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        ann_by_image.setdefault(image_id, []).append(ann)

    rel_map, basename_map = build_image_index(images_dir)
    new_images = []
    new_annotations = []
    next_image_id = 0
    next_ann_id = 0

    for img in images:
        image_id = img.get("id")
        file_name = img.get("file_name")
        if image_id is None or not file_name:
            continue

        image_path = resolve_image_path(file_name, rel_map, basename_map)
        if not image_path:
            continue

        with Image.open(image_path) as im:
            im = im.convert("RGB")
            width, height = im.size

            if grid_rows and grid_cols:
                xs, ys = compute_fixed_positions(
                    width, height, tile, grid_rows, grid_cols, grid_mode
                )
            else:
                xs, ys = compute_non_overlapping_positions(width, height, tile)

            for row_idx, y in enumerate(ys):
                for col_idx, x in enumerate(xs):
                    x2 = x + tile
                    y2 = y + tile
                    if x2 > width or y2 > height:
                        if not pad:
                            continue

                    clipped = []
                    for ann in ann_by_image.get(image_id, []):
                        bbox = ann.get("bbox")
                        if not bbox or len(bbox) != 4:
                            continue
                        bx, by, bw, bh = bbox
                        bx2 = bx + bw
                        by2 = by + bh
                        ix1 = max(bx, x)
                        iy1 = max(by, y)
                        ix2 = min(bx2, x2)
                        iy2 = min(by2, y2)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        nbx = ix1 - x
                        nby = iy1 - y
                        nbw = ix2 - ix1
                        nbh = iy2 - iy1
                        new_ann = dict(ann)
                        new_ann["bbox"] = [nbx, nby, nbw, nbh]
                        new_ann["area"] = float(nbw * nbh)
                        clipped.append(new_ann)

                    if not clipped:
                        continue

                    if x2 <= width and y2 <= height:
                        patch = im.crop((x, y, x2, y2))
                    else:
                        patch = Image.new("RGB", (tile, tile), (0, 0, 0))
                        part = im.crop((x, y, min(x2, width), min(y2, height)))
                        patch.paste(part, (0, 0))
                    row_out = row_idx + grid_index_base
                    col_out = col_idx + grid_index_base
                    if grid_order == "row_col":
                        suffix = f"_{row_out}_{col_out}"
                    else:
                        suffix = f"_{col_out}_{row_out}"

                    out_name = f"{image_path.stem}{suffix}{image_path.suffix}"
                    if preserve_dirs:
                        rel_parent = image_path.parent.relative_to(images_dir)
                        out_path = output_dir / rel_parent / out_name
                    else:
                        out_path = output_dir / out_name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    patch.save(out_path)

                    new_images.append(
                        {
                            "id": next_image_id,
                            "file_name": out_name,
                            "width": tile,
                            "height": tile,
                            "depth": 3,
                            "folder": None,
                            "path": None,
                        }
                    )
                    for ann in clipped:
                        ann["id"] = next_ann_id
                        ann["image_id"] = next_image_id
                        new_annotations.append(ann)
                        next_ann_id += 1
                    next_image_id += 1

    out = {"images": new_images, "annotations": new_annotations, "categories": categories}
    save_json(annotations_out, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice a high-resolution image into fixed-size tiles."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an image file or a directory of images",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save image tiles",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size in pixels (square)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1024,
        help="Stride in pixels (use smaller than tile-size for overlap)",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=3,
        help="Force a fixed number of rows (overrides stride-based grid)",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=5,
        help="Force a fixed number of cols (overrides stride-based grid)",
    )
    parser.add_argument(
        "--grid-mode",
        choices=["spread", "top_left"],
        default="top_left",
        help="Grid placement mode for fixed rows/cols",
    )
    parser.add_argument(
        "--name-style",
        choices=["index", "index_zero_pad", "grid_rc"],
        default="grid_rc",
        help="Naming style for tiles (index -> _1, index_zero_pad -> _0001, grid_rc -> _<col>_<row>)",
    )
    parser.add_argument(
        "--grid-order",
        choices=["col_row", "row_col"],
        default="row_col",
        help="Order for grid suffix when slicing with annotations",
    )
    parser.add_argument(
        "--grid-index-base",
        type=int,
        default=0,
        help="Index base for grid suffix when slicing with annotations (0 or 1)",
    )
    parser.add_argument(
        "--pad-width",
        type=int,
        default=4,
        help="Zero padding width when using index_zero_pad",
    )
    parser.add_argument(
        "--preserve-dirs",
        action="store_true",
        help="Preserve input subfolder structure under output-dir",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        default=True,
        help="Pad partial tiles at the borders instead of skipping them",
    )
    parser.add_argument(
        "--no-pad",
        action="store_false",
        dest="pad",
        help="Disable padding for partial tiles",
    )
    parser.add_argument(
        "--annotations-in",
        help="COCO-style annotations for HR images (enables patch-wise slicing)",
    )
    parser.add_argument(
        "--annotations-out",
        help="Output annotations for sliced patches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.annotations_in:
        if not args.annotations_out:
            raise SystemExit("--annotations-out is required when --annotations-in is set")
        annotations_in = Path(args.annotations_in).expanduser().resolve()
        annotations_out = Path(args.annotations_out).expanduser().resolve()
        slice_with_annotations(
            annotations_in=annotations_in,
            annotations_out=annotations_out,
            images_dir=input_path if input_path.is_dir() else input_path.parent,
            output_dir=output_dir,
            tile=args.tile_size,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            grid_order=args.grid_order,
            grid_index_base=args.grid_index_base,
            grid_mode=args.grid_mode,
            pad=args.pad,
            preserve_dirs=args.preserve_dirs,
        )
        print(f"Saved sliced annotations to {annotations_out}")
        return

    input_root = input_path if input_path.is_dir() else None
    total = 0
    for image_path in iter_images(input_path):
        total += slice_image(
            image_path=image_path,
            output_dir=output_dir,
            tile=args.tile_size,
            stride=args.stride,
            pad=args.pad,
            input_root=input_root,
            name_style=args.name_style,
            pad_width=args.pad_width,
            preserve_dirs=args.preserve_dirs,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            grid_order=args.grid_order,
            grid_index_base=args.grid_index_base,
            grid_mode=args.grid_mode,
        )
        #print(f"Processed {image_path}")

    print(f"Saved {total} tiles to {output_dir}")


if __name__ == "__main__":
    main()
