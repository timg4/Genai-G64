from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image


@dataclass(frozen=True)
class Tile:
    path: Path
    row: int
    col: int
    x: int
    y: int
    size: int


def tile_image(
    image_path: Path,
    output_dir: Path,
    tile_size: int = 1024,
    rows: int = 3,
    cols: int = 3,
    overlap: float = 0.2,
) -> List[Tile]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles: List[Tile] = []
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        width, height = im.size
        step = max(1, int(tile_size * (1.0 - overlap)))
        xs = [min(i * step, max(0, width - tile_size)) for i in range(cols)]
        ys = [min(i * step, max(0, height - tile_size)) for i in range(rows)]

        for row_idx, y in enumerate(ys):
            for col_idx, x in enumerate(xs):
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)
                patch = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                crop = im.crop((x, y, x2, y2))
                patch.paste(crop, (0, 0))
                out_name = f"{image_path.stem}_tile_{row_idx}_{col_idx}.png"
                out_path = output_dir / out_name
                patch.save(out_path)
                tiles.append(Tile(out_path, row_idx, col_idx, x, y, tile_size))
    return tiles
