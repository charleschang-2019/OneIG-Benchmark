"""Image splitting / black-image detection utilities."""
from __future__ import annotations

import os
from pathlib import Path

import megfile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def is_black_image(image: Image.Image) -> bool:
    """Return True iff every pixel is exactly (0, 0, 0).

    Used to skip the placeholder black tiles that ``text2image.py`` emits
    when generation failed.
    """
    pixels = image.load()
    for i in range(image.width):
        for j in range(image.height):
            if pixels[i, j] != (0, 0, 0):
                return False
    return True


def split_mxn_grid(image_path: str, grid_size: tuple[int, int], cache_dir: str | Path) -> list[str]:
    """Split a grid image into individual tiles, dropping all-black tiles.

    Saves each tile as ``{cache_dir}/{i}.jpg`` and returns the list of paths.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with megfile.smart_open(image_path, "rb") as f:
        grid_image = Image.open(f)
        width, height = grid_image.size
        individual_width = width // grid_size[0]
        individual_height = height // grid_size[1]

        image_list: list[Image.Image] = []
        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                box = (
                    j * individual_width,
                    i * individual_height,
                    (j + 1) * individual_width,
                    (i + 1) * individual_height,
                )
                tile = grid_image.crop(box)
                if is_black_image(tile):
                    print(f"Detected a black image at position ({i},{j}) in {image_path}")
                else:
                    image_list.append(tile)

    out_paths: list[str] = []
    for i, tile in enumerate(image_list):
        out = os.path.join(str(cache_dir), f"{i}.jpg")
        tile.save(out)
        out_paths.append(out)
    return out_paths
