import json
import unittest
from pathlib import Path

import cv2

from gamepigeon.config import WORD_HUNT_LAYOUT
from gamepigeon.vision.letters import read_tile_letter
from gamepigeon.vision.tiles import read_tile_grid

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_MANIFEST = _FIXTURES_DIR / "prototype_boards.json"


class TileLetterRecognitionTests(unittest.TestCase):
    def test_reads_rack_fixtures(self) -> None:
        if not _MANIFEST.exists():
            self.skipTest("fixture manifest missing")

        for entry in json.loads(_MANIFEST.read_text()):
            if entry.get("layout") != "rack":
                continue

            with self.subTest(file=entry["file"]):
                image_path = _FIXTURES_DIR / entry["file"]
                if not image_path.exists():
                    self.skipTest(f"fixture image missing: {entry['file']}")

                image = cv2.imread(str(image_path))
                self.assertIsNotNone(image)

                height, width = image.shape[:2]
                expected = entry["letters"]
                tile_width = width // len(expected)
                letters = [
                    read_tile_letter(image[:, index * tile_width : (index + 1) * tile_width])
                    for index in range(len(expected))
                ]

                self.assertEqual(letters, list(expected))

    def test_reads_grid_fixtures(self) -> None:
        if not _MANIFEST.exists():
            self.skipTest("fixture manifest missing")

        for entry in json.loads(_MANIFEST.read_text()):
            if entry.get("layout") != "grid":
                continue

            with self.subTest(file=entry["file"]):
                image_path = _FIXTURES_DIR / entry["file"]
                if not image_path.exists():
                    # Narrow no-break space in some macOS screenshot names
                    matches = [
                        path
                        for path in _FIXTURES_DIR.iterdir()
                        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                        and entry["file"][:8] in path.name
                    ]
                    if not matches:
                        self.skipTest(f"fixture image missing: {entry['file']}")
                    image_path = matches[0]
                image = cv2.imread(str(image_path))
                self.assertIsNotNone(image)

                grid = read_tile_grid(image, WORD_HUNT_LAYOUT)
                letters = "".join(
                    tile.letter
                    for tile in sorted(grid.tiles, key=lambda tile: (tile.row, tile.col))
                )
                self.assertEqual(letters, entry["letters"])


if __name__ == "__main__":
    unittest.main()
