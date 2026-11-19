"""Tests for Word Hunt irregular boards and grid solver."""

from __future__ import annotations

import unittest

from gamepigeon.config import (
    WORD_HUNT_CROSS_OCCUPIED,
    WORD_HUNT_DONUT_OCCUPIED,
    WORD_HUNT_LAYOUT_CROSS,
    WORD_HUNT_LAYOUT_DONUT,
    WORD_HUNT_LAYOUT_5X5,
)
from gamepigeon.vision.tile_detect import _assign_boxes_to_lattice
from gamepigeon.vision.tiles import _equal_split_in_bounds
from gamepigeon.words.anagrams import find_words_in_grid


class WordHuntShapeTests(unittest.TestCase):
    def test_cross_has_edge_middles_removed(self) -> None:
        self.assertEqual(len(WORD_HUNT_CROSS_OCCUPIED), 21)
        self.assertNotIn((0, 2), WORD_HUNT_CROSS_OCCUPIED)
        self.assertNotIn((2, 0), WORD_HUNT_CROSS_OCCUPIED)
        self.assertNotIn((2, 4), WORD_HUNT_CROSS_OCCUPIED)
        self.assertNotIn((4, 2), WORD_HUNT_CROSS_OCCUPIED)
        self.assertEqual(WORD_HUNT_LAYOUT_CROSS.grid.tile_count, 21)

    def test_donut_has_center_hole_and_missing_corners(self) -> None:
        self.assertEqual(len(WORD_HUNT_DONUT_OCCUPIED), 20)
        self.assertNotIn((2, 2), WORD_HUNT_DONUT_OCCUPIED)
        self.assertNotIn((0, 0), WORD_HUNT_DONUT_OCCUPIED)
        self.assertNotIn((0, 4), WORD_HUNT_DONUT_OCCUPIED)
        self.assertNotIn((4, 0), WORD_HUNT_DONUT_OCCUPIED)
        self.assertNotIn((4, 4), WORD_HUNT_DONUT_OCCUPIED)
        self.assertEqual(WORD_HUNT_LAYOUT_DONUT.grid.tile_count, 20)

    def test_equal_split_skips_unoccupied_cells(self) -> None:
        rects = _equal_split_in_bounds(
            0,
            0,
            500,
            500,
            rows=5,
            cols=5,
            occupied=WORD_HUNT_CROSS_OCCUPIED,
        )
        self.assertEqual(len(rects), 21)
        cells = {(row, col) for row, col, *_ in rects}
        self.assertEqual(cells, WORD_HUNT_CROSS_OCCUPIED)

    def test_assign_boxes_to_cross_lattice(self) -> None:
        boxes: list[tuple[int, int, int, int]] = []
        for row, col in sorted(WORD_HUNT_CROSS_OCCUPIED):
            x1 = col * 40
            y1 = row * 40
            boxes.append((x1, y1, x1 + 30, y1 + 30))

        assigned = _assign_boxes_to_lattice(
            boxes,
            rows=5,
            cols=5,
            occupied=WORD_HUNT_CROSS_OCCUPIED,
        )
        self.assertIsNotNone(assigned)
        assert assigned is not None
        self.assertEqual(len(assigned), 21)
        self.assertEqual({(r, c) for r, c, *_ in assigned}, WORD_HUNT_CROSS_OCCUPIED)

    def test_solver_skips_holes_on_cross_board(self) -> None:
        matrix = [
            list("AB.DE"),
            list("FGHIJ"),
            list(".KLM."),
            list("NOPQR"),
            list("ST.UV"),
        ]
        matrix = [[cell if cell != "." else "" for cell in row] for row in matrix]
        dictionary = ["bag", "bad", "bake", "xxx"]
        found = find_words_in_grid(matrix, dictionary, min_length=3, max_length=7)
        words = {word for word, _ in found}
        self.assertIn("bag", words)
        # "bad" would require crossing the hole between B and D on the top row.
        self.assertNotIn("bad", words)

    def test_solver_finds_word_around_hole(self) -> None:
        matrix = [
            list("AX.YZ"),
            list("BCDEF"),
            list(".GHI."),
            list("JKLMN"),
            list("OP.QR"),
        ]
        matrix = [[cell if cell != "." else "" for cell in row] for row in matrix]
        dictionary = ["cde", "ax"]
        prefixes = frozenset({"c", "cd", "cde", "a", "ax"})
        found = find_words_in_grid(
            matrix,
            dictionary,
            min_length=3,
            max_length=7,
            prefixes=prefixes,
        )
        self.assertEqual([word for word, _ in found], ["cde"])

    def test_5x5_layout_tile_count(self) -> None:
        self.assertEqual(WORD_HUNT_LAYOUT_5X5.grid.tile_count, 25)


if __name__ == "__main__":
    unittest.main()
