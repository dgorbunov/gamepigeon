"""Tests for alphagram index and anagram solver."""

from __future__ import annotations

import time
import unittest

from gamepigeon.words.alphagram_index import AlphagramIndex, normalize_word, to_alphagram
from gamepigeon.words.anagram_solver import AnagramSolver, iter_rack_alphagrams, normalize_rack
from gamepigeon.words.dictionary import load_scrabble_words


class AlphagramIndexTests(unittest.TestCase):
    def test_to_alphagram(self) -> None:
        self.assertEqual(to_alphagram("stop"), "opst")
        self.assertEqual(to_alphagram("  Spot "), "opst")

    def test_lookup_groups_anagrams(self) -> None:
        index = AlphagramIndex.from_words(["post", "spot", "stop", "tops", "cat"])
        self.assertEqual(index.lookup("opst"), ("post", "spot", "stop", "tops"))
        self.assertEqual(index.lookup("act"), ("cat",))

    def test_normalize_word_filters_invalid(self) -> None:
        self.assertIsNone(normalize_word("ab"))
        self.assertIsNone(normalize_word("abcdefgh"))
        self.assertIsNone(normalize_word("cat3"))
        self.assertEqual(normalize_word("  Cat "), "cat")


class AnagramSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            words = load_scrabble_words()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.index = AlphagramIndex.from_words(words)
        cls.solver = AnagramSolver(cls.index)

    def test_stop_rack_matches_example(self) -> None:
        matches = self.solver.solve("stop")
        self.assertIn("stop", matches)
        self.assertIn("post", matches)
        self.assertIn("spot", matches)
        self.assertIn("tops", matches)
        self.assertEqual(matches[0], max(matches, key=len))

    def test_duplicate_letters_in_rack(self) -> None:
        matches = self.solver.solve("aab")
        self.assertIn("aba", matches)
        self.assertTrue(all(word.count("a") <= 2 for word in matches))

    def test_exclude_already_played(self) -> None:
        matches = self.solver.solve("stop", exclude={"stop"})
        self.assertNotIn("stop", matches)
        self.assertIn("post", matches)

    def test_sort_longest_first_then_alpha(self) -> None:
        matches = self.solver.solve("aehncs")
        self.assertGreater(len(matches), 0)
        lengths = [len(word) for word in matches]
        self.assertEqual(lengths, sorted(lengths, reverse=True))
        for left, right in zip(matches, matches[1:]):
            if len(left) == len(right):
                self.assertLessEqual(left, right)

    def test_invalid_characters_ignored(self) -> None:
        self.assertEqual(normalize_rack(["A", "?", "E", "1", "H"]), "aeh")
        self.assertEqual(self.solver.solve("???"), [])

    def test_rack_sizes_three_to_seven(self) -> None:
        for size in range(3, 8):
            rack = "abcdefg"[:size]
            matches = self.solver.solve(rack)
            self.assertTrue(all(3 <= len(word) <= size for word in matches))

    def test_lookup_is_fast(self) -> None:
        start = time.perf_counter()
        for _ in range(200):
            self.solver.solve("aehncs")
        elapsed_ms = (time.perf_counter() - start) * 1000 / 200
        self.assertLess(elapsed_ms, 10.0)

    def test_csw21_includes_international_words(self) -> None:
        try:
            words = load_scrabble_words()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        self.assertIn("conche", words)
        self.assertIn("conches", words)


class RackAlphagramTests(unittest.TestCase):
    def test_unique_multiset_subsets(self) -> None:
        alphagrams = list(iter_rack_alphagrams("aab", 2, 3))
        self.assertEqual(alphagrams.count("ab"), 1)
        self.assertIn("aab", alphagrams)


if __name__ == "__main__":
    unittest.main()
