# GamePigeon Solver

Bot that plays GamePigeon Word Hunt and Anagrams through macOS iPhone Mirroring. It grabs the mirror window, reads tiles with Tesseract, solves against a popular Scrabble word list, and clicks/drags the answers in.

## Setup

- macOS with iPhone Mirroring app
- Python 3.11+
- `brew install tesseract` (the OCR binary — `pytesseract` alone isn't enough)
- Screen Recording + Accessibility enabled for Terminal / whatever runs the bot

The dictionary is Collins Scrabble Words 2021 (CSW21). It's copyrighted, so it can't be included here. Download it from the [scrabblewords](https://github.com/scrabblewords/scrabblewords) mirror:

```bash
mkdir -p gamepigeon/words/data
curl -L -o gamepigeon/words/data/CSW21.txt \
  https://raw.githubusercontent.com/scrabblewords/scrabblewords/main/words/British/CSW21.txt
```

Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

Open iPhone Mirroring and start a game:

```bash
python -m gamepigeon play --game word_hunt
python -m gamepigeon play --game anagrams
```

Word Hunt board shape auto-detects (`4x4`, `5x5`, `cross`, `donut`). Force one with `--board cross`. Use `--picker` if window auto-find fails, `--debug` to dump frames to `/tmp`. Ctrl+C stops it.

If OCR is off on your phone, tweak the rects in `gamepigeon/config.py` and run `calibrate` / `calibrate-start`.

## Testing

Crop the letter rack or Word Hunt grid from iPhone Mirroring, drop the PNGs in `tests/fixtures/`, and list them in `tests/fixtures/prototype_boards.json` with the expected letters:

```json
{ "file": "my_board.png", "layout": "grid", "rows": 4, "cols": 4, "letters": "ABCD..." }
```

Use `"layout": "rack"` for Anagrams. Then:

```bash
python -m unittest discover -s tests -v
```

to test OCR and board/letter detection.
