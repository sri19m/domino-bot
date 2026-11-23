# Domino Dreams bot

Setup (laptop): `pip install -r requirements.txt`, install Android platform-tools (`adb`).
Phone: enable Developer options > USB debugging, connect by USB, accept the prompt, check `adb devices`.
(Wireless debugging over a shared network also works; tethering is not needed for USB.)

1. `python main.py --once` while a level is on screen: prints detected tiles and writes `debug/once.png`. Verify the numbers.
2. Adjust `RULE`, `CHAIN`, regions in `config.py` if needed.
3. `python main.py --debug` to run. On any unknown screen it opens a window: drag a box around a unique element, click where to tap, name it. It remembers it.

## Testing without a phone

- `python selftest.py` draws a synthetic board (rotated tiles, vertical hand tile, deck, undo button), runs detection at several resolutions and as a portrait capture, and must print `ALL PASS`. PNG/JPG files in a `reference/` folder are also run through the pipeline and their results printed.
- `python main.py --image path\to\screenshot.png` runs the same analysis as `--once` on a saved screenshot.

## Troubleshooting: `hand: None board: []`

`python main.py --once` (or `--image`) always writes three files to `debug/`, and prints the capture size and orientation:

- `raw.png`: the exact full-resolution capture. If this is not the game (black, wrong app, cut off), the problem is the capture, not detection.
- `mask.png`: white where pixels look like cream tile. Tiles should appear as solid white rectangles. If they are speckled or missing, change `TILE_S_MAX` (raise it) or `TILE_V_MIN` (lower it) in `config.py`. A looser second pass is tried automatically.
- `once.png`: every candidate blob outlined. Green means accepted, red means rejected, with the reason (`area`, `ratio`, `fill`, `solid`). The magenta and cyan rectangles are `HAND_REGION` and `BOARD_REGION`. If the hand tile is outside the magenta box, change `HAND_REGION`.

Common causes:

- Portrait capture: detected automatically and rotated to landscape. If the image is upside down after rotation, tell me and the direction can be flipped.
- Tiles merged into one blob (red `area` or `ratio` reason): tiles touching or overlapping. Pause between animations and retry.
- Face-down tiles listed as rejected `solid`: expected, they are filtered on purpose.
- Rounded corners or the camera cut-out hiding part of the screen: adjust the regions in `config.py` (fractions of the screen).
- Wrong pip counts: send `debug/raw.png` and `debug/once.png`.
