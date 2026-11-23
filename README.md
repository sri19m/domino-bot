# Domino Dreams bot

Setup (laptop): `pip install -r requirements.txt`, install Android platform-tools (`adb`).
Phone: enable Developer options > USB debugging, connect by USB, accept the prompt, check `adb devices`.
(Wireless debugging over a shared network also works; tethering is not needed for USB.)

1. `python main.py --once` while a level is on screen: prints detected tiles and writes `debug/once.png`. Verify the numbers.
2. Adjust `RULE`, `CHAIN`, regions in `config.py` if needed.
3. `python main.py --debug` to run. On any unknown screen it opens a window: drag a box around a unique element, click where to tap, name it. It remembers it.
