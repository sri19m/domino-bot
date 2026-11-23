import argparse, time, os, cv2, numpy as np
import config, vision, strategy, screens
from device import Device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial")
    ap.add_argument("--debug", action="store_true", help="save annotated frames to ./debug")
    ap.add_argument("--once", action="store_true", help="analyse one frame, print tiles, no tapping")
    args = ap.parse_args()
    dev = Device(args.serial)
    if args.debug or args.once:
        os.makedirs("debug", exist_ok=True)
    last, same, n = None, 0, 0
    while True:
        img = dev.screenshot()
        small, s = vision.downscale(img)
        hand = vision.find_hand(small)
        tiles = vision.find_board(small) if hand else []
        if args.once:
            print("hand:", hand and hand["vals"], "board:", [t["vals"] for t in tiles])
            vision.annotate(small, hand, tiles, "debug/once.png")
            return
        if hand and len(tiles) > 0:
            i = strategy.choose(hand["vals"], tiles)
            if args.debug:
                vision.annotate(small, hand, tiles, f"debug/{n:04d}.png"); n += 1
            if i is None:
                dev.tap_frac(*config.DECK_TAP)
                print(f"hand {hand['vals']}: no match -> draw")
            else:
                x, y = tiles[i]["pos"]
                dev.tap_px(x / s, y / s)
                print(f"hand {hand['vals']}: tap {tiles[i]['vals']}")
            time.sleep(config.TAP_SETTLE)
            # stuck detection: nothing changed after our tap
            sig = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (64, 32)).astype(np.int16)
            same = same + 1 if last is not None and np.abs(sig - last).mean() < 1.0 else 0
            last = sig
            if same < config.STUCK_FRAMES:
                continue
            print("Board not changing; treating as unknown screen.")
            same = 0
        # not in play (or stuck): known screen?
        known = screens.match(img)
        if known:
            print(f"screen: {known['name']} -> tap")
            dev.tap_frac(*known["tap"])
            time.sleep(1.0)
            continue
        time.sleep(0.8)  # transitions/animations: re-check once before bothering the user
        img = dev.screenshot()
        if screens.match(img) or vision.find_hand(vision.downscale(img)[0]):
            continue
        screens.teach(img)

if __name__ == "__main__":
    main()
