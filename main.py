import argparse, time, os, cv2, numpy as np
import config, vision, strategy, screens
from device import Device

def tap_px(dev, x, y, rot, orig_w):
    """Tap a pixel given in the normalized (landscape) full-res frame."""
    dev.tap_px(*vision.unrotate_point(x, y, rot, orig_w))

def once(img, name_note=""):
    """Analyse one capture, print results, write debug images. Returns (hand, tiles)."""
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/raw.png", img)
    h, w = img.shape[:2]
    print(f"capture: {w}x{h} ({'portrait' if h > w else 'landscape'})")
    img, rot = vision.normalize(img)
    if rot:
        print("portrait capture: rotated to landscape for detection (taps are mapped back)")
    small, _ = vision.downscale(img)
    log = []
    hand = vision.find_hand(small, log)
    tiles = vision.find_board(small, log)
    print("hand:", hand and hand["vals"], "board:", [t["vals"] for t in tiles])
    vision.debug_dump(small, log, hand, tiles, "debug/mask.png", "debug/once.png")
    print(f"{len(log)} candidate blobs; rejected ones are red in debug/once.png with the reason")
    if hand is None or not tiles:
        print("Nothing found: open debug/raw.png and debug/once.png. If the game is not visible in raw.png, "
              "the capture is wrong. Otherwise adjust regions/TILE_S_MAX/TILE_V_MIN in config.py.")
    return hand, tiles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial")
    ap.add_argument("--debug", action="store_true", help="save annotated frames to ./debug")
    ap.add_argument("--image", help="run --once style detection on a saved screenshot, no phone needed")
    ap.add_argument("--once", action="store_true", help="analyse one frame, print tiles, no tapping")
    args = ap.parse_args()
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"cannot read image: {args.image}")
        once(img)
        return
    dev = Device(args.serial)
    if args.debug or args.once:
        os.makedirs("debug", exist_ok=True)
    last, same, n = None, 0, 0
    while True:
        img = dev.screenshot()
        if args.once:
            once(img)
            return
        raw = img
        orig_w = img.shape[1]
        img, rot = vision.normalize(img)
        small, s = vision.downscale(img)
        hand = vision.find_hand(small)
        tiles = vision.find_board(small) if hand else []
        if hand and len(tiles) > 0:
            i = strategy.choose(hand["vals"], tiles)
            if args.debug:
                vision.annotate(small, hand, tiles, f"debug/{n:04d}.png"); n += 1
            if i is None:
                tap_px(dev, config.DECK_TAP[0] * img.shape[1], config.DECK_TAP[1] * img.shape[0], rot, orig_w)
                print(f"hand {hand['vals']}: no match -> draw")
            else:
                x, y = tiles[i]["pos"]
                tap_px(dev, x / s, y / s, rot, orig_w)
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
        known = screens.match(raw)
        if known:
            print(f"screen: {known['name']} -> tap")
            dev.tap_frac(*known["tap"])
            time.sleep(1.0)
            continue
        time.sleep(0.8)  # transitions/animations: re-check once before bothering the user
        raw = dev.screenshot()
        if screens.match(raw) or vision.find_hand(vision.downscale(vision.normalize(raw)[0])[0]):
            continue
        screens.teach(raw)

if __name__ == "__main__":
    main()
