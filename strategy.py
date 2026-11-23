import config

def matches(hand, tile):
    h, t = hand, tile
    if config.RULE == "exact":
        return sorted(h) == sorted(t)
    if config.RULE == "plusminus":
        return any(abs(x - y) == 1 for x in h for y in t)
    return bool(set(h) & set(t))

def choose(hand_vals, tiles):
    """Pick the tile index whose tap leads to the longest chain among visible tiles. None => draw."""
    options = [i for i, t in enumerate(tiles) if matches(hand_vals, t["vals"])]
    if not options:
        return None
    if not config.CHAIN:
        return options[0]
    nodes = [0]
    memo = {}

    def best(hand, remaining):
        key = (hand, remaining)
        if key in memo:
            return memo[key]
        nodes[0] += 1
        if nodes[0] > config.SEARCH_NODE_CAP:
            return 0
        res = 0
        for i in remaining:
            if matches(hand, tiles[i]["vals"]):
                res = max(res, 1 + best(tiles[i]["vals"], remaining - {i}))
        memo[key] = res
        return res

    allr = frozenset(range(len(tiles)))
    return max(options, key=lambda i: best(tiles[i]["vals"], allr - {i}))
