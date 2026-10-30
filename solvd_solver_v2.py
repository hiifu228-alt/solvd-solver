#!/usr/bin/env python3
"""SOLVD nonogram solver module v2 — local only, stdlib only.

Optimized exact solver implementing the approved static-audit plan
(2026-10-30) on top of the v1 module design:

1. Per-line automaton TRANSITION TABLES are precomputed once per board
   (v1 recomputed offsets and re-scanned states inside a function call
   for every transition). Line DP uses integer BITMASKS over states.
2. Propagation is an incremental WORKLIST: a node re-evaluates only the
   lines whose fixed cells changed, cascading through crossing lines.
   Line evaluations are cached globally, keyed by (line, fixed-cell
   pattern) — safe because line evaluation is a pure function of those
   two inputs. (v1 re-evaluated all 112 lines on every fixpoint pass of
   every node: ~0.9 s/node, 466 nodes in 420 s on the saved board 0.)
3. Branching picks the cell whose row+column lines carry the least
   remaining ambiguity (from cached evaluations), and VALUE ORDER comes
   from DP placement counts of the two crossing lines.

Completeness is preserved: propagation only forces values shared by
every consistent placement, branching tries both values, contradiction
detection is exact. No network, no credentials, no subprocesses.

Usage: same CLI shape as v1 (engine is always the exact engine):
    python3 solvd_solver_v2.py input.json output.json
        [--time-limit SECONDS] [--report report.json]
Exit codes: 0 = all boards solved and verified; 2 = not solved / invalid.
"""
import argparse
import json
import sys
import time
from collections import deque

# ---------------------------------------------------------------- automaton

def _offs(R):
    o = [0]
    for r in R:
        o.append(o[-1] + r + 1)
    return o


def _trans(R, of, s, v):
    m = len(R)
    if s == of[m]:
        k, t = m, 0
    else:
        k = 0
        while k < m and s >= of[k + 1]:
            k += 1
        if k >= m:
            return -1
        t = s - of[k]
    if v == 0:
        if t == 0:
            return s
        if k < m and t == R[k]:
            return of[k + 1]
        return -1
    if k < m and t < R[k]:
        return s + 1
    return -1


def _accepting(R, of):
    a = {of[-1]}
    if R:
        a.add(of[-2] + R[-1])
    return a


class _Line:
    """A clue with its precomputed automaton transition tables."""
    __slots__ = ('clue', 'n', 'S', 't0', 't1', 'acc_mask', 'acc_states')

    def __init__(self, clue, n):
        self.clue = clue
        self.n = n
        of = _offs(clue)
        S = of[-1] + 1
        self.S = S
        self.t0 = [_trans(clue, of, s, 0) for s in range(S)]
        self.t1 = [_trans(clue, of, s, 1) for s in range(S)]
        acc = _accepting(clue, of)
        self.acc_states = sorted(acc)
        m = 0
        for s in acc:
            m |= 1 << s
        self.acc_mask = m


def _step(line, mask, v):
    """Bitmask of successor states of `mask` under cell value v."""
    t = line.t1 if v else line.t0
    out = 0
    while mask:
        lsb = mask & -mask
        ns = t[lsb.bit_length() - 1]
        if ns >= 0:
            out |= 1 << ns
        mask ^= lsb
    return out


def _line_eval(line, known):
    """Possible values per cell: tuple of 1 ({0}), 2 ({1}), 3 ({0,1}).

    None when the line is inconsistent with the fixed cells. Bitset DP
    over the precomputed automaton; no placement is enumerated.
    """
    n = line.n
    t0, t1 = line.t0, line.t1
    F = [0] * (n + 1)
    F[0] = 1
    for i in range(n):
        m = F[i]
        if not m:
            return None
        kv = known[i]
        nm = 0
        if kv != 1:
            nm |= _step(line, m, 0)
        if kv != 0:
            nm |= _step(line, m, 1)
        F[i + 1] = nm
    if not (F[n] & line.acc_mask):
        return None
    S = line.S
    B = [0] * (n + 1)
    B[n] = line.acc_mask
    for i in range(n - 1, -1, -1):
        target = B[i + 1]
        if not target:
            B[i] = 0
            continue
        kv = known[i]
        m = 0
        for s in range(S):
            if kv != 1:
                ns = t0[s]
                if ns >= 0 and (target >> ns) & 1:
                    m |= 1 << s
                    continue
            if kv != 0:
                ns = t1[s]
                if ns >= 0 and (target >> ns) & 1:
                    m |= 1 << s
        B[i] = m
    poss = []
    for i in range(n):
        live = F[i] & B[i]
        if not live:
            return None
        kv = known[i]
        vals = 0
        if kv != 1 and (_step(line, live, 0) & F[i + 1] & B[i + 1]):
            vals |= 1
        if kv != 0 and (_step(line, live, 1) & F[i + 1] & B[i + 1]):
            vals |= 2
        if not vals:
            return None
        poss.append(vals)
    return tuple(poss)


def _value_counts(line, known, idx):
    """(count0, count1): placements of this line with cell idx = 0 / 1,
    consistent with the fixed cells. DP counting over the automaton."""
    n, S = line.n, line.S
    t0, t1 = line.t0, line.t1
    FC = [[0] * S for _ in range(n + 1)]
    FC[0][0] = 1
    for i in range(n):
        kv = known[i]
        row, nxt = FC[i], FC[i + 1]
        for s in range(S):
            c = row[s]
            if not c:
                continue
            if kv != 1:
                ns = t0[s]
                if ns >= 0:
                    nxt[ns] += c
            if kv != 0:
                ns = t1[s]
                if ns >= 0:
                    nxt[ns] += c
    BC = [[0] * S for _ in range(n + 1)]
    for s in line.acc_states:
        BC[n][s] = 1
    for i in range(n - 1, -1, -1):
        kv = known[i]
        row = BC[i]
        nxt = BC[i + 1]
        for s in range(S):
            tot = 0
            if kv != 1:
                ns = t0[s]
                if ns >= 0:
                    tot += nxt[ns]
            if kv != 0:
                ns = t1[s]
                if ns >= 0:
                    tot += nxt[ns]
            row[s] = tot
    out = [0, 0]
    for v in (0, 1):
        if known[idx] not in (-1, v):
            continue
        t = t1 if v else t0
        tot = 0
        for s in range(S):
            c = FC[idx][s]
            if c:
                ns = t[s]
                if ns >= 0:
                    tot += c * BC[idx + 1][ns]
        out[v] = tot
    return out


# ------------------------------------------------------------------- engine

class _Timeout(Exception):
    pass


def solve_exact(b, deadline):
    """Worklist propagation + DFS. Returns (grid|None, nodes, status)."""
    W, H = b["width"], b["height"]
    rows, cols = b["rows"], b["cols"]
    lines = [_Line(rows[r], W) for r in range(H)] + \
            [_Line(cols[c], H) for c in range(W)]
    nodes = [0]
    eval_cache = {}

    def eval_line(lid, grid):
        line = lines[lid]
        if lid < H:
            known = grid[lid]
        else:
            c = lid - H
            known = [grid[r][c] for r in range(H)]
        key = (lid, bytes(2 if x < 0 else x for x in known))
        poss = eval_cache.get(key, False)
        if poss is False:
            poss = _line_eval(line, known)
            if len(eval_cache) > 300000:
                eval_cache.clear()
            eval_cache[key] = poss
        return poss

    def propagate(grid, cache, amb, seeds):
        queue = deque(seeds)
        inset = set(seeds)
        while queue:
            lid = queue.popleft()
            inset.discard(lid)
            poss = eval_line(lid, grid)
            if poss is None:
                return False
            if cache.get(lid) == poss:
                continue
            cache[lid] = poss
            amb[lid] = sum(1 for x in poss if x == 3)
            is_row = lid < H
            for i, vals in enumerate(poss):
                if vals == 3:
                    continue
                v = 0 if vals == 1 else 1
                r, c = (lid, i) if is_row else (i, lid - H)
                if grid[r][c] == -1:
                    grid[r][c] = v
                    cross = (H + c) if is_row else r
                    if cross not in inset:
                        queue.append(cross)
                        inset.add(cross)
                elif grid[r][c] != v:
                    return False
        return True

    def dfs(grid, cache, amb):
        if time.time() > deadline:
            raise _Timeout()
        nodes[0] += 1
        best = None
        for r in range(H):
            grow = grid[r]
            ra = amb[r]
            for c in range(W):
                if grow[c] == -1:
                    key = ra + amb[H + c]
                    if best is None or key < best[0]:
                        best = (key, r, c)
        if best is None:
            return grid
        _, r, c = best
        rc = _value_counts(lines[r], grid[r], c)
        cc = _value_counts(lines[H + c], [grid[rr][c] for rr in range(H)], r)
        support1 = rc[1] * cc[1]
        support0 = rc[0] * cc[0]
        order = (1, 0) if support1 >= support0 else (0, 1)
        for v in order:
            ng = [row[:] for row in grid]
            ng[r][c] = v
            ncache = dict(cache)
            namb = dict(amb)
            if propagate(ng, ncache, namb, (r, H + c)):
                res = dfs(ng, ncache, namb)
                if res is not None:
                    return res
        return None

    grid0 = [[-1] * W for _ in range(H)]
    cache0, amb0 = {}, {}
    if not propagate(grid0, cache0, amb0, range(H + W)):
        return None, 0, "unsat"
    try:
        grid = dfs(grid0, cache0, amb0)
    except _Timeout:
        return None, nodes[0], "timeout"
    except RecursionError:
        return None, nodes[0], "recursion_limit"
    if grid is None:
        return None, nodes[0], "unsat"
    return grid, nodes[0], "solved"


# ------------------------------------------------------------ verify / encode

def runs_of(line):
    out, cur = [], 0
    for v in line:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def verify_grid(grid, b):
    for r, clue in enumerate(b["rows"]):
        if runs_of(grid[r]) != list(clue):
            return False
    for c, clue in enumerate(b["cols"]):
        if runs_of([grid[r][c] for r in range(b["height"])]) != list(clue):
            return False
    return True


def hex_row(row):
    bits = "".join("1" if v else "0" for v in row)
    bits += "0" * (-len(bits) % 4)
    return "".join(format(int(bits[i:i + 4], 2), "x")
                   for i in range(0, len(bits), 4))


# ---------------------------------------------------------------- validation

def validate_board(b, idx):
    def fail(msg):
        raise ValueError(f"board {idx}: {msg}")
    if not isinstance(b, dict):
        fail("not an object")
    W, H = b.get("width"), b.get("height")
    if not (isinstance(W, int) and isinstance(H, int)) or not (1 <= W <= 512 and 1 <= H <= 512):
        fail("width/height must be integers in 1..512")
    rows, cols = b.get("rows"), b.get("cols")
    if not (isinstance(rows, list) and len(rows) == H):
        fail("rows must be a list of length height")
    if not (isinstance(cols, list) and len(cols) == W):
        fail("cols must be a list of length width")
    for name, lines_, n in (("rows", rows, W), ("cols", cols, H)):
        for i, clue in enumerate(lines_):
            if not isinstance(clue, list) or any(
                    not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in clue):
                fail(f"{name}[{i}] must be a list of positive integers")
            if sum(clue) + max(0, len(clue) - 1) > n:
                fail(f"{name}[{i}] cannot fit in length {n}")


# ---------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description="Local SOLVD nonogram solver v2 (exact)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--engine", choices=("auto", "exact"), default="exact")
    ap.add_argument("--time-limit", type=float, default=120.0,
                    help="per-board wall-clock limit in seconds")
    ap.add_argument("--seed", type=int, default=0, help="accepted for CLI parity; unused")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    sys.setrecursionlimit(20000)
    with open(args.input) as f:
        ch = json.load(f)
    if not isinstance(ch, dict) or not isinstance(ch.get("boards"), list) or not ch["boards"]:
        print("invalid input: missing non-empty 'boards'", file=sys.stderr)
        return 2

    boards_out = []
    report = {"boards": []}
    for bi, b in enumerate(ch["boards"]):
        try:
            validate_board(b, bi)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        deadline = time.time() + args.time_limit
        t0 = time.time()
        grid, work, status = solve_exact(b, deadline)
        dt = time.time() - t0
        entry = {"index": bi, "engine": "exact-v2", "status": status,
                 "seconds": round(dt, 3), "work": work}
        report["boards"].append(entry)
        print(f"board {bi}: {status} via exact-v2 in {dt:.2f}s (work={work})")
        if status != "solved" or grid is None:
            if args.report:
                with open(args.report, "w") as f:
                    json.dump(report, f, indent=1)
            return 2
        if not verify_grid(grid, b):
            print(f"board {bi}: INTERNAL verification failure", file=sys.stderr)
            return 2
        boards_out.append([hex_row(r) for r in grid])

    body = {"challengeId": ch.get("challengeId", "local"),
            "encoding": "hex", "boards": boards_out}
    limit = None
    if isinstance(ch.get("submit"), dict):
        limit = ch["submit"].get("maxBodyBytes")
    size = len(json.dumps(body))
    if limit is not None and size > limit:
        print(f"output {size} bytes exceeds maxBodyBytes {limit}", file=sys.stderr)
        return 2
    with open(args.output, "w") as f:
        json.dump(body, f)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    print(f"written {args.output} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
