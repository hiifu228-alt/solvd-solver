#!/usr/bin/env python3
"""SOLVD nonogram solver — parallel exact engine (local only, stdlib only).

Built on top of solvd_solver_v2.py. What is reused and how:

* Imported unchanged from solvd_solver_v2 (module level there):
  _Line (automaton transition tables), _line_eval (bitmask line DP),
  _value_counts (DP placement counts), _Timeout, verify_grid, hex_row,
  validate_board.
* v2's propagate()/dfs() are closures nested inside solve_exact() and
  solve_exact() offers no way to seed a partial assignment, so they
  cannot be imported. Their logic is copied verbatim into the _Engine
  class below and refactored minimally: the search now starts from an
  arbitrary initial grid (a partial assignment) instead of always from
  the all-unknown grid. Branching heuristic (cell minimising
  row-ambiguity + column-ambiguity) and value order (DP-count support)
  are identical to v2, so a worker searching one subproblem explores
  exactly the subtree v2's DFS would explore under those fixed cells.

Parallel scheme (per board):
  1. Parent runs the initial propagation. (Unsat or fully determined
     boards finish here, with no workers spawned.)
  2. Deterministic frontier expansion: repeatedly take the first
     expandable leaf in list order, branch it with the v2 heuristic
     (children appended in v2 value order), propagate each child, drop
     contradictions — until at least 4x the worker count of leaf
     subproblems exist (or the frontier is exhausted).
  3. Each leaf becomes a subproblem = the list of its fixed cells.
     Workers (multiprocessing, fork) pull subproblems from a queue,
     apply the fixed cells, re-run propagation (propagation computes a
     unique closure, so the worker reaches the identical state), then
     run the exact DFS. First complete grid wins; the parent stops the
     other workers and verifies the grid with verify_grid before
     accepting it.
  4. If every subproblem reports unsat, the board is unsat (the leaves
     partition the remaining search space). If the global deadline
     passes first, the status is timeout and all workers are stopped
     within a small margin.

Work accounting: parent expansion counts one unit per branch attempt;
each worker reports its cumulative DFS node count through a shared
array (updated every 256 nodes and at each subproblem end). The
reported "work" is expansion + sum over workers.

Usage (CLI-compatible with v2, plus --workers):
    python3 solvd_solver_par.py input.json output.json
        [--time-limit SECONDS] [--report report.json] [--workers N]
Exit codes: 0 = all boards solved and verified; 2 = not solved / invalid.
No network, no credentials, no blockchain anything.
"""
import argparse
import json
import os
import queue as queue_mod
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multiprocessing as mp

from solvd_solver_v2 import (_Line, _line_eval, _value_counts, _Timeout,
                             verify_grid, hex_row, validate_board)

# ------------------------------------------------------------------- engine
# Search scaffold copied from solvd_solver_v2.solve_exact's nested
# functions, refactored into a class whose search starts from a given
# grid (partial assignment). Line evaluation/propagation/branching
# semantics are identical to v2.


class _Engine:
    def __init__(self, b):
        self.W, self.H = b["width"], b["height"]
        self.lines = [_Line(b["rows"][r], self.W) for r in range(self.H)] + \
                     [_Line(b["cols"][c], self.H) for c in range(self.W)]
        self.eval_cache = {}
        self.nodes = 0
        self.tick = None  # optional callable(cumulative_nodes)

    def eval_line(self, lid, grid):
        line = self.lines[lid]
        if lid < self.H:
            known = grid[lid]
        else:
            c = lid - self.H
            known = [grid[r][c] for r in range(self.H)]
        key = (lid, bytes(2 if x < 0 else x for x in known))
        poss = self.eval_cache.get(key, False)
        if poss is False:
            poss = _line_eval(line, known)
            if len(self.eval_cache) > 300000:
                self.eval_cache.clear()
            self.eval_cache[key] = poss
        return poss

    def propagate(self, grid, cache, amb, seeds):
        queue = deque(seeds)
        inset = set(seeds)
        while queue:
            lid = queue.popleft()
            inset.discard(lid)
            poss = self.eval_line(lid, grid)
            if poss is None:
                return False
            if cache.get(lid) == poss:
                continue
            cache[lid] = poss
            amb[lid] = sum(1 for x in poss if x == 3)
            is_row = lid < self.H
            for i, vals in enumerate(poss):
                if vals == 3:
                    continue
                v = 0 if vals == 1 else 1
                r, c = (lid, i) if is_row else (i, lid - self.H)
                if grid[r][c] == -1:
                    grid[r][c] = v
                    cross = (self.H + c) if is_row else r
                    if cross not in inset:
                        queue.append(cross)
                        inset.add(cross)
                elif grid[r][c] != v:
                    return False
        return True

    def root_state(self, assignments=()):
        """Grid seeded with `assignments` ((r, c, v) triples), then fully
        propagated. Returns (grid, cache, amb) or None if contradictory."""
        grid = [[-1] * self.W for _ in range(self.H)]
        for r, c, v in assignments:
            grid[r][c] = v
        cache, amb = {}, {}
        if not self.propagate(grid, cache, amb, range(self.H + self.W)):
            return None
        return grid, cache, amb

    def pick_cell(self, grid, amb):
        """v2 branching heuristic: unknown cell with the least
        row-ambiguity + column-ambiguity. None when the grid is full."""
        best = None
        for r in range(self.H):
            grow = grid[r]
            ra = amb[r]
            for c in range(self.W):
                if grow[c] == -1:
                    key = ra + amb[self.H + c]
                    if best is None or key < best[0]:
                        best = (key, r, c)
        if best is None:
            return None
        return best[1], best[2]

    def value_order(self, grid, r, c):
        rc = _value_counts(self.lines[r], grid[r], c)
        cc = _value_counts(self.lines[self.H + c],
                           [grid[rr][c] for rr in range(self.H)], r)
        support1 = rc[1] * cc[1]
        support0 = rc[0] * cc[0]
        return (1, 0) if support1 >= support0 else (0, 1)

    def dfs(self, grid, cache, amb, deadline):
        if time.time() > deadline:
            raise _Timeout()
        self.nodes += 1
        if self.tick is not None and self.nodes % 256 == 0:
            self.tick(self.nodes)
        cell = self.pick_cell(grid, amb)
        if cell is None:
            return grid
        r, c = cell
        for v in self.value_order(grid, r, c):
            ng = [row[:] for row in grid]
            ng[r][c] = v
            ncache = dict(cache)
            namb = dict(amb)
            if self.propagate(ng, ncache, namb, (r, self.H + c)):
                res = self.dfs(ng, ncache, namb, deadline)
                if res is not None:
                    return res
        return None

    def solve_state(self, state, deadline):
        """DFS from a propagated state. Returns (grid|None, status)."""
        grid, cache, amb = state
        try:
            res = self.dfs(grid, cache, amb, deadline)
        except _Timeout:
            return None, "timeout"
        except RecursionError:
            return None, "recursion_limit"
        if res is None:
            return None, "unsat"
        return res, "solved"

    def expand_frontier(self, state, target, deadline):
        """Deterministic BFS frontier expansion. Returns the list of
        leaf states (grid, cache, amb), all propagated and mutually
        exclusive, whose fixed cells partition the remaining search.
        A leaf whose grid is complete is a solution candidate.
        Raises _Timeout if the deadline passes during expansion."""
        leaves = [state]
        while len(leaves) < target:
            if time.time() > deadline:
                raise _Timeout()
            spot = None
            for i, (g, ca, am) in enumerate(leaves):
                cell = self.pick_cell(g, am)
                if cell is not None:
                    spot = (i, cell)
                    break
            if spot is None:
                break
            idx, (r, c) = spot
            grid, cache, amb = leaves.pop(idx)
            for v in self.value_order(grid, r, c):
                ng = [row[:] for row in grid]
                ng[r][c] = v
                ncache = dict(cache)
                namb = dict(amb)
                self.nodes += 1
                if self.propagate(ng, ncache, namb, (r, self.H + c)):
                    leaves.append((ng, ncache, namb))
        return leaves


def _assignments_of(grid):
    return [(r, c, grid[r][c])
            for r, row in enumerate(grid)
            for c, v in enumerate(row) if v != -1]


# ------------------------------------------------------------------ workers

def _worker_main(wid, b, jobs, results, work_arr, deadline):
    sys.setrecursionlimit(20000)
    eng = _Engine(b)
    base = [0]

    def tick(n):
        work_arr[wid] = base[0] + n

    eng.tick = tick
    while True:
        item = jobs.get()
        if item is None:
            return
        idx, assigns = item
        if time.time() >= deadline:
            results.put((idx, "skipped", 0, None, wid))
            continue
        try:
            eng.nodes = 0
            state = eng.root_state(assigns)
            if state is None:
                results.put((idx, "unsat", 0, None, wid))
                continue
            grid, status = eng.solve_state(state, deadline)
            results.put((idx, status, eng.nodes,
                         grid if status == "solved" else None, wid))
            base[0] += eng.nodes
            work_arr[wid] = base[0]
        except Exception as e:  # keep the pool alive; parent decides
            results.put((idx, "error", eng.nodes, None, wid))
            print(f"worker {wid}: subproblem {idx} error: {e!r}",
                  file=sys.stderr)


# ------------------------------------------------------------- parallel API

def solve_board_parallel(b, time_limit, nworkers=None):
    """Solve one board with the parallel engine.

    Returns (grid|None, info) where info has keys: status, seconds,
    work (total, expansion + workers), subproblems, found_by
    (worker id int, "parent", or None), workers.
    """
    if nworkers is None:
        nworkers = os.cpu_count() or 1
    nworkers = max(1, int(nworkers))
    t_start = time.time()
    deadline = t_start + time_limit
    info = {"status": "timeout", "seconds": 0.0, "work": 0,
            "subproblems": 0, "found_by": None, "workers": nworkers}

    def finish(grid, status, work, subs, found_by):
        info.update(status=status, work=work, subproblems=subs,
                    found_by=found_by, seconds=round(time.time() - t_start, 3))
        return grid, info

    eng = _Engine(b)
    state = eng.root_state(())
    if state is None:
        return finish(None, "unsat", eng.nodes, 0, None)
    if eng.pick_cell(state[0], state[2]) is None:
        if verify_grid(state[0], b):
            return finish(state[0], "solved", eng.nodes, 0, "parent")
        return finish(None, "error", eng.nodes, 0, None)

    target = 4 * nworkers
    try:
        leaves = eng.expand_frontier(state, target, deadline)
    except _Timeout:
        return finish(None, "timeout", eng.nodes, 0, None)
    expansion_nodes = eng.nodes

    for g, ca, am in leaves:
        if eng.pick_cell(g, am) is None:
            if verify_grid(g, b):
                return finish(g, "solved", expansion_nodes, len(leaves),
                              "parent")
    if not leaves:
        return finish(None, "unsat", expansion_nodes, 0, None)

    subs = [_assignments_of(g) for g, _, _ in leaves]
    info["subproblems"] = len(subs)

    ctx = mp.get_context("fork")
    jobs = ctx.Queue()
    results = ctx.Queue()
    work_arr = ctx.Array("l", [0] * nworkers)
    procs = [ctx.Process(target=_worker_main,
                         args=(w, b, jobs, results, work_arr, deadline),
                         daemon=True)
             for w in range(nworkers)]
    for p in procs:
        p.start()
    for i, a in enumerate(subs):
        jobs.put((i, a))
    for _ in range(nworkers):
        jobs.put(None)

    received = {}
    found = None
    wait_until = deadline + 8.0
    try:
        while time.time() < wait_until:
            if found is not None or len(received) >= len(subs):
                break
            try:
                idx, status, _nodes, grid, wid = results.get(timeout=0.25)
            except queue_mod.Empty:
                if time.time() > deadline and not any(p.is_alive()
                                                      for p in procs):
                    break
                continue
            received[idx] = status
            if status == "solved" and grid is not None:
                if verify_grid(grid, b):
                    found = (grid, wid)
                    break
                print(f"board: worker {wid} grid failed verification; "
                      "continuing", file=sys.stderr)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)

    total_work = expansion_nodes + sum(work_arr[:])
    if found is not None:
        return finish(found[0], "solved", total_work, len(subs), found[1])
    if len(received) >= len(subs) and \
            all(s == "unsat" for s in received.values()):
        return finish(None, "unsat", total_work, len(subs), None)
    return finish(None, "timeout", total_work, len(subs), None)


# ---------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Local SOLVD nonogram solver (parallel exact)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--engine", choices=("auto", "exact"), default="exact")
    ap.add_argument("--time-limit", type=float, default=120.0,
                    help="per-board wall-clock limit in seconds")
    ap.add_argument("--seed", type=int, default=0,
                    help="accepted for CLI parity; unused")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes (default: os.cpu_count())")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    sys.setrecursionlimit(20000)
    with open(args.input) as f:
        ch = json.load(f)
    if not isinstance(ch, dict) or not isinstance(ch.get("boards"), list) \
            or not ch["boards"]:
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
        grid, info = solve_board_parallel(b, args.time_limit, args.workers)
        entry = {"index": bi, "engine": "exact-par",
                 "status": info["status"], "seconds": info["seconds"],
                 "work": info["work"], "subproblems": info["subproblems"],
                 "found_by": info["found_by"], "workers": info["workers"]}
        report["boards"].append(entry)
        print(f"board {bi}: {info['status']} via exact-par in "
              f"{info['seconds']:.2f}s (work={info['work']}, "
              f"subproblems={info['subproblems']}, "
              f"workers={info['workers']})")
        if info["status"] != "solved" or grid is None:
            _write_report(report, args.report)
            return 2
        if not verify_grid(grid, b):
            print(f"board {bi}: INTERNAL verification failure",
                  file=sys.stderr)
            _write_report(report, args.report)
            return 2
        boards_out.append([hex_row(r) for r in grid])

    body = {"challengeId": ch.get("challengeId", "local"),
            "encoding": "hex", "boards": boards_out}
    limit = None
    if isinstance(ch.get("submit"), dict):
        limit = ch["submit"].get("maxBodyBytes")
    size = len(json.dumps(body))
    if limit is not None and size > limit:
        print(f"output {size} bytes exceeds maxBodyBytes {limit}",
              file=sys.stderr)
        _write_report(report, args.report)
        return 2
    with open(args.output, "w") as f:
        json.dump(body, f)
    _write_report(report, args.report)
    print(f"written {args.output} ({size} bytes)")
    return 0


def _write_report(report, path):
    if not path:
        return
    agg_status = "solved"
    for e in report["boards"]:
        if e["status"] != "solved":
            agg_status = e["status"]
            break
    report["status"] = agg_status
    report["seconds"] = round(sum(e["seconds"] for e in report["boards"]), 3)
    report["work"] = sum(e["work"] for e in report["boards"])
    report["subproblems"] = sum(e["subproblems"] for e in report["boards"])
    found = [e["found_by"] for e in report["boards"]
             if e["found_by"] is not None]
    report["found_by"] = found[0] if found else None
    with open(path, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
