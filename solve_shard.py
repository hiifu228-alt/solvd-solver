#!/usr/bin/env python3
"""SOLVD nonogram solver — one deterministic shard of the parallel search.

Sharding scheme (built entirely on solvd_solver_par / solvd_solver_v2,
which are imported unchanged):

* Every shard process rebuilds the SAME deterministic frontier as the
  parallel engine would: initial propagation, then BFS frontier
  expansion (solvd_solver_par._Engine.expand_frontier) to
  NSHARDS * LEAVES_PER_SHARD leaf subproblems, in the same order.
  A leaf subproblem is the list of its fixed cells; leaves are mutually
  exclusive and partition the remaining search space.
* Shard K of NSHARDS owns exactly the leaves whose global index i
  satisfies i % NSHARDS == K. It runs the par worker pool
  (solvd_solver_par._worker_main) over only those leaves.
* First verified complete grid wins inside the shard. If every owned
  leaf reports unsat, the shard is unsat (its region has no solution).
  Anything else at the deadline is timeout.

Usage:
    python3 solve_shard.py <challenge.json> <nshards> <kshard> \
        <time_limit> <result.json>

result.json is written in ALL cases (solved / unsat / timeout / error)
with keys: shard, nshards, board_index, status, seconds, work
(expansion + this shard's DFS nodes), dfs_work, expansion_work,
subproblems_total, subproblems_mine, found_subproblem, and on success
grid (rows of "0"/"1" strings) + submit_body (hex encoding).
Exit code is 0 whenever result.json was written.
No network, no credentials, no blockchain anything.
"""
import argparse
import json
import os
import queue as queue_mod
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multiprocessing as mp

from solvd_solver_v2 import _Timeout, hex_row, validate_board, verify_grid
from solvd_solver_par import _Engine, _assignments_of, _worker_main

# Leaves each shard should own. Mirrors the par engine's ratio
# (frontier target = 4 x workers) for a 4-core runner: 4 x 4 = 16.
LEAVES_PER_SHARD = 16


def solve_board_shard(b, nshards, kshard, time_limit):
    """Solve board `b` restricted to shard kshard of nshards.

    Returns (grid|None, info dict). Mirrors
    solvd_solver_par.solve_board_parallel, except the frontier target
    is NSHARDS*LEAVES_PER_SHARD and the worker pool only receives this
    shard's leaves (global index % nshards == kshard).
    """
    t_start = time.time()
    deadline = t_start + time_limit
    nworkers = max(1, os.cpu_count() or 1)
    info = {"status": "timeout", "seconds": 0.0, "work": 0, "dfs_work": 0,
            "expansion_work": 0, "subproblems_total": 0,
            "subproblems_mine": 0, "found_subproblem": None,
            "workers": nworkers}

    def finish(grid, status, work, dfs_work, exp_work, total, mine, found):
        info.update(status=status, work=work, dfs_work=dfs_work,
                    expansion_work=exp_work, subproblems_total=total,
                    subproblems_mine=mine, found_subproblem=found,
                    seconds=round(time.time() - t_start, 3))
        return grid, info

    eng = _Engine(b)
    state = eng.root_state(())
    if state is None:
        return finish(None, "unsat", eng.nodes, 0, eng.nodes, 0, 0, None)
    if eng.pick_cell(state[0], state[2]) is None:
        if verify_grid(state[0], b):
            return finish(state[0], "solved", eng.nodes, 0, eng.nodes,
                          0, 0, None)
        return finish(None, "error", eng.nodes, 0, eng.nodes, 0, 0, None)

    target = nshards * LEAVES_PER_SHARD
    try:
        leaves = eng.expand_frontier(state, target, deadline)
    except _Timeout:
        return finish(None, "timeout", eng.nodes, 0, eng.nodes, 0, 0, None)
    expansion_nodes = eng.nodes
    if not leaves:
        return finish(None, "unsat", expansion_nodes, 0, expansion_nodes,
                      0, 0, None)

    subs = [_assignments_of(g) for g, _, _ in leaves]
    mine = [(i, subs[i]) for i in range(len(subs)) if i % nshards == kshard]
    if not mine:
        return finish(None, "unsat", expansion_nodes, 0, expansion_nodes,
                      len(subs), 0, None)

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
    for i, a in mine:
        jobs.put((i, a))
    for _ in range(nworkers):
        jobs.put(None)

    mine_ids = {i for i, _ in mine}
    received = {}
    found = None
    wait_until = deadline + 8.0
    try:
        while time.time() < wait_until:
            if found is not None or len(received) >= len(mine):
                break
            try:
                idx, status, _nodes, grid, wid = results.get(timeout=0.25)
            except queue_mod.Empty:
                if time.time() > deadline and not any(p.is_alive()
                                                      for p in procs):
                    break
                continue
            if idx not in mine_ids:
                continue
            received[idx] = status
            if status == "solved" and grid is not None:
                if verify_grid(grid, b):
                    found = (grid, idx)
                    break
                print(f"shard {kshard}: worker {wid} grid failed "
                      "verification; continuing", file=sys.stderr)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)

    dfs_work = sum(work_arr[:])
    total_work = expansion_nodes + dfs_work
    if found is not None:
        return finish(found[0], "solved", total_work, dfs_work,
                      expansion_nodes, len(subs), len(mine), found[1])
    if len(received) >= len(mine) and \
            all(s == "unsat" for s in received.values()):
        return finish(None, "unsat", total_work, dfs_work,
                      expansion_nodes, len(subs), len(mine), None)
    return finish(None, "timeout", total_work, dfs_work,
                  expansion_nodes, len(subs), len(mine), None)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="SOLVD nonogram solver — single deterministic shard")
    ap.add_argument("input")
    ap.add_argument("nshards", type=int)
    ap.add_argument("kshard", type=int)
    ap.add_argument("time_limit", type=float)
    ap.add_argument("result")
    args = ap.parse_args(argv)

    sys.setrecursionlimit(20000)
    result = {"shard": args.kshard, "nshards": args.nshards,
              "board_index": 0, "status": "error", "seconds": 0.0,
              "work": 0, "dfs_work": 0, "expansion_work": 0,
              "subproblems_total": 0, "subproblems_mine": 0,
              "found_subproblem": None}
    t0 = time.time()
    try:
        with open(args.input) as f:
            ch = json.load(f)
        if not isinstance(ch, dict) or not ch.get("boards"):
            raise ValueError("invalid input: missing non-empty 'boards'")
        b = ch["boards"][0]
        validate_board(b, 0)
        grid, info = solve_board_shard(b, args.nshards, args.kshard,
                                       args.time_limit)
        result.update(info)
        result["status"] = info["status"]
        if info["status"] == "solved" and grid is not None:
            result["grid"] = ["".join(str(v) for v in row) for row in grid]
            result["submit_body"] = {
                "challengeId": ch.get("challengeId", "local"),
                "encoding": "hex",
                "boards": [[hex_row(r) for r in grid]],
            }
    except Exception as e:  # result.json must exist in all cases
        result["status"] = "error"
        result["error"] = repr(e)
        print(f"solve_shard: {e!r}", file=sys.stderr)
    result["seconds"] = round(time.time() - t0, 3) \
        if result.get("seconds") in (0.0, None) else result["seconds"]
    with open(args.result, "w") as f:
        json.dump(result, f, indent=1)
    print(f"shard {args.kshard}/{args.nshards}: {result['status']} "
          f"in {result['seconds']:.2f}s (work={result['work']}, "
          f"mine={result['subproblems_mine']}/"
          f"{result['subproblems_total']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
