"""Codabench leaderboard helper (uses the session of the Phase 2 client; never prints credentials).

  python leaderboard.py show [--phase 30685] [--top 10]
  python leaderboard.py set <submission id>      # put this submission on the leaderboard (replaces ours)
"""
import argparse
import sys

sys.path.insert(0, "/home/mani/cs781_phase1/Legal_LLM/scripts")
import codabench_phase2_client as c  # noqa: E402


def client():
    return [v for v in vars(c).values() if isinstance(v, type) and hasattr(v, "get_phase_info")][0]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["show", "set"])
    ap.add_argument("sub", nargs="?", type=int)
    ap.add_argument("--phase", type=int, default=30685)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    cl = client()
    if args.cmd == "set":
        r = cl.session.post(f"{c.BASE_URL}/api/submissions/{args.sub}/submission_leaderboard_connection/",
                            headers={"Content-Type": "application/json"}, json={})
        s = cl.get_submission(args.sub)
        print("status", r.status_code, "| submission", args.sub, "on leaderboard:", s.get("on_leaderboard"))
    j = cl.session.get(f"{c.BASE_URL}/api/phases/{args.phase}/get_leaderboard/").json()
    rows = []
    for s in j.get("submissions", []):
        sc = {x.get("column_key", x.get("index")): x.get("score") for x in s.get("scores", [])}
        rows.append((float(sc.get("primary_metric_1", 0)), s.get("owner"), sc.get("secondary_metric_1"),
                     sc.get("secondary_metric_2")))
    rows.sort(reverse=True)
    for rank, (score, owner, p, r) in enumerate(rows[:args.top], 1):
        print(rank, owner, score, p, r)
    print("entries", len(rows))


if __name__ == "__main__":
    main()
