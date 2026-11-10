"""Upload Phase 5 (default) or Phase 4 (--task cjpe) submission zips to Codabench in a fixed order and wait for
each score.

The login is read from the environment (never stored in a file, never printed):
  export CODABENCH_CSRF=...      # cookie "csrftoken" from a logged-in browser on codabench.org
  export CODABENCH_SESSION=...   # cookie "sessionid"
  python upload.py               # default order below; or pass zip paths explicitly

The first upload of a phase goes on the leaderboard automatically, so the planned entry (P5_S04 / P4_S04, chosen on
dev before any test score was seen) is uploaded first and the leaderboard entry is never moved afterwards.
"""
import argparse
import os
import sys

sys.path.insert(0, "/home/mani/cs781_phase1/Legal_LLM/scripts")
import codabench_phase2_client as c  # noqa: E402

CAPSTONE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
TASKS = {  # phase id, task id, submissions folder, upload order (planned leaderboard entry first)
    "lsi": (30687, 36368, os.path.join(CAPSTONE, "Phase5_LSI", "submissions"),
            ["P5_S04_mix_all", "P5_S03_mix_tapt", "P5_S02_scratch_xf_final", "P5_S01_tfidfxf2_lm3",
             "P5_S05_wordcount_only", "P5_S06_finetuned_only"]),
    "sum": (30688, 36369, os.path.join(CAPSTONE, "Phase6_SUM", "submissions"),
            ["P6_S02_hybrid150", "P6_S01_extractive", "P6_S03_hybrid250", "P6_S05_extractive_C16",
             "P6_S04_abstractive", "P6_S06_abstractive_trainonly"]),
    "cjpe": (30686, 36367, os.path.join(CAPSTONE, "Phase4_CJPE", "submissions"),
             ["P4_S04_tapt_fix_all", "P4_S03_tapt_fix", "P4_S02_lm3_tfidf_scratch", "P4_S01_lm3_tfidf",
              "P4_S06_finetuned_only", "P4_S05_wordcount_only"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="*", help="zip files in upload order (default: the six P5 files, P5_S04 first)")
    ap.add_argument("--check_only", action="store_true", help="only check the login and eligibility")
    ap.add_argument("--task", choices=list(TASKS), default="lsi")
    args = ap.parse_args()
    PHASE, TASK, SUB, ORDER = TASKS[args.task]
    csrf, session = os.environ.get("CODABENCH_CSRF"), os.environ.get("CODABENCH_SESSION")
    if not csrf or not session:
        sys.exit("set CODABENCH_CSRF and CODABENCH_SESSION (browser cookies csrftoken and sessionid)")
    cl = c.CodabenchClient(csrf_token=csrf, session_id=session)
    can = cl.can_make_submission(PHASE)
    print(f"logged in; can submit to phase {PHASE}:", can.get("can"), can.get("reason") or "")
    done = {s.get("id"): s for s in cl.list_submissions(PHASE)}
    print(f"existing submissions in phase {PHASE}:", len(done))
    if args.check_only or not can.get("can"):
        return
    zips = args.zips or [os.path.join(SUB, n, n + ".zip") for n in ORDER]
    assert all(os.path.exists(z) for z in zips), zips
    for z in zips:
        name = os.path.basename(z)
        print(f"\n=== uploading {name}", flush=True)
        sid = cl.submit(z, phase_id=PHASE, task_id=TASK, description=name.replace(".zip", ""))
        sub = cl.monitor(sid, poll_interval=15, timeout=1800)
        if sub is None or sub.get("status") != "Finished":
            print(f"stopping: {name} did not finish scoring ({sub and sub.get('status')})")
            return
    for s in cl.list_submissions(PHASE):
        print(f"  #{s.get('id')} {s.get('status'):9s} on leaderboard: {s.get('on_leaderboard')} | "
              f"{ {x.get('column_key'): x.get('score') for x in s.get('scores', [])} }")


if __name__ == "__main__":
    main()
