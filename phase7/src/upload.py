"""Upload Phase 7 Prior Case Retrieval (PCR) submissions to Codabench.

Competition: CS-781 Capstone (Competition ID: 18186)
Phase ID: 30701
Task ID: 36382

Planned Leaderboard Entry: P7_S06_apex_consensus_ensemble (highest Dev Micro-F1@7: 0.220122)

The login is read from the environment:
  export CODABENCH_CSRF=...      # cookie "csrftoken" from a logged-in browser session on codabench.org
  export CODABENCH_SESSION=...   # cookie "sessionid"
  python upload.py               # default upload order: planned entry first, then remaining 5
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/mani/cs781_phase1/Legal_LLM/scripts")
import codabench_phase2_client as c  # noqa: E402

PHASE_ID = 30701
TASK_ID = 36382
SUB_DIR = Path(__file__).resolve().parent.parent / "submissions"

DEFAULT_ORDER = [
    "P7_S06_apex_citation_ensemble",
    "P7_S07_rrf_consensus_ensemble",
    "P7_S05_phrase_bm25_from_scratch",
    "P7_S03_bge_m3_dense",
    "P7_S04_hybrid_phase1_bm25",
    "P7_S02_phase1_sc295m_dense",
    "P7_S01_bm25_unigram_baseline",
]


def main():
    parser = argparse.ArgumentParser(description="Upload Phase 7 submissions to Codabench")
    parser.add_argument("zips", nargs="*", help="Optional explicit list of zip files to upload")
    parser.add_argument("--check_only", action="store_true", help="Only verify login and submission quota")
    args = parser.parse_args()

    csrf = os.environ.get("CODABENCH_CSRF")
    session = os.environ.get("CODABENCH_SESSION")

    if not csrf or not session:
        print("\n[!] ERROR: CODABENCH_CSRF and/or CODABENCH_SESSION environment variables are missing.")
        print("To upload via API, get fresh cookies from your browser session on codabench.org:")
        print("  1. Log into https://www.codabench.org")
        print("  2. Open DevTools (F12) -> Application/Storage -> Cookies -> https://www.codabench.org")
        print("  3. Copy 'csrftoken' and 'sessionid'")
        print("  4. Run:")
        print("     export CODABENCH_CSRF=\"<csrftoken>\"")
        print("     export CODABENCH_SESSION=\"<sessionid>\"")
        print("     python upload.py\n")
        print("Alternatively, you can manually upload the generated ZIP archives from:")
        print(f"  {SUB_DIR.resolve()}\n")
        sys.exit(1)

    client = c.CodabenchClient(csrf_token=csrf, session_id=session)
    can = client.can_make_submission(PHASE_ID)
    print(f"Logged in. Can submit to phase {PHASE_ID}:", can.get("can"), can.get("reason") or "")

    existing = client.list_submissions(PHASE_ID)
    print(f"Existing submissions in phase {PHASE_ID}: {len(existing)}")

    if args.check_only or not can.get("can"):
        return

    targets = args.zips or [str(SUB_DIR / name / f"{name}.zip") for name in DEFAULT_ORDER]
    for z in targets:
        if not os.path.exists(z):
            sys.exit(f"Archive not found: {z}")

    print(f"\nUploading {len(targets)} submissions in sequence...")
    for zpath in targets:
        zname = os.path.basename(zpath)
        sub_desc = zname.replace(".zip", "")
        print(f"\n{'='*60}\nUploading {zname}...", flush=True)

        sid = client.submit(zpath, phase_id=PHASE_ID, task_id=TASK_ID, description=sub_desc)
        print(f"Submitted. Submission ID: #{sid}. Waiting for scoring...", flush=True)

        sub_info = client.monitor(sid, poll_interval=15, timeout=1800)
        status = sub_info.get("status") if sub_info else "Unknown"
        scores = {x.get("column_key"): x.get("score") for x in (sub_info.get("scores", []) if sub_info else [])}
        print(f"Result for #{sid} ({zname}): {status} | Scores: {scores}", flush=True)

        if status != "Finished":
            print(f"WARNING: #{sid} finished with status '{status}'. Stopping sequence.")
            break

    print("\nSummary of all submissions in Phase 7:")
    for s in client.list_submissions(PHASE_ID):
        print(
            f"  #{s.get('id')} {s.get('status'):9s} | Leaderboard: {s.get('on_leaderboard')} | "
            f"File: {s.get('filename')} | Scores: { {x.get('column_key'): x.get('score') for x in s.get('scores', [])} }"
        )


if __name__ == "__main__":
    main()
