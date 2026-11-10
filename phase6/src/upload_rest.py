"""Upload the remaining Phase 6 zips one at a time, waiting for each score through the submission list (the
single-submission endpoint returned HTTP 500 on 2 Nov). Cookies come from CODABENCH_CSRF / CODABENCH_SESSION."""
import os, sys, time
sys.path.insert(0, '/home/mani/cs781_phase1/Legal_LLM/scripts')
import codabench_phase2_client as c
PHASE, TASK = 30688, 36369
SUB = '/home/mani/cs781/Capstone/Phase6_SUM/submissions'
REST = ["P6_S03_hybrid250", "P6_S05_extractive_C16", "P6_S04_abstractive", "P6_S06_abstractive_trainonly"]
cl = c.CodabenchClient(csrf_token=os.environ['CODABENCH_CSRF'], session_id=os.environ['CODABENCH_SESSION'])

def wait(sid):
    for _ in range(150):
        for s in cl.list_submissions(PHASE):
            if s.get('id') == sid and s.get('status') in ('Finished', 'Failed', 'Cancelled'):
                print(f"#{sid} {s.get('status')} {({x.get('column_key'): x.get('score') for x in s.get('scores', [])})}", flush=True)
                return s.get('status')
        time.sleep(20)
    print(f"#{sid} timed out waiting", flush=True)


for n in REST:
    print(f"=== uploading {n}", flush=True)
    sid = cl.submit(f"{SUB}/{n}/{n}.zip", phase_id=PHASE, task_id=TASK, description=n)
    if wait(sid) != 'Finished':
        print('stopping'); break
for s in cl.list_submissions(PHASE):
    print(f"  #{s.get('id')} {s.get('status')} on leaderboard {s.get('on_leaderboard')} {s.get('filename')} "
          f"{({x.get('column_key'): x.get('score') for x in s.get('scores', [])})}", flush=True)
