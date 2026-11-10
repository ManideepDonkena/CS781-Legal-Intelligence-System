#!/usr/bin/env python3
"""Resumable GPU/CPU job queue for the Legal-LLM studies.

  python studies/run_queue.py            # run the scheduler (foreground; use setsid nohup to detach)
  python studies/run_queue.py --status   # print queue state
  python studies/run_queue.py --reset ID # mark a failed/blocked job pending again

Jobs live in studies/jobs.json (re-read every cycle, so a `plan` job may append new jobs):
  {"id", "kind": "gpu"|"cpu", "cmd": [..] or "shell string", "deps": [ids], "gpu_mem_gb": 20,
   "priority": 0 (lower runs first), "min_gpu_free_gb": optional, "gpus": optional allow-list}
State lives in results/queue_state.json. A job whose process died while 'running' (e.g. reboot)
is re-queued; training jobs resume from last.pt automatically (engine.py).
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "studies" / "jobs.json"
STATE = ROOT / "results" / "queue_state.json"
LOGS = ROOT / "logs_studies"
PY = "/home/mani/miniconda3/envs/phase_1_capstone/bin/python"
MAX_CPU_JOBS = 3
ALL_GPUS = [0, 1, 2, 3, 4, 5]
GPU_MARGIN_GB = 2.0


def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def load(p, default):
    try: return json.loads(Path(p).read_text())
    except Exception: return default
def save(p, v):
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(v, indent=1)); tmp.replace(p)


def gpu_free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    free = {}
    for line in out.strip().splitlines():
        i, tot, used = (int(x) for x in line.split(","))
        free[i] = (tot - used) / 1024
    return free


def alive(pid):
    try: os.kill(pid, 0); return True
    except OSError: return False


def exit_code(pid_file):
    try: return int(Path(pid_file).read_text().strip())
    except Exception: return None


def launch(job, gpu):
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"{job['id']}.log"
    rc_file = LOGS / f"{job['id']}.rc"
    rc_file.unlink(missing_ok=True)
    cmd = job["cmd"] if isinstance(job["cmd"], str) else " ".join(map(str, job["cmd"]))
    # wrapper writes the exit code so a restarted scheduler can learn the outcome of detached jobs
    wrapped = f"({cmd}); echo $? > {rc_file}"
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="true", PYTHON=PY)
    if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    f = open(log, "a"); f.write(f"\n=== {now()} launch on gpu={gpu}: {cmd}\n"); f.flush()
    p = subprocess.Popen(["bash", "-c", wrapped], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
                         start_new_session=True, stdin=subprocess.DEVNULL)
    return p.pid


def status(jobs, state):
    for j in jobs:
        s = state.get(j["id"], {})
        print(f"{j['id']:<44} {s.get('status','pending'):<9} gpu={s.get('gpu')} {s.get('start','')} {s.get('end','')} rc={s.get('rc')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset", nargs="*")
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--exclude-gpus", default="", help="comma list never used by the queue")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    STATE.parent.mkdir(exist_ok=True)
    jobs, state = load(JOBS, []), load(STATE, {})
    if a.status:
        status(jobs, state); return
    if a.reset is not None:
        for i in a.reset: state.pop(i, None)
        save(STATE, state); return
    excluded = {int(x) for x in a.exclude_gpus.split(",") if x}
    print(f"[{now()}] queue scheduler up; {len(jobs)} jobs", flush=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    while not stop["flag"]:
        jobs, state = load(JOBS, []), load(STATE, {})
        byid = {j["id"]: j for j in jobs}
        # --- reap finished / dead jobs ---
        for jid, s in list(state.items()):
            if s["status"] != "running": continue
            rc = exit_code(LOGS / f"{jid}.rc")
            if rc is not None:
                s.update(status="done" if rc == 0 else "failed", rc=rc, end=now())
                print(f"[{now()}] {jid} -> {s['status']} (rc={rc})", flush=True)
            elif not alive(s["pid"]):
                s.update(status="pending"); s.pop("pid", None)   # died without rc (reboot/kill): requeue
                print(f"[{now()}] {jid} lost; requeued", flush=True)
        save(STATE, state)
        # --- schedule ---
        free = gpu_free_gb()
        busy_gpus = {s["gpu"] for s in state.values() if s["status"] == "running" and s.get("gpu") is not None}
        cpu_running = sum(1 for jid, s in state.items() if s["status"] == "running" and s.get("gpu") is None)
        pending = [j for j in jobs if state.get(j["id"], {}).get("status", "pending") == "pending"]
        pending.sort(key=lambda j: (j.get("priority", 0), jobs.index(j)))
        for j in pending:
            deps = j.get("deps", [])
            if any(state.get(d, {}).get("status") in ("failed", "blocked") for d in deps):
                state[j["id"]] = {"status": "blocked", "end": now()}; continue
            if any(state.get(d, {}).get("status") != "done" for d in deps): continue
            if j["kind"] == "cpu":
                if cpu_running >= MAX_CPU_JOBS: continue
                pid = launch(j, None); cpu_running += 1
                state[j["id"]] = {"status": "running", "pid": pid, "gpu": None, "start": now()}
            else:
                need = j.get("gpu_mem_gb", 20) + GPU_MARGIN_GB
                cands = [g for g in (j.get("gpus") or ALL_GPUS) if g not in excluded and g not in busy_gpus and free.get(g, 0) >= need]
                if not cands: continue
                g = max(cands, key=lambda x: free[x])
                pid = launch(j, g); busy_gpus.add(g); free[g] -= need
                state[j["id"]] = {"status": "running", "pid": pid, "gpu": g, "start": now()}
            print(f"[{now()}] started {j['id']} on gpu={state[j['id']]['gpu']}", flush=True)
        save(STATE, state)
        remaining = [j for j in jobs if state.get(j["id"], {}).get("status", "pending") in ("pending", "running")]
        if a.once or not remaining:
            print(f"[{now()}] queue idle/finished", flush=True); break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
