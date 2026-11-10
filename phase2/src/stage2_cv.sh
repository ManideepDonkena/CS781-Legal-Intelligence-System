#!/bin/bash
# Repeated cross-validation of the second stage: four document splits in parallel, mean F1 per
# threshold.  A single split moves by about +-1 F1, too much to compare model sets with.
# usage: stage2_cv.sh <name> <stage2.py arguments...>
PYS=/home/mani/miniconda3/envs/cs781/bin/python
cd "$(dirname "$0")"
name=$1; shift
for r in 0 1 2 3; do
  OMP_NUM_THREADS=12 $PYS stage2.py --data ../data/Phase_2_LNER --runs ../runs --thrs 0.05,0.07,0.1,0.15,0.2 --seeds ${STAGE2_SEEDS:-1} \
    --cv_seed $r "$@" > ../runs/stage2_cv/${name}_r$r.txt 2>&1 &
done
wait
$PYS - "$name" <<'PY'
import collections, glob, re, sys
acc = collections.defaultdict(list)
for p in sorted(glob.glob(f"../runs/stage2_cv/{sys.argv[1]}_r?.txt")):
    for line in open(p):
        m = re.match(r"thr=([\d.]+)\s+F1=([\d.]+) P=([\d.]+) R=([\d.]+)", line)
        if m:
            acc[m.group(1)].append([float(x) for x in m.groups()[1:]])
if not acc:
    print(open(p).read()[-800:])
for thr, v in acc.items():
    f = [x[0] for x in v]
    print(f"{sys.argv[1]:12} thr={thr} F1 mean={sum(f) / len(f):.2f} (min {min(f):.2f}, max {max(f):.2f}, n={len(f)}) "
          f"P={sum(x[1] for x in v) / len(v):.1f} R={sum(x[2] for x in v) / len(v):.1f}")
PY
