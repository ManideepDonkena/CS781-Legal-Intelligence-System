"""Package test summaries ({id: summary} JSON) as a Codabench submission: submissions/<name>/<name>.zip holding
prediction.json = [{"id": ..., "summary": ...}, ...] at the archive root, in test-file order.

  python make_sub.py ../runs/ext_v1_all/test_summaries.json P6_S01_extractive
"""
import json
import os
import sys
import zipfile

import summ


def main(path, name):
    test = summ.load("test")
    s = json.load(open(path))
    assert set(s) == {d["id"] for d in test}, "ids differ from the test file"
    assert all(s[d["id"]].strip() for d in test), "empty summary"
    pred = [{"id": d["id"], "summary": s[d["id"]]} for d in test]
    out = f"/home/mani/cs781/Capstone/Phase6_SUM/submissions/{name}"
    os.makedirs(out, exist_ok=True)
    json.dump(pred, open(f"{out}/prediction.json", "w"), ensure_ascii=False)
    with zipfile.ZipFile(f"{out}/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{out}/prediction.json", "prediction.json")
    words = sum(len(p["summary"].split()) for p in pred) / len(pred)
    print(f"{out}/{name}.zip: {len(pred)} summaries, mean {words:.0f} words, from {path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
