"""데모 트랙: 규칙 기반 클루 추출 → post_demo (basis를 분리 기록).

  python run/05_5_demo.py
출력: data/stage/post_demo.parquet, 콘솔에 basis별 커버리지
"""
import json
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data/stage"
RULES = yaml.safe_load((ROOT / "config/demo_clue_rules.yaml").read_text(encoding="utf-8"))
COMPILED = {k: [(r["value"], r.get("priority", 0), re.compile(r["pattern"])) for r in v] for k, v in RULES.items()}


def first_match(dim, text):
    hits = sorted((p, v, m.group(0)) for v, p, rx in COMPILED[dim] if (m := rx.search(text)))
    return (hits[0][1], hits[0][2]) if hits else (None, None)


def all_matches(dim, text):
    return [(v, m.group(0)) for v, _, rx in COMPILED[dim] if (m := rx.search(text))]


def extract(row):
    meta = json.loads(row.meta)
    out = []
    v, q = first_match("life_stage", row.text)
    if v:
        out.append(("life_stage", v, "text", q))
    elif meta.get("channel"):
        v, q = first_match("channel_life_stage", meta["channel"])
        if v:
            out.append(("life_stage", v, "channel", q))
    g = all_matches("gender", row.text)
    if len({x for x, _ in g}) == 1:  # 상충 단서면 판단 불가
        out.append(("gender", g[0][0], "text", g[0][1]))
    vehicle_text = " ".join(str(meta.get(k, "")) for k in ("vehicle", "product")) + " " + row.text
    v, q = first_match("ev", vehicle_text)
    if v:
        out.append(("ev", v, "meta" if meta.get("vehicle") and q in str(meta.get("vehicle")) else "text", q))
    for v, q in all_matches("brand", vehicle_text):
        out.append(("brand", v, "meta" if meta.get("vehicle") else "text", q))
    return [(row.post_id, d, val, b, q) for d, val, b, q in out]


if __name__ == "__main__":
    c = pd.read_parquet(STAGE / "corpus.parquet")
    rows = [x for r in c.itertuples() for x in extract(r)]
    demo = pd.DataFrame(rows, columns=["post_id", "dimension", "value", "basis", "clue"])
    demo.to_parquet(STAGE / "post_demo.parquet", index=False)
    n = len(c)
    for dim, g in demo.groupby("dimension"):
        cov = g.groupby("basis")["post_id"].nunique()
        print(f"{dim:10} " + "  ".join(f"{b} {k:,} ({k / n:.1%})" for b, k in cov.items()))
    ls = demo[demo.dimension == "life_stage"]
    print(ls.groupby(["value", "basis"]).size().unstack(fill_value=0))
