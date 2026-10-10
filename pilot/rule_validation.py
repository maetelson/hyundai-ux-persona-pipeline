"""규칙(키워드 사전) 판정 검증: 표본 추출 → 사람(또는 Claude Code 세션) 정답 → 정밀도·재현율.
  V1 E4 '없어서 불만'  V2 E4 '있어서 기쁨'  V3 E1 가격 반응(비쌈/적정)  → 정밀도
  V4 E2 스토어 문제 유형 5개 (무작위 리뷰) → 유형별 정밀도·재현율

  python pilot/rule_validation.py sample   → data/gold/rule_val_{v1..v4}.jsonl (정답 칸 비어 있음)
  python pilot/rule_validation.py score    → data/gold/rule_val_{..}_gold.json 과 비교 → data/output/tables/extra/v_rule_validation.csv
정답 파일 형식: {"id": true/false} (V1~V3), {"id": ["유형", ...]} (V4)
"""
import importlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
A = importlib.import_module("06_analyze")
X1 = importlib.import_module("07_extra")
G = ROOT / "data/gold"
N = 100


def flagged(df, rx_ctx, sent_sign):
    out = []
    for r in df.itertuples():
        if (int(r.sentiment) > 0) != (sent_sign > 0) or int(r.sentiment) == 0:
            continue
        for f, rx in X1.FEAT_RE.items():
            m = rx.search(r.text)
            if not m:
                continue
            ctx = r.text[max(0, m.start() - X1.WIN):m.end() + X1.WIN]
            if rx_ctx.search(ctx):
                out.append({"id": f"{r.post_id}:{f}", "feature": f, "context": ctx.replace("\n", " ")})
    return out


def sample():
    random.seed(A.SEED)
    df = A.load()
    sets = {"v1": flagged(df, X1.ABSENT, -1), "v2": flagged(df, X1.DELIGHT, 1)}
    men = pd.read_csv(X1.OUT / "e1_price_mentions.csv")
    men = men[men.react.notna() & (men.react != "")].drop_duplicates(["post_id", "feature"])
    txt = dict(zip(df.post_id, df.text))
    v3 = []
    for r in men.itertuples():
        t = txt.get(r.post_id, "")
        m = X1.FEAT_RE[r.feature].search(t)
        if m:
            v3.append({"id": f"{r.post_id}:{r.feature}", "feature": r.feature, "label": r.react, "amount": r.amount,
                       "context": t[max(0, m.start() - 90):m.end() + 90].replace("\n", " ")})
    sets["v3"] = v3
    st = df[df.source == "kia_store"]
    sets["v4"] = [{"id": r.post_id, "text": r.text.replace("\n", " "), "pred": [k for k, rx in X1.ISS_RE.items() if rx.search(r.text)]}
                  for r in st.sample(N, random_state=A.SEED).itertuples()]
    for k, v in sets.items():
        v = random.sample(v, min(N, len(v)))
        (G / f"rule_val_{k}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in v), encoding="utf-8")
        print(k, len(v))


def score():
    rows = []
    rx_now = {"v1": X1.ABSENT, "v2": X1.DELIGHT}
    for k, name in (("v1", "E4 없어서 불만"), ("v2", "E4 있어서 기쁨"), ("v3", "E1 가격 반응")):
        gold = json.loads((G / f"rule_val_{k}_gold.json").read_text(encoding="utf-8"))
        ctx = {json.loads(l)["id"]: json.loads(l)["context"] for l in (G / f"rule_val_{k}.jsonl").read_text(encoding="utf-8").splitlines()}
        for ver, keep in (("v1 규칙", gold), ("현재 규칙(같은 표본 재측정)", {i: g for i, g in gold.items() if k not in rx_now or rx_now[k].search(ctx[i])})):
            n, tp = len(keep), sum(keep.values())
            lo, hi = A.wilson(tp, n)
            rows.append({"check": name, "rule": ver, "type": "정밀도", "n": n, "value": round(tp / n, 3) if n else None, "ci_lo": lo, "ci_hi": hi})
    items = {json.loads(l)["id"]: json.loads(l) for l in (G / "rule_val_v4.jsonl").read_text(encoding="utf-8").splitlines()}
    gold = json.loads((G / "rule_val_v4_gold.json").read_text(encoding="utf-8"))
    for t in X1.ISSUES:
        pred = {i for i, x in items.items() if X1.ISS_RE[t].search(x["text"])}
        true = {i for i, g in gold.items() if t in g}
        tp = len(pred & true)
        for typ, num, den in (("정밀도", tp, len(pred)), ("재현율", tp, len(true))):
            lo, hi = A.wilson(num, den)
            rows.append({"check": f"E2 {t}", "rule": "현재 규칙", "type": typ, "n": den, "value": round(num / den, 3) if den else None, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(X1.OUT / "v_rule_validation.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
