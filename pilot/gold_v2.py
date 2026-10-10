"""정답 v2: 본 코퍼스(카페·블로그 포함)에서 새 표본을 뽑아 코드북 v2로 layer·상황 정답을 붙이고 본 라벨(Haiku)과 비교한다.
  python pilot/gold_v2.py sample   → data/gold/gold_v2_input.jsonl (본 라벨 없이 원문만, 블라인드)
  python pilot/gold_v2.py score    → data/gold/gold_v2_claude.json 과 post_labels 비교 → data/output/tables/extra/v_label_accuracy.csv
정답 형식: {"post_id": {"layer": "...", "situation": ["S_..."]}}
"""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
A = importlib.import_module("06_analyze")
G = ROOT / "data/gold"
OUT = ROOT / "data/output/tables/extra"
QUOTA = {"naver_cafe": 110, "naver_blog": 50, "app_review": 20, "kia_store": 20}  # ponytail: 카페 비중(76%)보다 블로그·리뷰를 약간 과대표집


def sample():
    cor = pd.read_parquet(A.STAGE / "corpus.parquet")
    lab = pd.read_parquet(A.STAGE / "post_labels.parquet")[["post_id"]]
    cor = cor.merge(lab, on="post_id")
    rows = pd.concat([cor[cor.source == s].sample(n, random_state=A.SEED + 2) for s, n in QUOTA.items()])
    rows = rows.sample(frac=1, random_state=A.SEED)
    (G / "gold_v2_input.jsonl").write_text("\n".join(json.dumps({"post_id": r.post_id, "source": r.source, "text": r.text}, ensure_ascii=False)
                                                       for r in rows.itertuples()), encoding="utf-8")
    print(len(rows), rows.source.value_counts().to_dict())


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0


def score():
    gold = json.loads((G / "gold_v2_claude.json").read_text(encoding="utf-8"))
    lab = pd.read_parquet(A.STAGE / "post_labels.parquet").set_index("post_id")
    src = {json.loads(l)["post_id"]: json.loads(l)["source"] for l in (G / "gold_v2_input.jsonl").read_text(encoding="utf-8").splitlines()}
    ids = [i for i in gold if i in lab.index]
    gl = [gold[i]["layer"] for i in ids]
    pl = [lab.loc[i, "layer"] for i in ids]
    valid = lambda x: x in ("A_FOD", "B_SITUATION")
    rows = [
        {"metric": "layer 정확도", "n": len(ids), "value": round(sum(a == b for a, b in zip(gl, pl)) / len(ids), 3)},
        {"metric": "layer 카파(κ)", "n": len(ids), "value": round(cohen_kappa_score(gl, pl), 3)},
        {"metric": "유효/제외 정확도", "n": len(ids), "value": round(sum(valid(a) == valid(b) for a, b in zip(gl, pl)) / len(ids), 3)},
    ]
    for s in sorted(set(src.values())):
        sid = [i for i in ids if src[i] == s]
        rows.append({"metric": f"layer 정확도 · {s}", "n": len(sid), "value": round(sum(gold[i]["layer"] == lab.loc[i, "layer"] for i in sid) / len(sid), 3)})
    # 상황 코드: 둘 다 유효로 본 글에서 비교(군집 입력과 같은 조건)
    both = [i for i in ids if valid(gold[i]["layer"]) and valid(lab.loc[i, "layer"])]
    codes = sorted({c for i in both for c in gold[i]["situation"]} | {c for i in both for c in lab.loc[i, "situation"]})
    TP = FP = FN = 0
    per = []
    for c in codes:
        tp = sum(c in gold[i]["situation"] and c in lab.loc[i, "situation"] for i in both)
        fp = sum(c not in gold[i]["situation"] and c in lab.loc[i, "situation"] for i in both)
        fn = sum(c in gold[i]["situation"] and c not in lab.loc[i, "situation"] for i in both)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        p, r, f = prf(tp, fp, fn)
        per.append({"metric": f"상황 F1 · {c}", "n": tp + fn, "value": f, "precision": p, "recall": r})
    p, r, f = prf(TP, FP, FN)
    rows.append({"metric": "상황 마이크로 F1", "n": len(both), "value": f, "precision": p, "recall": r})
    macro = [x["value"] for x in per if x["n"] >= 5]
    rows.append({"metric": "상황 매크로 F1 (정답 5건 이상 코드)", "n": len(macro), "value": round(sum(macro) / len(macro), 3) if macro else None})
    out = pd.DataFrame(rows + sorted(per, key=lambda x: -x["n"]))
    out.to_csv(OUT / "v_label_accuracy.csv", index=False, encoding="utf-8-sig")
    print(out.head(12).to_string(index=False))
    print(pd.crosstab(pd.Series(gl, name="정답"), pd.Series(pl, name="본 라벨")))


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
