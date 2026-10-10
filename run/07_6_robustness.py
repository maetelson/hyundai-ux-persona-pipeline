"""포화 점검용 견고성 3종 (API 호출 없음).
  P1 라벨 오차 보정: 정답 v2(200건)의 코드별 정밀도·재현율로 유병률 보정 = 관측 × 정밀도 / 재현율, 정답 부트스트랩 95% 구간
  P2 출처별 재군집: 카페만 / 카페 외(블로그·리뷰) 표본으로 LCA(k 동일) 재적합 → 본 세그먼트 프로필과 헝가리안 매칭 코사인 유사도, 공통 글 ARI
  P3 작성자 일관성: 블로그 작성자(2글 이상)의 두 글이 같은 세그먼트일 확률 vs 무작위 짝 기준선
  python run/07_6_robustness.py
출력: data/output/tables/extra/r_label_correction.csv, r_source_refit.csv, r_meta.json
"""
import importlib
import json
import sys
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from stepmix.stepmix import StepMix

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
A = importlib.import_module("06_analyze")
X1 = importlib.import_module("07_extra")
RNG = np.random.default_rng(A.SEED)


def label_correction(df):
    gold = json.loads((A.ROOT / "data/gold/gold_v2_claude.json").read_text(encoding="utf-8"))
    lab = pd.read_parquet(A.STAGE / "post_labels.parquet").set_index("post_id")
    valid = lambda x: x in ("A_FOD", "B_SITUATION")
    both = [i for i in gold if i in lab.index and valid(gold[i]["layer"]) and valid(lab.loc[i, "layer"])]
    G = {i: set(gold[i]["situation"]) for i in both}
    P = {i: set(lab.loc[i, "situation"]) for i in both}
    n_all = len(df)
    obs = Counter(c for xs in df.situation for c in set(xs))
    rows = []
    for c in sorted({c for s in G.values() for c in s}):
        n_true = sum(c in G[i] for i in both)
        if n_true < 5:
            continue

        def factor(ids):
            tp = sum(c in G[i] and c in P[i] for i in ids)
            fp = sum(c not in G[i] and c in P[i] for i in ids)
            fn = sum(c in G[i] and c not in P[i] for i in ids)
            return (tp / (tp + fp)) / (tp / (tp + fn)) if tp else np.nan

        f = factor(both)
        boot = [factor(list(RNG.choice(both, len(both)))) for _ in range(1000)]
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        o = obs[c] / n_all
        rows.append({"code": c, "name": X1.A_SIT.get(c, c), "gold_n": n_true, "observed": round(o, 4), "factor": round(f, 2),
                     "corrected": round(o * f, 4), "corrected_lo": round(o * lo, 4), "corrected_hi": round(o * hi, 4)})
    return pd.DataFrame(rows).sort_values("factor", ascending=False)


def fit_profile(X, k, seed):
    m = StepMix(n_components=k, measurement="binary", n_init=5, random_state=seed, verbose=0, progress_bar=0).fit(X)
    return m, np.array(m.get_parameters()["measurement"]["pis"])


def source_refit(df, codes, k, main_prof):
    rows, aris = [], {}
    X_all = np.array([[int(c in xs) for c in codes] for xs in df.situation])
    core = X_all.sum(1) >= 2
    for name, mask in (("카페만", df.source.values == "naver_cafe"), ("카페 외(블로그·리뷰)", df.source.values != "naver_cafe")):
        sel = core & mask
        m, prof = fit_profile(X_all[sel], k, A.SEED)
        sim = np.array([[np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) for b in prof] for a in main_prof])
        r, c = linear_sum_assignment(-sim)
        pred = m.predict(X_all[sel])
        mapped = np.array([dict(zip(c, r))[p] for p in pred])
        aris[name] = round(adjusted_rand_score(df.segment.values[sel], mapped), 3)
        for a, b in zip(r, c):
            rows.append({"subset": name, "n": int(sel.sum()), "segment": int(a), "cosine": round(float(sim[a, b]), 3)})
    return pd.DataFrame(rows), aris


def author_consistency(df):
    a = df[df.author_hash.notna() & (df.segment >= 0)]
    groups = [g.segment.values for _, g in a.groupby("author_hash") if len(g) >= 2]
    same = [x == y for g in groups for x, y in combinations(g, 2)]
    share = a.segment.value_counts(normalize=True)
    base = float((share ** 2).sum())  # 무작위 두 글이 같은 세그먼트일 확률
    return {"authors": len(groups), "pairs": len(same), "same_segment": round(float(np.mean(same)), 3), "random_baseline": round(base, 3),
            "ci": A.wilson(int(sum(same)), len(same))}


def main():
    import yaml
    cb = yaml.safe_load((A.ROOT / "config/codebook_v2.yaml").read_text(encoding="utf-8"))
    X1.A_SIT.update({k: v.split(":")[0] for k, v in cb["families"]["situation"]["values"].items()})
    df = A.load().merge(pd.read_csv(A.OUT / "post_segment.csv"), on="post_id")
    lc = label_correction(df)
    lc.to_csv(X1.OUT / "r_label_correction.csv", index=False, encoding="utf-8-sig")
    print(lc.to_string(index=False))
    prof = pd.read_csv(A.OUT / "segment_code_probs.csv", index_col=0)
    codes = list(prof.columns)
    sr, aris = source_refit(df, codes, len(prof), prof.values)
    sr.to_csv(X1.OUT / "r_source_refit.csv", index=False, encoding="utf-8-sig")
    print(sr.pivot(index="segment", columns="subset", values="cosine"), aris)
    au = author_consistency(df)
    print(au)
    (X1.OUT / "r_meta.json").write_text(json.dumps({"source_ari": aris, "author": au}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
