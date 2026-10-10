"""세그먼트 수(k) 선택: k별 BIC·엔트로피·부트스트랩 안정성 → 안정성 기준으로 k 결정.

규칙(v2, 2026-10-10): 평균 안정성 ≥0.8 그리고 최소 안정성 ≥0.7인 k 중 가장 큰 k (페르소나 해상도 우선).
  v1 규칙(최소 ≥0.5 중 평균 최대)은 k=4를 골랐으나 페르소나로 쓰기에 거칠어 결과 확인 후 변경 — 보고서에 명시.
  python run/06_1_select_k.py [hybrid|strict]
출력: data/output/tables{_strict}/k_selection.csv, 콘솔에 선택 k
"""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from stepmix.stepmix import StepMix

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("an", ROOT / "run/06_analyze.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)
K_RANGE, N_BOOT, SEED = range(4, 9), 8, 20261010

if __name__ == "__main__":
    df = an.load()
    codes = [c for c, _ in Counter(x for xs in df["situation"] for x in set(xs)).most_common(an.TOP_CODES)]
    X = np.array([[int(c in xs) for c in codes] for xs in df["situation"]])
    X = X[X.sum(1) >= 2]
    rows, rng = [], np.random.default_rng(SEED)
    for k in K_RANGE:
        m = StepMix(n_components=k, measurement="binary", n_init=10, random_state=SEED, verbose=0, progress_bar=0).fit(X)
        base = m.predict(X)
        jac = {c: [] for c in range(k)}
        for b in range(N_BOOT):
            idx = rng.choice(len(X), len(X), replace=True)
            pb = StepMix(n_components=k, measurement="binary", n_init=3, random_state=b, verbose=0, progress_bar=0).fit(X[idx]).predict(X)
            for c in range(k):
                a = set(np.where(base == c)[0])
                jac[c].append(max(len(a & set(np.where(pb == d)[0])) / max(len(a | set(np.where(pb == d)[0])), 1) for d in range(k)))
        st = [float(np.mean(v)) for v in jac.values()]
        sizes = np.bincount(base, minlength=k) / len(base)
        rows.append({"k": k, "bic": round(m.bic(X), 1), "stab_mean": round(np.mean(st), 3), "stab_min": round(min(st), 3),
                     "n_stable_075": sum(s >= 0.75 for s in st), "min_size": round(sizes.min(), 3)})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    ok = t[(t["stab_mean"] >= 0.8) & (t["stab_min"] >= 0.7)]
    best = ok.sort_values("k", ascending=False).iloc[0] if len(ok) else t.sort_values("stab_mean", ascending=False).iloc[0]
    t["selected"] = t["k"] == best["k"]
    t.to_csv(an.OUT / "k_selection.csv", index=False, encoding="utf-8-sig")
    print("선택 k =", int(best["k"]))
