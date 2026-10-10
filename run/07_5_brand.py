"""브랜드별 FoD 반응 비교: 현대 · 기아 · 제네시스 · 테슬라 · 기타 수입 (API 호출 없음).
  - 브랜드 = 데모 트랙 brand 단서(본문·메타·채널). 수입 중 단서가 '테슬라'면 테슬라로 분리. 한 글에 여러 브랜드면 제외.
  - A층(FoD 직접) 글: 태도 10개 비율, 부정 비율, 지불 신호(수용/거부), 저니 단계 분포 — Wilson 95% CI
  - 브랜드 × 태도 카이제곱 독립성 검정(scipy)
  python run/07_5_brand.py
출력: data/output/tables/extra/e10_brand_attitude.csv, e10_brand_summary.csv, e10_meta.json
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).parent))
A = importlib.import_module("06_analyze")
X1 = importlib.import_module("07_extra")
ATT = ["T_RESIST_HWLOCK", "T_RESIST_DOUBLEPAY", "T_ACCEPT_SW", "T_HESITATE", "T_SATISFIED", "T_DISAPPOINT", "T_DISTRUST", "T_CALCULATE", "T_OWNERSHIP_WORRY", "T_WANT"]


def brand_of(demo):
    b = demo[demo.dimension == "brand"].copy()
    b["brand"] = np.where((b.value == "수입") & b.clue.str.contains("테슬라", na=False), "테슬라", np.where(b.value == "수입", "기타 수입", b.value))
    n = b.groupby("post_id").brand.nunique()
    return b[b.post_id.isin(n[n == 1].index)].drop_duplicates("post_id")[["post_id", "brand"]]


def main():
    df = A.load()
    df = df[df.layer == "A_FOD"].merge(brand_of(pd.read_parquet(A.STAGE / "post_demo.parquet")), on="post_id")
    # 민감도: 기아 스토어 리뷰(구매자 후기, 긍정 쏠림)를 뺀 기아
    kia_ns = df[(df.brand == "기아") & (df.source != "kia_store")].assign(brand="기아(스토어 리뷰 제외)")
    df = pd.concat([df, kia_ns])
    order = ["현대", "기아", "기아(스토어 리뷰 제외)", "제네시스", "테슬라", "기타 수입"]
    rows, summ = [], []
    for br in order:
        g = df[df.brand == br]
        n = len(g)
        if n < 30:
            continue
        att = g.attitude.map(set)
        for t in ATT:
            k = int(att.map(lambda s: t in s).sum())
            lo, hi = A.wilson(k, n)
            rows.append({"brand": br, "attitude": t, "n": n, "share": round(k / n, 3), "ci_lo": lo, "ci_hi": hi})
        neg = int((g.sentiment.astype(int) < 0).sum())
        acc, rej = int((g.wtp_signal == "accept").sum()), int((g.wtp_signal == "reject").sum())
        top_j = g.journey.explode().value_counts(normalize=True).head(3)
        summ.append({"brand": br, "n": n, "neg_share": round(neg / n, 3), "neg_ci": A.wilson(neg, n),
                     "accept_share": round(acc / n, 3), "reject_share": round(rej / n, 3),
                     "accept_vs_reject": round(acc / rej, 2) if rej else None,
                     "top_journey": " | ".join(f"{X1.A_J.get(j, j)} {v:.0%}" for j, v in top_j.items()),
                     "top_source": g.source.value_counts(normalize=True).round(2).to_dict()})
    at = pd.DataFrame(rows)
    at_main = at[at.brand != "기아(스토어 리뷰 제외)"]
    piv = at_main.assign(k=(at.share * at.n).round()).pivot(index="brand", columns="attitude", values="k").fillna(0)
    chi2, p, dof, _ = chi2_contingency(piv.values + 0.5)
    meta = {"n_afod_with_brand": int((df.brand != "기아(스토어 리뷰 제외)").sum()), "chi2": round(float(chi2), 1), "dof": int(dof), "p": float(p),
            "cramers_v": round(float(np.sqrt(chi2 / (piv.values.sum() * (min(piv.shape) - 1)))), 3)}
    at.to_csv(X1.OUT / "e10_brand_attitude.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summ).to_csv(X1.OUT / "e10_brand_summary.csv", index=False, encoding="utf-8-sig")
    (X1.OUT / "e10_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(summ).drop(columns=["top_source"]).to_string(index=False))
    print(at.pivot(index="attitude", columns="brand", values="share")[[b for b in order if b in at.brand.unique()]].to_string())
    print(meta)


if __name__ == "__main__":
    main()
