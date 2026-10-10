"""종합: E1~E9 결과를 기능별 FoD 후보 점수표와 세그먼트별 FoD 처방으로 묶는다 (API 호출 없음).
  S1 점수표: 기준 6개(수요·불편·매력 +, 불안·대안·유료화 반감 −) 순위 평균(Borda) + 가중치 몬테카를로 민감도
     판정 규칙(결과 보기 전 고정): 유료화 반감 ≥10% 또는 Kano '당연'(안정성 ≥0.7) → 기본 탑재 권장,
     나머지 중 점수 상위 절반 → FoD 판매 후보, 하위 절반 → 보류
  S2 처방: 세그먼트별 과대 언급 기능(lift), 4가지 힘, 대체 유형, 가격 앵커, 막는 태도 → 판매 조건 규칙

  python run/07_3_synthesis.py
출력: data/output/tables/extra/s1_scorecard.csv, s1_sensitivity.csv, s2_prescriptions.csv
"""
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
A = importlib.import_module("06_analyze")
X1 = importlib.import_module("07_extra")
X2 = importlib.import_module("07_2_extra")
X = X1.OUT
RNG = np.random.default_rng(A.SEED)
CRIT = {"demand": 1, "push": 1, "pull": 1, "anxiety": -1, "habit": -1, "pay_resistance": -1}
CRIT_KO = {"demand": "수요(언급 수)", "push": "불편", "pull": "매력", "anxiety": "불안", "habit": "기존 대안", "pay_resistance": "유료화 반감"}
COND = {
    "T_HESITATE": "무료 체험·짧은 기간권으로 망설임 낮추기",
    "T_CALCULATE": "월·연·평생 가격 비교와 짧은 기간권",
    "T_DISTRUST": "설치 후 7일 전액 환불 보장",
    "T_OWNERSHIP_WORRY": "계정 귀속·다음 차 이전 허용",
    "T_RESIST_HWLOCK": "기본 탑재 또는 하드웨어 포함 패키지",
    "T_RESIST_DOUBLEPAY": "기본 탑재 또는 하드웨어 포함 패키지",
}


def price_ref(pa, f):
    g = pa[(pa.feature == f) & (pa.period.isin(["월", "평생", "연"])) & (pa.n_posts >= 5)].sort_values("n_posts", ascending=False)
    return "; ".join(f"{r.period} {r.median:,.0f}원(n={r.n_posts})" for r in g.itertuples())


def scorecard():
    kn, fo, pa = (pd.read_csv(X / f"{n}.csv") for n in ("e4_kano", "e6_forces", "e1_price_anchors"))
    sc = kn[["feature", "n", "kano_class", "boot_stability", "pay_resistance"]].merge(
        fo[["feature", "push", "pull", "anxiety", "habit", "top_anxiety"]], on="feature")
    sc["demand"] = np.log(sc.n)
    R = pd.DataFrame({c: (sc[c] * s).rank(ascending=False) for c, s in CRIT.items()})  # 1 = 가장 좋음
    sc["mean_rank"] = R.mean(1).round(2)
    base = (sc.pay_resistance >= 0.10) | ((sc.kano_class.str.startswith("당연")) & (sc.boot_stability >= 0.7))
    rest = sc[~base].mean_rank
    sc["decision"] = np.where(base, "기본 탑재 권장", np.where(sc.mean_rank <= rest.median(), "FoD 판매 후보", "보류"))
    sc["price_ref"] = sc.feature.map(lambda f: price_ref(pa, f))
    # 가중치 민감도: 디리클레(1) 가중치 2,000회, FoD 판정 대상(기본 탑재 제외) 안에서의 순위
    Rr = R[~base].values
    names = sc.feature[~base].values
    W = RNG.dirichlet(np.ones(len(CRIT)), 2000)
    ranks = np.argsort(np.argsort(W @ Rr.T, axis=1), axis=1) + 1
    half = int(np.ceil(len(names) / 2))
    sens = pd.DataFrame({"feature": names, "p_top_half": (ranks <= half).mean(0).round(3), "p_top3": (ranks <= 3).mean(0).round(3),
                         "rank_p05": np.percentile(ranks, 5, axis=0), "rank_median": np.median(ranks, axis=0), "rank_p95": np.percentile(ranks, 95, axis=0)})
    sc = sc.merge(sens, on="feature", how="left")
    return sc.sort_values(["decision", "mean_rank"]), sens


def prescriptions(df, sc):
    seg = pd.read_csv(A.OUT / "post_segment.csv")
    nar = A.yaml.safe_load((A.ROOT / "config/report_narrative.yaml").read_text(encoding="utf-8")).get("segments", {})
    sx = pd.read_csv(X / "e3_sub_segment.csv").set_index("segment")
    pa = pd.read_csv(X / "e1_price_anchors.csv")
    df = df.merge(seg, on="post_id")
    M = pd.DataFrame({f: df.text.str.contains(rx) for f, rx in X1.FEAT_RE.items()})
    overall = M.mean()
    dec = dict(zip(sc.feature, sc.decision))
    rows = []
    SUBK = {"aftermarket": "사제 용품", "diy": "직접 해결", "other_app": "다른 앱", "give_up": "포기"}
    for s in sorted(x for x in df.segment.unique() if x >= 0):
        m = (df.segment == s).values
        g, Mg = df[m], M[m]
        lift = (Mg.mean() / overall).round(2)
        cnt = Mg.sum()
        feats = [f for f in lift.sort_values(ascending=False).index if cnt[f] >= 30 and lift[f] > 1.2][:3]
        att = g.attitude.map(set)
        forces = {"push": ((g.layer == "B_SITUATION") & (g.sentiment.astype(int) < 0) & (g.intensity.astype(int) >= 2)).mean(),
                  "pull": att.map(lambda x: bool(x & X2.PULL)).mean(), "anxiety": att.map(lambda x: bool(x & X2.ANX)).mean(),
                  "habit": g.substitute.isin(["aftermarket", "diy", "other_app"]).mean()}
        top_att = Counter(t for x in att for t in x & X2.ANX).most_common(2)
        subs = sx.loc[s].drop("n").astype(float) if s in sx.index else pd.Series(dtype=float)
        block = "weak" if max(forces["habit"], forces["anxiety"]) < 0.03 else "habit" if forces["habit"] > forces["anxiety"] else "anxiety"
        sub_top = f"{SUBK.get(subs.idxmax(), subs.idxmax())} {subs.max():.0%}" if len(subs) else ""
        cond = ("막는 힘이 약함(3% 미만) — 판매 조건보다 상황 기반 노출로 필요 환기" if block == "weak"
                else f"대안 대비 가격·설치 편의·순정 연동 강조 (대체 행동 중 {sub_top})" if block == "habit"
                else " / ".join(dict.fromkeys(COND[t] for t, _ in top_att if t in COND)) or "판매 조건보다 기능 인지 확대")
        rows.append({"segment": s, "name": nar.get(s, {}).get("name", f"세그먼트 {s}"), "n": int(m.sum()),
                     "features": " | ".join(f"{f} (lift {lift[f]}, {dec.get(f, '')})" for f in feats) or "과대 언급 FoD 기능 없음",
                     **{k: round(v, 3) for k, v in forces.items()}, "blocking": block,
                     "top_anxiety": "|".join(f"{t}:{c}" for t, c in top_att),
                     "substitute_top": sub_top,
                     "price_ref": " / ".join(filter(None, (f"{f}: {price_ref(pa, f)}" if price_ref(pa, f) else "" for f in feats))),
                     "condition": cond})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = A.load()
    sc, sens = scorecard()
    sc.to_csv(X / "s1_scorecard.csv", index=False, encoding="utf-8-sig")
    print(sc[["feature", "decision", "mean_rank", "kano_class", "pay_resistance", "p_top_half", "rank_p05", "rank_p95", "price_ref"]].to_string(index=False))
    pr = prescriptions(df, sc)
    pr.to_csv(X / "s2_prescriptions.csv", index=False, encoding="utf-8-sig")
    print(pr.drop(columns=["price_ref"]).to_string(index=False))
