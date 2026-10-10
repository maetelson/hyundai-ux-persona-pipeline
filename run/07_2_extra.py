"""추가 분석 2차 (API 호출 없음, 기존 라벨 재사용).
  E5 앱 리뷰 IPA: 별점 ~ 측면 언급 회귀 → 중요도(별점 영향) × 성과(4~5점 비율)  [Cheng·Shen·Bi 2022 Kano-IPA 방식 단순화]
  E6 전환의 4가지 힘(JTBD Forces of Progress): 기능별 Push·Pull·Anxiety·Habit 비율
  E7 FSD 구독 전환(2026-08-10) 중단 시계열: 주별 부정 비율, 비교 계열(다른 주행 보조 글)과 차이
  E8 비지도 토픽(NMF) ↔ 코드북 교차 검증: 토픽별 지배 코드·순도, 코드가 못 덮는 토픽
  E9 대응 분석: 누가(생애단계·성별·브랜드·EV) × 어떤 상황

  python run/07_2_extra.py
출력: data/output/tables/extra/e5~e9_*.csv
"""
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
A = importlib.import_module("06_analyze")
X1 = importlib.import_module("07_extra")  # FEAT_RE 재사용
OUT = X1.OUT


# ---------------- E5 앱 리뷰 IPA ----------------
ASPECTS = {
    "로그인·인증": r"로그인|인증|비밀번호|간편\s*인증|재로그인",
    "차량 연결·통신": r"연결|통신|서버|먹통|접속|응답\s*없",
    "원격 제어": r"원격|시동|공조|문\s*(잠금|열림)|도어",
    "디지털 키": r"디지털\s*키|폰\s*키",
    "차량 상태·위치": r"위치|주차\s*위치|상태\s*확인|배터리|충전\s*상태|주행\s*가능",
    "업데이트 후 오류": r"업데이트\s*(후|이후|하고)|업뎃",
    "속도·로딩": r"느리|로딩|오래\s*걸|버벅|렉",
    "위젯·워치": r"위젯|워치|갤럭시\s*워치|애플\s*워치",
    "UI·디자인": r"UI|디자인|화면\s*구성|메뉴|복잡",
    "알림": r"알림|푸시",
}
ASP_RE = {k: re.compile(v, re.I) for k, v in ASPECTS.items()}


def ols(X, y):
    X1_ = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1_, y, rcond=None)
    res = y - X1_ @ b
    cov = np.linalg.inv(X1_.T @ X1_) * (res @ res) / (len(y) - X1_.shape[1])
    return b[1:], np.sqrt(np.diag(cov))[1:], 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))


def app_ipa(cor):
    a = cor[cor.source == "app_review"].copy()
    a["rating"] = a.meta.map(lambda m: json.loads(m).get("rating")).astype(float)
    a["app"] = a.meta.map(lambda m: json.loads(m).get("app"))
    a = a.dropna(subset=["rating"])
    M = np.column_stack([a.text.str.contains(rx).values for rx in ASP_RE.values()]).astype(float)
    b, se, r2 = ols(M, a.rating.values)
    rows = []
    for i, k in enumerate(ASPECTS):
        m = M[:, i] == 1
        rows.append({"aspect": k, "n": int(m.sum()), "share": round(m.mean(), 3), "coef": round(b[i], 3), "se": round(se[i], 3),
                     "importance": round(-b[i], 3), "mean_rating": round(a.rating[m].mean(), 2),
                     "performance": round((a.rating[m] >= 4).mean(), 3)})
    df = pd.DataFrame(rows)
    by_app = a.groupby("app").rating.agg(["count", "mean"]).round(2).reset_index()
    return df, {"n": len(a), "r2": round(r2, 3), "mean_rating": round(a.rating.mean(), 2)}, by_app


# ---------------- E6 Forces of Progress ----------------
PULL = {"T_WANT", "T_SATISFIED", "T_ACCEPT_SW"}
ANX = {"T_HESITATE", "T_DISTRUST", "T_OWNERSHIP_WORRY", "T_CALCULATE", "T_RESIST_HWLOCK", "T_RESIST_DOUBLEPAY"}


def forces(df):
    rows = []
    for f, rx in X1.FEAT_RE.items():
        g = df[df.text.str.contains(rx)]
        if len(g) < 50:
            continue
        att = g.attitude.map(set)
        push = ((g.layer == "B_SITUATION") & (g.sentiment.astype(int) < 0) & (g.intensity.astype(int) >= 2)).mean()
        pull = att.map(lambda s: bool(s & PULL)).mean()
        anx = att.map(lambda s: bool(s & ANX)).mean()
        habit = (g.substitute.isin(["aftermarket", "diy", "other_app"])).mean()
        rows.append({"feature": f, "n": len(g), "push": round(push, 3), "pull": round(pull, 3), "anxiety": round(anx, 3),
                     "habit": round(habit, 3), "net": round(push + pull - anx - habit, 3),
                     "top_anxiety": "|".join(f"{k}:{v}" for k, v in Counter(t for s in att for t in s & ANX).most_common(3))})
    return pd.DataFrame(rows).sort_values("net", ascending=False)


# ---------------- E7 FSD 중단 시계열 ----------------
EVENT = pd.Timestamp("2026-08-10")
FSD = re.compile(r"FSD", re.I)


def its(df):
    # 비교 계열: 같은 기간 날짜 있는 다른 FoD 기능 글(주행 보조 외) — 주행 보조 비FSD 글은 주당 0~2건이라 못 씀
    other = re.compile("|".join(v for k, v in X1.FEATURES.items() if k != "주행 보조"), re.I)
    d = df[df.created_at.notna()].copy()
    d["grp"] = np.where(d.text.str.contains(FSD), "FSD", np.where(d.text.str.contains(other), "다른 FoD 기능", ""))
    d = d[d.grp != ""]
    d["t"] = pd.to_datetime(d.created_at).dt.tz_localize(None)
    d = d[(d.t >= EVENT - pd.Timedelta(weeks=26)) & (d.t < EVENT + pd.Timedelta(weeks=9))]
    d["neg"] = (d.sentiment.astype(int) < 0).astype(float)
    d["anx"] = d.attitude.map(lambda s: bool(set(s) & ANX)).astype(float)
    d["week"] = ((d.t - EVENT).dt.days // 7).astype(int)
    w = d.groupby(["grp", "week"]).agg(n=("neg", "size"), neg=("neg", "mean"), anx=("anx", "mean")).reset_index()
    rows = []
    for y in ("neg", "anx"):
        for g in ("FSD", "다른 FoD 기능"):
            s = w[(w.grp == g) & (w.n >= 3)]
            post = (s.week >= 0).astype(float)
            X = np.column_stack([s.week, post, s.week * post])
            b, se, _ = ols(X, s[y].values)  # ponytail: OLS 표준오차(자기상관 미보정) — 주 수가 늘면 Newey-West로
            rows.append({"outcome": y, "group": g, "weeks": len(s), "pre_mean": round(s[y][s.week < 0].mean(), 3),
                         "post_mean": round(s[y][s.week >= 0].mean(), 3), "level_change": round(b[1], 3), "level_se": round(se[1], 3),
                         "slope_change": round(b[2], 4)})
    r = pd.DataFrame(rows)
    did = {y: round(r[(r.outcome == y) & (r.group == "FSD")].level_change.iloc[0] - r[(r.outcome == y) & (r.group != "FSD")].level_change.iloc[0], 3) for y in ("neg", "anx")}
    return w, r, did


# ---------------- E8 NMF 토픽 ↔ 코드북 ----------------
def topics(df, k=25):
    from kiwipiepy import Kiwi
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import normalized_mutual_info_score
    kiwi = Kiwi()
    docs = [" ".join(t.form for t in kiwi.tokenize(x) if t.tag in ("NNG", "NNP") and len(t.form) > 1) for x in df.text]
    vec = TfidfVectorizer(min_df=20, max_df=0.3, max_features=8000)
    Xt = vec.fit_transform(docs)
    nmf = NMF(n_components=k, random_state=A.SEED, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(Xt)
    vocab = np.array(vec.get_feature_names_out())
    top = W.argmax(1)
    has = W.max(1) > 0
    prim = df.situation.map(lambda s: s[0] if len(s) else "(없음)")
    rows = []
    for t in range(k):
        m = (top == t) & has
        cnt = Counter(c for cs in df.situation[m] for c in cs)
        n = int(m.sum())
        dom, dk = cnt.most_common(1)[0] if cnt else ("(없음)", 0)
        rows.append({"topic": t, "n": n, "words": " ".join(vocab[nmf.components_[t].argsort()[::-1][:10]]),
                     "dominant_code": dom, "dominant_name": X1.A_SIT.get(dom, dom), "purity": round(dk / n, 3) if n else 0,
                     "no_code_share": round((df.situation[m].map(len) == 0).mean(), 3) if n else 0,
                     "a_fod_share": round((df.layer[m] == "A_FOD").mean(), 3) if n else 0})
    nmi = normalized_mutual_info_score(prim[has], top[has])
    return pd.DataFrame(rows).sort_values("n", ascending=False), round(nmi, 3)


# ---------------- E9 대응 분석 ----------------
def corresp(df):
    demo = pd.read_parquet(A.STAGE / "post_demo.parquet")
    demo = demo[demo.dimension.isin(["life_stage", "gender", "brand", "ev"])].drop_duplicates(["post_id", "value"])
    sit = df[["post_id", "situation"]].explode("situation").dropna()
    m = demo.merge(sit, on="post_id")
    top_codes = m.situation.value_counts().head(18).index
    N = pd.crosstab(m.value, m.situation)[top_codes].astype(float)
    N = N[N.sum(1) >= 200]
    P = N / N.values.sum()
    r, c = P.sum(1).values, P.sum(0).values
    S = (P.values - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    inertia = s ** 2 / (s ** 2).sum()
    rowc = (U[:, :2] * s[:2]) / np.sqrt(r)[:, None]
    colc = (Vt.T[:, :2] * s[:2]) / np.sqrt(c)[:, None]
    pts = pd.concat([pd.DataFrame({"kind": "who", "label": N.index, "x": rowc[:, 0], "y": rowc[:, 1], "mass": r}),
                     pd.DataFrame({"kind": "situation", "label": N.columns, "x": colc[:, 0], "y": colc[:, 1], "mass": c})])
    pts["name"] = pts.label.map(lambda v: X1.A_SIT.get(v, v).split(",")[0])
    return pts.round(4), [round(v, 3) for v in inertia[:3]], N


if __name__ == "__main__":
    import yaml
    cb = yaml.safe_load((A.ROOT / "config/codebook_v2.yaml").read_text(encoding="utf-8"))
    X1.A_SIT.update({k: v.split(":")[0] for k, v in cb["families"]["situation"]["values"].items()})
    df = A.load()
    cor = pd.read_parquet(A.STAGE / "corpus.parquet")
    df = df.merge(cor[["post_id", "created_at"]], on="post_id", how="left")
    w = lambda d, n: d.to_csv(OUT / f"{n}.csv", index=False, encoding="utf-8-sig")
    meta = {}

    ipa, meta["e5"], by_app = app_ipa(cor)
    w(ipa, "e5_app_ipa"), w(by_app, "e5_app_ratings")
    print("E5 IPA", meta["e5"], "\n", ipa.to_string(index=False))

    fo = forces(df)
    w(fo, "e6_forces")
    print("E6 forces\n", fo.drop(columns="top_anxiety").to_string(index=False))

    wk, itsr, meta["e7_did"] = its(df)
    w(wk, "e7_its_weekly"), w(itsr, "e7_its_model")
    print("E7 ITS\n", itsr.to_string(index=False), "\nDiD", meta["e7_did"])

    tp, meta["e8_nmi"] = topics(df)
    w(tp, "e8_topics")
    print("E8 NMI", meta["e8_nmi"], "\n", tp[["topic", "n", "dominant_code", "purity", "no_code_share", "words"]].to_string(index=False))

    pts, meta["e9_inertia"], _ = corresp(df)
    w(pts, "e9_ca_points")
    print("E9 inertia", meta["e9_inertia"])

    (OUT / "e5_e9_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print("→", OUT.relative_to(A.ROOT))
