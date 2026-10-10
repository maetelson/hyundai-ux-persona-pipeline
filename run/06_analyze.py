"""분석: 유병률 → 테마 네트워크(PMI+Louvain) → 니즈·순간 세그먼트(LCA) → 페르소나 표.

  python run/06_analyze.py            # 혼합 시간 창(코퍼스 그대로)
  python run/06_analyze.py strict     # 민감도: 엄격 시간 창 부분집합
출력: data/output/tables{_strict}/*.csv
"""
import json
import math
import os
import sys
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from stepmix.stepmix import StepMix

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data/stage"
MODE = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
OUT = ROOT / f"data/output/tables{'_strict' if MODE == 'strict' else ''}"
SEED, K_RANGE, TOP_CODES, N_BOOT = 20261010, range(3, 10), 22, 10
NEEDS = yaml.safe_load((ROOT / "config/need_hierarchy_v1.yaml").read_text(encoding="utf-8"))
NEED_INFO = {n["id"]: {"hashtag": n["hashtag"], "name": n["name"], "code": c} for c, v in NEEDS.items() for n in v["needs"]}
STRICT = {"in", "cafe_recent", "cafe_id_recent"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def load():
    lab = pd.read_parquet(STAGE / "post_labels.parquet")
    cor = pd.read_parquet(STAGE / "corpus.parquet")[["post_id", "source", "text", "meta", "date_status", "author_hash"]]
    df = lab.merge(cor, on="post_id")
    if MODE == "strict":
        df = df[df["date_status"].isin(STRICT)]
    df = df[df["layer"].isin(["A_FOD", "B_SITUATION"])].copy()
    for f in ("situation", "needs", "journey", "attitude"):
        df[f] = df[f].map(list)
    df["channel"] = df["meta"].map(lambda m: json.loads(m).get("channel") or "")
    return df.reset_index(drop=True)


def prevalence(df):
    n, rows = len(df), []
    for fam in ("situation", "needs", "journey", "attitude"):
        for code, k in Counter(x for xs in df[fam] for x in xs).items():
            lo, hi = wilson(k, n)
            src = df[df[fam].map(lambda xs: code in xs)]["source"].value_counts(normalize=True)
            rows.append({"family": fam, "code": code, "label": NEED_INFO.get(code, {}).get("hashtag", ""),
                         "n": k, "share": round(k / n, 4), "ci_low": lo, "ci_high": hi,
                         "top_source": src.index[0], "top_source_share": round(src.iloc[0], 3)})
    return pd.DataFrame(rows).sort_values(["family", "n"], ascending=[True, False])


def theme_network(df, fam="situation", min_cooc=None):
    n = len(df)
    min_cooc = min_cooc or max(5, round(n * 0.0005))  # 6만 건이면 30
    cnt = Counter(x for xs in df[fam] for x in set(xs))
    pair = Counter(p for xs in df[fam] for p in combinations(sorted(set(xs)), 2))
    edges = []
    for (a, b), k in pair.items():
        if k < min_cooc:
            continue
        pmi = math.log((k / n) / ((cnt[a] / n) * (cnt[b] / n)))
        lift = (k / n) / ((cnt[a] / n) * (cnt[b] / n))
        if pmi > 0:
            edges.append({"a": a, "b": b, "cooc": k, "pmi": round(pmi, 3), "lift": round(lift, 2)})
    edges = pd.DataFrame(edges, columns=["a", "b", "cooc", "pmi", "lift"])
    if edges.empty:
        return pd.DataFrame(columns=["code", "n", "community", "coassign_stability"]), edges, 0.0
    g = nx.Graph()
    g.add_weighted_edges_from((r.a, r.b, r.pmi) for r in edges.itertuples())
    # 다중 seed Louvain → 공동배정 비율로 안정성
    runs = [nx.community.louvain_communities(g, weight="weight", seed=s) for s in range(50)]
    nodes = list(g.nodes)
    co = {(a, b): 0 for a, b in combinations(sorted(nodes), 2)}
    for comms in runs:
        lab = {v: i for i, c in enumerate(comms) for v in c}
        for a, b in co:
            co[(a, b)] += lab[a] == lab[b]
    best = max(runs, key=lambda c: nx.community.modularity(g, c, weight="weight"))
    q = nx.community.modularity(g, best, weight="weight")
    lab = {v: i for i, c in enumerate(best) for v in c}
    stab = {v: np.mean([co[tuple(sorted((v, u)))] / 50 for u in nodes if u != v and lab[u] == lab[v]] or [1.0]) for v in nodes}
    node_df = pd.DataFrame([{"code": v, "n": cnt[v], "community": lab[v], "coassign_stability": round(stab[v], 3)} for v in nodes])
    return node_df, edges, round(q, 3)


def segments(df):
    codes = [c for c, _ in Counter(x for xs in df["situation"] for x in set(xs)).most_common(TOP_CODES)]
    X_all = np.array([[int(c in xs) for c in codes] for xs in df["situation"]])
    fit_mask = X_all.sum(1) >= 2  # 군집 적합은 상황 2개 이상 글로
    X = X_all[fit_mask]
    sel = []
    for k in K_RANGE:
        m = StepMix(n_components=k, measurement="binary", n_init=10, random_state=SEED, verbose=0, progress_bar=0).fit(X)
        p = m.predict_proba(X)
        ent = 1 + (p * np.log(np.clip(p, 1e-12, 1))).sum() / (len(X) * math.log(k))
        sel.append({"k": k, "bic": round(m.bic(X), 1), "entropy": round(ent, 3), "model": m})
    sel_df = pd.DataFrame([{k: v for k, v in s.items() if k != "model"} for s in sel])
    k_env = int(os.getenv("K", "0"))  # run/06_1_select_k.py 결과로 지정
    best = next(s for s in sel if s["k"] == k_env) if k_env else min(sel, key=lambda s: s["bic"])
    m, k = best["model"], best["k"]
    base = m.predict(X)
    # 부트스트랩 안정성: 재표본 적합 → 원 표본 예측 → 군집별 최대 Jaccard
    rng = np.random.default_rng(SEED)
    jac = {c: [] for c in range(k)}
    for b in range(N_BOOT):
        idx = rng.choice(len(X), len(X), replace=True)
        mb = StepMix(n_components=k, measurement="binary", n_init=3, random_state=b, verbose=0, progress_bar=0).fit(X[idx])
        pb = mb.predict(X)
        for c in range(k):
            a = set(np.where(base == c)[0])
            jac[c].append(max(len(a & set(np.where(pb == d)[0])) / max(len(a | set(np.where(pb == d)[0])), 1) for d in range(k)))
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)  # ponytail: k-modes 대신 KMeans 교차확인
    ari = adjusted_rand_score(base, km)
    # 상황 1개 이상인 모든 글에 배정
    # 핵심 구성원 = 상황 2개 이상(적합 표본). 확장 = 상황 1개이면서 소속 확률 ≥ 0.6
    any_mask = X_all.sum(1) >= 1
    seg = np.full(len(df), -1)
    proba = m.predict_proba(X_all[any_mask])
    pred, pmax = proba.argmax(1), proba.max(1)
    single = X_all[any_mask].sum(1) == 1
    pred[single & (pmax < 0.6)] = -1
    seg[any_mask] = pred
    df["segment"] = seg
    df["core"] = X_all.sum(1) >= 2
    prof = pd.DataFrame(m.get_parameters()["measurement"]["pis"], columns=codes) if "measurement" in m.get_parameters() else None
    info = {"k": k, "ari_vs_kmeans": round(ari, 3), "n_fit": int(fit_mask.sum()), "n_assigned": int((seg >= 0).sum()),
            "stability": {c: round(float(np.mean(v)), 3) for c, v in jac.items()}}
    return df, sel_df, info, prof, codes


def persona_tables(df, codes):
    d = df[df["segment"] >= 0]
    n_all = len(d)
    demo = pd.read_parquet(STAGE / "post_demo.parquet")
    rows_sum, prof, needs_t, cooc_t, feat_t, quote_t, jb_t, opp_t = [], [], [], [], [], [], [], []
    base_sit = Counter(x for xs in d["situation"] for x in set(xs))
    for s, g in d.groupby("segment"):
        n = len(g)
        src = g["source"].value_counts(normalize=True)
        sit = Counter(x for xs in g["situation"] for x in set(xs))
        lift = {c: (sit[c] / n) / (base_sit[c] / n_all) for c in sit if base_sit[c] >= 30}
        top_sit = sorted(sit, key=lambda c: -sit[c])[:5]
        rows_sum.append({"segment": s, "n": n, "n_core": int(g["core"].sum()), "share": round(n / n_all, 4), "top_source": src.index[0],
                         "top_source_share": round(src.iloc[0], 3), "top_situations": "|".join(top_sit),
                         "top_lift": "|".join(sorted(lift, key=lambda c: -lift[c])[:5]),
                         "a_fod_share": round((g["layer"] == "A_FOD").mean(), 3)})
        # 프로필: LLM 생애단계·성별(본문 근거) + 규칙 데모(basis 분리)
        for dim in ("life_stage", "gender"):
            known = g[~g[dim].str.endswith("UNKNOWN")]
            for v, k in known[dim].value_counts().items():
                prof.append({"segment": s, "dimension": dim, "value": v, "basis": "llm_text", "n": k,
                             "share_of_known": round(k / max(len(known), 1), 3), "coverage": round(len(known) / n, 3)})
        dm = demo[demo["post_id"].isin(g["post_id"])]
        for (dim, basis), dg in dm.groupby(["dimension", "basis"]):
            cov = dg["post_id"].nunique()
            for v, k in dg.drop_duplicates(["post_id", "value"])["value"].value_counts().items():
                prof.append({"segment": s, "dimension": dim, "value": v, "basis": f"rule_{basis}", "n": k,
                             "share_of_known": round(k / max(cov, 1), 3), "coverage": round(cov / n, 3)})
        for ch, k in g["channel"].value_counts().head(5).items():
            if ch:
                prof.append({"segment": s, "dimension": "channel", "value": ch, "basis": "meta", "n": k,
                             "share_of_known": round(k / n, 3), "coverage": 1.0})
        # #해시태그 니즈
        nd = Counter(x for xs in g["needs"] for x in set(xs))
        for nid, k in nd.most_common(8):
            info = NEED_INFO.get(nid, {})
            needs_t.append({"segment": s, "need_id": nid, "hashtag": info.get("hashtag"), "name": info.get("name"),
                            "n": k, "share_in_segment": round(k / n, 3)})
        # 니즈 교차 언급 상위 5쌍
        pc = Counter(p for xs in g["needs"] for p in combinations(sorted(set(xs)), 2))
        for (a, b), k in pc.most_common(5):
            cooc_t.append({"segment": s, "need_a": NEED_INFO.get(a, {}).get("hashtag", a), "need_b": NEED_INFO.get(b, {}).get("hashtag", b),
                           "n": k, "pct_of_segment": round(k / n, 3)})
        # 핵심 특징 후보(규칙): 최고 lift 상황, 최대 니즈 교차, 패싯 특이값
        for c in sorted(lift, key=lambda c: -lift[c])[:3]:
            feat_t.append({"segment": s, "kind": "lift", "metric": f"{c} {sit[c] / n:.0%} (전체 대비 {lift[c]:.1f}배)"})
        for f in ("substitute", "wtp_signal", "intensity", "actor"):
            vc = g[f].value_counts(normalize=True)
            allv = d[f].value_counts(normalize=True)
            for v, p in vc.items():
                if v not in ("none", "unknown", "0", "1", "self") and allv.get(v, 0) > 0 and p / allv[v] >= 1.5 and (g[f] == v).sum() >= 20:
                    feat_t.append({"segment": s, "kind": f, "metric": f"{f}={v} {p:.0%} (전체 대비 {p / allv[v]:.1f}배)"})
        # 대표 인용: 니즈별 강도 높고 근거 있는 글, 출처 다양성
        for nid, _ in nd.most_common(4):
            cand = g[g["needs"].map(lambda xs: nid in xs)].copy()
            cand["score"] = cand["intensity"].astype(int) * 2 + (cand["confidence"] == "high") + cand["text"].str.len().clip(0, 300) / 300
            picked = []
            for src_name in cand.sort_values("score", ascending=False)["source"].unique():
                picked += list(cand[cand["source"] == src_name].nlargest(2, "score").itertuples())
            for r in sorted(picked, key=lambda r: -r.score)[:6]:
                ev = json.loads(r.evidence)
                quote = ev.get(NEED_INFO.get(nid, {}).get("code"))  # 니즈의 상황 코드에 대한 근거만
                if not quote:
                    continue
                quote_t.append({"segment": s, "need_id": nid, "hashtag": NEED_INFO.get(nid, {}).get("hashtag"),
                                "post_id": r.post_id, "source": r.source, "quote": quote[:80], "outcome": r.outcome})
        # FoD 저니 장벽 (A층)
        a = g[g["layer"] == "A_FOD"]
        for j in {x for xs in a["journey"] for x in xs}:
            aj = a[a["journey"].map(lambda xs: j in xs)]
            for t, k in Counter(x for xs in aj["attitude"] for x in xs).items():
                lo, hi = wilson(k, len(aj))
                jb_t.append({"segment": s, "journey": j, "attitude": t, "n": k, "n_stage": len(aj),
                             "pct": round(k / len(aj), 3), "ci_low": lo, "ci_high": hi, "hypothesis": len(aj) < 30})
        # 텍스트 기회 신호 지수 (순위 비교용, ODI 아님)
        for nid, k in nd.most_common(8):
            sub = g[g["needs"].map(lambda xs: nid in xs)]
            opp_t.append({"segment": s, "need_id": nid, "hashtag": NEED_INFO.get(nid, {}).get("hashtag"), "n": k,
                          "prevalence": round(k / n, 3), "intensity_mean": round(sub["intensity"].astype(int).mean(), 2),
                          "neg_share": round((sub["sentiment"] == "-1").mean(), 3),
                          "substitute_share": round((sub["substitute"] != "none").mean(), 3),
                          "signal_index": round(k / n * sub["intensity"].astype(int).mean() * (0.5 + (sub["sentiment"] == "-1").mean())
                                                * (1 + (sub["substitute"] != "none").mean()), 4)})
    return {k: pd.DataFrame(v) for k, v in dict(persona_summary=rows_sum, persona_profile=prof, persona_needs=needs_t,
                                                persona_cooccurrence=cooc_t, persona_features=feat_t, persona_quotes=quote_t,
                                                journey_barriers=jb_t, opportunities=opp_t).items()}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"[{MODE}] 분석 대상(A+B) {len(df):,}")
    prevalence(df).to_csv(OUT / "prevalence.csv", index=False, encoding="utf-8-sig")
    nodes, edges, q = theme_network(df)
    nodes.to_csv(OUT / "theme_nodes.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(OUT / "theme_edges.csv", index=False, encoding="utf-8-sig")
    print(f"테마 네트워크: 노드 {len(nodes)}, 엣지 {len(edges)}, 모듈성 Q {q}, 커뮤니티 {nodes['community'].nunique()}")
    df, sel, info, prof, codes = segments(df)
    sel.to_csv(OUT / "segment_selection.csv", index=False, encoding="utf-8-sig")
    if prof is not None:
        prof.to_csv(OUT / "segment_code_probs.csv", encoding="utf-8-sig")
    (OUT / "segment_info.json").write_text(json.dumps({**info, "theme_modularity": q, "mode": MODE}, ensure_ascii=False, indent=1), encoding="utf-8")
    df[["post_id", "segment"]].to_csv(OUT / "post_segment.csv", index=False)
    print(f"세그먼트: k={info['k']}, 적합 {info['n_fit']:,}, 배정 {info['n_assigned']:,}, KMeans ARI {info['ari_vs_kmeans']}, 안정성 {info['stability']}")
    for name, t in persona_tables(df, codes).items():
        t.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print("→", OUT.relative_to(ROOT))
