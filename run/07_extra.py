"""추가 분석 4종 (API 호출 없음, 기존 라벨 재사용).
  E1 가격 앵커: 본문 금액 표현 → 기능별·기간별 분포와 비쌈/적정 반응
  E2 스토어 상품 성과: 기아 커넥트 스토어 리뷰 → 상품군·차종별 감성·문제 유형·저니 장벽
  E3 대체재 경쟁 지도: substitute 패싯 × 상황·세그먼트, 대체 용품, 특징어(log-odds)
  E4 Kano 추정: 기능별 '없어서 불만' vs '있어서 기쁨' 비대칭 + 유료화 반감, 부트스트랩 안정성

  python run/07_extra.py
출력: data/output/tables/extra/*.csv
"""
import importlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
A = importlib.import_module("06_analyze")  # load(), wilson() 재사용
OUT = A.ROOT / "data/output/tables/extra"
RNG = np.random.default_rng(A.SEED)

# D1-2·A-27 후보와 맞춘 기능 사전
FEATURES = {
    "원격 주차": r"원격\s*(스마트\s*)?주차|RSPA|리모트\s*(스마트\s*)?주차",
    "디지털 키": r"디지털\s*키",
    "원격 제어(시동·공조)": r"원격\s*(시동|공조|제어)|블루링크|기아\s*커넥트|커넥티드\s*서비스",
    "주차 감시·블랙박스": r"빌트인\s*캠|블랙박스|주차\s*(감시|녹화)",
    "주행 보조": r"HDA|고속도로\s*주행\s*보조|스마트\s*크루즈|차로\s*유지|FSD|오토\s*파일럿",
    "OTA 업데이트": r"OTA|무선\s*(소프트웨어\s*)?업데이트",
    "영상·게임": r"스트리밍|넷플릭스|영상\s*이용권|아케이드|차량용?\s*게임",
    "화면 테마": r"디스플레이\s*테마|테마\s*(구매|적용)|디즈니|마블|픽사",
    "라이팅": r"라이팅|웰컴\s*라이트|앰비언트|무드등",
    "회생 제동·가상 변속": r"회생\s*제동|스마트\s*회생|가상\s*(기어|변속)",
    "열선·통풍": r"열선|통풍\s*시트",
    "운전자 프로필": r"운전자\s*(프로필|설정)|프로필\s*설정",
    "실내 전원·캠핑 모드": r"V2L|유틸리티\s*모드|캠핑\s*모드",
}
FEAT_RE = {k: re.compile(v, re.I) for k, v in FEATURES.items()}
WIN = 60


def snip(text, s, e, w=25):
    return text[max(0, s - w):e + w].replace("\n", " ")


# ---------------- E1 가격 앵커 ----------------
AMT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(만|천)?\s*(?:(\d[\d,]*)\s*(천)?)?\s*원")
PERIOD = [("월", re.compile(r"월\s*$|매달|한\s*달|월\s*구독|월정액|달에")),
          ("연", re.compile(r"연\s*$|1년|일\s*년|년에|연간")),
          ("평생", re.compile(r"평생|영구|일시불|한\s*번에"))]
EXPENSIVE = re.compile(r"비싸|부담|사악|창렬|아깝(?!지\s*않)|돈\s*아까")
FAIR = re.compile(r"저렴|싸(다|요|네|게)|괜찮|가성비|아깝지\s*않|안\s*아깝|할\s*만")


def parse_amount(m):
    a, u1, b, u2 = m.groups()
    a = float(a.replace(",", ""))
    v = a * {"만": 1e4, "천": 1e3}.get(u1, 1)
    if b:
        v += float(b.replace(",", "")) * (1e3 if u2 else 1)
    return v


def price_anchors(df):
    rows = []
    for r in df.itertuples():
        t = r.text
        for m in AMT.finditer(t):
            v = parse_amount(m)
            if not 1e3 <= v <= 5e6:  # ponytail: 차값·할부 금액은 범위로만 거른다
                continue
            ctx = t[max(0, m.start() - WIN):m.end() + WIN]
            feats = [k for k, rx in FEAT_RE.items() if rx.search(ctx)]
            if not feats:
                continue
            before = t[max(0, m.start() - 12):m.start()]
            period = next((p for p, rx in PERIOD if rx.search(before)), "미상")
            react = "비쌈" if EXPENSIVE.search(ctx) else "적정" if FAIR.search(ctx) else ""
            for f in feats:
                rows.append({"feature": f, "period": period, "amount": v, "react": react, "post_id": r.post_id,
                             "source": r.source, "wtp": r.wtp_signal, "snippet": snip(t, m.start(), m.end())})
    men = pd.DataFrame(rows)
    out = []
    for (f, p), g in men.groupby(["feature", "period"]):
        n_r = (g.react != "").sum()
        out.append({"feature": f, "period": p, "n_mentions": len(g), "n_posts": g.post_id.nunique(),
                    "p25": g.amount.quantile(.25), "median": g.amount.median(), "p75": g.amount.quantile(.75),
                    "n_reaction": n_r, "expensive_share": round((g.react == "비쌈").sum() / n_r, 3) if n_r else None,
                    "examples": " | ".join(g.sort_values("amount").snippet.iloc[[0, len(g) // 2, -1]].unique()[:3])})
    return men, pd.DataFrame(out).sort_values(["feature", "n_mentions"], ascending=[True, False])


# ---------------- E2 스토어 상품 성과 ----------------
# v2(2026-10-11): 규칙 검증 V4 — '설치' 단독(정밀도 0.25)·가격 재현율 0.21 보완
ISSUES = {
    "적용·설치 실패": r"(설치|다운로드|적용|다운)\S*\s*(이\s*)?(안\s*되|안\s*돼|안됩|실패|불가)|오류|에러|먹통|풀리더니|로딩",
    "호환·차종 차이": r"클러스터|차종|미지원|지원\s*안|호환|EV만|일부\s*화면|전체\s*화면",
    "가격·결제": r"비싸|가격|쿠폰|할인|포인트|환불|결제|1\+1|이벤트|무료|블프|블랙\s*프라이데이|행사|돈값|\d\s*만\s*원",
    "디자인 만족": r"예쁘|이쁘|귀엽|멋지|멋있|만족|좋아요|최고|강추|화사|깔끔",
    "기능 효용": r"편하|편리|유용|신기|잘\s*(돼|됩|작동)|안전",
}
ISS_RE = {k: re.compile(v) for k, v in ISSUES.items()}


def store_group(cat):
    if "테마" in cat:
        return "디스플레이 테마"
    for k in ("RSPA", "회생", "라이팅", "게임", "스트리밍"):
        if k in cat:
            return {"RSPA": "원격 주차(RSPA 2)", "회생": "스마트 회생+", "라이팅": "라이팅 패턴",
                    "게임": "아케이드 게임", "스트리밍": "스트리밍"}[k]
    return "기타"


def store_perf(df):
    s = df[df.source == "kia_store"].copy()
    meta = s.meta.map(json.loads)
    s["group"] = meta.map(lambda d: store_group(d.get("category", "")))
    s["product"] = meta.map(lambda d: d.get("product", ""))
    s["vehicle"] = meta.map(lambda d: re.split(r"[\s(]", d.get("vehicle", ""))[0])
    s["sent"] = s.sentiment.astype(int)
    for k, rx in ISS_RE.items():
        s[k] = s.text.str.contains(rx)

    def summ(g):
        n, neg = len(g), (g.sent < 0).sum()
        lo, hi = A.wilson(neg, n)
        d = {"n": n, "neg_share": round(neg / n, 3), "neg_ci_lo": lo, "neg_ci_hi": hi, "pos_share": round((g.sent > 0).mean(), 3)}
        d.update({k: round(g[k].mean(), 3) for k in ISSUES})
        jn = Counter(j for js in g[g.sent < 0].journey for j in js)
        d["neg_top_journey"] = "|".join(f"{A_J.get(j, j)}:{c}" for j, c in jn.most_common(3))
        return pd.Series(d)

    by_group = s.groupby("group").apply(summ, include_groups=False).reset_index().sort_values("n", ascending=False)
    top_v = s.vehicle.value_counts().head(12).index
    by_vehicle = s[s.vehicle.isin(top_v)].groupby("vehicle").apply(summ, include_groups=False).reset_index().sort_values("n", ascending=False)
    by_product = s.groupby(["group", "product"]).apply(summ, include_groups=False).reset_index()
    by_product = by_product[by_product.n >= 20].sort_values("neg_share", ascending=False)
    neg_ex = s[(s.sent < 0)].groupby("group").text.apply(lambda x: " | ".join(t[:70].replace("\n", " ") for t in x.head(3))).rename("neg_examples")
    return by_group.merge(neg_ex, on="group", how="left"), by_vehicle, by_product


A_J = {"J_DISCOVER": "발견", "J_UNDERSTAND": "이해", "J_CHECK": "확인", "J_TRIAL": "체험", "J_PURCHASE": "구매",
       "J_ACTIVATE": "활성화", "J_USE": "사용", "J_MANAGE": "관리", "J_SETTLE": "정산", "J_TRANSFER": "양도"}

# ---------------- E3 대체재 경쟁 지도 ----------------
PRODUCTS = {
    "사제 블랙박스": r"블랙박스", "사제 원격시동기": r"원격\s*시동기|리모컨\s*시동|스타트\s*키트",
    "무선 카플레이·안드로이드오토": r"카플레이|안드로이드\s*오토|동글|미러링", "폰 거치대·태블릿": r"거치대|태블릿|아이패드",
    "하이패스": r"하이패스", "후방·전방·360 카메라(사제)": r"(후방|전방|사이드)\s*카메라|어라운드\s*뷰|360",
    "무드등·LED": r"무드등|LED|엠비언트|앰비언트", "열선·통풍 방석": r"방석|시트\s*커버|쿨링\s*시트",
    "썬팅·PPF": r"썬팅|틴팅|PPF|랩핑", "트렁크·수납 용품": r"정리함|수납|트렁크\s*매트|루프\s*박스",
    "카시트·유아 용품": r"카시트|유아|선쉐이드", "내비·지도 앱": r"티맵|카카오\s*내비|네이버\s*지도|T맵",
    "주차 앱·발렛": r"발렛|주차\s*앱|모두의\s*주차",
}
PROD_RE = {k: re.compile(v, re.I) for k, v in PRODUCTS.items()}


def log_odds(tok_a, tok_b, top=15, prior=0.01):
    # ponytail: 균일 사전 log-odds z — 정보 사전(Monroe et al.)은 어휘 편향이 문제 될 때
    ca, cb = Counter(tok_a), Counter(tok_b)
    na, nb, V = sum(ca.values()), sum(cb.values()), len(set(ca) | set(cb))
    rows = []
    for w, a in ca.items():
        if a < 5:
            continue
        b = cb.get(w, 0)
        la = math.log((a + prior) / (na + prior * V - a - prior))
        lb = math.log((b + prior) / (nb + prior * V - b - prior))
        z = (la - lb) / math.sqrt(1 / (a + prior) + 1 / (b + prior))
        rows.append((w, a, round(z, 2)))
    return sorted(rows, key=lambda x: -x[2])[:top]


def substitutes(df):
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    seg = pd.read_csv(A.OUT / "post_segment.csv")
    df = df.merge(seg, on="post_id", how="left")
    base_sit = Counter(c for cs in df.situation for c in cs)
    n_all = len(df)
    sub = df[df.substitute != "none"]
    rows, prods, terms = [], [], []
    toks = {pid: [t.form for t in kiwi.tokenize(txt) if t.tag in ("NNG", "NNP") and len(t.form) > 1]
            for pid, txt in zip(df.post_id, df.text)}
    for typ, g in sub.groupby("substitute"):
        n = len(g)
        sit = Counter(c for cs in g.situation for c in cs)
        for c, k in sit.most_common(6):
            rows.append({"substitute": typ, "n": n, "situation": c, "situation_name": A_SIT.get(c, c),
                         "share": round(k / n, 3), "lift": round((k / n) / (base_sit[c] / n_all), 2)})
        for p, rx in PROD_RE.items():
            k = g.text.str.contains(rx).sum()
            if k:
                prods.append({"substitute": typ, "product": p, "n": int(k), "share": round(k / n, 3)})
        rest = [t for pid in df.post_id[df.substitute != typ] for t in toks[pid]]
        for w, a, z in log_odds([t for pid in g.post_id for t in toks[pid]], rest):
            terms.append({"substitute": typ, "term": w, "count": a, "z": z})
    segx = pd.crosstab(sub.segment.fillna(-1).astype(int), sub.substitute, normalize="index").round(3)
    segx["n"] = sub.segment.fillna(-1).astype(int).value_counts()
    sent = sub.groupby("substitute").sentiment.apply(lambda x: round((x.astype(int) < 0).mean(), 3)).rename("neg_share")
    return pd.DataFrame(rows), pd.DataFrame(prods).sort_values(["substitute", "n"], ascending=[True, False]), \
        pd.DataFrame(terms), segx.reset_index(), sent.reset_index()


A_SIT = {}

# ---------------- E4 Kano 추정 ----------------
# v2(2026-10-11): 규칙 검증 V1 정밀도 0.45 → 고장('안 들어와')·출금('빠져나가') 오탐 제거
ABSENT = re.compile(r"없어서|없는\s*게|없으니|없어\s*(아쉽|불편)|빠져\s*(있|서|버)|빠진|미적용|미지원|옵션에\s*없|넣을\s*걸|안\s*넣|뺐더니|빼서|없어졌")
DELIGHT = re.compile(r"신세계|감동|생각보다\s*좋|꿀\s*기능|강추|최고|편하더|편해요|좋더라|좋네요|만족|신기")
RESIST = {"T_RESIST_HWLOCK", "T_RESIST_DOUBLEPAY"}


def kano(df):
    rows, per = [], {}
    for f, rx in FEAT_RE.items():
        hit = []
        for r in df.itertuples():
            m = rx.search(r.text)
            if not m:
                continue
            ctx = r.text[max(0, m.start() - WIN):m.end() + WIN]
            s = int(r.sentiment)
            hit.append((s < 0 and bool(ABSENT.search(ctx)), s > 0 and bool(DELIGHT.search(ctx)),
                        bool(RESIST & set(r.attitude))))
        if len(hit) < 30:
            continue
        per[f] = np.array(hit, dtype=float)
    ab_med = np.median([v[:, 0].mean() for v in per.values()])
    de_med = np.median([v[:, 1].mean() for v in per.values()])

    def cls(ab, de):
        return {(True, False): "당연(Must-be)", (False, True): "매력(Attractive)",
                (True, True): "일원(One-dimensional)", (False, False): "무관심(Indifferent)"}[(ab >= ab_med, de >= de_med)]

    for f, v in per.items():
        ab, de, rs = v.mean(0)
        c = cls(ab, de)
        boot = [cls(*v[RNG.integers(0, len(v), len(v))][:, :2].mean(0)) == c for _ in range(300)]
        rows.append({"feature": f, "n": len(v), "absent_complaint": round(ab, 4), "present_delight": round(de, 4),
                     "kano_class": c, "boot_stability": round(np.mean(boot), 2), "pay_resistance": round(rs, 4),
                     "resist_ci": A.wilson(int(v[:, 2].sum()), len(v))})
    return pd.DataFrame(rows).sort_values("n", ascending=False), {"absent_median": ab_med, "delight_median": de_med}


if __name__ == "__main__":
    import yaml
    cb = yaml.safe_load((A.ROOT / "config/codebook_v2.yaml").read_text(encoding="utf-8"))
    A_SIT.update({k: v.split(":")[0] for k, v in cb["families"]["situation"]["values"].items()})
    OUT.mkdir(parents=True, exist_ok=True)
    df = A.load()
    w = lambda d, n: d.to_csv(OUT / f"{n}.csv", index=False, encoding="utf-8-sig")

    men, price = price_anchors(df)
    w(price, "e1_price_anchors")
    w(men.drop(columns="snippet"), "e1_price_mentions")
    print(f"E1 가격: 언급 {len(men):,}건, 글 {men.post_id.nunique():,}건, 기능 {men.feature.nunique()}개")

    g, v, p = store_perf(df)
    w(g, "e2_store_by_group"), w(v, "e2_store_by_vehicle"), w(p, "e2_store_by_product")
    print(f"E2 스토어: {int(g.n.sum()):,}건, 상품군 {len(g)}")

    sit, prods, terms, segx, sent = substitutes(df)
    w(sit, "e3_sub_situation"), w(prods, "e3_sub_products"), w(terms, "e3_sub_terms"), w(segx, "e3_sub_segment"), w(sent, "e3_sub_sentiment")
    print(f"E3 대체재: {sit.substitute.nunique()}유형")

    k, th = kano(df)
    w(k, "e4_kano")
    (OUT / "e4_kano_thresholds.json").write_text(json.dumps(th), encoding="utf-8")
    print("E4 Kano:\n", k[["feature", "n", "kano_class", "boot_stability", "pay_resistance"]].to_string(index=False))
    print("→", OUT.relative_to(A.ROOT))
