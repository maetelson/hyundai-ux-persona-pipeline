"""분석 표 → 단일 HTML 보고서 (최종 정리본 + 방법·신뢰도). 외부 라이브러리·네트워크 없이 열린다.

  python run/08_report.py
입력: data/output/tables/*.csv, data/output/tables_strict/post_segment.csv, data/stage/*,
      config/report_narrative.yaml (세그먼트 이름·인사이트·아이디어; 없으면 표에서 자동 문구)
출력: data/output/report.html
"""
import html
import json
import math
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "data/output/tables"
TS = ROOT / "data/output/tables_strict"
STAGE = ROOT / "data/stage"
OUT = ROOT / "data/output/report.html"
e = lambda s: html.escape(str(s))


def csv(name, base=T):
    p = base / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def narrative():
    p = ROOT / "config/report_narrative.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


CODEBOOK = yaml.safe_load((ROOT / "config/codebook_v2.yaml").read_text(encoding="utf-8"))
SIT_NAME = {k: v.split(":")[0] for k, v in CODEBOOK["families"]["situation"]["values"].items()}
JOURNEY = list(CODEBOOK["families"]["journey"]["values"])
ATT = list(CODEBOOK["families"]["attitude"]["values"])
LS_NAME = {"L_FIRSTCAR": "첫차·초보", "L_NEWLYWED": "신혼", "L_INFANT": "영유아 부모", "L_SCHOOLKID": "학령기 부모",
           "L_EMPTYNEST": "부모 돌봄·은퇴·60+", "G_F": "여성 단서", "G_M": "남성 단서", "EV": "전기차"}
J_NAME = {"J_DISCOVER": "발견", "J_UNDERSTAND": "이해", "J_CHECK": "확인", "J_TRIAL": "체험", "J_PURCHASE": "구매",
          "J_ACTIVATE": "활성화", "J_USE": "사용", "J_MANAGE": "관리", "J_SETTLE": "정산", "J_TRANSFER": "양도"}
T_NAME = {"T_RESIST_HWLOCK": "HW잠금 반감", "T_RESIST_DOUBLEPAY": "이중결제 반감", "T_ACCEPT_SW": "SW진보 수용",
          "T_HESITATE": "망설임", "T_SATISFIED": "만족", "T_DISAPPOINT": "실망", "T_DISTRUST": "불신",
          "T_CALCULATE": "본전 계산", "T_OWNERSHIP_WORRY": "귀속 우려", "T_WANT": "요구"}


def bar(p, label="", color="var(--accent)"):
    w = max(0, min(100, p * 100))
    return f'<div class="bar"><span style="width:{w:.1f}%;background:{color}"></span><em>{e(label)}</em></div>'


def heat(v, vmax):
    a = 0 if vmax == 0 else min(1, v / vmax)
    return f"background:color-mix(in oklab, var(--accent) {a * 85:.0f}%, transparent)"


def method_link(mid):
    return f'<a class="mlink" href="#{mid}" data-go="{mid}">방법·신뢰도 보기 →</a>'


# ---------------- 데이터 적재 ----------------
info = json.loads((T / "segment_info.json").read_text(encoding="utf-8"))
summ, prof, needs, cooc = csv("persona_summary"), csv("persona_profile"), csv("persona_needs"), csv("persona_cooccurrence")
feats, quotes, jb, opp = csv("persona_features"), csv("persona_quotes"), csv("journey_barriers"), csv("opportunities")
prev, nodes, edges = csv("prevalence"), csv("theme_nodes"), csv("theme_edges")
ksel, segsel = csv("k_selection"), csv("segment_selection")
funnel = pd.read_csv(STAGE / "funnel.csv", index_col=0)
tw = pd.read_csv(STAGE / "time_window.csv")
labels = pd.read_parquet(STAGE / "post_labels.parquet")
corpus = pd.read_parquet(STAGE / "corpus.parquet")[["post_id", "source"]]
lab = labels.merge(corpus, on="post_id")
NAR = narrative()
segs = list(summ.sort_values("n", ascending=False)["segment"])
SEG_NAR = {int(k): v for k, v in (NAR.get("segments") or {}).items()}


def seg_name(s):
    n = SEG_NAR.get(int(s), {})
    return n.get("name") or f"세그먼트 {s}"


def seg_tag(s):
    return SEG_NAR.get(int(s), {}).get("hashtag", "")


# 민감도: 혼합 vs 엄격 배정의 ARI (공통 글)
sens = None
if (TS / "post_segment.csv").exists():
    a = pd.read_csv(T / "post_segment.csv").merge(pd.read_csv(TS / "post_segment.csv"), on="post_id", suffixes=("_h", "_s"))
    a = a[(a.segment_h >= 0) & (a.segment_s >= 0)]
    sens = (round(adjusted_rand_score(a.segment_h, a.segment_s), 3), len(a))

# 규칙 vs LLM 생애단계 일치
demo = pd.read_parquet(STAGE / "post_demo.parquet")
rule_ls = demo[(demo.dimension == "life_stage") & (demo.basis == "text")][["post_id", "value"]]
cmp_ = rule_ls.merge(labels[["post_id", "life_stage"]], on="post_id")
cmp_known = cmp_[cmp_.life_stage != "L_UNKNOWN"]
ls_agree = (round((cmp_known.value == cmp_known.life_stage).mean(), 3), len(cmp_known), len(cmp_))

evals = pd.read_csv(ROOT / "data/eval/v0/summary.csv") if (ROOT / "data/eval/v0/summary.csv").exists() else pd.DataFrame()

# ---------------- 최종 정리본 ----------------
S = []  # (id, 메뉴명, 그룹, html)

f_rows = "".join(f"<tr><th>{e(src)}</th>" + "".join(f"<td>{int(v):,}</td>" for v in r) + "</tr>" for src, r in funnel.iterrows())
n_ab = int(lab.layer.isin(["A_FOD", "B_SITUATION"]).sum())
S.append(("s1", "1. 수집·분석 프로세스", "final", f"""
<h2>1. 수집 → 분석 전체 프로세스</h2>
<div class="flow">
 <div><b>[1] 원천 수집</b><span>네이버 카페·블로그 검색, 기아 커넥트 스토어 리뷰, 앱 리뷰</span><strong>{int(funnel.iloc[:, 0].sum()):,}</strong></div>
 <div><b>[2] 중복·길이 정제</b><span>URL·본문 중복 제거, 15자 미만 제외</span><strong>{int(funnel.iloc[:, -1].sum()):,}</strong></div>
 <div><b>[3] 시간 창·할당 표본</b><span>2021-10-10 이후(혼합), 칸별 상한·채널 30% 상한</span><strong>{len(corpus):,}</strong></div>
 <div><b>[4] LLM 전수 라벨링</b><span>유효(FoD 직접 + 운전 상황) 판정, 근거 인용 검증</span><strong>{n_ab:,}</strong></div>
 <div><b>[5] 세그먼트·페르소나</b><span>LCA 니즈·순간 세그먼트 → 페르소나</span><strong>{info['k']}개</strong></div>
</div>
<table class="t"><thead><tr><th>소스</th>{''.join(f'<th>{e(c)}</th>' for c in funnel.columns)}</tr></thead><tbody>{f_rows}</tbody></table>
{method_link('m1')}"""))

need_rows = ""
hier = yaml.safe_load((ROOT / "config/need_hierarchy_v1.yaml").read_text(encoding="utf-8"))
pv = prev[prev.family == "needs"].set_index("code")["n"].to_dict() if len(prev) else {}
for code, v in hier.items():
    tags = " ".join(f'<span class="tag">{e(n["hashtag"])} <small>{pv.get(n["id"], 0):,}</small></span>' for n in v["needs"])
    need_rows += f"<tr><th>{e(SIT_NAME.get(code, code))}<br><small>{e(code)}</small></th><td>{tags}</td></tr>"
S.append(("s2", "2. 코드 체계", "final", f"""
<h2>2. 니즈 분석 · 데모 분석 두 트랙과 코드 체계</h2>
<div class="tracks"><div><b>니즈 트랙</b> 오픈 코딩(결과 문장 2,000+) → 3차 니즈 도출 → 전수 라벨링 → 테마 네트워크 → LCA 세그먼트</div>
<div><b>데모 트랙</b> 본문 단서(LLM 근거 + 규칙) · 메타(차종·상품) · 채널명 단서를 분리 수집 → 세그먼트에 대입</div></div>
<p>상황 2차 코드 {len(hier)}개 × 3차 #해시태그 니즈 {sum(len(v['needs']) for v in hier.values())}개. 숫자는 유효 글 중 해당 니즈가 붙은 건수.</p>
<table class="t codes"><tbody>{need_rows}</tbody></table>{method_link('m3')}"""))

ins = NAR.get("insights") or []
ins_html = "".join(f'<div class="ins"><b>{e(i["headline"])}</b><small>{e(i["evidence"])}</small></div>' for i in ins) or "<p class='muted'>config/report_narrative.yaml의 insights를 채우면 표시됩니다.</p>"
S.append(("s3", "3. 인사이트 요약", "final", f"<h2>3. 페르소나 & 니즈 분석 인사이트 요약</h2>{ins_html}"))

# 테마 네트워크 SVG (원형 배치, 커뮤니티 색)
def network_svg():
    if nodes.empty:
        return "<p class='muted'>네트워크 없음</p>"
    W, H, R = 760, 520, 210
    comms = sorted(nodes.community.unique())
    pal = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#9c755f"]
    order = nodes.sort_values(["community", "n"], ascending=[True, False]).reset_index(drop=True)
    pos = {r.code: (W / 2 + R * math.cos(2 * math.pi * i / len(order)), H / 2 + R * math.sin(2 * math.pi * i / len(order)))
           for i, r in order.iterrows()}
    nmax = nodes.n.max()
    lines = "".join(f'<line x1="{pos[r.a][0]:.0f}" y1="{pos[r.a][1]:.0f}" x2="{pos[r.b][0]:.0f}" y2="{pos[r.b][1]:.0f}" stroke="var(--muted)" stroke-opacity="{min(0.8, 0.15 + r.pmi / 3):.2f}" stroke-width="{0.5 + min(4, r.pmi * 1.5):.1f}"/>'
                    for r in edges.itertuples() if r.a in pos and r.b in pos)
    circles = "".join(f'<circle cx="{pos[r.code][0]:.0f}" cy="{pos[r.code][1]:.0f}" r="{6 + 22 * math.sqrt(r.n / nmax):.0f}" fill="{pal[comms.index(r.community) % len(pal)]}" fill-opacity=".85"><title>{e(r.code)} n={r.n:,} 안정성 {r.coassign_stability}</title></circle>'
                      f'<text x="{pos[r.code][0]:.0f}" y="{pos[r.code][1] + 4:.0f}" text-anchor="middle" class="nl">{e(SIT_NAME.get(r.code, r.code)[:8])}</text>'
                      for r in order.itertuples())
    return f'<svg viewBox="0 0 {W} {H}" class="net" role="img" aria-label="상황 코드 공출현 네트워크">{lines}{circles}</svg>'


S.append(("s4", "4. 니즈 테마 지도", "final", f"""
<h2>4. 니즈 테마 지도</h2><p>한 글에 함께 나온 상황 코드를 PMI로 정규화해 연결(선 굵기 = PMI), 버블 크기 = 언급량, 색 = Louvain 커뮤니티. 모듈성 Q = {info.get('theme_modularity')}.</p>
{network_svg()}{method_link('m5')}"""))

rows = ""
for s in segs:
    r = summ[summ.segment == s].iloc[0]
    nt = needs[needs.segment == s].head(4)
    tags = " ".join(f'<span class="tag">{e(x)}</span>' for x in nt.hashtag)
    stab = info["stability"].get(str(s), info["stability"].get(s, ""))
    rows += f"""<tr><td><b>{e(seg_name(s))}</b><br><small>{e(seg_tag(s))}</small></td><td>{r.share:.1%}<br><small>{int(r.n):,}건 (핵심 {int(r.get('n_core', 0)):,})</small></td>
<td>{e(SEG_NAR.get(int(s), {}).get('one_liner', ' · '.join(SIT_NAME.get(c, c) for c in str(r.top_situations).split('|')[:3])))}</td><td>{tags}</td><td>{stab}</td></tr>"""
S.append(("s5", "5. 페르소나 요약", "final", f"""
<h2>5. 페르소나 요약</h2><p>비중은 <b>텍스트 비중</b>(유효 글 중 세그먼트 배정 비율)이며 시장 비중이 아님. 설문 후 확정.</p>
<table class="t"><thead><tr><th>페르소나</th><th>비중</th><th>한 줄 프로필</th><th>#니즈</th><th>안정성</th></tr></thead><tbody>{rows}</tbody></table>{method_link('m6')}"""))


def profile_block(s):
    out = ""
    for dim, title in (("life_stage", "생애단계"), ("gender", "성별"), ("ev", "전기차"), ("brand", "브랜드")):
        p = prof[(prof.segment == s) & (prof.dimension == dim)]
        if p.empty:
            continue
        basis_pref = ["llm_text", "rule_text", "rule_meta", "rule_channel"]
        p = p.assign(o=p.basis.map(lambda b: basis_pref.index(b) if b in basis_pref else 9)).sort_values(["o", "n"], ascending=[True, False])
        b = p.basis.iloc[0]
        top = p[p.basis == b].nlargest(2, "n")
        cov = top.coverage.iloc[0]
        gate = cov >= 0.3 and top.n.sum() >= 100
        vals = " &gt; ".join(f"{e(LS_NAME.get(v, v))} {sh:.0%}" for v, sh in zip(top.value, top.share_of_known)) if gate else "판단 불가"
        out += f"<tr><th>{title}</th><td>{vals}<br><small>근거 {e(b)} · 단서 보유 {cov:.0%}</small></td></tr>"
    return out


cards = ""
for s in segs:
    r = summ[summ.segment == s].iloc[0]
    nar = SEG_NAR.get(int(s), {})
    nt = needs[needs.segment == s].head(5)
    cc = cooc[cooc.segment == s].head(5)
    ft = nar.get("features") or [f"{x}" for x in feats[feats.segment == s].metric.head(3)]
    need_rows = ""
    for nr in nt.itertuples():
        qs = quotes[(quotes.segment == s) & (quotes.need_id == nr.need_id)].head(6)
        qhtml = " · ".join(f'"{e(q)}"' for q in qs.quote)
        need_rows += f'<tr><td><b>{e(nr.hashtag)}</b><br><small>{nr.share_in_segment:.1%} · {int(nr.n):,}건</small></td><td>{e(nr.name)}</td><td class="q">{qhtml}</td></tr>'
    cbars = "".join(f'<div class="crow"><span>{e(c.need_a)} ↔ {e(c.need_b)}</span>{bar(c.pct_of_segment * 5, f"{c.pct_of_segment:.1%}")}</div>' for c in cc.itertuples())
    jbs = jb[jb.segment == s]
    jtab = ""
    if not jbs.empty:
        piv = jbs.pivot_table(index="attitude", columns="journey", values="pct", aggfunc="max").reindex(columns=[j for j in JOURNEY if j in jbs.journey.unique()])
        nstage = jbs.groupby("journey").n_stage.max()
        vmax = piv.max().max()
        head = "".join(f"<th>{J_NAME.get(j, j)}<br><small>n={int(nstage.get(j, 0))}{'*' if nstage.get(j, 0) < 30 else ''}</small></th>" for j in piv.columns)
        body = "".join(f"<tr><th>{T_NAME.get(a, a)}</th>" + "".join(f'<td style="{heat(v, vmax)}">{"" if pd.isna(v) else f"{v:.0%}"}</td>' for v in rr) + "</tr>" for a, rr in piv.iterrows())
        jtab = f'<table class="t heat"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table><small>* n&lt;30 단계는 가설로 해석</small>'
    stab = info["stability"].get(str(s), "")
    insight = "".join(f"<li>{e(x)}</li>" for x in nar.get("insight", []))
    cards += f"""
<article class="card" id="p{s}">
 <header><div><small>{e(seg_tag(s))}</small><h3>{e(seg_name(s))}</h3><p>{e(nar.get('one_liner', ''))}</p></div>
 <div class="kpi"><small>전체 분석수</small><b>{int(r.n):,}개</b><span>유효 글 대비 {r.share:.1%}</span></div></header>
 {f'<div class="insight"><b>INSIGHT</b><ul>{insight}</ul></div>' if insight else ''}
 <div class="grid3">
  <section><h4>인구통계 프로필</h4><table class="mini">{profile_block(s)}</table></section>
  <section><h4>니즈 교차언급 (동일 게시글 내)</h4>{cbars or '<p class=muted>교차 부족</p>'}</section>
  <section class="feat"><h4>핵심 특징</h4>{''.join(f'<p>{e(x)}</p>' for x in ft)}</section>
 </div>
 <h4 class="band">니즈 상세</h4>
 <table class="t needs"><tbody>{need_rows}</tbody></table>
 <details><summary>뒷면: FoD 저니 장벽 · 근거 · 경계</summary>
  <div class="back"><section><h4>FoD 저니 × 태도 (A층 글)</h4>{jtab or '<p class=muted>A층 글 부족</p>'}</section>
  <section><h4>근거 등급</h4><p>부트스트랩 안정성 {stab} · 최다 출처 {e(r.top_source)} {r.top_source_share:.0%} · FoD 직접 글 비율 {r.a_fod_share:.0%}</p>
  <h4>이 카드로 말할 수 없는 것</h4><p>{e(nar.get('boundary', '시장 내 실제 비중, 연령 분포의 모집단 대표성, 지불 금액 수준 — 설문으로 확인 필요.'))}</p></section></div>
 </details>
</article>"""
S.append(("s6", "6. 페르소나 상세", "final", f"<h2>6. 페르소나 상세</h2>{cards}{method_link('m6')}"))

ideas = NAR.get("ideas") or []
ihtml = "".join(f"""<div class="idea"><small>{e(seg_name(i['segment']))}</small><h4>{e(i['title'])}</h4><p class="tags">{' '.join(f'<span class=tag>{e(t)}</span>' for t in i.get('needs', []))}</p>
<p>{e(i['concept'])}</p><ul>{''.join(f'<li>{e(x)}</li>' for x in i.get('specifics', []))}</ul><small>검증 실험: {e(i.get('experiment', '-'))}</small></div>""" for i in ideas)
S.append(("s7", "7. FoD 아이디어", "final", f"<h2>7. 페르소나별 FoD 아이디어 (결과 → 기회 → 해법 → 실험)</h2><div class='ideas'>{ihtml or '<p class=muted>narrative ideas 미작성</p>'}</div>"))

if not opp.empty:
    top_needs = opp.groupby("need_id").signal_index.max().nlargest(18).index
    piv = opp[opp.need_id.isin(top_needs)].pivot_table(index="hashtag", columns="segment", values="signal_index", aggfunc="max")
    piv = piv.reindex(columns=segs)
    vmax = piv.max().max()
    head = "".join(f"<th>{e(seg_name(s))}</th>" for s in piv.columns)
    body = "".join(f"<tr><th>{e(h)}</th>" + "".join(f'<td style="{heat(0 if pd.isna(v) else v, vmax)}">{"" if pd.isna(v) else f"{v:.3f}"}</td>' for v in rr) + "</tr>" for h, rr in piv.iterrows())
    S.append(("s8", "8. 기회 매트릭스", "final", f"""<h2>8. 기회 매트릭스 (텍스트 기회 신호 지수)</h2>
<p>지수 = 세그먼트 내 유병률 × 평균 강도 × (0.5 + 부정 비율) × (1 + 대체 행동 비율). <b>순위 비교용이며 ODI가 아님</b> — 설문 ODI로 확정.</p>
<table class="t heat"><thead><tr><th>#니즈</th>{head}</tr></thead><tbody>{body}</tbody></table>{method_link('m7')}"""))

if not jb.empty:
    a = lab[lab.layer == "A_FOD"]
    rowsj = []
    for j in JOURNEY:
        sub = a[a.journey.map(lambda xs: j in list(xs))]
        if len(sub) == 0:
            continue
        cnt = pd.Series([x for xs in sub.attitude for x in xs]).value_counts()
        rowsj.append((j, len(sub), {t: cnt.get(t, 0) / len(sub) for t in ATT}))
    vmax = max(max(d.values()) for _, _, d in rowsj)
    head = "".join(f"<th>{T_NAME[t]}</th>" for t in ATT)
    body = "".join(f"<tr><th>{J_NAME.get(j, j)}<br><small>n={n:,}</small></th>" + "".join(f'<td style="{heat(d[t], vmax)}">{d[t]:.0%}</td>' for t in ATT) + "</tr>" for j, n, d in rowsj)
    S.append(("s9", "9. FoD 저니 장벽", "final", f"""<h2>9. FoD 저니 단계 × 태도 (전체 A층 {len(a):,}건)</h2>
<table class="t heat"><thead><tr><th>단계</th>{head}</tr></thead><tbody>{body}</tbody></table>{method_link('m2')}"""))

# ---------------- 추가 분석 (run/07_extra.py) ----------------
X = T / "extra"
won = lambda v: "" if pd.isna(v) else f"{v / 1e4:,.1f}만" if v >= 1e4 else f"{v:,.0f}"
pct = lambda v: "" if pd.isna(v) else f"{v:.0%}"


def tbl(df, cols, fmt={}):
    head = "".join(f"<th>{e(h)}</th>" for _, h in cols)
    body = "".join("<tr>" + "".join(f"<td>{e(fmt.get(c, str)(r[c]))}</td>" for c, _ in cols) + "</tr>" for _, r in df.iterrows())
    return f'<table class="t"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


if (X / "e1_price_anchors.csv").exists():
    pa = csv("e1_price_anchors", X)
    pa = pa[pa.n_mentions >= 10].sort_values("n_mentions", ascending=False)
    S.append(("x1", "E1. 가격 앵커", "extra", f"""<h2>E1. 사람들이 말하는 기능 가격 (가격 앵커)</h2>
<p>본문에서 기능 이름 앞뒤 60자 안에 있는 금액 표현을 뽑았습니다(1천 원~500만 원, 언급 10건 이상만). '기간 미상'에는 선택 사양·용품 가격이 섞여 있습니다.</p>
<div class="ins"><b>테슬라 FSD 월 15만 원 전환(2026-08)이 가장 큰 가격 기준점</b>주행 보조 가격 언급의 대부분이 이 소식이고, 반응이 있는 글의 약 65%가 '비싸다'입니다. 커넥티드 서비스는 월 5,500~9,900원, 원격 주차 평생 이용권은 50만 원이 반복해서 나옵니다.</div>
{tbl(pa, [("feature", "기능"), ("period", "기간"), ("n_posts", "글"), ("p25", "하위 25%"), ("median", "중앙값"), ("p75", "상위 25%"), ("n_reaction", "반응 글"), ("expensive_share", "'비싸다' 비율")],
     {"p25": won, "median": won, "p75": won, "expensive_share": pct})}{method_link('m9')}"""))

if (X / "e2_store_by_group.csv").exists():
    sg, sv, sp = csv("e2_store_by_group", X), csv("e2_store_by_vehicle", X), csv("e2_store_by_product", X)
    sg = sg[sg.n >= 10]
    iss = ["적용·설치 실패", "호환·차종 차이", "가격·결제", "디자인 만족", "기능 효용"]
    S.append(("x2", "E2. 스토어 상품 성과", "extra", f"""<h2>E2. 기아 커넥트 스토어 상품 성과 (리뷰 {int(sg.n.sum()):,}건)</h2>
<div class="ins"><b>테마는 '디자인', 기능형은 '효용'으로 만족 — 불만은 사용 단계에 몰림</b>원격 주차의 부정 비율(11%)은 테마(6%)의 두 배 가까이이고, 가격·결제 언급이 17%로 가장 높습니다. 스트리밍은 표본이 작지만(12건) 부정이 압도적입니다.</div>
{tbl(sg, [("group", "상품군"), ("n", "리뷰"), ("neg_share", "부정"), ("neg_ci_lo", "부정 95% 하한"), ("neg_ci_hi", "상한"), ("pos_share", "긍정")] + [(c, c) for c in iss] + [("neg_top_journey", "부정 글의 저니 단계")],
     {c: pct for c in ["neg_share", "neg_ci_lo", "neg_ci_hi", "pos_share"] + iss})}
<h4>차종별 (상위 12)</h4>{tbl(sv, [("vehicle", "차종"), ("n", "리뷰"), ("neg_share", "부정"), ("neg_ci_lo", "95% 하한"), ("neg_ci_hi", "상한")], {c: pct for c in ["neg_share", "neg_ci_lo", "neg_ci_hi"]})}
<h4>부정 비율이 높은 상품 (리뷰 20건 이상)</h4>{tbl(sp.head(8), [("group", "상품군"), ("product", "상품"), ("n", "리뷰"), ("neg_share", "부정")], {"neg_share": pct})}
<p class="muted">Opposites United는 무료 테마라 '유료 고급판을 만들어 달라'는 요구가 부정으로 잡혔습니다.</p>{method_link('m10')}"""))

if (X / "e3_sub_situation.csv").exists():
    ss, spd, st, sx, sn = (csv(n, X) for n in ("e3_sub_situation", "e3_sub_products", "e3_sub_terms", "e3_sub_segment", "e3_sub_sentiment"))
    SUB = {"aftermarket": "사제 용품", "diy": "직접 해결", "other_app": "다른 앱", "give_up": "포기"}
    neg = dict(zip(sn.substitute, sn.neg_share))
    cards = ""
    for k, nm in SUB.items():
        g = ss[ss.substitute == k]
        if g.empty:
            continue
        cards += f"""<div class="idea"><small>n={int(g.n.iloc[0]):,} · 부정 {pct(neg.get(k))}</small><h4>{nm}</h4>
<p><b>상황</b> {' · '.join(f"{r.situation_name.split(',')[0]} (lift {r.lift})" for r in g.head(3).itertuples())}</p>
<p><b>대체 수단</b> {' · '.join(f"{r.product} {r.n}" for r in spd[spd.substitute == k].head(4).itertuples())}</p>
<p><b>특징어</b> {' '.join(f'<span class="tag">{e(t)}</span>' for t in st[st.substitute == k].term.head(8))}</p></div>"""
    sx = sx[sx.segment >= 0].copy()
    sx["name"] = sx.segment.map(seg_name)
    S.append(("x3", "E3. 대체재 경쟁 지도", "extra", f"""<h2>E3. FoD 대신 무엇으로 해결하나 (대체재 경쟁 지도)</h2>
<div class="ins"><b>경쟁 상대는 다른 브랜드가 아니라 사제 용품과 티맵</b>사제 용품은 부정 17%로 만족스러운 대안이고, 다른 앱(티맵·카플레이)은 순정 내비 불만(부정 51%)에서 나옵니다. '포기'는 구독 해지·환불과 함께 나오고 부정이 66%로 가장 높습니다.</div>
<div class="ideas">{cards}</div>
<h4>세그먼트별 대체 유형</h4>{tbl(sx, [("name", "세그먼트"), ("n", "대체 글")] + [(k, v) for k, v in SUB.items()], {k: pct for k in SUB})}{method_link('m11')}"""))

if (X / "e4_kano.csv").exists():
    kn = csv("e4_kano", X)
    th = json.loads((X / "e4_kano_thresholds.json").read_text())
    W, H, P = 640, 380, 50
    xm, ym = kn.absent_complaint.max() * 1.15, kn.present_delight.max() * 1.15
    sx_ = lambda v: P + v / xm * (W - 2 * P)
    sy_ = lambda v: H - P - v / ym * (H - 2 * P)
    pts = "".join(f'<circle cx="{sx_(r.absent_complaint):.0f}" cy="{sy_(r.present_delight):.0f}" r="{4 + math.sqrt(r.n) / 8:.0f}" fill="var(--accent)" opacity=".55"/>'
                  f'<text class="nl" x="{sx_(r.absent_complaint) + 6:.0f}" y="{sy_(r.present_delight) - 6:.0f}">{e(r.feature)}</text>' for r in kn.itertuples())
    svg = f"""<svg class="net" viewBox="0 0 {W} {H}"><line x1="{sx_(th['absent_median']):.0f}" y1="{P}" x2="{sx_(th['absent_median']):.0f}" y2="{H - P}" stroke="var(--line)" stroke-dasharray="4"/>
<line x1="{P}" y1="{sy_(th['delight_median']):.0f}" x2="{W - P}" y2="{sy_(th['delight_median']):.0f}" stroke="var(--line)" stroke-dasharray="4"/>
<text class="nl" x="{W - P}" y="{P}" text-anchor="end">일원</text><text class="nl" x="{P + 4}" y="{P}">매력</text><text class="nl" x="{W - P}" y="{H - P - 4}" text-anchor="end">당연</text><text class="nl" x="{P + 4}" y="{H - P - 4}">무관심</text>
<text class="nl" x="{W / 2}" y="{H - 12}" text-anchor="middle">없어서 불만 비율 →</text><text class="nl" x="14" y="{H / 2}" transform="rotate(-90 14 {H / 2})" text-anchor="middle">있어서 기쁨 비율 →</text>{pts}</svg>"""
    S.append(("x4", "E4. Kano 추정", "extra", f"""<h2>E4. 텍스트로 추정한 Kano 분류와 유료화 반감</h2>
<div class="ins"><b>열선·통풍은 '당연' + 유료화 반감 22% — FoD 후보에서 빼야 할 신호</b>테마·라이팅·영상은 '매력'으로 유료화 반감이 낮고, 원격 주차·회생 제동은 '일원'(있으면 만족, 없으면 불만)입니다. 디지털 키·주차 감시는 '당연'에 가까워 기본 탑재 기대가 큽니다.</div>
{svg}
{tbl(kn, [("feature", "기능"), ("n", "언급 글"), ("absent_complaint", "없어서 불만"), ("present_delight", "있어서 기쁨"), ("kano_class", "추정 분류"), ("boot_stability", "부트스트랩 안정성"), ("pay_resistance", "유료화 반감")],
     {"absent_complaint": lambda v: f"{v:.1%}", "present_delight": lambda v: f"{v:.1%}", "pay_resistance": lambda v: f"{v:.1%}"})}
<p class="muted">점선은 기능 간 중앙값. 안정성 0.7 미만 기능(원격 제어, 영상·게임, 캠핑 모드, OTA)은 분류가 경계에 있어 설문 Kano 문항으로 확정해야 합니다.</p>{method_link('m12')}"""))

# ---------------- 방법·신뢰도 (논문형) ----------------
def method(mid, title, methods, variables, reliability, limits):
    li = lambda xs: "".join(f"<li>{x}</li>" for x in xs)
    return (mid, title, "method", f"""<h2>{title}</h2>
<details open><summary>방법</summary><ul>{li(methods)}</ul></details>
<details open><summary>사용 변수</summary><ul>{li(variables)}</ul></details>
<details open><summary>신뢰도·타당도 지표</summary><ul>{li(reliability)}</ul></details>
<details open><summary>한계</summary><ul>{li(limits)}</ul></details>""")


twp = tw.pivot_table(index="source", columns="date_status", values="n", fill_value=0)
S.append(method("m1", "M1. 데이터 수집·정제·시간 창",
    ["네이버 검색 API(카페·블로그) 시드 뱅크 9종(FoD 직접·운전 상황·생애단계·신규 상황·경쟁·대체재·이벤트), 쿼리·정렬당 1,000건 상한, 저수율 조기 중단",
     "기아 커넥트 스토어 리뷰 전수, 앱 리뷰(Google Play·App Store)",
     "정규화 → URL·정규화 본문 중복 제거 → 15자 미만 제외 → 할당 표본(A층·리뷰 전수, 생애단계 칸 3,700, 상황 칸 카페 2,600·블로그 600, 채널 ≤30%)",
     "시간 창 2021-10-10 이후 '혼합' 모드: 블로그·리뷰는 작성일, 카페는 최신순 결과·카페별 글번호 기준선, 판정 불가 카페 글 유지"],
    ["source, seed, seed_code, channel(카페·블로그명), created_at, date_status"],
    [f"시간 창 판정 분포: {e(twp.to_dict())}",
     f"민감도(혼합 vs 엄격 시간 창) 세그먼트 배정 ARI = {sens[0] if sens else 'n/a'} (공통 {sens[1] if sens else 0:,}건)",
     "블로그 대리 측정: 최신순 1,000건은 시드 중앙값 기준 최근 1주 내 → 카페 최신순 결과는 5년 내로 판정"],
    ["네이버 카페 스니펫에는 작성일이 없어 판정 불가 글에 5년 밖 글이 약 10~15% 섞였을 수 있음",
     "검색 API 결과는 검색 엔진 랭킹에 의존(대표 표본 아님). 네이버 카페 본문 미수집(스니펫 1~2줄)",
     "네이버 검색 API 약관(2.3: 저장·AI 입력 금지)과의 충돌을 인지하고 팀 결정으로 사용, 2026-10-30 원문 삭제 예정"]))

ev_rows = ""
if not evals.empty:
    for r in evals.itertuples():
        ev_rows += f"<li>{e(r.cell)}: layer {r.layer_acc} · 상황 F1 {r.situation_F1} · 저니 F1 {r.journey_F1} · 태도 F1 {r.attitude_F1} · 근거 유효 {r.evidence_valid} · $/1천건 {r.cost_per_1k_items}</li>"
S.append(method("m2", "M2. LLM 라벨링",
    ["Claude Haiku 5.5(effort medium), 10건 묶음, Message Batches, 구조화 출력(코드 enum 강제)",
     "코드북 v2 + 니즈 계층 v1을 시스템 프롬프트로 제공, 코드마다 원문 부분 문자열 근거 요구",
     "수집 후 근거가 원문에 없는 코드는 자동 삭제"],
    ["layer(A_FOD/B_SITUATION/INFO/EXCLUDE), situation[], needs[], journey[], attitude[], life_stage, gender",
     "패싯: outcome, intensity(0~3), frequency, wtp_signal, substitute, sentiment(-1/0/1), actor"],
    [f"라벨 {len(labels):,}건, 근거 미검증으로 삭제된 코드 1,638개, 실패 요청 13/10,988",
     "모델 선택 시험(임시 정답 300건, 코드북 v0):<ul>" + ev_rows + "</ul>",
     "정답은 Claude Code 세션 초벌 라벨 — 팀 검수 전, Opus 유리 편향 가능"],
    ["코드북 v2·니즈 계층 기준 정답 세트 재측정 전(현재 F1은 v0 기준)", "스니펫이 짧아 상황 코드 2개 이상인 글은 유효 글의 22%"]))

S.append(method("m3", "M3. 니즈 계층(오픈 코딩)",
    ["층화 표본 2,000 + 얇은 코드 보충 → Claude Opus 5.5가 해법 없는 결과 문장(방향+지표+대상) 작성",
     "상황 코드별로 결과 문장을 3차 니즈 2~5개로 귀납 도출(#해시태그 별칭·정의·포함/제외 기준)",
     "앞 절반으로 도출 → 뒤 절반 배정해 NEW 비율로 포화 확인 → 전체로 재도출 → 3건 미만 니즈 제외"],
    ["outcome(결과 문장), situation(2차), need_l3(3차)"],
    ["포화 검사 NEW 비율 전체 13.1%(코드별 0~50%) → 얇은 코드 보충 후 재도출", "3차 니즈 102개(제외 4개)"],
    ["LLM 귀납 결과로 사람 검토 전", "문장 수가 적은 코드(반려동물·발렛 등)는 니즈 경계가 불안정"]))

S.append(method("m4", "M4. 데모 클루",
    ["본문 자기 진술(LLM 근거 + 정규식 규칙), 메타(차종·상품), 채널명 단서를 basis로 분리 기록",
     "카드 표시 게이트: 단서 보유율 ≥30% 그리고 ≥100건, 미달은 '판단 불가'"],
    ["life_stage(첫차·신혼·영유아·학령기·부모돌봄/은퇴), gender, ev, brand, channel"],
    [f"규칙 본문 단서와 LLM 생애단계 일치율 {ls_agree[0]:.1%} (둘 다 판정 {ls_agree[1]:,}건 / 규칙 판정 {ls_agree[2]:,}건)"],
    ["단서가 있는 글만의 분포 — 모집단 연령 분포가 아님", "입력 습관 단서는 쓰지 않음(롯데마트 대비)"]))

S.append(method("m5", "M5. 니즈 테마 네트워크",
    ["상황 코드 공출현 → PMI·lift 정규화(원시 빈도 사용 안 함), 최소 공출현 = 유효 글의 0.05%",
     "Louvain 50 seed, 최고 모듈성 분할 채택, 노드별 공동배정 안정성"],
    ["situation[] (유효 글)"],
    [f"모듈성 Q = {info.get('theme_modularity')}", f"노드 {len(nodes)}, 엣지 {len(edges)}"],
    ["코드 테마는 '사람 유형'이 아님 — 페르소나는 M6"]))

ksel_html = ksel.to_html(index=False, classes="t") if not ksel.empty else segsel.to_html(index=False, classes="t")
S.append(method("m6", "M6. 니즈·순간 세그먼트(LCA)",
    ["상황 코드 상위 22개 이진 지표, 상황 2개 이상 글로 LCA 적합(stepmix, n_init 10)",
     "k 선택(v2): k=4~8마다 BIC·부트스트랩 안정성(8회) → 평균 안정성 ≥0.8 그리고 최소 ≥0.7인 k 중 가장 큰 k. <b>v1 규칙(최소 ≥0.5 중 평균 최대)은 k=4(평균 0.93)를 골랐으나 페르소나 해상도가 낮아 결과 확인 후 v2로 변경했음을 밝힌다.</b> BIC는 k=3~9에서 계속 감소해 선택 기준으로 쓰지 않음",
     "배정: 핵심(상황 ≥2) + 확장(상황 1개이면서 소속 확률 ≥0.6)",
     "교차 확인: KMeans 동일 k와 ARI, 엄격 시간 창 재적합과 ARI"],
    ["situation 상위 22개 0/1"],
    [f"선택 k = {info['k']}, 적합 {info['n_fit']:,}, 배정 {info['n_assigned']:,}",
     f"최종 적합 군집별 부트스트랩 Jaccard(10회): {e(info['stability'])} — k 선택 단계(8회) 측정치와 차이는 재표본 변동",
     f"KMeans ARI = {info['ari_vs_kmeans']} (보통 수준 — 이진 지표에 KMeans는 약한 비교 기준)",
     f"시간 창 민감도 ARI = {sens[0] if sens else 'n/a'}", "k 선택 표:" + ksel_html],
    ["글 단위 세그먼트 = '니즈·순간 유형'. 사람 페르소나는 설문 LCA로 확정", "텍스트 비중 ≠ 시장 비중"]))

S.append(method("m7", "M7. 텍스트 기회 신호 지수",
    ["세그먼트 × 3차 니즈별: 유병률 × 평균 강도 × (0.5 + 부정 비율) × (1 + 대체 행동 비율)"],
    ["needs, intensity, sentiment, substitute"],
    ["순위 비교용 — 설문 ODI(중요도·만족도)와 순위 상관으로 검증 예정"],
    ["언급량은 '말하기 쉬움'을 반영, 감성 ≠ 만족도 — ODI라고 부르지 않음"]))

S.append(method("m8", "M8. 한계와 주장 경계",
    [], [],
    ["주장 등급: 검토 가능(현재) → 근거 충분(정답 v3·팀 검수 후) → 설문 확인됨"],
    ["온라인 글 작성자 편향(말 많은 사람·카페 활동층)", "네이버 데이터 약관 리스크와 삭제 일정",
     "카페 스니펫 위주(본문 미수집)", "연령·성별은 단서 기반 추정", "LLM 라벨·니즈 계층은 팀 검토 전"]))

S.append(method("m9", "M9. 가격 앵커 추출 (E1)",
    ["정규식으로 금액 표현 파싱(12,000원 · 12만 원 · 1만2천 원 · 1.2만 원), 1천 원~500만 원만 사용",
     "금액 앞뒤 60자 안의 기능 사전(13개) 일치로 기능 배정, 금액 앞 12자의 '월·연·평생' 표현으로 기간 판정",
     "같은 창의 '비싸다/부담'·'저렴/가성비' 표현으로 반응 판정"],
    ["text, wtp_signal(라벨), 기능 사전(run/07_extra.py FEATURES)"],
    ["파서 단위 검사 7종 통과", "기능·기간별 사분위수, 언급 10건 이상만 표시"],
    ["창 안의 다른 금액(차값·옵션 묶음가)이 섞일 수 있음 — '기간 미상'은 해석 주의", "반응 판정 글이 적어(기능당 수~수십 건) '비싸다' 비율은 방향만 참고",
     "같은 기사·글이 여러 카페에 퍼진 경우 중복 언급"]))

S.append(method("m10", "M10. 스토어 상품 성과 (E2)",
    ["기아 커넥트 스토어 리뷰를 상품군(테마·RSPA 2·스마트 회생+·라이팅·게임·스트리밍)과 차종으로 묶음",
     "LLM 감성 라벨로 부정·긍정 비율, Wilson 95% 신뢰구간", "문제 유형 5개는 키워드 규칙, 부정 글의 저니 단계는 라벨"],
    ["sentiment, journey, meta(product·category·vehicle)"],
    ["상품군·차종별 부정 비율 Wilson CI 제시", "스토어 리뷰에는 별점이 없어 별점 기반 검증은 불가"],
    ["리뷰를 남긴 구매자만 — 산 뒤 불만으로 리뷰를 안 쓴 사람, 사지 않은 사람은 없음", "리뷰 이벤트(쿠폰) 영향으로 긍정 쏠림 가능"]))

S.append(method("m11", "M11. 대체재 경쟁 지도 (E3)",
    ["substitute 라벨(사제·직접 해결·다른 앱·포기)별 상황 코드 비율과 lift(전체 대비)",
     "대체 수단 사전 13개 일치 비율", "Kiwi 명사 + 균일 사전 log-odds z로 유형별 특징어", "세그먼트 × 대체 유형 교차표"],
    ["substitute, situation, sentiment, segment(M6), text"],
    ["lift > 2인 상황만 해석", "특징어 z > 8(빈도 5 이상)"],
    ["'대체 없음'(none) 글이 대부분이라 대체 행동은 언급된 경우만", "세그먼트 미배정 글(-1)은 교차표에서 제외"]))

S.append(method("m12", "M12. 텍스트 기반 Kano 추정 (E4)",
    ["기능 언급 글에서 (부정 감성 + 기능 앞뒤 60자 안 '없어서·빠져·안 넣' 표현) = 없어서 불만, (긍정 감성 + '편하더라·꿀기능·만족' 표현) = 있어서 기쁨",
     "두 비율을 기능 간 중앙값으로 나눠 4분면: 당연(불만↑기쁨↓)·매력(불만↓기쁨↑)·일원(둘 다↑)·무관심(둘 다↓)",
     "부트스트랩 300회로 같은 분류가 나오는 비율 = 안정성", "유료화 반감 = 태도 라벨 HW잠금 반감·이중결제 반감 비율(Wilson CI)"],
    ["sentiment, attitude, text, 기능 사전"],
    ["분류 안정성 0.54~1.00 (0.7 미만 4개 기능은 경계)"],
    ["정식 Kano는 기능 있음/없음 짝 질문이 필요 — 이 분류는 가설이며 설문으로 확정", "기준이 기능 간 상대값(중앙값)이라 '절대적으로 당연'이라는 뜻이 아님",
     "불만·기쁨 비율이 1~10%로 작아 표현 사전 범위에 민감"]))

# ---------------- 페이지 조립 ----------------
nav_final = "".join(f'<a href="#{i}" data-go="{i}">{e(t)}</a>' for i, t, g, _ in S if g == "final")
nav_extra = "".join(f'<a href="#{i}" data-go="{i}">{e(t)}</a>' for i, t, g, _ in S if g == "extra")
nav_method = "".join(f'<a href="#{i}" data-go="{i}">{e(t)}</a>' for i, t, g, _ in S if g == "method")
panels = "".join(f'<section class="panel" id="{i}">{h}</section>' for i, _, _, h in S)
title = NAR.get("title", "현대차 FoD 데이터 기반 페르소나")
page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FoD 페르소나 보고서</title><style>
:root{{--bg:#fbfaf8;--fg:#1c1b19;--muted:#77736c;--card:#ffffff;--line:#e6e2db;--accent:#d2452a;--band:#1f2a44;--soft:#fdeee9}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#141414;--fg:#ece9e4;--muted:#9a958d;--card:#1d1d1d;--line:#333;--accent:#ff6a4d;--band:#2d3a5c;--soft:#2a1d1a}}}}
:root[data-theme="dark"]{{--bg:#141414;--fg:#ece9e4;--muted:#9a958d;--card:#1d1d1d;--line:#333;--accent:#ff6a4d;--band:#2d3a5c;--soft:#2a1d1a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"Pretendard","Apple SD Gothic Neo","Malgun Gothic",sans-serif}}
.wrap{{display:grid;grid-template-columns:250px 1fr;min-height:100vh}}nav{{position:sticky;top:0;height:100vh;overflow:auto;padding:20px 16px;border-right:1px solid var(--line);background:var(--card)}}
nav h1{{font-size:16px;margin:0 0 4px}}nav p{{font-size:12px;color:var(--muted);margin:0 0 16px}}nav b{{display:block;margin:16px 0 6px;font-size:12px;color:var(--muted);letter-spacing:.04em}}
nav a{{display:block;padding:6px 10px;border-radius:8px;color:var(--fg);text-decoration:none;font-size:14px}}nav a.on,nav a:hover{{background:var(--soft);color:var(--accent)}}
main{{padding:28px 32px;max-width:1180px}}.panel{{display:none}}.panel.on{{display:block}}h2{{font-size:22px;margin:0 0 12px}}h3{{margin:2px 0;font-size:24px}}h4{{margin:0 0 8px;font-size:14px}}
.t{{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px}}.t th,.t td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}.t thead th{{color:var(--muted);font-weight:600}}
.heat td{{text-align:center}}small{{color:var(--muted)}}.muted{{color:var(--muted)}}.tag{{display:inline-block;background:var(--soft);color:var(--accent);border-radius:999px;padding:1px 8px;margin:2px;font-size:12px}}
.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:12px 0}}.flow div{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}}.flow b,.flow span,.flow strong{{display:block}}.flow span{{font-size:12px;color:var(--muted);margin:4px 0}}.flow strong{{font-size:20px;color:var(--accent)}}
.tracks{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}.tracks div{{background:var(--card);border-left:4px solid var(--band);padding:10px 12px;border-radius:6px}}
.ins{{border-left:5px solid var(--band);padding:8px 14px;margin:12px 0;background:var(--card)}}.ins b{{display:block;font-size:17px}}
.net{{width:100%;max-width:820px;background:var(--card);border:1px solid var(--line);border-radius:12px}}.nl{{font-size:10px;fill:var(--fg)}}
.card{{background:var(--card);border:1px solid var(--line);border-top:5px solid var(--accent);border-radius:14px;padding:18px;margin:20px 0}}
.card header{{display:flex;justify-content:space-between;gap:16px}}.kpi{{background:var(--soft);border-radius:10px;padding:10px 16px;text-align:center;min-width:170px}}.kpi b{{display:block;font-size:26px;color:var(--accent)}}
.insight{{background:var(--soft);border-radius:10px;padding:8px 14px;margin:10px 0}}.insight ul{{margin:4px 0;padding-left:18px}}
.grid3{{display:grid;grid-template-columns:1fr 1.2fr 1fr;gap:12px;margin:12px 0}}.grid3 section{{background:var(--bg);border-radius:10px;padding:10px}}.mini th{{width:70px;text-align:left;font-weight:600;font-size:12px;vertical-align:top}}.mini td{{font-size:13px}}
.feat p{{margin:0 0 6px;font-size:13px}}.band{{background:var(--band);color:#fff;padding:6px 12px;border-radius:6px}}.needs td{{font-size:13px}}.q{{color:var(--muted);font-size:12px}}
.crow{{font-size:12px;margin:4px 0}}.bar{{position:relative;height:16px;background:var(--line);border-radius:4px;overflow:hidden}}.bar span{{position:absolute;inset:0 auto 0 0}}.bar em{{position:absolute;right:4px;top:-1px;font-size:11px;font-style:normal}}
details{{margin:10px 0}}summary{{cursor:pointer;font-weight:600}}.back{{display:grid;grid-template-columns:1.4fr 1fr;gap:12px;margin-top:8px}}
.ideas{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}.idea{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}}.idea h4{{font-size:16px;margin:4px 0}}
.mlink{{display:inline-block;margin-top:10px;font-size:13px;color:var(--accent)}}
@media (max-width:820px){{.wrap{{grid-template-columns:1fr}}nav{{position:static;height:auto}}main{{padding:16px}}.flow,.grid3,.tracks,.back{{grid-template-columns:1fr}}.card header{{flex-direction:column}}}}
</style></head><body><div class="wrap"><nav><h1>{e(title)}</h1><p>생성 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 텍스트 기반(설문 확정 전)</p>
<b>최종 정리본</b>{nav_final}<b>추가 분석</b>{nav_extra}<b>방법·신뢰도</b>{nav_method}</nav><main>{panels}</main></div>
<script>
const show=id=>{{document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.id===id));document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('on',a.dataset.go===id));window.scrollTo(0,0)}};
document.addEventListener('click',ev=>{{const a=ev.target.closest('[data-go]');if(a){{ev.preventDefault();history.replaceState(null,'','#'+a.dataset.go);show(a.dataset.go)}}}});
show(location.hash.slice(1)||'s1');
</script></body></html>"""
OUT.write_text(page, encoding="utf-8")
print("→", OUT.relative_to(ROOT), f"({len(page) / 1e3:.0f} KB)")
