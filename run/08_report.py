"""분석 표 → 단일 HTML 보고서 (최종 정리본 + 방법·신뢰도). 외부 라이브러리·네트워크 없이 열린다.

  python run/08_report.py
입력: data/output/tables/*.csv, data/output/tables_strict/post_segment.csv, data/stage/*,
      config/report_narrative.yaml (세그먼트 이름·인사이트·아이디어; 없으면 표에서 자동 문구)
출력: data/output/report.html
"""
import html
import json
import math
import re
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
    return f"background:rgba(var(--heat),{a * 0.42:.2f})"


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
        ql = [f'"{e(q)}"' for q in qs.quote]
        qhtml = " · ".join(ql[:2]) + (f'<details><summary>인용 {len(ql) - 2}개 더</summary>{" · ".join(ql[2:])}</details>' if len(ql) > 2 else "")
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
 <header><div><h3>{e(seg_name(s))}</h3><p class="tagline">{e(seg_tag(s))}</p><p>{e(nar.get('one_liner', ''))}</p></div>
 <div class="kpi"><small>전체 분석수</small><b>{int(r.n):,}개</b><span>유효 글 대비 {r.share:.1%}</span></div></header>
 {f'<div class="insight"><b>인사이트</b><ul>{insight}</ul></div>' if insight else ''}
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
chips = "".join(f'<a class="tag" href="#p{sg}" data-go="s6" data-card="p{sg}">{e(seg_name(sg))}</a>' for sg in segs)
S.append(("s6", "6. 페르소나 상세", "final", f"<h2>6. 페르소나 상세</h2><p>{chips}</p>{cards}{method_link('m6')}"))

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
<div class="ins"><b>테슬라 FSD 월 15만 원 전환(2026-08)이 가장 큰 가격 기준점</b>주행 보조 가격 언급의 대부분이 이 소식입니다. 커넥티드 서비스는 월 5,500~9,900원, 원격 주차 평생 이용권은 50만 원이 반복해서 나옵니다. '비싸다/적정' 반응 판정은 규칙 검증 정밀도가 0.29라 표시하지 않습니다(V 메뉴).</div>
{tbl(pa, [("feature", "기능"), ("period", "기간"), ("n_posts", "글"), ("p25", "하위 25%"), ("median", "중앙값"), ("p75", "상위 25%"), ],
     {"p25": won, "median": won, "p75": won})}{method_link('m9')}"""))

if (X / "e2_store_by_group.csv").exists():
    sg, sv, sp = csv("e2_store_by_group", X), csv("e2_store_by_vehicle", X), csv("e2_store_by_product", X)
    sg = sg[sg.n >= 10]
    iss = ["적용·설치 실패", "호환·차종 차이", "가격·결제", "디자인 만족", "기능 효용"]
    S.append(("x2", "E2. 스토어 상품 성과", "extra", f"""<h2>E2. 기아 커넥트 스토어 상품 성과 (리뷰 {int(sg.n.sum()):,}건)</h2>
<div class="ins"><b>테마는 '디자인', 기능형은 '효용'으로 만족 — 불만은 사용 단계에 몰림</b>원격 주차의 부정 비율(11%)은 테마(6%)의 두 배 가까이입니다. 가격·결제 언급(1+1·무료 이벤트 포함)은 테마 28%, 원격 주차 22%로 할인·이벤트가 구매 계기로 자주 나옵니다. 스트리밍은 표본이 작지만(12건) 부정이 압도적입니다. '호환·차종 차이' 열은 검증 정밀도 0이라 해석하지 않습니다.</div>
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
<div class="ins"><b>안정적인 분류는 4개뿐 — 디지털 키·열선은 '당연', 테마는 '매력', 주행 보조는 '무관심'</b>열선·통풍은 유료화 반감도 22%로 FoD 후보에서 빼야 할 신호입니다. 나머지 기능은 부트스트랩 안정성이 0.7 미만이고, '없어서 불만' 규칙 정밀도가 0.57(수정 후)이라 분류는 가설로만 봅니다. 규칙을 고치자 원격 주차(일원→매력) 등 여러 기능의 분류가 바뀌었습니다.</div>
{svg}
{tbl(kn, [("feature", "기능"), ("n", "언급 글"), ("absent_complaint", "없어서 불만"), ("present_delight", "있어서 기쁨"), ("kano_class", "추정 분류"), ("boot_stability", "부트스트랩 안정성"), ("pay_resistance", "유료화 반감")],
     {"absent_complaint": lambda v: f"{v:.1%}", "present_delight": lambda v: f"{v:.1%}", "pay_resistance": lambda v: f"{v:.1%}"})}
<p class="muted">점선은 기능 간 중앙값. 안정성 0.7 미만 기능은 분류가 경계에 있어 설문 Kano 문항으로 확정해야 합니다.</p>{method_link('m12')}"""))

def scatter(df, xcol, ycol, label, W=640, H=400, P=50, color=None, xlab="", ylab="", xline=None, yline=None, size=None):
    xs, ys = df[xcol], df[ycol]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = lambda a, b: ((b - a) or 1) * 0.12
    x0, x1, y0, y1 = x0 - pad(x0, x1), x1 + pad(x0, x1), y0 - pad(y0, y1), y1 + pad(y0, y1)
    sx_ = lambda v: P + (v - x0) / (x1 - x0) * (W - 2 * P)
    sy_ = lambda v: H - P - (v - y0) / (y1 - y0) * (H - 2 * P)
    g = ""
    if xline is not None:
        g += f'<line x1="{sx_(xline):.0f}" y1="{P}" x2="{sx_(xline):.0f}" y2="{H - P}" stroke="var(--line)" stroke-dasharray="4"/>'
    if yline is not None:
        g += f'<line x1="{P}" y1="{sy_(yline):.0f}" x2="{W - P}" y2="{sy_(yline):.0f}" stroke="var(--line)" stroke-dasharray="4"/>'
    for _, r in df.iterrows():
        c = color(r) if color else "var(--accent)"
        rad = size(r) if size else 6
        g += (f'<circle cx="{sx_(r[xcol]):.0f}" cy="{sy_(r[ycol]):.0f}" r="{rad:.0f}" fill="{c}" opacity=".6"/>'
              f'<text class="nl" x="{sx_(r[xcol]) + rad + 2:.0f}" y="{sy_(r[ycol]) + 3:.0f}">{e(r[label])}</text>')
    g += f'<text class="nl" x="{W / 2}" y="{H - 12}" text-anchor="middle">{e(xlab)}</text><text class="nl" x="14" y="{H / 2}" transform="rotate(-90 14 {H / 2})" text-anchor="middle">{e(ylab)}</text>'
    return f'<svg class="net" viewBox="0 0 {W} {H}">{g}</svg>'


if (X / "e5_app_ipa.csv").exists():
    meta2 = json.loads((X / "e5_e9_meta.json").read_text(encoding="utf-8"))
    ipa = csv("e5_app_ipa", X)
    m5 = meta2["e5"]
    S.append(("x5", "E5. 앱 리뷰 IPA", "extra", f"""<h2>E5. 커넥티드 앱 리뷰 IPA (리뷰 {m5['n']:,}건, 평균 별점 {m5['mean_rating']})</h2>
<div class="ins"><b>별점을 깎는 건 원격 제어 기능이 아니라 '연결·로그인·속도'</b>연결·통신, 로그인·인증, 속도·로딩 언급은 각각 별점을 약 0.65~0.74점 낮춥니다. 원격 제어 자체는 −0.09점으로 거의 영향이 없습니다. 커넥티드 관리자 페르소나의 '먹통' 불만은 기능이 아니라 기반(연결·인증) 문제입니다.</div>
{scatter(ipa, "performance", "importance", "aspect", xlab="성과: 언급 리뷰 중 4~5점 비율 →", ylab="중요도: 별점 하락폭 →", xline=ipa.performance.mean(), yline=ipa.importance.mean(), size=lambda r: 4 + math.sqrt(r.n) / 6)}
<p class="muted">오른쪽 위 = 유지, 왼쪽 위 = 집중 개선, 왼쪽 아래 = 낮은 우선순위, 오른쪽 아래 = 과잉. 점 크기 = 언급 수.</p>
{tbl(ipa.sort_values("importance", ascending=False), [("aspect", "측면"), ("n", "언급"), ("coef", "별점 계수"), ("se", "표준오차"), ("mean_rating", "평균 별점"), ("performance", "4~5점 비율")], {"performance": pct})}{method_link('m13')}"""))

if (X / "e6_forces.csv").exists():
    fo = csv("e6_forces", X)
    fb = lambda v: f'<div class="bar"><span style="width:{min(100, v * 150):.0f}%;background:var(--accent)"></span><em>{v:.0%}</em></div>'
    fbn = lambda v: f'<div class="bar"><span style="width:{min(100, v * 150):.0f}%;background:var(--band)"></span><em>{v:.0%}</em></div>'
    body = "".join(f"<tr><th>{e(r.feature)}<br><small>n={r.n:,}</small></th><td>{fb(r.push)}</td><td>{fb(r.pull)}</td><td>{fbn(r.anxiety)}</td><td>{fbn(r.habit)}</td><td><b>{r.net:+.2f}</b></td></tr>" for r in fo.itertuples())
    S.append(("x6", "E6. 전환의 4가지 힘", "extra", f"""<h2>E6. 기능별 전환의 4가지 힘 (JTBD Forces of Progress)</h2>
<p>살 쪽으로 미는 힘(불편 Push + 매력 Pull)과 붙잡는 힘(불안 Anxiety + 기존 습관·대안 Habit)을 기능별로 비교했습니다.</p>
<div class="ins"><b>디지털 키는 '불편'이, 테마는 '매력'이 끌고 — 주행 보조는 '불안'이, 영상·주차 감시는 '기존 대안'이 막는다</b>주행 보조의 불안 41%는 대부분 망설임·본전 계산(FSD 가격)이고, 열선·통풍의 불안은 거의 전부 HW잠금 반감입니다. 주차 감시는 사제 블랙박스, 영상은 폰·태블릿이 이미 자리를 차지하고 있어 FoD가 이기려면 '갈아탈 이유'가 필요합니다.</div>
<table class="t"><thead><tr><th>기능</th><th>Push 불편</th><th>Pull 매력</th><th>Anxiety 불안</th><th>Habit 대안</th><th>순힘</th></tr></thead><tbody>{body}</tbody></table>{method_link('m14')}"""))

if (X / "e7_its_model.csv").exists():
    wk, im = csv("e7_its_weekly", X), csv("e7_its_model", X)
    wf = wk[(wk.grp == "FSD") & (wk.n >= 3)]
    W_, H_, P_ = 640, 260, 40
    xmin, xmax = wf.week.min(), wf.week.max()
    sx_ = lambda v: P_ + (v - xmin) / ((xmax - xmin) or 1) * (W_ - 2 * P_)
    sy_ = lambda v: H_ - P_ - v * (H_ - 2 * P_)
    line = lambda col, c: f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{" ".join(f"{sx_(r.week):.0f},{sy_(r[col]):.0f}" for _, r in wf.iterrows())}"/>'
    svg = f"""<svg class="net" viewBox="0 0 {W_} {H_}"><line x1="{sx_(0):.0f}" y1="{P_}" x2="{sx_(0):.0f}" y2="{H_ - P_}" stroke="var(--fg)" stroke-dasharray="4"/>
<text class="nl" x="{sx_(0) + 4:.0f}" y="{P_ + 10}">2026-08-10 FSD 월 구독 전환</text>{line("neg", "var(--accent)")}{line("anx", "var(--band)")}
<text class="nl" x="{P_}" y="{H_ - 10}">주(전환일 기준) · 빨강 = 부정 비율 · 남색 = 불안 태도 비율</text></svg>"""
    did = meta2["e7_did"]
    S.append(("x7", "E7. FSD 구독 전환 시계열", "extra", f"""<h2>E7. 테슬라 FSD 월 구독 전환 전후 (중단 시계열)</h2>
<div class="ins"><b>전환 발표 뒤 부정·불안이 '튀지' 않았다 — 통계적으로 유의한 변화 없음</b>FSD 글의 부정 비율 수준 변화 {im.query("outcome=='neg' and group=='FSD'").level_change.iloc[0]:+.3f} (표준오차 {im.query("outcome=='neg' and group=='FSD'").level_se.iloc[0]:.3f}), 비교 계열 대비 차이 {did['neg']:+.3f}. 일시불(904만 원) → 월 15만 원 전환이 '반발'보다 '계산'(구독 vs 일시불)으로 받아들여졌다는 해석과 맞습니다. 불안 비율은 전 0.60 → 후 0.45로 낮아졌지만 추세 안의 변화입니다.</div>
{svg}
{tbl(im, [("outcome", "결과"), ("group", "계열"), ("weeks", "주 수"), ("pre_mean", "전 평균"), ("post_mean", "후 평균"), ("level_change", "수준 변화"), ("level_se", "표준오차"), ("slope_change", "기울기 변화")])}{method_link('m15')}"""))

if (X / "e8_topics.csv").exists():
    tp = csv("e8_topics", X)
    gaps = tp[(tp.purity < 0.3) | (tp.no_code_share >= 0.1)]
    S.append(("x8", "E8. 토픽 ↔ 코드북 검증", "extra", f"""<h2>E8. 비지도 토픽(NMF 25개)으로 코드북 교차 검증</h2>
<div class="ins"><b>토픽 대부분이 코드 하나에 대응 (NMI {meta2['e8_nmi']}) — 코드북 바깥 후보는 '보험 할인'과 '서비스 유료 전환'</b>아이·적재·테마·정비 토픽은 순도 0.8 이상으로 코드북과 잘 맞습니다. '자동차 보험·커넥티드 할인' 토픽은 돈 계산(S_COST)에 묶여 있어 커넥티드 데이터 기반 보험 할인(UBI)을 별도 상황 코드로 둘지 검토할 만합니다. '블루링크 무료 기간 종료·유료 전환' 토픽은 코드 없는 글이 13%로 저니(관리·정산)에만 잡힙니다.</div>
<h4>코드북과 어긋나는 토픽 (순도 &lt;0.3 또는 코드 없음 ≥10%)</h4>
{tbl(gaps, [("topic", "#"), ("n", "글"), ("words", "상위 단어"), ("dominant_name", "가장 많은 코드"), ("purity", "순도"), ("no_code_share", "코드 없음")], {"purity": pct, "no_code_share": pct})}
<details><summary>전체 토픽 25개</summary>{tbl(tp, [("topic", "#"), ("n", "글"), ("words", "상위 단어"), ("dominant_name", "가장 많은 코드"), ("purity", "순도"), ("a_fod_share", "A층")], {"purity": pct, "a_fod_share": pct})}</details>{method_link('m16')}"""))

if (X / "e9_ca_points.csv").exists():
    ca = csv("e9_ca_points", X)
    ca["nm"] = ca.apply(lambda r: LS_NAME.get(r.label, r.label) if r.kind == "who" else r["name"], axis=1)
    inr = meta2["e9_inertia"]
    who, sit_ = ca[ca.kind == "who"], ca[ca.kind == "situation"]
    near = "".join(f"<tr><th>{e(w.nm)}</th><td>" + " · ".join(e(x.split(',')[0]) for x in sit_.assign(d=(sit_.x - w.x) ** 2 + (sit_.y - w.y) ** 2).nsmallest(3, "d")["name"]) + "</td></tr>" for w in who.itertuples())
    ca["lbl"] = ca.apply(lambda r: r.nm if r.kind == "who" or r.mass >= 0.07 else "", axis=1)
    S.append(("x9", "E9. 대응 분석", "extra", f"""<h2>E9. 누가 어떤 상황을 말하나 (대응 분석)</h2>
<div class="ins"><b>가로축 = '가족' 대 '차 기술', 세로축 = '현대·기아 커넥티드' 대 '입문·전기차'</b>영유아·학령기 부모와 여성 단서는 아이 동승·적재와 함께 왼쪽에 모이고, 현대·기아는 개인화·원격 제어·내비와 위쪽에, 첫차·신혼은 주차 실력·교체 고민과 아래쪽에 모입니다. 부모 단서가 아이 동승·적재와 붙는 것은 아이 동승 부모 세그먼트와 일치하고, EV·수입 단서는 돈 계산·주행 보조·교체와 같은 쪽(구매 결정자 세그먼트의 상황)에 있습니다.</div>
{scatter(ca, "x", "y", "lbl", W=720, H=480, xlab=f"1축 ({inr[0]:.0%})", ylab=f"2축 ({inr[1]:.0%})", xline=0, yline=0, color=lambda r: "var(--band)" if r.kind == "who" else "var(--accent)", size=lambda r: 3 + math.sqrt(r.mass) * 18)}
<p class="muted">남색 = 사람 단서(생애단계·성별·브랜드·EV), 빨강 = 상황 코드. 가까울수록 함께 나오는 경향. 1·2축 설명력 합 {inr[0] + inr[1]:.0%}. 큰 상황(비중 7% 이상)만 이름을 붙였습니다.</p><h4>사람 단서별 가장 가까운 상황 3개</h4><table class="t"><thead><tr><th>사람 단서</th><th>가까운 상황</th></tr></thead><tbody>{near}</tbody></table>{method_link('m17')}"""))

if (X / "s1_scorecard.csv").exists():
    sc = csv("s1_scorecard", X)
    DEC_C = {"FoD 판매 후보": "var(--accent)", "기본 탑재 권장": "var(--band)", "보류": "var(--muted)"}
    rows_ = ""
    for d in ("FoD 판매 후보", "기본 탑재 권장", "보류"):
        for r in sc[sc.decision == d].sort_values("mean_rank").itertuples():
            sens_ = "" if pd.isna(r.p_top_half) else f"{r.p_top_half:.0%} (순위 {r.rank_p05:.0f}~{r.rank_p95:.0f})"
            rows_ += (f'<tr><th>{e(r.feature)}</th><td><span class="tag" style="color:{DEC_C[d]}">{d}</span></td><td>{r.mean_rank:.2f}</td>'
                      f'<td>{r.push:.0%}</td><td>{r.pull:.0%}</td><td>{r.anxiety:.0%}</td><td>{r.habit:.0%}</td><td>{r.pay_resistance:.1%}</td>'
                      f'<td>{e(r.kano_class.split("(")[0])} <small>{r.boot_stability:.2f}</small></td><td style="white-space:nowrap">{e(sens_)}</td><td><small>{e(r.price_ref if isinstance(r.price_ref, str) else "")}</small></td></tr>')
    S.append(("x10", "S1. FoD 후보 점수표", "extra", f"""<h2>S1. 기능별 FoD 후보 점수표 (가격·기대 유형·4가지 힘 종합)</h2>
<div class="ins"><b>FoD로 팔 것: 회생 제동·화면 테마가 가중치와 무관하게 상위 / 기본 탑재: 디지털 키·열선 / 보류: 영상·게임·라이팅·OTA·주행 보조</b>회생 제동과 테마는 가중치를 2,000번 무작위로 바꿔도 92~97%가 상위 절반입니다. 원격 주차·원격 제어·캠핑 모드·주차 감시는 58~63%로 '후보이나 가중치에 민감'합니다. 주행 보조(FSD)는 수요는 크지만 불안이 커서 보류입니다.</div>
<p>기준 6개(수요·불편·매력은 높을수록, 불안·기존 대안·유료화 반감은 낮을수록 좋음)의 순위 평균. 판정 규칙은 결과를 보기 전에 정했습니다: 유료화 반감 ≥10% 또는 Kano '당연'(안정성 ≥0.7) → 기본 탑재 권장, 나머지는 순위 평균 중앙값으로 판매 후보/보류.</p>
<table class="t"><thead><tr><th>기능</th><th>판정</th><th>순위 평균</th><th>불편</th><th>매력</th><th>불안</th><th>대안</th><th>유료화 반감</th><th>Kano <small>안정성</small></th><th>상위 절반 확률</th><th>가격 앵커</th></tr></thead><tbody>{rows_}</tbody></table>{method_link('m18')}"""))

if (X / "s2_prescriptions.csv").exists():
    pr = csv("s2_prescriptions", X)
    BLK = {"habit": "기존 대안", "anxiety": "불안", "weak": "약함"}
    cards = "".join(f"""<div class="idea"><small>n={r.n:,} · 막는 힘: {BLK.get(r.blocking, r.blocking)} · 대체 행동 중 {e(r.substitute_top)}</small><h4>{e(seg_name(r.segment))}</h4>
<p><b>맞는 기능</b> {e(r.features)}</p>
<p><b>4가지 힘</b> 불편 {r.push:.0%} · 매력 {r.pull:.0%} · 불안 {r.anxiety:.0%} · 대안 {r.habit:.0%}</p>
<p><b>가격 앵커</b> {e(r.price_ref) if isinstance(r.price_ref, str) else "—"}</p>
<p class="band">{e(r.condition)}</p></div>""" for r in pr.itertuples())
    S.append(("x11", "S2. 페르소나별 FoD 처방", "extra", f"""<h2>S2. 페르소나별 FoD 처방</h2>
<div class="ins"><b>대부분의 세그먼트를 막는 건 '불안'이 아니라 '기존 대안'</b>사제 보충파(사제 91%)·문콕 방어자(사제 61%)는 블랙박스 등 사제 용품, 커넥티드 관리자·공간 활용족은 직접 해결이 자리를 차지하고 있습니다. 불안이 막는 건 구매 결정자뿐이고(망설임·본전 계산), 여기에는 체험·짧은 기간권이 맞습니다. 아이 동승 부모는 FoD 기능 언급 자체가 적어 판매 조건보다 상황 노출이 먼저입니다.</div>
<div class="ideas">{cards}</div>
<p class="muted">맞는 기능 = 세그먼트 안 언급 비율이 전체의 1.2배 넘고 30건 이상. 판매 조건은 막는 힘·주요 불안 태도에서 규칙으로 정했습니다(M18).</p>{method_link('m18')}"""))

if (X / "v_rule_validation.csv").exists():
    vr = csv("v_rule_validation", X)
    S.append(("x12", "V. 규칙 판정 검증", "extra", f"""<h2>V. 키워드 규칙 판정 검증 (사람 정답 399건)</h2>
<div class="ins"><b>가격 반응(0.29)·없어서 불만(0.45)은 원래 규칙이 부정확 — 가격 반응은 보고서에서 빼고, 나머지 규칙은 고쳐서 다시 돌렸습니다</b>'없어서 불만'은 고장('안 들어와요')·출금('돈이 빠져나가') 오탐을 지워 0.57로, 스토어 '가격·결제'는 1+1·무료 이벤트를 넣어 재현율 0.21 → 0.96으로 올렸습니다. 단 수정 후 수치는 <b>같은 표본으로 다시 잰 것이라 낙관적</b>입니다. '호환·차종 차이'는 정답이 2건뿐이라 판단할 수 없습니다.</div>
{tbl(vr, [("check", "판정"), ("rule", "규칙"), ("type", "지표"), ("n", "n"), ("value", "값"), ("ci_lo", "95% 하한"), ("ci_hi", "상한")], {"value": lambda v: "" if pd.isna(v) else f"{v:.2f}", "ci_lo": lambda v: f"{v:.2f}", "ci_hi": lambda v: f"{v:.2f}"})}{method_link('m19')}"""))

if (X / "s3_survey_items.csv").exists():
    si = csv("s3_survey_items", X)
    sm = json.loads((X / "s3_survey_meta.json").read_text(encoding="utf-8"))
    rec = " · ".join(f"{seg_name(int(k))} {v:.0%}" for k, v in sm["recall_by_segment_12"].items())
    S.append(("x13", "S3. 설문 판별 문항", "extra", f"""<h2>S3. 세그먼트를 가르는 설문 문항 후보</h2>
<div class="ins"><b>상황 문항 5개면 6개 세그먼트를 90% 가른다 — 그중 2개(원격·앱 실패, 캠핑·레저)는 지금 설문에 없음</b>문콕 칸 주차 → 사제 장착 → 아이 동승 → 캠핑·레저 → 원격·앱 사용 순서로 넣으면 균형 정확도가 0.33 → 0.90으로 오릅니다(우연 기준 {sm['chance_balanced']:.2f}, 22개 전부 {sm['full22_balanced']:.2f}). 구매 결정자는 '다른 상황이 없음'으로 가려지므로, 설문에는 돈 계산·교체 고민을 직접 묻는 문항을 따로 둬야 합니다.</div>
{tbl(si, [("rank", "순서"), ("name", "상황"), ("balanced_acc", "누적 균형 정확도"), ("points_to_segment", "가리키는 세그먼트"), ("survey_item", "설문 문항"), ("status", "상태"), ("item_text", "문항 내용")], {"points_to_segment": seg_name, "balanced_acc": lambda v: f"{v:.2f}"})}
<p class="muted">12개 사용 시 세그먼트별 재현율: {e(rec)}. 세그먼트가 같은 코드로 만들어졌으므로 이 정확도는 '몇 개 코드로 재현되나'를 뜻하며, 실제 설문 타당도는 응답자로 다시 학습·검증해야 합니다.</p>{method_link('m20')}"""))

if (X / "v_label_accuracy.csv").exists():
    la = csv("v_label_accuracy", X)
    top = la[~la.metric.str.startswith("상황 F1 ·")]
    per = la[la.metric.str.startswith("상황 F1 ·") & (la.n >= 3)].assign(code=lambda d: d.metric.str.replace("상황 F1 · ", "").map(lambda c: SIT_NAME.get(c, c)))
    S.append(("x14", "V2. LLM 라벨 정확도", "extra", f"""<h2>V2. 본 라벨(Haiku) 정확도 — 코드북 v2, 본 코퍼스 200건</h2>
<div class="ins"><b>유효/제외 판정 0.94, 층 판정 κ 0.83, 상황 코드 F1 0.78 — 아이 동승은 놓치는 쪽(재현율 0.54)</b>본 라벨은 상황 코드를 붙일 때 정밀도(0.83)가 재현율(0.74)보다 높아 '덜 붙이는' 쪽입니다. 특히 아이 동승·교체 고민·전기차 코드를 자주 놓쳐, 아이 동승 부모와 구매 결정자 세그먼트는 실제보다 작게 잡혔을 수 있습니다. 돈 계산 코드는 F1 0.36으로 가장 불안정합니다.</div>
{tbl(top, [("metric", "지표"), ("n", "n"), ("value", "값"), ("precision", "정밀도"), ("recall", "재현율")], {"precision": lambda v: "" if pd.isna(v) else f"{v:.2f}", "recall": lambda v: "" if pd.isna(v) else f"{v:.2f}"})}
<h4>상황 코드별 (정답 3건 이상)</h4>{tbl(per, [("code", "상황"), ("n", "정답 수"), ("value", "F1"), ("precision", "정밀도"), ("recall", "재현율")])}{method_link('m21')}"""))

if (X / "e10_brand_summary.csv").exists():
    bs, ba = csv("e10_brand_summary", X), csv("e10_brand_attitude", X)
    bm = json.loads((X / "e10_meta.json").read_text(encoding="utf-8"))
    bp = ba.pivot(index="attitude", columns="brand", values="share")
    cols = [c for c in ["현대", "기아", "기아(스토어 리뷰 제외)", "제네시스", "테슬라", "기타 수입"] if c in bp.columns]
    bp = bp[cols]
    vmax = bp.max().max()
    head = "".join(f"<th>{e(c)}</th>" for c in cols)
    body = "".join(f"<tr><th>{T_NAME.get(a, a)}</th>" + "".join(f'<td style="{heat(v, vmax)}">{v:.0%}</td>' for v in r) + "</tr>" for a, r in bp.iterrows())
    S.append(("x15", "E10. 브랜드별 FoD 반응", "extra", f"""<h2>E10. 브랜드별 FoD 반응 (FoD 직접 글 {bm['n_afod_with_brand']:,}건)</h2>
<div class="ins"><b>기아가 유독 만족해 보이는 건 스토어 구매 후기 때문 — 빼면 현대·기아·테슬라의 부정 비율은 25~31%로 비슷</b>브랜드마다 막히는 지점은 다릅니다: 현대·제네시스는 '관리'(블루링크 유료 전환·연장), 테슬라는 '구매' 단계의 망설임(23%)·본전 계산(17%), BMW 등 기타 수입은 HW잠금 반감 32%(열선 구독 논란)로 부정이 55%입니다. 브랜드와 태도는 독립이 아닙니다(χ²={bm['chi2']:,}, Cramér's V {bm['cramers_v']}).</div>
{tbl(bs, [("brand", "브랜드"), ("n", "글"), ("neg_share", "부정"), ("accept_share", "지불 수용"), ("reject_share", "지불 거부"), ("top_journey", "많이 나오는 저니 단계")], {"neg_share": pct, "accept_share": pct, "reject_share": pct})}
<h4>브랜드 × 태도</h4><table class="t heat"><thead><tr><th>태도</th>{head}</tr></thead><tbody>{body}</tbody></table>{method_link('m22')}"""))

if (X / "r_meta.json").exists():
    rl, rs = csv("r_label_correction", X), csv("r_source_refit", X)
    rm = json.loads((X / "r_meta.json").read_text(encoding="utf-8"))
    au = rm["author"]
    rsp = rs.pivot(index="segment", columns="subset", values="cosine").reset_index()
    rsp["name"] = rsp.segment.map(seg_name)
    S.append(("x16", "R. 견고성 점검", "extra", f"""<h2>R. 결과는 얼마나 단단한가 — 라벨 오차·출처·작성자</h2>
<div class="ins"><b>6개 중 4개 세그먼트는 출처를 바꿔도 다시 나온다 — 사제 보충파·공간 활용족은 카페 커뮤니티에서만 보이는 페르소나</b>블로그·리뷰만으로 다시 묶으면 커넥티드 관리자·아이 동승 부모·구매 결정자·문콕 방어자는 다시 나오지만(코사인 0.73~0.91), 사제 보충파(0.15)와 공간 활용족(0.53)은 사라집니다. 같은 블로거의 두 글이 같은 세그먼트일 확률은 {au['same_segment']:.0%}로 무작위 {au['random_baseline']:.0%}보다 높아, 세그먼트가 글의 주제만이 아니라 사람의 성향도 반영합니다. 라벨 오차를 보정하면 아이 동승·교체 고민·전기차 상황은 실제보다 1.6~2.0배 적게 잡혔을 수 있습니다.</div>
<h3 class="sub">출처별 재군집 — 본 세그먼트와의 프로필 유사도(코사인)</h3>
{tbl(rsp, [("name", "세그먼트"), ("카페만", "카페만"), ("카페 외(블로그·리뷰)", "카페 외")], {"카페만": lambda v: f"{v:.2f}", "카페 외(블로그·리뷰)": lambda v: f"{v:.2f}"})}
<p class="muted">공통 글 배정 ARI: 카페만 {rm['source_ari']['카페만']}, 카페 외 {rm['source_ari']['카페 외(블로그·리뷰)']}.</p>
<h3 class="sub">라벨 오차 보정 유병률 (정답 5건 이상 코드)</h3>
{tbl(rl, [("name", "상황"), ("gold_n", "정답 수"), ("observed", "관측"), ("factor", "보정 배수"), ("corrected", "보정값"), ("corrected_lo", "95% 하한"), ("corrected_hi", "상한")], {"observed": pct, "corrected": pct, "corrected_lo": pct, "corrected_hi": pct})}
<p class="muted">작성자 일관성: 블로그 작성자 {au['authors']}명, 글 짝 {au['pairs']}개, 같은 세그먼트 {au['same_segment']:.0%} (95% CI {au['ci'][0]:.0%}~{au['ci'][1]:.0%}).</p>{method_link('m23')}"""))

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
    ["창 안의 다른 금액(차값·옵션 묶음가)이 섞일 수 있음 — '기간 미상'은 해석 주의", "반응(비싸다/적정) 판정은 사람 검증 정밀도 0.29(99건) — 금액이 다른 기능·차값을 가리키거나 질문형 문장이 많아 보고서에서 제외",
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
    ["부트스트랩 분류 안정성은 E4 표 참고(0.7 이상만 해석)", "규칙 정밀도(사람 검증 100건): 없어서 불만 v1 0.45 → v2 0.57(같은 표본 재측정), 있어서 기쁨 0.76"],
    ["정식 Kano는 기능 있음/없음 짝 질문이 필요 — 이 분류는 가설이며 설문으로 확정", "기준이 기능 간 상대값(중앙값)이라 '절대적으로 당연'이라는 뜻이 아님",
     "불만·기쁨 비율이 1~10%로 작아 표현 사전 범위에 민감"]))

S.append(method("m13", "M13. 앱 리뷰 IPA (E5)",
    ["마이현대·기아·MY GENESIS 앱 리뷰(Google Play·App Store) 별점 1~5를 종속변수로, 측면 10개 언급(키워드 사전) 더미를 독립변수로 OLS",
     "중요도 = −계수(언급 시 별점 하락폭), 성과 = 언급 리뷰 중 4~5점 비율, 평균선으로 4분면 (Cheng·Shen·Bi 2022 Kano-IPA의 단순화)"],
    ["rating, text(측면 사전 run/07_2_extra.py ASPECTS)"],
    [f"R² = {meta2['e5']['r2'] if (X / 'e5_e9_meta.json').exists() else 'n/a'} — 측면 언급이 별점 변동의 일부만 설명(나머지는 측면 사전 밖 내용)", "계수 표준오차 0.05~0.12, 상위 3개 측면은 |t| > 8"],
    ["앱 리뷰는 불만 쏠림(1점 50%)", "측면 언급 = 키워드 일치(문맥 미고려)", "속성별 감성이 아닌 언급 여부만 사용"]))

S.append(method("m14", "M14. 전환의 4가지 힘 (E6)",
    ["Moesta·Spiek의 Forces of Progress(Push + Pull > Anxiety + Habit)를 기존 라벨에 매핑",
     "Push = 상황층 글 중 부정·강도 2 이상 / Pull = 요구·만족·SW진보 수용 태도 / Anxiety = 망설임·불신·귀속 우려·본전 계산·HW잠금·이중결제 반감 / Habit = 대체 행동(사제·직접·다른 앱)",
     "기능 사전(13개) 언급 글 50건 이상인 기능만"],
    ["layer, sentiment, intensity, attitude, substitute, text"],
    ["힘별 비율은 한 글이 여러 힘에 동시에 들어갈 수 있음(합 ≠ 1)"],
    ["매핑은 연구자 정의 — 원 이론은 전환 인터뷰 기반", "Push는 상황층 글에서, Pull·Anxiety는 주로 A층 태도에서 나와 층 구성 차이의 영향을 받음"]))

S.append(method("m15", "M15. FSD 구독 전환 중단 시계열 (E7)",
    ["사건: 2026-08-10 테슬라 국내 FSD 일시불 → 월 구독 전환. 전 26주·후 9주, 주별 집계(글 3건 이상 주만)",
     "분절 회귀 y = β0 + β1·주 + β2·사후 + β3·주×사후, β2 = 수준 변화, β3 = 기울기 변화",
     "비교 계열: 같은 기간 날짜 있는 다른 FoD 기능 글 → 수준 변화 차이(DiD 근사)"],
    ["created_at(블로그·리뷰), sentiment, attitude"],
    ["모든 수준 변화가 표준오차 안(유의하지 않음)"],
    ["수집이 최신순 검색이라 최근 글이 과대 — 글 수가 아닌 비율만 사용", "자기상관 미보정 OLS 표준오차", "카페 글은 날짜가 없어 제외",
     "비교 계열이 주행 보조가 아닌 다른 기능이라 평행 추세 가정이 약함"]))

S.append(method("m16", "M16. 비지도 토픽 ↔ 코드북 (E8)",
    ["Kiwi 명사(2자 이상) → TF-IDF(min_df 20, max_df 0.3) → NMF 25토픽(nndsvda, seed 고정)",
     "글별 최대 가중 토픽과 첫 상황 코드의 NMI, 토픽별 지배 코드 순도·코드 없음 비율로 코드북 공백 탐지"],
    ["text, situation, layer"],
    [f"NMI = {meta2['e8_nmi'] if (X / 'e5_e9_meta.json').exists() else 'n/a'} (코드 23개·토픽 25개, 다중 라벨을 첫 코드로 단순화해 하한에 가까움)"],
    ["BERTopic(문장 임베딩) 대비 의미 포착이 약함 — 한국어 SBERT 설치 시 교체 가능", "토픽 수 25는 고정(민감도 미검토)"]))

S.append(method("m17", "M17. 대응 분석 (E9)",
    ["사람 단서(생애단계·성별·브랜드·EV, 데모 트랙) × 상황 코드 상위 18개 교차표, 행 합 200 이상",
     "표준화 잔차 행렬 SVD → 대칭 지도(행·열 주좌표)"],
    ["post_demo(value), situation"],
    [f"관성 설명력 1축 {meta2['e9_inertia'][0]:.0%} · 2축 {meta2['e9_inertia'][1]:.0%} · 3축 {meta2['e9_inertia'][2]:.0%}" if (X / 'e5_e9_meta.json').exists() else ""],
    ["단서가 있는 글만(데모 트랙 커버리지 한계)", "한 글이 여러 단서·코드를 가지면 중복 계산", "대칭 지도에서 행-열 거리는 직접 해석하지 않고 방향만 해석"]))

S.append(method("m18", "M18. FoD 후보 점수표·처방 (S1·S2)",
    ["기준 6개(수요 = log 언급 수, 불편·매력·불안·기존 대안 = E6, 유료화 반감 = E4)의 기능 간 순위를 평균(Borda)",
     "판정 규칙(결과 보기 전 고정): 유료화 반감 ≥10% 또는 Kano '당연'(안정성 ≥0.7) → 기본 탑재 권장 / 나머지 중 순위 평균 ≤ 중앙값 → FoD 판매 후보 / 그 외 보류",
     "민감도: 기준 가중치를 디리클레(1) 분포로 2,000회 뽑아 판매 대상 기능 안에서 순위 재계산 → 상위 절반 확률, 순위 5~95% 구간",
     "처방: 세그먼트 안 기능 언급 lift > 1.2(30건 이상) / 4가지 힘 중 '불안' vs '기존 대안' 큰 쪽이 막는 힘(둘 다 3% 미만이면 '약함') / 주요 불안 태도 → 판매 조건 대응표(망설임 → 체험·짧은 기간권, 본전 계산 → 기간별 가격 비교, 불신 → 설치 후 환불, 귀속 우려 → 계정 이전, HW잠금·이중결제 반감 → 기본 탑재)"],
    ["E1 가격 앵커, E4 Kano·유료화 반감, E6 4가지 힘, E3 대체 유형, post_segment"],
    ["가중치 민감도(상위 절반 확률)를 표에 함께 표시"],
    ["가격 수용도는 규칙 정밀도가 낮아 점수에 넣지 않고 참고로만 표시", "기준이 모두 텍스트 언급 기반 — 설문 Kano·가격 문항으로 확정 필요",
     "Kano 분류는 규칙 수정에 민감(E4) — 판정에는 안정성 0.7 이상만 사용"]))

S.append(method("m19", "M19. 키워드 규칙 검증 (V)",
    ["표본(seed 고정): E4 없어서 불만·있어서 기쁨 판정 글 각 100, E1 가격 반응 판정 99, 기아 스토어 무작위 리뷰 100(문제 유형 5개 다중 라벨)",
     "정답: Claude Code 세션(Opus 5.5)이 문맥을 읽고 판정 — 팀 검수 전",
     "규칙 수정 뒤 같은 표본으로 재측정(V1 정밀도, V4 정밀도·재현율), Wilson 95% CI"],
    ["data/gold/rule_val_v1~v4(.jsonl, _gold.json)"],
    ["원래 규칙: 없어서 불만 0.45, 있어서 기쁨 0.76, 가격 반응 0.29, 스토어 가격 재현율 0.21"],
    ["수정 후 수치는 같은 표본 재측정이라 낙관 편향 — 새 표본으로 재검증 필요", "정답자가 한 명(LLM)이라 일치도(κ) 없음", "희귀 유형(호환, 설치 실패)은 표본이 작아 CI가 넓음"]))

S.append(method("m20", "M20. 설문 판별 문항 선택 (S3)",
    ["입력: LCA 적합 표본(상황 2개 이상, 13,594건)의 상황 상위 22개 이진 지표와 배정 세그먼트",
     "앞으로 선택: 다항 로지스틱(class_weight=balanced) 5겹 층화 CV의 균형 정확도를 가장 많이 올리는 코드를 하나씩 추가(최대 12개)",
     "각 코드를 현행 설문 문항(B2-1, B2-3, C-1a 등) 또는 신규 문항 제안에 대응"],
    ["situation 22개 0/1, segment"],
    ["기준선: 우연 1/6, 22개 전부 사용", "선택 12개의 세그먼트별 재현율"],
    ["순환성: 세그먼트가 같은 코드로 만들어져 정확도가 높게 나옴 — 문항 '후보'를 고르는 용도", "글의 '언급' 지표와 설문의 '빈도' 문항은 다른 측정 — 응답자 자료로 재학습 필요(PLAN 8.1)"]))

S.append(method("m21", "M21. 본 라벨 정확도 재측정 (V2)",
    ["본 코퍼스에서 층화 표본 200건(카페 110·블로그 50·앱 리뷰 20·스토어 20, seed 고정)",
     "Claude Code 세션(Opus 5.5)이 본 라벨을 보지 않고 코드북 v2로 layer·상황 정답 작성(블라인드)",
     "layer: 정확도·Cohen κ·유효/제외 정확도 / 상황: 둘 다 유효로 본 글에서 코드별·마이크로·매크로 F1"],
    ["post_labels(layer, situation), data/gold/gold_v2_claude.json"],
    ["정답자 1명(LLM) — 사람 검수·이중 코딩 κ 없음"],
    ["정답 5건 미만 코드는 F1이 불안정", "카페 스니펫이 짧아 정답 자체도 애매한 글이 있음"]))

S.append(method("m22", "M22. 브랜드별 비교 (E10)",
    ["브랜드 = 데모 트랙 brand 단서(본문·메타·채널), 수입 중 '테슬라' 단서는 테슬라로 분리, 여러 브랜드 글은 제외",
     "FoD 직접(A층) 글에서 태도 10개 비율·부정 비율·지불 신호·저니 단계, Wilson 95% CI",
     "브랜드 × 태도 카이제곱 검정, Cramér's V / 민감도: 기아 스토어 리뷰 제외"],
    ["brand(post_demo), attitude, sentiment, wtp_signal, journey, source"],
    ["기아 스토어 리뷰 제외 시 기아 부정 비율 12.5% → 27.6%로, 원천 구성이 브랜드 차이를 크게 만든다"],
    ["브랜드 단서가 있는 글만", "브랜드별 원천(카페·스토어·앱) 구성이 달라 순수 브랜드 효과가 아님"]))

S.append(method("m23", "M23. 견고성 점검 (R)",
    ["라벨 오차 보정: 코드별 관측 유병률 × (정밀도 / 재현율), 정답 v2 200건 부트스트랩 1,000회로 95% 구간",
     "출처별 재군집: 카페만 / 카페 외 표본으로 LCA(k=6, n_init 5) 재적합 → 본 세그먼트 코드 확률 프로필과 헝가리안 매칭 코사인, 매칭 후 공통 글 ARI",
     "작성자 일관성: 블로그 작성자(배정 글 2개 이상)의 모든 글 짝이 같은 세그먼트인 비율 vs 세그먼트 비율 제곱합(무작위 기준)"],
    ["situation, segment, source, author_hash(블로그만), 정답 v2"],
    ["카페만 ARI 0.81 / 카페 외 ARI 0.60"],
    ["정답이 작아(코드당 5~19건) 보정 구간이 넓음 — 방향만 해석", "카페 외 표본은 작고 리뷰 위주라 레저·사제 상황이 적음", "작성자 식별은 블로그만 가능(전체의 4.6%)"]))

# ---------------- 한눈에 보기 (첫 화면) ----------------
def overview():
    sc = csv("s1_scorecard", X) if (X / "s1_scorecard.csv").exists() else pd.DataFrame()
    pr = csv("s2_prescriptions", X) if (X / "s2_prescriptions.csv").exists() else pd.DataFrame()
    ipa_ = csv("e5_app_ipa", X) if (X / "e5_app_ipa.csv").exists() else pd.DataFrame()
    facts = [(f"{int(funnel.iloc[:, 0].sum()):,}", "건 수집"), (f"{len(corpus):,}", "건 분석"), (f"{n_ab:,}", "건 유효"), (f"{info['k']}", "개 페르소나")]
    fh_ = "".join(f"<span><b>{v}</b>{k}</span>" for v, k in facts)
    dec = lambda d: " · ".join(sc[sc.decision == d].sort_values("mean_rank").feature) if not sc.empty else ""
    biggest = summ.sort_values("n", ascending=False).iloc[0]
    top_asp = ipa_.nlargest(3, "importance") if not ipa_.empty else pd.DataFrame()
    asp = " · ".join(f"{r.aspect} −{r.importance:.2f}점" for r in top_asp.itertuples())
    rest = " · ".join(sc[sc.decision == "FoD 판매 후보"].sort_values("mean_rank").feature[2:]) if not sc.empty else ""
    finds = [
        ("FoD로 먼저 팔 기능은 회생 제동과 화면 테마", f"기준 가중치를 2,000번 바꿔도 상위를 지킵니다. 다음 후보는 {rest}. 디지털 키·열선은 기본 탑재, {dec('보류')}는 보류입니다.", "x10"),
        (f"가장 큰 그룹은 {seg_name(biggest.segment)}({biggest.share:.0%})", f"앱 별점을 깎는 것은 기능이 아니라 기반입니다 — {asp}.", "x5"),
        ("구매를 막는 건 불안보다 '이미 쓰는 대안'", "사제 보충파는 사제 용품 91%, 문콕 방어자 61%. 불안이 막는 그룹은 구매 결정자뿐입니다.", "x11"),
        ("가격 기준점은 커넥티드 월 5,500~9,900원", "원격 주차 평생 이용권 50만 원, 테슬라 FSD 월 15만 원이 반복해서 나옵니다.", "x1"),
        ("비중은 온라인 글의 비중이지 시장 비중이 아님", "카페 글 위주이고, 사제 보충파·공간 활용족은 카페에서만 재현됩니다. LLM 상황 라벨 F1 0.78 — 아이 동승은 적게 잡혔을 수 있습니다. 설문으로 확정합니다.", "x16"),
    ]
    rows_ = "".join(f'<a class="row{" lead" if k == 0 else ""}" href="#{g}" data-go="{g}"><b>{e(h)}</b><p>{e(b_)}</p><span class="go">근거 보기 →</span></a>' for k, (h, b_, g) in enumerate(finds))
    pmap = {int(r.segment): r for r in pr.itertuples()} if not pr.empty else {}
    ph = ""
    for sg in segs:
        r = summ[summ.segment == sg].iloc[0]
        rx = pmap.get(int(sg))
        ph += f"""<a class="prow" href="#s6" data-go="s6" data-card="p{sg}"><div><b>{e(seg_name(sg))}</b><small>{e(seg_tag(sg))}</small></div>
<div class="bar"><span style="width:{r.share * 100 / summ.share.max():.0f}%;background:var(--accent)"></span><em>{r.share:.1%}</em></div>
<div><p>{e(SEG_NAR.get(int(sg), {}).get('one_liner', ''))}</p><p class="rx">{e(rx.condition) if rx is not None else ''}</p></div></a>"""
    return ("s0", "한눈에 보기", "summary", f"""<div class="lede"><h2>{e(NAR.get('title', '현대차 FoD 데이터 기반 페르소나'))}</h2>
<p class="stand">온라인 리뷰와 커뮤니티 글로 FoD(출고 후 기능 구매)를 둘러싼 니즈와 순간을 찾고, 6개 페르소나와 기능별 판매 판단을 만들었습니다. 텍스트 기반 1차 결과이며 설문으로 확정합니다.</p>
<div class="facts">{fh_}</div></div>
<h3 class="sec">핵심 결론</h3><div class="rows">{rows_}</div>
<h3 class="sec">6개 페르소나와 처방</h3><div class="rows">{ph}</div>
<p class="muted">행을 누르면 근거 화면으로 이동합니다. 각 분석 화면 아래에 방법과 신뢰도가 있습니다.</p>""")


S.insert(0, overview())

# ---------------- 페이지 조립 ----------------
# 메뉴는 '읽는 사람의 질문'으로, 방법·신뢰도는 각 분석 화면 아래에 접어서 넣는다
NAV_TITLE = {
    "s0": "한눈에 보기", "s3": "핵심 인사이트", "s5": "페르소나 한눈에", "x10": "어떤 기능을 FoD로 팔까", "x11": "페르소나별 판매 전략",
    "s6": "페르소나 상세 카드", "s7": "FoD 아이디어", "s8": "니즈별 기회 크기", "s9": "구매 단계별 장벽", "s4": "니즈는 어떻게 묶이나", "s2": "니즈 코드 체계",
    "x1": "사람들이 말하는 가격", "x2": "스토어 상품은 잘 팔렸나", "x3": "FoD 대신 무엇을 쓰나", "x6": "사게 하는 힘 vs 막는 힘", "x4": "기능별 기대 유형 (Kano)",
    "x5": "앱 별점을 깎는 것", "x7": "FSD 구독 전환 반응", "x9": "누가 어떤 상황을 말하나", "x8": "코드북이 놓친 주제", "x13": "설문에 넣을 판별 문항", "x14": "LLM 라벨 정확도", "x15": "브랜드별 FoD 반응", "x16": "결과는 얼마나 단단한가",
    "s1": "데이터를 어떻게 모았나", "x12": "규칙 판정 검증", "m8": "한계와 주장 경계", "m4": "인구 단서 추출",
}
GROUPS = [("summary", "결론", ["s0", "s3", "s5", "x10", "x11", "x13"]),
          ("persona", "페르소나", ["s6", "s7", "s8", "s9", "s4", "s2"]),
          ("extra", "심화 분석", ["x1", "x2", "x3", "x6", "x4", "x5", "x15", "x7", "x9", "x8"]),
          ("appendix", "부록", ["s1", "x16", "x14", "x12", "m8", "m4"])]
html_of = {i: h for i, _, _, h in S}
order = [i for _, _, ids in GROUPS for i in ids if i in html_of]
grp_of = {i: k for k, _, ids in GROUPS for i in ids}
title_of = {i: NAV_TITLE.get(i, t) for i, t, _, _ in S}
nav = "".join(f'<details open><summary>{lab}</summary>' + "".join(f'<a href="#{i}" data-go="{i}">{e(title_of[i])}</a>' for i in ids if i in html_of) + "</details>" for _, lab, ids in GROUPS)
TBL = re.compile(r'(<table class="t[ "][^>]*>.*?</table>)', re.S)
MLINK = re.compile(r'<a class="mlink" href="#(m\d+)" data-go="m\d+">[^<]*</a>')
H2 = re.compile(r"<h2>(?:[A-Z]?\d+\.|V\.|[A-Z]\d+\.)\s*")


def method_box(m):
    body = re.sub(r"<h2>.*?</h2>", "", html_of.get(m, ""), count=1, flags=re.S)
    body = re.sub(r"<details open><summary>(.*?)</summary>(.*?)</details>", lambda m: f'<div class="msec"><h3 class="sub">{m.group(1)}</h3>{m.group(2)}</div>', body, flags=re.S)
    return f'<details class="method"><summary>이 분석은 어떻게 했나 · 신뢰도</summary><div class="mbody">{body}</div></details>'


def panel(i):
    k = order.index(i)
    prv, nxt = (order[k - 1] if k else None), (order[k + 1] if k + 1 < len(order) else None)
    pager = (f'<a href="#{prv}" data-go="{prv}"><small>이전</small>{e(title_of[prv])}</a>' if prv else "<span></span>") + \
            (f'<a class="nx" href="#{nxt}" data-go="{nxt}"><small>다음</small>{e(title_of[nxt])}</a>' if nxt else "")
    crumb = next(lab for k_, lab, _ in GROUPS if k_ == grp_of[i])
    body = TBL.sub(r'<div class="tw">\1</div>', html_of[i])
    body = MLINK.sub(lambda m: method_box(m.group(1)), body)
    body = H2.sub("<h2>", body)
    body = body.replace("<h4", '<h3 class="sub"').replace("</h4>", "</h3>")
    return f'<section class="panel" id="{i}" aria-label="{e(crumb)} · {e(title_of[i])}">{body}<nav class="pager">{pager}</nav></section>'


panels = "".join(panel(i) for i in order)
title = NAR.get("title", "현대차 FoD 데이터 기반 페르소나")
page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FoD 페르소나 보고서</title><link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"><style>:root{{--bg:#f6f5f2;--fg:#191a1d;--muted:#5f6268;--faint:#8b8e94;--card:#ffffff;--line:#e2dfd8;--accent:#b8361f;--accent-ink:#9c2c18;--band:#1d2a47;--soft:#f6e9e5;--zebra:#faf9f6;--heat:184,54,31;--rail:#1d2a47}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#131416;--fg:#ebe9e5;--muted:#a9abb0;--faint:#80838a;--card:#1b1c1f;--line:#2e3034;--accent:#ff7a5c;--accent-ink:#ff9a80;--band:#9fb3e0;--soft:#2b1f1c;--zebra:#18191b;--heat:255,122,92;--rail:#161d2e}}}}
:root[data-theme="dark"]{{--bg:#131416;--fg:#ebe9e5;--muted:#a9abb0;--faint:#80838a;--card:#1b1c1f;--line:#2e3034;--accent:#ff7a5c;--accent-ink:#ff9a80;--band:#9fb3e0;--soft:#2b1f1c;--zebra:#18191b;--heat:255,122,92;--rail:#161d2e}}
*{{box-sizing:border-box}}
html,nav.side,.tw,main{{scrollbar-width:none}}html::-webkit-scrollbar,nav.side::-webkit-scrollbar,.tw::-webkit-scrollbar{{display:none}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.72 "Pretendard Variable","Pretendard",-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;word-break:keep-all;-webkit-font-smoothing:antialiased}}
::selection{{background:var(--accent);color:#fff}}
a{{color:var(--accent-ink);text-underline-offset:3px}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}}

/* 틀: 왼쪽 목차 레일 + 읽기 기둥 */
.wrap{{display:grid;grid-template-columns:272px minmax(0,1fr);min-height:100vh}}
nav.side{{position:sticky;top:0;height:100vh;overflow:auto;padding:28px 18px 40px;background:var(--rail);color:#e8ecf5}}
nav.side h1{{font-size:15px;line-height:1.45;margin:0 0 6px;color:#fff;letter-spacing:-.01em}}
nav.side p{{font-size:12.5px;line-height:1.5;color:#b9c2d6;margin:0 0 18px}}
nav.side details{{margin:0;border-top:1px solid #ffffff1f}}
nav.side summary{{font-size:12px;font-weight:700;color:#b9c2d6;letter-spacing:.04em;padding:14px 10px 6px;list-style:none;cursor:pointer;display:flex;align-items:center;gap:8px}}
nav.side summary::-webkit-details-marker{{display:none}}
nav.side summary::after{{content:"";width:6px;height:6px;margin-left:auto;border-right:1.5px solid currentColor;border-bottom:1.5px solid currentColor;transform:rotate(45deg) translateY(-2px);transition:transform .2s}}
nav.side details:not([open]) summary::after{{transform:rotate(-45deg)}}
nav.side a{{display:block;padding:6px 10px;margin:1px 0 1px;border-radius:8px;color:#d7dded;text-decoration:none;font-size:14.5px;line-height:1.45}}
nav.side a:hover{{background:#ffffff14;color:#fff}}
nav.side a.on{{background:#fff;color:#1d2a47;font-weight:650}}
nav.side details:last-child{{padding-bottom:8px}}

main{{padding:56px 56px 80px;max-width:1080px;width:100%;margin:0 auto}}
.panel{{display:none}}.panel.on{{display:block}}

/* 타이포 */
h2{{font-size:30px;line-height:1.3;margin:0 0 18px;letter-spacing:-.025em;text-wrap:balance}}
h3{{font-size:21px;line-height:1.4;margin:0;letter-spacing:-.015em}}
h3.sub{{font-size:16px;margin:34px 0 10px;letter-spacing:-.01em}}
h3.sec{{font-size:20px;margin:56px 0 14px}}
p{{margin:10px 0;max-width:70ch}}
small,.muted{{color:var(--muted)}}.muted{{font-size:14.5px}}

/* 결론 블록: 굵은 결론 + 설명. 테두리 강조 대신 바탕 톤 */
.ins{{background:var(--soft);border-radius:14px;padding:20px 24px;margin:18px 0 28px;color:var(--muted);font-size:15.5px;max-width:78ch}}
.ins b{{display:block;font-size:19px;line-height:1.5;color:var(--fg);margin-bottom:8px;letter-spacing:-.01em;text-wrap:balance}}
.ins small{{display:block;font-size:14.5px}}

/* 표 */
.tw{{overflow-x:auto;margin:16px 0 8px;border:1px solid var(--line);border-radius:14px;background:var(--card)}}
.t{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}
.t th,.t td{{border-bottom:1px solid var(--line);padding:11px 14px;text-align:left;vertical-align:top}}
.t thead th{{color:var(--muted);font-weight:600;font-size:13px;white-space:nowrap;background:var(--zebra)}}
.t tbody tr:last-child>*{{border-bottom:0}}
.t tbody th{{font-weight:600}}
.heat td{{text-align:center}}
.tag{{display:inline-block;white-space:nowrap;background:var(--soft);color:var(--accent-ink);border-radius:999px;padding:3px 10px;margin:2px 3px 2px 0;font-size:12.5px;line-height:1.4;text-decoration:none}}

/* 1장 흐름 */
.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:0;margin:20px 0;border:1px solid var(--line);border-radius:14px;background:var(--card);overflow:hidden}}
.flow div{{padding:16px 18px;border-right:1px solid var(--line)}}.flow div:last-child{{border-right:0}}
.flow b,.flow span,.flow strong{{display:block}}.flow span{{font-size:13px;color:var(--muted);margin:6px 0 10px;line-height:1.5}}.flow strong{{font-size:22px;color:var(--fg);font-variant-numeric:tabular-nums;letter-spacing:-.01em}}
.tracks{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.tracks div{{background:var(--card);border:1px solid var(--line);padding:16px 18px;border-radius:14px}}

/* 그림 */
.net{{display:block;width:100%;max-width:880px;margin:18px 0;padding:8px;background:var(--card);border:1px solid var(--line);border-radius:14px}}.nl{{font-size:12px;fill:var(--fg)}}

/* 페르소나 카드 */
.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:28px;margin:28px 0;scroll-margin-top:24px}}
.card header{{display:flex;justify-content:space-between;gap:24px;align-items:flex-start}}
.card header .tagline{{color:var(--accent-ink);font-weight:600;font-size:14px;margin:6px 0 0}}
.kpi{{border-left:1px solid var(--line);padding:4px 0 4px 20px;min-width:170px}}.kpi small{{display:block}}.kpi b{{display:block;font-size:28px;color:var(--fg);font-variant-numeric:tabular-nums;letter-spacing:-.02em}}
.insight{{background:var(--soft);border-radius:12px;padding:14px 18px;margin:18px 0}}.insight ul{{margin:6px 0;padding-left:18px}}
.grid3{{display:grid;grid-template-columns:1fr 1.2fr 1fr;gap:14px;margin:18px 0}}.grid3 section{{background:var(--zebra);border-radius:12px;padding:16px}}.grid3 h3.sub{{margin-top:0}}
.mini th{{width:70px;text-align:left;font-weight:600;font-size:13px;vertical-align:top;padding:4px 8px 4px 0}}.mini td{{font-size:14px;padding:4px 0}}
.feat p{{margin:0 0 10px;font-size:14px}}
.band{{background:var(--fg);color:var(--bg);padding:10px 16px;border-radius:10px;font-size:15px}}
.needs td{{font-size:14px}}.q{{color:var(--muted);font-size:13.5px}}.q details{{margin:6px 0 0}}.q summary{{font-weight:500;color:var(--accent-ink)}}
.crow{{font-size:13px;margin:8px 0}}
.bar{{position:relative;height:20px;background:var(--line);border-radius:6px;overflow:hidden;min-width:60px}}.bar span{{position:absolute;inset:0 auto 0 0;border-radius:6px}}.bar em{{position:absolute;right:4px;top:3px;padding:0 5px;border-radius:4px;background:var(--card);font-size:11.5px;line-height:14px;font-style:normal;font-variant-numeric:tabular-nums}}
details{{margin:14px 0}}summary{{cursor:pointer;font-weight:600}}
.back{{display:grid;grid-template-columns:1.4fr 1fr;gap:16px;margin-top:10px}}

/* 아이디어·처방 묶음 */
.ideas{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;margin:18px 0}}
.idea{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;font-size:14.5px}}.idea h4,.idea h3.sub{{font-size:17px;margin:6px 0 10px}}.idea p{{margin:8px 0}}

/* 첫 화면: 문서형 */
.lede{{max-width:66ch}}
.lede h2{{font-size:38px;line-height:1.25;margin:0 0 14px;letter-spacing:-.03em}}
.lede .stand{{font-size:18px;line-height:1.7;color:var(--muted);margin:0}}
.facts{{display:flex;flex-wrap:wrap;gap:8px 28px;margin:22px 0 0;padding:16px 0 0;border-top:1px solid var(--line);font-size:14.5px;color:var(--muted)}}
.facts b{{color:var(--fg);font-size:17px;font-variant-numeric:tabular-nums;margin-right:6px}}
.rows{{border-top:1px solid var(--fg);margin-top:8px}}
.row{{display:grid;grid-template-columns:minmax(220px,1fr) 2fr auto;gap:28px;align-items:baseline;padding:20px 0;border-bottom:1px solid var(--line);color:var(--fg);text-decoration:none}}
.row:hover .go{{color:var(--accent)}}
.row b{{font-size:18px;line-height:1.45;letter-spacing:-.01em}}
.row p{{margin:0;color:var(--muted);font-size:15px}}
.row .go{{font-size:13.5px;font-weight:600;color:var(--faint);white-space:nowrap}}
.row.lead b{{font-size:24px;color:var(--accent-ink)}}
.prow{{display:grid;grid-template-columns:minmax(200px,1.1fr) 140px 2fr;gap:24px;align-items:center;padding:16px 0;border-bottom:1px solid var(--line);color:var(--fg);text-decoration:none}}
.prow:hover b{{color:var(--accent-ink)}}
.prow b{{font-size:16.5px;display:block}}.prow small{{display:block;font-size:13px}}
.prow p{{margin:0;font-size:14.5px;color:var(--muted)}}.prow p.rx{{color:var(--fg);margin-top:4px}}

/* 방법 상자 */
.mlink{{display:inline-block;margin-top:14px;font-size:14px;color:var(--accent-ink);font-weight:600;text-decoration:none}}
details.method{{margin:44px 0 0;border:1px solid var(--line);border-radius:14px;background:var(--card)}}
details.method>summary{{padding:16px 20px;font-size:14.5px;color:var(--muted);list-style:none;display:flex;align-items:center}}
details.method>summary::-webkit-details-marker{{display:none}}
details.method>summary::after{{content:"";width:7px;height:7px;margin-left:auto;border-right:1.5px solid var(--accent);border-bottom:1.5px solid var(--accent);transform:rotate(45deg);transition:transform .2s}}
details.method[open]>summary::after{{transform:rotate(-135deg)}}
details.method .mbody{{padding:0 20px 18px;font-size:14.5px;display:grid;grid-template-columns:1fr 1fr;gap:4px 32px}}
details.method .mbody ul{{margin:6px 0;padding-left:18px}}
.msec h3.sub{{margin:16px 0 4px;font-size:13px;color:var(--accent-ink);letter-spacing:.02em}}

/* 이전·다음 */
nav.pager{{display:flex;justify-content:space-between;gap:12px;margin-top:56px;padding-top:20px;border-top:1px solid var(--line)}}
nav.pager a{{display:flex;flex-direction:column;gap:2px;max-width:48%;padding:12px 16px;border:1px solid var(--line);border-radius:12px;background:var(--card);color:var(--fg);font-weight:600;text-decoration:none;font-size:14.5px}}
nav.pager a:hover{{border-color:var(--accent)}}nav.pager a small{{font-weight:500}}nav.pager a.nx{{text-align:right;margin-left:auto}}

nav.side .menu{{display:none;align-items:center;justify-content:space-between;gap:12px;width:100%;background:none;border:0;color:#fff;font:inherit;padding:0;cursor:pointer}}
nav.side .menu span{{font-size:13px;color:#b9c2d6;border:1px solid #ffffff40;border-radius:999px;padding:4px 12px}}
@media (max-width:860px){{
 .wrap{{grid-template-columns:minmax(0,1fr)}}
 nav.side{{position:sticky;top:0;z-index:5;height:auto;max-height:none;padding:12px 16px}}
 nav.side h1{{font-size:14px;margin:0}}nav.side p{{display:none}}
 nav.side .groups{{display:none}}nav.side.open .groups{{display:block;max-height:70vh;overflow:auto;margin-top:8px}}
 nav.side .menu{{display:flex}}
 main{{padding:28px 16px 56px}}
 .flow,.grid3,.tracks,.back,details.method .mbody{{grid-template-columns:1fr}}.flow div{{border-right:0;border-bottom:1px solid var(--line)}}
 .card{{padding:20px}}.card header{{flex-direction:column}}.kpi{{border-left:0;padding:0}}
 h2{{font-size:24px}}.lede h2{{font-size:28px}}.lede .stand{{font-size:16px}}
 .row,.prow{{grid-template-columns:1fr;gap:6px}}.row .go{{display:none}}
}}
</style></head><body><div class="wrap"><nav class="side" aria-label="목차"><h1>{e(title)}</h1><p>생성 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 텍스트 기반(설문 확정 전)</p>
<button class="menu" type="button" aria-expanded="false" onclick="const n=this.closest('nav');n.classList.toggle('open');this.setAttribute('aria-expanded',n.classList.contains('open'))"><span>목차</span></button><div class="groups">{nav}</div></nav><main>{panels}</main></div>
<script>
const show=(id,card)=>{{if(!document.getElementById(id))id='s0';document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.id===id));
document.querySelectorAll('nav.side a').forEach(a=>{{const on=a.dataset.go===id;a.classList.toggle('on',on);if(on)a.closest('details').open=true}});
const c=card&&document.getElementById(card);c?c.scrollIntoView():window.scrollTo(0,0)}};
document.addEventListener('click',ev=>{{const a=ev.target.closest('[data-go]');if(a){{ev.preventDefault();document.querySelector('nav.side').classList.remove('open');history.replaceState(null,'','#'+a.dataset.go);show(a.dataset.go,a.dataset.card)}}}});
show(location.hash.slice(1)||'s0');
</script></body></html>"""
OUT.write_text(page, encoding="utf-8")
print("→", OUT.relative_to(ROOT), f"({len(page) / 1e3:.0f} KB)")
