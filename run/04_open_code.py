"""아래→위 오픈 코딩: 결과 문장 → 상황 코드별 #해시태그 니즈 도출 → 포화 확인.

  python run/04_open_code.py statements   # 표본 2,000건 결과 문장 (data/stage/open_statements.jsonl)
  python run/04_open_code.py induce       # 앞 절반으로 상황 코드별 3차 니즈 도출 (config/need_hierarchy_draft.yaml)
  python run/04_open_code.py saturation   # 뒤 절반을 배정, NEW 비율 보고
  python run/04_open_code.py extra        # 얇은 코드 표적 보충(half=C)
  python run/04_open_code.py induce all   # 전체(A+B+C)로 최종 도출
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data/stage"
MODEL, EFFORT = "claude-opus-5-5", "low"
CODEBOOK = yaml.safe_load((ROOT / "config/codebook_v2.yaml").read_text(encoding="utf-8"))
SIT = CODEBOOK["families"]["situation"]["values"]
SIT_CODES = list(SIT) + ["NONE"]
N_SAMPLE, SEED = 2000, 20261010


def client():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
    return anthropic.Anthropic()


def ask(c, system, user, schema, max_tokens=16000):
    """구조화 출력 1회 호출. 거절 시 서버 측 fallback(기본 라우팅) 사용."""
    for attempt in range(5):
        try:
            r = c.beta.messages.create(
                model=MODEL, max_tokens=max_tokens, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": user}],
            )
            if r.stop_reason != "end_turn":
                raise RuntimeError(f"stop_reason={r.stop_reason}")
            return json.loads(next(b.text for b in r.content if b.type == "text")), r.usage
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
            time.sleep(3 * 2 ** attempt)
    raise RuntimeError("retries exhausted")


def render(r):
    return f'<post id="{r.post_id}" source="{r.source}">\n{r.text[:1200]}\n</post>'


def obj(props, req=None):
    return {"type": "object", "additionalProperties": False, "properties": props, "required": req or list(props)}


SYS_STMT = (
    "너는 한국 자동차 사용자 글에서 고객 니즈를 뽑는 연구자다. 각 글마다:\n"
    "1) outcome: 작성자가 바라는 결과를 '방향 + 지표 + 대상/상황' 형식의 한 문장으로 쓴다. "
    "예: '좁은 칸에서 아이를 태우고 내릴 때 문을 다 여는 데 드는 수고를 줄인다'. "
    "기능명·제품명·해법(원격주차, 앱 등)을 쓰지 않는다. 니즈가 없으면 빈 문자열.\n"
    "2) situation: 아래 상황 코드 중 가장 맞는 하나, 없으면 NONE.\n"
    "3) need_label: 니즈를 3~8자 한국어 이름으로(예: '승하차 공간 확보'). 니즈가 없으면 빈 문자열.\n\n# 상황 코드\n"
    + "\n".join(f"{k}: {v}" for k, v in SIT.items())
)
STMT_SCHEMA = obj({"items": {"type": "array", "items": obj({
    "post_id": {"type": "string"}, "outcome": {"type": "string"},
    "situation": {"type": "string", "enum": SIT_CODES}, "need_label": {"type": "string"}})}})


def statements():
    corpus = pd.read_parquet(STAGE / "corpus.parquet")
    corpus["code"] = corpus["seed_codes"].map(lambda x: x[0] if len(x) else corpus.get("source"))
    corpus["stratum"] = corpus["source"] + "|" + corpus["code"].astype(str)
    sample = corpus.groupby("stratum", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(10, round(N_SAMPLE * len(g) / len(corpus)))), random_state=SEED))
    sample = sample.sample(n=min(N_SAMPLE, len(sample)), random_state=SEED)
    c, rows = client(), list(sample.itertuples())
    chunks = [rows[i:i + 20] for i in range(0, len(rows), 20)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(lambda ch: ask(c, SYS_STMT, "\n\n".join(render(r) for r in ch), STMT_SCHEMA), chunks))
    src = {r.post_id: r for r in rows}
    out = STAGE / "open_statements.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, (data, _) in enumerate(res):
            for it in data["items"]:
                if it["post_id"] in src:
                    it.update({"half": "A" if i < len(res) // 2 else "B", "source": src[it["post_id"]].source})
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
    report_cost(res)
    print("→", out.relative_to(ROOT))


THIN = ["S_PET", "S_INCAR_REST", "S_HANDOVER", "S_PARK_REMOTE", "S_ENTERTAIN", "S_FAMILY_SHARE",
        "S_LONG_DRIVE", "S_PARK_TIGHT", "S_PARK_SKILL"]


def extra(per_code=60):
    """얇은 코드 표적 보충: seed_code가 해당 코드인 글에서 코드당 per_code건."""
    done = {json.loads(l)["post_id"] for l in (STAGE / "open_statements.jsonl").open(encoding="utf-8")}
    corpus = pd.read_parquet(STAGE / "corpus.parquet")
    corpus = corpus[~corpus["post_id"].isin(done)]
    picks = [corpus[corpus["seed_codes"].map(lambda x: code in list(x))].sample(frac=1, random_state=SEED).head(per_code)
             for code in THIN]
    rows = list(pd.concat(picks).drop_duplicates("post_id").itertuples())
    c = client()
    chunks = [rows[i:i + 20] for i in range(0, len(rows), 20)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(lambda ch: ask(c, SYS_STMT, "\n\n".join(render(r) for r in ch), STMT_SCHEMA), chunks))
    src = {r.post_id: r for r in rows}
    with (STAGE / "open_statements.jsonl").open("a", encoding="utf-8") as f:
        for data, _ in res:
            for it in data["items"]:
                if it["post_id"] in src:
                    it.update({"half": "C", "source": src[it["post_id"]].source})
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
    report_cost(res)


SYS_INDUCE = (
    "너는 고객 니즈 계층을 설계하는 연구자다. 같은 상황 코드에 속한 결과 문장 목록을 받는다. "
    "서로 겹치지 않는 3차 니즈 2~5개로 묶어라(문장 20건 미만이면 2~3개). 각 니즈는:\n"
    "- id: 상황코드_N (예: S_PARK_TIGHT_1)\n- hashtag: '#'로 시작하는 6~14자 한국어 별칭, 롯데마트식 '#상황+정체성' 느낌 (예: #좁은칸탈출아빠, #문콕공포러)\n"
    "- name: 니즈 이름(명사구)\n- definition: 포함·제외 기준 한 문장\n- member_ids: 해당 문장 번호 목록\n"
    "전체의 3% 미만인 묶음은 만들지 말고 가장 가까운 니즈에 합친다. 어디에도 안 맞는 문장은 member_ids에서 빼라. 해법·기능명 금지."
)
INDUCE_SCHEMA = obj({"needs": {"type": "array", "items": obj({
    "id": {"type": "string"}, "hashtag": {"type": "string"}, "name": {"type": "string"},
    "definition": {"type": "string"}, "member_ids": {"type": "array", "items": {"type": "integer"}}})}})


def induce():
    st = [json.loads(l) for l in (STAGE / "open_statements.jsonl").open(encoding="utf-8")]
    halves = {"A"} if len(sys.argv) < 3 or sys.argv[2] != "all" else {"A", "B", "C"}
    st = [s for s in st if s["half"] in halves and s["outcome"] and s["situation"] != "NONE"]
    by = {}
    for s in st:
        by.setdefault(s["situation"], []).append(s)
    c, hierarchy, usage = client(), {}, []
    for code, items in sorted(by.items()):
        if len(items) < 8:
            hierarchy[code] = {"name": SIT[code], "n_statements": len(items), "needs": [], "note": "표본 부족(8건 미만)"}
            continue
        user = "\n".join(f"{i}. {s['outcome']}" for i, s in enumerate(items))
        data, u = ask(c, SYS_INDUCE, f"상황 코드 {code}: {SIT[code]}\n\n{user}", INDUCE_SCHEMA)
        usage.append((data, u))
        for n in data["needs"]:
            n["n_members"] = len(n["member_ids"])
            n["examples"] = [items[i]["outcome"] for i in n["member_ids"][:3] if i < len(items)]
            del n["member_ids"]
        hierarchy[code] = {"name": SIT[code], "n_statements": len(items), "needs": data["needs"]}
        print(f"{code:15} 문장 {len(items):>4} → 니즈 {len(data['needs'])}")
    out = ROOT / "config/need_hierarchy_draft.yaml"
    out.write_text("# 오픈 코딩 초안 (사람 검토 필요). 1차=situation 대분류, 2차=situation 코드, 3차=아래 needs\n"
                   + yaml.safe_dump(hierarchy, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report_cost(usage)
    print("→", out.relative_to(ROOT))


SYS_ASSIGN = "결과 문장마다 주어진 3차 니즈 목록 중 가장 맞는 id를 고른다. 어떤 니즈에도 맞지 않으면 NEW."


def saturation():
    hier = yaml.safe_load((ROOT / "config/need_hierarchy_draft.yaml").read_text(encoding="utf-8"))
    st = [json.loads(l) for l in (STAGE / "open_statements.jsonl").open(encoding="utf-8")]
    st = [s for s in st if s["half"] == "B" and s["outcome"] and s["situation"] in hier and hier[s["situation"]]["needs"]]
    c, usage, rows = client(), [], []
    by = {}
    for s in st:
        by.setdefault(s["situation"], []).append(s)
    for code, items in sorted(by.items()):
        needs = hier[code]["needs"]
        ids = [n["id"] for n in needs] + ["NEW"]
        schema = obj({"items": {"type": "array", "items": obj({"i": {"type": "integer"}, "need": {"type": "string", "enum": ids}})}})
        menu = "\n".join(f"{n['id']}: {n['name']} — {n['definition']}" for n in needs)
        data, u = ask(c, SYS_ASSIGN, f"# 니즈\n{menu}\n\n# 문장\n" + "\n".join(f"{i}. {s['outcome']}" for i, s in enumerate(items)), schema)
        usage.append((data, u))
        new = sum(x["need"] == "NEW" for x in data["items"])
        rows.append({"code": code, "n": len(items), "new": new, "new_rate": round(new / max(len(items), 1), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(STAGE / "open_saturation.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"전체 NEW 비율 {df['new'].sum() / max(df['n'].sum(), 1):.1%} (낮을수록 포화)")
    report_cost(usage)


def report_cost(res):
    # Opus 5.5: $4 in / $20 out / cache read $0.20 / cache write 1.25x
    t = sum(((u.input_tokens or 0) * 4 + (u.cache_creation_input_tokens or 0) * 5 + (u.cache_read_input_tokens or 0) * 0.2
             + (u.output_tokens or 0) * 20) / 1e6 for _, u in res)
    print(f"비용 약 ${t:.2f}")


if __name__ == "__main__":
    {"statements": statements, "extra": extra, "induce": induce, "saturation": saturation}[sys.argv[1]]()
