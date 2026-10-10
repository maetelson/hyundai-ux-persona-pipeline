"""본 라벨링: 배치 API(Haiku 5.5 medium, 10건 묶음) → post_labels. 근거 인용 검증.

  python run/05_label.py submit pilot 500     # 표본 500건 배치 제출
  python run/05_label.py submit all           # 코퍼스 전체(이미 라벨된 글 제외)
  python run/05_label.py status               # 진행 상태
  python run/05_label.py collect              # 끝난 배치 결과 → data/stage/post_labels.parquet (누적)
상태 파일: data/stage/label_batches.json
"""
import json
import os
import re
import sys
from pathlib import Path

import anthropic
import pandas as pd
import yaml
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data/stage"
STATE = STAGE / "label_batches.json"
OUT = STAGE / "post_labels.parquet"
MODEL, EFFORT, PACK = "claude-haiku-5-5", "medium", 10
PRICE = (0.10, 0.50, 0.01)  # $/MTok in, out, cache read (배치 50% 별도)
CODEBOOK = yaml.safe_load((ROOT / "config/codebook_v2.yaml").read_text(encoding="utf-8"))
NEEDS = yaml.safe_load((ROOT / "config/need_hierarchy_v1.yaml").read_text(encoding="utf-8"))
FAM = CODEBOOK["families"]
CODES = {f: list(FAM[f]["values"]) for f in ("layer", "situation", "journey", "attitude", "life_stage", "gender")}
NEED_IDS = [n["id"] for v in NEEDS.values() for n in v["needs"]]
EVID = CODES["situation"] + CODES["journey"] + CODES["attitude"] + CODES["life_stage"] + CODES["gender"]
FACETS = {
    "intensity": ["0", "1", "2", "3"],  # 0 없음, 1 언급, 2 불편, 3 강한 고통·위험
    "frequency": ["daily", "periodic", "occasional", "once", "unknown"],
    "wtp_signal": ["amount_mentioned", "accept", "reject", "none"],
    "substitute": ["aftermarket", "other_app", "diy", "give_up", "none"],
    "sentiment": ["-1", "0", "1"],
    "actor": ["self", "spouse", "child", "parent", "other", "unknown"],
}


def arr(enum):
    return {"type": "array", "items": {"type": "string", "enum": enum}}


def obj(props):
    return {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}


SCHEMA = obj({"items": {"type": "array", "items": obj({
    "id": {"type": "string"},
    "layer": {"type": "string", "enum": CODES["layer"]},
    "situation": arr(CODES["situation"]), "needs": arr(NEED_IDS),
    "journey": arr(CODES["journey"]), "attitude": arr(CODES["attitude"]),
    "life_stage": {"type": "string", "enum": CODES["life_stage"]},
    "gender": {"type": "string", "enum": CODES["gender"]},
    "outcome": {"type": "string"},
    **{k: {"type": "string", "enum": v} for k, v in FACETS.items()},
    "evidence": {"type": "array", "items": obj({"code": {"type": "string", "enum": EVID}, "quote": {"type": "string"}})},
    "confidence": {"type": "string", "enum": ["high", "low"]},
})}})

NEED_MENU = "\n".join(f"{n['id']} {n['hashtag']} {n['name']}: {n['definition']}"
                      for v in NEEDS.values() for n in v["needs"])
SYSTEM = (
    "너는 한국어 자동차 사용자 글을 현대차 FoD(Features on Demand) 페르소나 연구용으로 코딩하는 연구 보조다. "
    "아래 코드북과 니즈 목록을 그대로 따른다.\n"
    "- 글에 근거가 있는 코드만 붙인다. situation·journey·attitude·life_stage·gender 코드마다 evidence에 원문을 그대로 복사한 부분 문자열(10~80자, 의역·띄어쓰기 수정 금지)을 넣는다.\n"
    "- needs: 니즈 목록에서 글에 해당하는 3차 니즈 id(0~3개). 해당 situation과 같은 계열을 우선한다.\n"
    "- journey·attitude는 layer가 A_FOD일 때만. INFO·EXCLUDE면 모든 코드 배열을 비우고 outcome도 빈 문자열.\n"
    "- outcome: 작성자의 중심 니즈 한 가지를 '방향 + 지표 + 대상/상황' 한 문장으로. 기능명·해법 금지.\n"
    "- 패싯: intensity(0 없음/1 언급/2 불편/3 강한 고통·위험), frequency(상황 빈도), wtp_signal(금액 언급·지불 수용·거부·없음), "
    "substitute(사제 장착·다른 앱·직접 해결·포기·없음), sentiment(FoD·차량 기능에 대한 -1/0/1), actor(니즈의 주인: 본인·배우자·자녀·부모·기타).\n"
    "- life_stage·gender는 본문 단서가 없으면 L_UNKNOWN·G_UNKNOWN.\n"
    "- 입력 글마다 items에 정확히 하나씩, 같은 id로 출력한다.\n\n"
    "# 코드북\n" + yaml.safe_dump(CODEBOOK, allow_unicode=True, sort_keys=False)
    + "\n# 3차 니즈 목록 (id #해시태그 이름: 정의)\n" + NEED_MENU
)


def client():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
    return anthropic.Anthropic()


def render(r):
    meta = json.loads(r.meta)
    ctx = " / ".join(f"{k}={v}" for k, v in meta.items() if v and k in ("channel", "product", "vehicle", "app"))
    return f'<post id="{r.post_id}" source="{r.source}" context="{ctx}">\n{r.text[:1500]}\n</post>'


def load_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"batches": []}


def submit(scope, n=None):
    corpus = pd.read_parquet(STAGE / "corpus.parquet")
    done = set(pd.read_parquet(OUT)["post_id"]) if OUT.exists() else set()
    pending = {pid for b in load_state()["batches"] if b["status"] != "collected" for pid in b["post_ids"]}
    todo = corpus[~corpus["post_id"].isin(done | pending)]
    if scope == "pilot":
        todo = todo.sample(n=min(int(n), len(todo)), random_state=20261010)
    rows = list(todo.itertuples())
    c, state = client(), load_state()
    for start in range(0, len(rows), 50_000):  # 배치당 요청 수를 넉넉히 100k 미만으로
        part = rows[start:start + 50_000]
        chunks = [part[i:i + PACK] for i in range(0, len(part), PACK)]
        reqs = [Request(custom_id=f"r{start + i * PACK}", params=MessageCreateParamsNonStreaming(
            model=MODEL, max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": "\n\n".join(render(r) for r in ch)}],
        )) for i, ch in enumerate(chunks)]
        b = c.messages.batches.create(requests=reqs)
        state["batches"].append({"id": b.id, "scope": scope, "n_requests": len(reqs),
                                 "post_ids": [r.post_id for r in part], "status": "submitted"})
        print(f"제출 {b.id}: 요청 {len(reqs)}, 글 {len(part)}")
    STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def status():
    c = client()
    for b in load_state()["batches"]:
        r = c.messages.batches.retrieve(b["id"])
        print(b["id"], b["scope"], r.processing_status, r.request_counts.model_dump(), "| 로컬:", b["status"])


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def collect():
    c, state = client(), load_state()
    corpus = pd.read_parquet(STAGE / "corpus.parquet").set_index("post_id")["text"].map(norm).to_dict()
    new_rows, cost, dropped_codes, errors = [], 0.0, 0, 0
    for b in state["batches"]:
        if b["status"] == "collected" or c.messages.batches.retrieve(b["id"]).processing_status != "ended":
            continue
        for res in c.messages.batches.results(b["id"]):
            if res.result.type != "succeeded":
                errors += 1
                continue
            msg = res.result.message
            u = msg.usage
            cost += ((u.input_tokens or 0) * PRICE[0] + (u.cache_creation_input_tokens or 0) * PRICE[0] * 1.25
                     + (u.cache_read_input_tokens or 0) * PRICE[2] + (u.output_tokens or 0) * PRICE[1]) / 1e6 / 2
            if msg.stop_reason != "end_turn":
                errors += 1
                continue
            for it in json.loads(next(x.text for x in msg.content if x.type == "text"))["items"]:
                text = corpus.get(it["id"])
                if text is None:
                    continue
                ok = {e["code"]: e["quote"] for e in it["evidence"] if norm(e["quote"]) and norm(e["quote"]) in text}
                for fam in ("situation", "journey", "attitude"):  # 근거 없는 코드 제거
                    keep = [x for x in it[fam] if x in ok]
                    dropped_codes += len(it[fam]) - len(keep)
                    it[fam] = keep
                for fam, unk in (("life_stage", "L_UNKNOWN"), ("gender", "G_UNKNOWN")):
                    if it[fam] != unk and it[fam] not in ok:
                        it[fam], dropped_codes = unk, dropped_codes + 1
                it["evidence"] = json.dumps(ok, ensure_ascii=False)
                it["post_id"] = it.pop("id")
                new_rows.append(it)
        b["status"] = "collected"
    if new_rows:
        df = pd.DataFrame(new_rows)
        if OUT.exists():
            df = pd.concat([pd.read_parquet(OUT), df]).drop_duplicates("post_id", keep="last")
        df.to_parquet(OUT, index=False)
    STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    print(f"수집 {len(new_rows):,}건, 근거 없어 제거된 코드 {dropped_codes:,}, 실패 요청 {errors}, 비용 약 ${cost:.2f} (배치 할인 반영)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "submit":
        submit(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        {"status": status, "collect": collect}[cmd]()
