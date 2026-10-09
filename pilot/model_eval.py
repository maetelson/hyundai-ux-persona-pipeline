"""모델·effort·묶음 크기별 라벨링 정확도와 비용 비교 (정답: data/gold/gold_v0_claude.jsonl).

  python pilot/model_eval.py                 # 기본 셀 전부
  python pilot/model_eval.py haiku-low-p10   # 특정 셀만
결과: data/eval/{cell}.jsonl (예측), data/eval/summary.csv
"""
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import yaml

ROOT = Path(__file__).resolve().parents[1]
GOLD_IN = ROOT / "data/gold/gold_v0_input.jsonl"
VERSION = os.getenv("CODEBOOK_VERSION", "v1")  # 코드북·정답 버전
GOLD = ROOT / f"data/gold/gold_{VERSION}_claude.jsonl"
OUT = ROOT / "data/eval" / VERSION
CODEBOOK = yaml.safe_load((ROOT / f"config/codebook_{VERSION}.yaml").read_text(encoding="utf-8"))

# $/MTok (2026-10 Claude API 표준가). cache write 5분 = 1.25x input
PRICE = {"claude-haiku-5-5": (0.10, 0.50, 0.01), "claude-sonnet-5-5": (2.0, 10.0, 0.20), "claude-opus-5-5": (4.0, 20.0, 0.20)}
CELLS = {  # name: (model, effort, pack)
    "haiku-low-p10": ("claude-haiku-5-5", "low", 10),
    "haiku-medium-p10": ("claude-haiku-5-5", "medium", 10),
    "haiku-low-p1": ("claude-haiku-5-5", "low", 1),
    "sonnet-low-p10": ("claude-sonnet-5-5", "low", 10),
    "sonnet-medium-p10": ("claude-sonnet-5-5", "medium", 10),
    "opus-low-p10": ("claude-opus-5-5", "low", 10),
    "opus-medium-p10": ("claude-opus-5-5", "medium", 10),
}
FAM = CODEBOOK["families"]
CODES = {f: list(FAM[f]["values"]) for f in ("layer", "situation", "journey", "attitude", "life_stage", "gender")}
ALL_EVID = CODES["situation"] + CODES["journey"] + CODES["attitude"] + CODES["life_stage"] + CODES["gender"]

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", "layer", "situation", "journey", "attitude", "life_stage", "gender", "evidence", "confidence"],
        "properties": {
            "id": {"type": "string"},
            "layer": {"type": "string", "enum": CODES["layer"]},
            "situation": {"type": "array", "items": {"type": "string", "enum": CODES["situation"]}},
            "journey": {"type": "array", "items": {"type": "string", "enum": CODES["journey"]}},
            "attitude": {"type": "array", "items": {"type": "string", "enum": CODES["attitude"]}},
            "life_stage": {"type": "string", "enum": CODES["life_stage"]},
            "gender": {"type": "string", "enum": CODES["gender"]},
            "evidence": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["code", "quote"],
                "properties": {"code": {"type": "string", "enum": ALL_EVID}, "quote": {"type": "string"}}}},
            "confidence": {"type": "string", "enum": ["high", "low"]},
        }}}},
}

SYSTEM = (
    "너는 한국어 자동차 커뮤니티 글을 현대차 FoD(Features on Demand) 페르소나 연구용으로 코딩하는 연구 보조다.\n"
    "아래 코드북을 그대로 따른다. 글에 근거가 있는 코드만 붙이고, 코드마다 evidence에 원문을 그대로 복사한 부분 문자열"
    "(10~80자, 의역·요약·띄어쓰기 수정 금지)을 넣는다. life_stage·gender는 단서가 없으면 L_UNKNOWN·G_UNKNOWN이고 evidence를 넣지 않는다. "
    "journey·attitude는 layer가 A_FOD일 때만 붙인다. INFO·EXCLUDE면 모든 코드 배열을 비운다.\n"
    "입력 글마다 items에 정확히 하나씩, 같은 id로 출력한다.\n\n# 코드북\n"
    + yaml.safe_dump(CODEBOOK, allow_unicode=True, sort_keys=False)
)


def load_env():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def render(r):
    ctx = " / ".join(f"{k}={v}" for k, v in r["meta"].items() if k in ("cafename", "bloggername", "product", "vehicle", "app") and v)
    return f'<post id="{r["id"]}" context="{ctx}">\n{r["text"]}\n</post>'


def call(client, model, effort, chunk):
    for attempt in range(5):
        try:
            t0 = time.time()
            resp = client.messages.create(
                model=model, max_tokens=16000,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": "\n\n".join(render(r) for r in chunk)}],
            )
            if resp.stop_reason != "end_turn":
                return {"error": f"stop_reason={resp.stop_reason}", "ids": [r["id"] for r in chunk], "usage": resp.usage.model_dump()}
            text = next(b.text for b in resp.content if b.type == "text")
            return {"items": json.loads(text)["items"], "usage": resp.usage.model_dump(), "sec": time.time() - t0}
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
            time.sleep(2 ** attempt * 3)
    return {"error": "retries exhausted", "ids": [r["id"] for r in chunk], "usage": {}}


def cost(model, u):
    pin, pout, pread = PRICE[model]
    return ((u.get("input_tokens") or 0) * pin + (u.get("cache_creation_input_tokens") or 0) * pin * 1.25
            + (u.get("cache_read_input_tokens") or 0) * pread + (u.get("output_tokens") or 0) * pout) / 1e6


def prf(gold, pred):
    tp = sum(len(g & p) for g, p in zip(gold, pred))
    fp = sum(len(p - g) for g, p in zip(gold, pred))
    fn = sum(len(g - p) for g, p in zip(gold, pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(cell, preds, inputs, gold):
    ids = [i for i in gold if i in preds]
    row = {"cell": cell, "n_pred": len(ids), "n_missing": len(gold) - len(ids)}
    row["layer_acc"] = sum(preds[i]["layer"] == gold[i]["layer"] for i in ids) / len(ids)
    useful = lambda l: l in ("A_FOD", "B_SITUATION")
    row["useful_acc"] = sum(useful(preds[i]["layer"]) == useful(gold[i]["layer"]) for i in ids) / len(ids)
    for fam in ("situation", "journey", "attitude"):
        p, r, f = prf([set(gold[i].get(fam, [])) for i in ids], [set(preds[i].get(fam, [])) for i in ids])
        row[f"{fam}_P"], row[f"{fam}_R"], row[f"{fam}_F1"] = round(p, 3), round(r, 3), round(f, 3)
    q = [(i, e["quote"]) for i in ids for e in preds[i].get("evidence", [])]
    row["evidence_valid"] = round(sum(norm(qt) in norm(inputs[i]["text"]) for i, qt in q) / max(len(q), 1), 3)
    row["layer_acc"], row["useful_acc"] = round(row["layer_acc"], 3), round(row["useful_acc"], 3)
    return row


def run(cell, client, inputs, gold):
    model, effort, pack = CELLS[cell]
    rows = [inputs[i] for i in sorted(gold)]
    chunks = [rows[k:k + pack] for k in range(0, len(rows), pack)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda c: call(client, model, effort, c), chunks))
    preds, usage, errors = {}, [], 0
    for res in results:
        usage.append(res.get("usage") or {})
        if "error" in res:
            errors += 1
            continue
        for it in res["items"]:
            preds[it["id"]] = it
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{cell}.jsonl").write_text("".join(json.dumps(preds[k], ensure_ascii=False) + "\n" for k in sorted(preds)), encoding="utf-8")
    row = score(cell, preds, inputs, gold)
    c = sum(cost(model, u) for u in usage)
    row.update({"model": model, "effort": effort, "pack": pack, "request_errors": errors,
                "cost_usd": round(c, 4), "cost_per_1k_items": round(c / len(rows) * 1000, 3),
                "out_tokens": sum((u.get("output_tokens") or 0) for u in usage),
                "cache_read": sum((u.get("cache_read_input_tokens") or 0) for u in usage)})
    print(json.dumps(row, ensure_ascii=False))
    return row


if __name__ == "__main__":
    load_env()
    client = anthropic.Anthropic()
    inputs = {r["id"]: r for r in map(json.loads, GOLD_IN.open(encoding="utf-8"))}
    gold = {r["id"]: r for r in map(json.loads, GOLD.open(encoding="utf-8"))}
    cells = sys.argv[1:] or list(CELLS)
    summary = OUT / "summary.csv"
    prev = list(csv.DictReader(summary.open(encoding="utf-8-sig"))) if summary.exists() else []
    rows = [r for r in prev if r["cell"] not in cells] + [run(c, client, inputs, gold) for c in cells]
    with summary.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1]))
        w.writeheader()
        w.writerows(rows)
    print("→", summary.relative_to(ROOT))
