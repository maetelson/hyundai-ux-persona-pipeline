"""라벨(JSON 배열 파일)을 data/gold/gold_v0_claude.jsonl에 추가. 근거 인용이 원문 부분 문자열인지 검증.
오류가 있는 건은 저장하지 않고 출력만 한다. 같은 id는 덮어쓴다.
  python pilot/gold_append.py data/gold/batch_000.json
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = {r["id"]: r for r in map(json.loads, (ROOT / "data/gold/gold_v0_input.jsonl").open(encoding="utf-8"))}
OUT = ROOT / "data/gold/gold_v0_claude.jsonl"


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def errors(lab):
    text, errs = norm(SRC[lab["id"]]["text"]), []
    for code, quote in lab.get("evidence", {}).items():
        if norm(quote) not in text:
            errs.append(f"{code}: 인용 불일치 '{quote}'")
    codes = lab.get("situation", []) + lab.get("journey", []) + lab.get("attitude", [])
    if lab.get("life_stage", "L_UNKNOWN") != "L_UNKNOWN":
        codes.append(lab["life_stage"])
    errs += [f"{c}: 근거 없음" for c in codes if c not in lab.get("evidence", {})]
    return errs


if __name__ == "__main__":
    saved = {r["id"]: r for r in map(json.loads, OUT.open(encoding="utf-8"))} if OUT.exists() else {}
    bad = 0
    for lab in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")):
        errs = errors(lab)
        if errs:
            bad += 1
            print(f"  ✗ {lab['id']}: " + " | ".join(errs))
        else:
            saved[lab["id"]] = lab
    OUT.write_text("".join(json.dumps(saved[k], ensure_ascii=False) + "\n" for k in sorted(saved)), encoding="utf-8")
    print(f"누적 저장 {len(saved)}건, 이번 오류 {bad}건")
