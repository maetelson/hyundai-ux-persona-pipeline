"""네이버 원자료 삭제(2026-10-30) 전 동결본: 집계표·보고서·설정을 복사하고 체크섬을 남긴다. 삭제 대상 목록만 출력한다(직접 지우지 않음).
  python run/09_freeze.py freeze        → data/frozen/{오늘}/ (tables, report.html, config, MANIFEST.json)
  python run/09_freeze.py delete-plan   → 10-30에 지울 원문 포함 파일 목록과 크기
동결본에 들어가는 것: 집계표(글 단위 원문 없음), 보고서(짧은 인용 포함 — 팀 결정), 코드북·니즈 계층.
"""
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
# 원문(네이버 본문·스니펫)이 들어 있어 삭제 대상인 위치
RAW = [D / "raw/naver", D / "stage", D / "gold", D / "eval", D / "output/tables/extra/e1_price_mentions.csv"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def freeze():
    out = D / "frozen" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for src in sorted((D / "output").rglob("*")):
        if src.is_file() and src.suffix in (".csv", ".json", ".html") and src not in RAW and "post_segment" not in src.name:
            dst = out / src.relative_to(D / "output")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files.append(dst)
    for src in sorted((ROOT / "config").rglob("*.yaml")):
        dst = out / "config" / src.relative_to(ROOT / "config")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files.append(dst)
    man = {"frozen_at": date.today().isoformat(), "note": "네이버 원문 삭제 전 집계 동결본. 글 단위 원문·post_id 표 제외.",
           "files": {str(f.relative_to(out)): sha(f) for f in files}}
    (out / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out.relative_to(ROOT)} ({len(files)} files)")


def delete_plan():
    total = 0
    for p in RAW:
        if p.exists():
            size = sum(f.stat().st_size for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file())
            total += size
            print(f"{p.relative_to(ROOT)}  {size / 1e6:,.1f} MB")
    print(f"합계 {total / 1e6:,.1f} MB — 2026-10-30까지 사람이 직접 삭제(휴지통 비우기 포함)")


if __name__ == "__main__":
    {"freeze": freeze, "delete-plan": delete_plan}[sys.argv[1]]()
