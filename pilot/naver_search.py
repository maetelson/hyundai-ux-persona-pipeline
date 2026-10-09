"""NAVER API HUB 검색(blog, cafearticle) 수집기. 소스별 시드 뱅크, 저수율 조기 중단, 시드 수율 집계.

환경변수(.env): NAVER_HUB_CLIENT_ID, NAVER_HUB_CLIENT_SECRET, NAVER_DAILY_BUDGET
  python pilot/naver_search.py count   config/seeds/naver_cafe.yaml config/seeds/naver_blog.yaml
  python pilot/naver_search.py collect config/seeds/naver_cafe.yaml config/seeds/naver_blog.yaml
  python pilot/naver_search.py rejudge config/seeds/naver_cafe.yaml config/seeds/naver_blog.yaml  # API 호출 없음
산출: data/raw/naver/{source_id}.jsonl (kept 플래그 포함 전체), data/seed_yield.csv
"""
import csv
import html
import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "naver"
LEDGER = ROOT / "data" / "naver_calls.json"  # 일별 호출 수
YIELD = ROOT / "data" / "seed_yield.csv"
BASE = "https://naverapihub.apigw.ntruss.com/search/v1/"
DAILY_BUDGET = int(os.getenv("NAVER_DAILY_BUDGET", "20000"))  # 공식 일 25,000보다 낮게
MIN_INTERVAL = 0.1  # 10 RPS (공식 키당 50)


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def calls_today(add=0):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    today = date.today().isoformat()
    data[today] = data.get(today, 0) + add
    LEDGER.write_text(json.dumps(data, indent=1))
    return data[today]


def search(endpoint, query, start=1, display=100, sort="sim"):
    if calls_today() >= DAILY_BUDGET:
        sys.exit(f"오늘 호출 예산 {DAILY_BUDGET}회 소진. 내일 다시.")
    headers = {"X-NCP-APIGW-API-KEY-ID": os.environ["NAVER_HUB_CLIENT_ID"],
               "X-NCP-APIGW-API-KEY": os.environ["NAVER_HUB_CLIENT_SECRET"]}
    params = {"query": query, "display": display, "start": start, "sort": sort}
    for attempt in range(4):
        time.sleep(MIN_INTERVAL)
        r = requests.get(BASE + endpoint, headers=headers, params=params, timeout=20)
        calls_today(1)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:  # RPS 초과면 잠시 뒤 회복, 한도 초과면 계속 429
            if attempt == 3:
                sys.exit(f"429 연속: 한도 초과로 판단하고 중단. {r.text[:200]}")
            time.sleep(2 ** attempt)
            continue
        sys.exit(f"{r.status_code} {endpoint} '{query}': {r.text[:300]}")


def clean(s):
    return html.unescape(re.sub(r"</?b>", "", s or ""))


def load_bank(path):
    bank = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    bank["seeds"] = [s["seed"] for s in bank.get("core_seeds", []) + bank.get("optional_templates", [])]
    return bank


def make_judge(bank):
    rel = yaml.safe_load((ROOT / "config" / "relevance_fod.yaml").read_text(encoding="utf-8"))
    strong, weak, car = (re.compile(rel[k]) for k in ("fod_strong", "fod_weak", "car"))
    neg = [t for t in bank.get("negative_terms", [])]
    neg_any = re.compile(rel["negative_any"]) if rel.get("negative_any") else None

    def judge(text):
        if any(t in text for t in neg) or (neg_any and neg_any.search(text)):
            return False, "negative_term"
        if strong.search(text):
            return True, ""
        if not weak.search(text):
            return False, "no_fod_term"
        if not car.search(text):
            return False, "no_car_context"
        return True, ""
    return judge


def judge_text(it):
    return " ".join(clean(it.get(k, "")) for k in ("title", "description", "cafename", "bloggername"))


def rejudge(banks):
    """API 호출 없이 저장된 원문에 현재 관련성 규칙을 다시 적용."""
    for bank in banks:
        judge, path = make_judge(bank), OUT / f"{bank['source_id']}.jsonl"
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        before = sum(r["kept"] for r in rows)
        for r in rows:
            r["kept"], r["drop_reason"] = judge(judge_text(r))
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{bank['source_id']}: kept {before:,} → {sum(r['kept'] for r in rows):,} / {len(rows):,}")


def classify(precision, new_kept):
    # ponytail: 고정 임계값, 2차 수집 후 분포 보고 조정
    if precision < 0.10 or new_kept < 10:
        return "noise"
    if precision >= 0.40 and new_kept >= 50:
        return "high"
    return "medium"


def count(banks):
    for bank in banks:
        judge = make_judge(bank)
        print(f"== {bank['source_id']}")
        for q in bank["seeds"]:
            res = search(bank["endpoint"], q, display=100)
            kept = sum(judge(judge_text(i))[0] for i in res["items"])
            print(f"{res['total']:>9,}  p@100={kept / max(len(res['items']), 1):.2f}  {q}")


def collect(banks):
    OUT.mkdir(parents=True, exist_ok=True)
    yield_rows = []
    for bank in banks:
        ep, sid, judge = bank["endpoint"], bank["source_id"], make_judge(bank)
        path = OUT / f"{sid}.jsonl"
        seen = {json.loads(l)["link"] for l in path.open(encoding="utf-8")} if path.exists() else set()
        with path.open("a", encoding="utf-8") as f:
            for q in bank["seeds"]:
                for sort in bank.get("sorts", ["sim"]):
                    pages = fetched = kept = new_kept = low_streak = 0
                    stop = "exhausted"
                    for start in range(1, bank.get("max_pages", 10) * 100, 100):
                        res = search(ep, q, start=start, sort=sort)
                        items = res.get("items", [])
                        pages += 1
                        page_kept = 0
                        for it in items:
                            text = judge_text(it)
                            ok, why = judge(text)
                            page_kept += ok
                            if it["link"] in seen:
                                continue
                            seen.add(it["link"])
                            new_kept += ok
                            f.write(json.dumps({
                                "source": sid, "layer": bank.get("layer"), "seed": q, "sort": sort, "page_start": start,
                                "kept": ok, "drop_reason": why, "fetched_at": datetime.now().isoformat(timespec="seconds"),
                                **{k: clean(v) if k in ("title", "description") else v for k, v in it.items()},
                            }, ensure_ascii=False) + "\n")
                        fetched += len(items)
                        kept += page_kept
                        low_streak = low_streak + 1 if page_kept < bank["early_stop_min_kept_ratio"] * max(len(items), 1) else 0
                        if low_streak >= bank["early_stop_low_yield_pages"]:
                            stop = "low_yield"
                            break
                        if len(items) < 100 or start + 100 > res["total"]:
                            break
                    else:
                        stop = "max_pages"
                    precision = kept / max(fetched, 1)
                    yield_rows.append({"source_id": sid, "seed": q, "sort": sort, "pages": pages, "fetched": fetched,
                                       "kept": kept, "new_unique_kept": new_kept, "precision": round(precision, 3),
                                       "stop_reason": stop, "class": classify(precision, new_kept)})
                    r = yield_rows[-1]
                    print(f"{sid:10} {sort:4} p={r['precision']:.2f} new={new_kept:>4} pages={pages:>2} {stop:9} {r['class']:6} {q}")
    with YIELD.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(yield_rows[0]))
        w.writeheader()
        w.writerows(yield_rows)
    print(f"시드 수율 → {YIELD.relative_to(ROOT)}, 오늘 호출 {calls_today():,}")


if __name__ == "__main__":
    load_env()
    if not os.getenv("NAVER_HUB_CLIENT_ID") or not os.getenv("NAVER_HUB_CLIENT_SECRET"):
        sys.exit(".env에 NAVER_HUB_CLIENT_ID / NAVER_HUB_CLIENT_SECRET를 넣어 주세요 (.env.example 참고)")
    mode, files = sys.argv[1], sys.argv[2:]
    {"count": count, "collect": collect, "rejudge": rejudge}[mode]([load_bank(p) for p in files])
