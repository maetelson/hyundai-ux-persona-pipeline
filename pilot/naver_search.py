"""NAVER API HUB 검색(blog, cafearticle) 수집기. 무료 한도 안에서만 돈다.

환경변수(.env 또는 셸): NAVER_HUB_CLIENT_ID, NAVER_HUB_CLIENT_SECRET
  python pilot/naver_search.py count   config/queries_fod.txt   # 쿼리당 1회 호출로 total(모수)만
  python pilot/naver_search.py collect config/queries_fod.txt   # sim+date 페이지 수집
"""
import html
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "naver"
LEDGER = ROOT / "data" / "naver_calls.json"  # 일별 호출 수 기록
BASE = "https://naverapihub.apigw.ntruss.com/search/v1/"
ENDPOINTS = ("blog", "cafearticle")
DAILY_BUDGET = int(os.getenv("NAVER_DAILY_BUDGET", "20000"))  # 공식 한도 25,000보다 낮게
MIN_INTERVAL = 0.1  # 10 RPS, 공식 키당 50 RPS


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
        if r.status_code == 429:  # RPS 초과면 1초 뒤 재시도, 한도 초과면 반복해도 429
            if attempt == 3:
                sys.exit(f"429 연속: 한도 초과로 판단하고 중단. {r.text[:200]}")
            time.sleep(2 ** attempt)
            continue
        sys.exit(f"{r.status_code} {endpoint} '{query}': {r.text[:300]}")


def clean(s):
    return html.unescape(re.sub(r"</?b>", "", s or ""))


def read_queries(path):
    return [q.strip() for q in Path(path).read_text(encoding="utf-8").splitlines()
            if q.strip() and not q.lstrip().startswith("#")]


def count(queries):
    rows = []
    for q in queries:
        row = {"query": q, **{ep: search(ep, q, display=1)["total"] for ep in ENDPOINTS}}
        rows.append(row)
        print(f"{row['blog']:>9,} {row['cafearticle']:>9,}  {q}")
    print(f"합계 blog {sum(r['blog'] for r in rows):,} / cafe {sum(r['cafearticle'] for r in rows):,} "
          f"(쿼리 간 중복 포함, API로 받을 수 있는 건 쿼리·정렬당 최대 1,000건)")
    return rows


def collect(queries):
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in ENDPOINTS:
        seen, path = set(), OUT / f"{ep}.jsonl"
        if path.exists():
            seen = {json.loads(l)["link"] for l in path.open(encoding="utf-8")}
        with path.open("a", encoding="utf-8") as f:
            for q in queries:
                for sort in ("sim", "date"):
                    for start in range(1, 1001, 100):
                        res = search(ep, q, start=start, sort=sort)
                        items = res.get("items", [])
                        for it in items:
                            if it["link"] in seen:
                                continue
                            seen.add(it["link"])
                            f.write(json.dumps({"source": f"naver_{ep}", "query": q, "sort": sort,
                                                **{k: clean(v) if k in ("title", "description") else v
                                                   for k, v in it.items()}}, ensure_ascii=False) + "\n")
                        if start + 100 > res["total"] or len(items) < 100:
                            break
                print(f"{ep} '{q}' 누적 {len(seen):,}건, 오늘 호출 {calls_today():,}")


if __name__ == "__main__":
    load_env()
    if not os.getenv("NAVER_HUB_CLIENT_ID") or not os.getenv("NAVER_HUB_CLIENT_SECRET"):
        sys.exit(".env에 NAVER_HUB_CLIENT_ID / NAVER_HUB_CLIENT_SECRET를 넣어 주세요 (.env.example 참고)")
    mode, qfile = sys.argv[1], sys.argv[2]
    {"count": count, "collect": collect}[mode](read_queries(qfile))
