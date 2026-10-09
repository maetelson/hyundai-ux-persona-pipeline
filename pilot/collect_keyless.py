"""키 없이 수집 가능한 소스의 모수 파일럿: 기아 커넥트 스토어 리뷰, App Store RSS, Google Play.

python pilot/collect_keyless.py  ->  data/raw/pilot/*.jsonl, 콘솔에 건수 요약
"""
import html
import json
import re
import time
from pathlib import Path

import requests
from google_play_scraper import Sort, app as play_app, reviews as play_reviews

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "pilot"
OUT.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0 (research pilot; DECK FoD persona study)"
DELAY = 1.0  # 요청 간격(초)

APPS = {  # name: (ios_id, android_pkg)
    "myhyundai": ("6714472723", "com.hyundai.oneapp.kr"),
    "kia": ("6590612538", "com.kia.oneapp.kr"),
    "mygenesis": ("6444664964", "com.genesis.oneapp"),
}


def write(name, rows):
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return len(rows)


def kia_reviews():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    page_html = s.get("https://connectstore.kia.com/kr/info/review", timeout=20).text
    csrf = re.search(r'name="_csrf"\s+content="([^"]+)"', page_html).group(1)
    rows, page, total = [], 1, None
    while total is None or len(rows) < total:
        r = s.post(
            "https://connectstore.kia.com/kr/review/getFilteredReviews.do",
            json={"fodNos": [], "photoFilter": "all", "myCarOnly": False,
                  "sortType": "latest", "page": page, "carInfo3": None},
            headers={"OWASP_CSRFTOKEN": csrf, "X-Requested-With": "XMLHttpRequest",
                     "Referer": "https://connectstore.kia.com/kr/info/review"},
            timeout=20,
        )
        data = r.json()["data"]
        total = data["totCnt"]
        batch = data["reviewList"]
        if not batch:
            break
        for x in batch:
            rows.append({
                "source": "kia_connect_store", "id": x["reviewId"],
                "created_ms": x.get("vbgRgstTismp"), "product": x.get("fodNm"),
                "category": x.get("fodCtyTitlNm"), "vehicle": x.get("vehicleName"),
                "text": html.unescape(x.get("content") or ""),
            })
        page += 1
        time.sleep(DELAY)
    return rows, total


def ios_reviews(app_id):
    rows = []
    for page in range(1, 11):  # RSS 상한: 10페이지(약 500건)
        r = requests.get(
            f"https://itunes.apple.com/kr/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json",
            headers={"User-Agent": UA}, timeout=20)
        if r.status_code != 200:
            break
        entries = r.json().get("feed", {}).get("entry", [])
        entries = [e for e in (entries if isinstance(entries, list) else [entries]) if "im:rating" in e]
        if not entries:
            break
        for e in entries:
            rows.append({"source": "appstore", "id": e["id"]["label"], "created": e["updated"]["label"],
                         "rating": int(e["im:rating"]["label"]), "version": e["im:version"]["label"],
                         "text": f'{e["title"]["label"]}\n{e["content"]["label"]}'})
        time.sleep(DELAY)
    return rows


def play(pkg, cap=3000):
    meta = play_app(pkg, lang="ko", country="kr")
    rows, token = [], None
    while len(rows) < cap:  # ponytail: cap으로 파일럿 제한, 본수집 때 해제
        batch, token = play_reviews(pkg, lang="ko", country="kr", sort=Sort.NEWEST,
                                    count=200, continuation_token=token)
        if not batch:
            break
        rows += [{"source": "googleplay", "id": b["reviewId"], "created": b["at"], "rating": b["score"],
                  "version": b.get("reviewCreatedVersion"), "text": b["content"]} for b in batch]
        if token is None or token.token is None:
            break
        time.sleep(DELAY)
    return rows, meta.get("reviews"), meta.get("ratings")


if __name__ == "__main__":
    summary = {}
    rows, total = kia_reviews()
    summary["kia_connect_store"] = {"collected": write("kia_connect_store", rows), "site_total": total}
    for name, (ios_id, pkg) in APPS.items():
        summary[f"ios_{name}"] = {"collected": write(f"ios_{name}", ios_reviews(ios_id))}
        rows, n_reviews, n_ratings = play(pkg)
        summary[f"play_{name}"] = {"collected": write(f"play_{name}", rows),
                                   "store_text_reviews": n_reviews, "store_ratings": n_ratings}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
