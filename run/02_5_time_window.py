"""5개년 시간 창(기본 2021-10-10 이후). posts에 date_status를 붙이고 모드에 따라 거른다.

  python run/02_5_time_window.py [hybrid|strict]   (기본 hybrid)
date_status:
  in / out          : 작성일이 있는 글(블로그·리뷰)의 판정
  cafe_recent       : 카페 최신순 검색 결과에 나온 글 (블로그 대리 측정상 최근 수개월)
  cafe_id_recent    : 카페별 글번호 기준선 이후 (2022년 이후 연도 언급 글의 하위 2% 글번호, 단서 20건 이상 카페만)
  cafe_id_old       : 카페별 글번호 기준선 이전 → 5년 밖 추정
  cafe_unknown      : 판정 불가 (정확도순으로만 수집, 기준선 없는 카페)
hybrid: out·cafe_id_old 제외, cafe_unknown 유지(민감도 분석으로 확인)
strict: in·cafe_recent·cafe_id_recent만 유지
출력: data/stage/posts.parquet 갱신(date_status 열), data/stage/time_window.csv
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data/stage"
START = "2021-10-10"
MIN_ANCHORS, ANCHOR_Q = 20, 0.02


def mentioned_year(text):
    ys = [int(a) for a in re.findall(r"(?<!\d)(20[12]\d)년", text)]
    ys += [2000 + int(b) for b in re.findall(r"(?<![\d.])([12]\d)년(?:식|형)", text)]
    return max(ys) if ys else np.nan


def date_sorted_links():
    links = set()
    for f in glob.glob(str(ROOT / "data/raw/naver/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if "cafename" in r and r["sort"] == "date":
                links.add(r["link"])
    return links


def classify(p):
    st = pd.Series("na", index=p.index)
    dated = pd.to_datetime(p["created_at"], errors="coerce")
    st[dated.notna()] = np.where(dated[dated.notna()] >= START, "in", "out")
    cafe = p["source"] == "naver_cafe"
    c = p[cafe].copy()
    c["cafe"] = c["url"].str.extract(r"cafe\.naver\.com/([^/]+)/")[0]
    c["aid"] = pd.to_numeric(c["url"].str.extract(r"/(\d+)(?:\?|$)")[0], errors="coerce")
    c["yr"] = c["text"].map(mentioned_year)
    anchors = c[(c["yr"] >= 2022) & (c["yr"] <= 2027)].groupby("cafe")["aid"]
    cutoff = anchors.quantile(ANCHOR_Q)[anchors.size() >= MIN_ANCHORS]
    c["cut"] = c["cafe"].map(cutoff)
    ds = c["url"].isin(date_sorted_links())
    s = np.where(c["cut"].notna() & (c["aid"] < c["cut"]), "cafe_id_old",
         np.where(ds, "cafe_recent", np.where(c["cut"].notna(), "cafe_id_recent", "cafe_unknown")))
    st[cafe] = s
    return st


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    p = pd.read_parquet(STAGE / "posts.parquet")
    p["date_status"] = classify(p)
    p.to_parquet(STAGE / "posts.parquet", index=False)
    keep = {"hybrid": ~p["date_status"].isin(["out", "cafe_id_old"]),
            "strict": p["date_status"].isin(["in", "cafe_recent", "cafe_id_recent"])}[mode]
    t = p.groupby(["source", "date_status"]).size().rename("n").reset_index()
    t.to_csv(STAGE / "time_window.csv", index=False, encoding="utf-8-sig")
    print(t.to_string(index=False))
    print(f"{mode}: 유지 {keep.sum():,} / {len(p):,}")
    # 코퍼스에도 반영
    c = pd.read_parquet(STAGE / "corpus.parquet").drop(columns=["date_status"], errors="ignore")
    c = c.merge(p[["post_id", "date_status"]], on="post_id", how="left")
    ck = {"hybrid": ~c["date_status"].isin(["out", "cafe_id_old"]),
          "strict": c["date_status"].isin(["in", "cafe_recent", "cafe_id_recent"])}[mode]
    c[ck].to_parquet(STAGE / "corpus.parquet", index=False)
    print(f"corpus {mode}: 유지 {ck.sum():,} / {len(c):,}")
