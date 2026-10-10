"""모든 원천을 공통 posts 표로 정규화하고 중복을 제거한다.

  python run/02_normalize.py
입력: data/raw/naver/*.jsonl (kept만), data/raw/pilot/*.jsonl
출력: data/stage/posts.parquet, data/stage/funnel.csv
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW, STAGE = ROOT / "data/raw", ROOT / "data/stage"
TIER = {"naver_cafe": "core", "naver_blog": "core", "kia_store": "support", "app_review": "support"}


def h(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def norm_text(s):
    return re.sub(r"\s+", " ", s or "").strip()


def seed_codes():
    """시드 → 코드(시드 파일의 code 필드, 상황 조합 시드는 situation_bases 코드)."""
    m = {}
    for p in (ROOT / "config/seeds").glob("*.yaml"):
        b = yaml.safe_load(p.read_text(encoding="utf-8"))
        for s in b.get("core_seeds", []) + b.get("optional_templates", []):
            m[s["seed"]] = s.get("code", "")
        for code, bases in (b.get("situation_bases") or {}).items():
            for base in bases:
                for suf in b.get("suffixes", [""]):
                    m.setdefault(f"{base} {suf}".strip(), code)
    return m


def naver_rows(codes):
    for p in sorted((RAW / "naver").glob("*.jsonl")):
        bank = p.stem
        layer = "A" if bank in ("naver_cafe", "naver_blog", "naver_benchmark_cafearticle", "naver_benchmark_blog", "naver_event") else "B"
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if not r["kept"]:
                continue
            is_cafe = "cafename" in r
            yield {
                "url": r["link"], "source": "naver_cafe" if is_cafe else "naver_blog",
                "created_at": (f"{r['postdate'][:4]}-{r['postdate'][4:6]}-{r['postdate'][6:]}" if r.get("postdate") else None),
                "author_hash": h(r["bloggerlink"]) if r.get("bloggerlink") else None,
                "text": norm_text(f"{r['title']}\n{r['description']}"),
                "meta": {"channel": r.get("cafename") or r.get("bloggername")},
                "seed": r["seed"], "seed_code": codes.get(r["seed"], ""), "bank": bank, "layer_hint": layer,
            }


def review_rows():
    for p in sorted((RAW / "pilot").glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if p.stem == "kia_connect_store":
                created = datetime.fromtimestamp(r["created_ms"] / 1000, timezone.utc).date().isoformat() if r.get("created_ms") else None
                yield {"url": f"kia:{r['id']}", "source": "kia_store", "created_at": created, "author_hash": None,
                       "text": norm_text(r["text"]), "meta": {"product": r["product"], "category": r["category"], "vehicle": r["vehicle"]},
                       "seed": "", "seed_code": "", "bank": p.stem, "layer_hint": "A"}
            else:
                yield {"url": f"{p.stem}:{r['id']}", "source": "app_review", "created_at": str(r.get("created"))[:10],
                       "author_hash": None, "text": norm_text(r["text"]),
                       "meta": {"app": p.stem, "rating": r.get("rating")}, "seed": "", "seed_code": "", "bank": p.stem, "layer_hint": "AB"}


if __name__ == "__main__":
    STAGE.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(naver_rows(seed_codes())) + list(review_rows()))
    funnel = [("raw_kept", df.groupby("source").size())]
    # 같은 URL이 여러 시드에서 걸린 경우: 첫 행을 남기고 시드·코드는 목록으로 합친다
    agg = df.groupby("url", sort=False).agg(
        seeds=("seed", lambda s: sorted({x for x in s if x})),
        seed_codes=("seed_code", lambda s: sorted({x for x in s if x})),
        banks=("bank", lambda s: sorted(set(s))),
    )
    df = df.drop_duplicates("url").set_index("url").join(agg).reset_index()
    funnel.append(("dedupe_url", df.groupby("source").size()))
    df["text_hash"] = df["text"].str.replace(r"[^0-9A-Za-z가-힣]", "", regex=True).map(h)
    df = df.drop_duplicates("text_hash")  # 퍼나르기·복붙(정확 일치)
    funnel.append(("dedupe_text", df.groupby("source").size()))
    df = df[df["text"].str.len() >= 15]
    funnel.append(("min_length", df.groupby("source").size()))
    df["post_id"] = df["url"].map(h)
    df["source_tier"] = df["source"].map(TIER)
    df["meta"] = df["meta"].map(lambda m: json.dumps(m, ensure_ascii=False))
    cols = ["post_id", "source", "source_tier", "url", "created_at", "author_hash", "text", "meta",
            "seeds", "seed_codes", "banks", "layer_hint"]
    df[cols].to_parquet(STAGE / "posts.parquet", index=False)
    f = pd.concat({k: v for k, v in funnel}, axis=1).fillna(0).astype(int)
    f.to_csv(STAGE / "funnel.csv", encoding="utf-8-sig")
    print(f)
    print("posts →", len(df))
