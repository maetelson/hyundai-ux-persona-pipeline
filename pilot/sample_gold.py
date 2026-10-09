"""정답 세트 표본 추출 (층화, 고정 seed). → data/gold/gold_v0_input.jsonl

  python pilot/sample_gold.py
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "gold" / "gold_v0_input.jsonl"
FOD_KW = re.compile(r"스토어|구독|유료|결제|요금|무료\s?기간|해지|환불|테마|원격\s?(스마트\s?)?주차|RSPA|스트리밍|라이팅")
random.seed(20261009)


def jl(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def naver(sid):
    rows = jl(RAW / "naver" / f"{sid}.jsonl")
    for r in rows:
        r["_text"] = f"{r['title']}\n{r['description']}"
        r["_meta"] = {k: r.get(k) for k in ("link", "cafename", "bloggername", "postdate", "seed", "kept")}
        r["_id"] = r["link"]
    return [r for r in rows if r["kept"]], [r for r in rows if not r["kept"]]


def apps():
    rows = []
    for p in sorted((RAW / "pilot").glob("*.jsonl")):
        if p.stem == "kia_connect_store":
            continue
        for r in jl(p):
            r["_text"], r["_id"] = r["text"], f"{p.stem}:{r['id']}"
            r["_meta"] = {"app": p.stem, "rating": r.get("rating"), "created": str(r.get("created"))[:10]}
            rows.append(r)
    return rows


def kia():
    rows = jl(RAW / "pilot" / "kia_connect_store.jsonl")
    for r in rows:
        r["_text"], r["_id"] = r["text"], f"kia:{r['id']}"
        r["_meta"] = {"product": r["product"], "category": r["category"], "vehicle": r["vehicle"]}
    return rows


def pick(rows, n):
    rows = [r for r in rows if len(r["_text"].strip()) >= 15]
    return random.sample(rows, min(n, len(rows)))


if __name__ == "__main__":
    cafe_k, cafe_d = naver("naver_cafe")
    blog_k, blog_d = naver("naver_blog")
    k = kia()
    k_theme = [r for r in k if "테마" in (r["category"] or "")]
    k_other = [r for r in k if r not in k_theme]
    a = apps()
    a_kw = [r for r in a if FOD_KW.search(r["_text"])]
    a_rest = [r for r in a if not FOD_KW.search(r["_text"])]
    strata = {
        "cafe_kept": pick(cafe_k, 100), "cafe_dropped": pick(cafe_d, 20),
        "blog_kept": pick(blog_k, 50), "blog_dropped": pick(blog_d, 10),
        "kia_theme": pick(k_theme, 25), "kia_other": pick(k_other, 35),
        "app_fodkw": pick(a_kw, 40), "app_other": pick(a_rest, 20),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs = [(s, r) for s, rs in strata.items() for r in rs]
    random.shuffle(pairs)  # id·순서에서 층 정보가 드러나지 않게
    rows = [{"id": f"g{i:03d}", "stratum": s, "src_id": r["_id"], "source": r.get("source", s.split("_")[0]),
             "text": r["_text"].strip(), "meta": r["_meta"]} for i, (s, r) in enumerate(pairs)]
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print({s: len(v) for s, v in strata.items()}, "→", len(rows), OUT.relative_to(ROOT))
