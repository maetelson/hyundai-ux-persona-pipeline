"""분석 코퍼스 할당 표본 (양보다 구성): A층·리뷰 전수 + 생애단계·상황 코드별 상한 + 채널 상한.

  python run/03_sample.py
출력: data/stage/corpus.parquet, data/stage/corpus_quota.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data/stage"
SEED = 20261010
CAP = {  # code 접두사 → 소스별 상한
    "L_": {"naver_cafe": 3700, "naver_blog": 0},
    "S_": {"naver_cafe": 2600, "naver_blog": 600},
}
CHANNEL_SHARE = 0.30  # 한 카페·블로그가 한 칸에서 차지할 수 있는 최대 비율


def cap_channel(g, n):
    """채널별 상한을 지키며 n개 뽑기. 남는 자리는 다른 채널에서 채운다."""
    g = g.sample(frac=1, random_state=SEED)
    per = max(int(n * CHANNEL_SHARE), 1)
    g = g.assign(_rank=g.groupby("channel").cumcount())
    first = g[g["_rank"] < per]
    return first.head(n).drop(columns="_rank")


if __name__ == "__main__":
    df = pd.read_parquet(STAGE / "posts.parquet")
    df["code"] = df["seed_codes"].map(lambda x: x[0] if len(x) else "")
    df["channel"] = df["meta"].str.extract(r'"channel": "([^"]*)"')[0].fillna(df["source"])
    a = df[df["layer_hint"].isin(["A", "AB"])]
    b = df[df["layer_hint"] == "B"]
    parts, quota = [a], []
    for (code, src), g in b.groupby(["code", "source"]):
        n = CAP.get(code[:2], {}).get(src, 0)
        if n:
            s = cap_channel(g, n)
            parts.append(s)
            quota.append({"code": code, "source": src, "available": len(g), "sampled": len(s),
                          "top_channel_share": round(s["channel"].value_counts(normalize=True).iloc[0], 3) if len(s) else 0})
    corpus = pd.concat(parts).drop_duplicates("post_id")
    corpus.drop(columns=["channel"]).to_parquet(STAGE / "corpus.parquet", index=False)
    pd.DataFrame(quota).to_csv(STAGE / "corpus_quota.csv", index=False, encoding="utf-8-sig")
    print(corpus.groupby(["layer_hint", "source"]).size())
    print("corpus →", len(corpus))
