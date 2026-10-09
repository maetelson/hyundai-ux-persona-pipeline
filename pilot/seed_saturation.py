"""시드 충분성 진단: 상한 도달률, 시드 추가에 따른 신규 고유 글 곡선, 시드에 없는 빈출 표현.

  python pilot/seed_saturation.py naver_situation_blog naver_situation_cafearticle
"""
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = set("그리고 그런데 그래서 하지만 있는 없는 하는 있어요 있습니다 합니다 했는데 하고 너무 정말 진짜 이번 지금 오늘 어제 그냥 많이 조금 같아요 같은 어떻게 어떤 이런 저런 그런 혹시 다들 저는 제가 저희 우리 여기 거기 때문에 경우 정도 관련 대한 위해 통해 대해 이후 이전 부분 사용 확인 문의 질문 안녕하세요 감사합니다 생각 이렇게 해서 하면 하는데 했어요 해요 되는 되나요 할까요 인데 입니다 있나요 없나요 그럼 근데 아니 계속 처음 다시 바로 모두 가장 다른 아직 이제 함께 하나 둘 후기 고민 불편".split())


_KIWI = None


GENERIC = "차량 자동차 운전 주차 차 생각 사람 시간 경우 정도 방법 이유 부분 문제 사용 가격 구매 정보 추천 비교 총정리 후기 고민 불편 오늘 이번 요즘 처음 최근 기능 상황 확인 필요 가능 진행 이용 관리 기준 내용 결과 서비스 제품 브랜드 모델 신형 출시 정리 포스팅 블로그 카페 회원 질문 답변 안녕 감사".split()


def nouns(text):
    """명사(NNG·NNP·SL)와 인접 명사 2개 묶음."""
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi()
    toks = [t for t in _KIWI.tokenize(text) if t.tag in ("NNG", "NNP", "SL")]
    words = [t.form for t in toks if len(t.form) >= 2 and t.form not in STOP]
    pairs = [f"{a.form} {b.form}" for a, b in zip(toks, toks[1:])
             if a.end == b.start or text[a.end:b.start] == " "]
    return words + [p for p in pairs if all(len(x) >= 2 and x not in STOP for x in p.split())]


def main(sids):
    for sid in sids:
        rows = [json.loads(l) for l in (ROOT / f"data/raw/naver/{sid}.jsonl").open(encoding="utf-8")]
        kept = [r for r in rows if r["kept"]]
        y = list(csv.DictReader((ROOT / f"data/seed_yield_{sid}.csv").open(encoding="utf-8-sig")))
        cap = sum(r["stop_reason"] == "max_pages" for r in y) / max(len(y), 1)
        print(f"\n== {sid}: 고유 {len(rows):,}, kept {len(kept):,}, 시드×정렬 {len(y)}, 1,000건 상한 도달 {cap:.0%}")
        # 수집 순서대로 시드 추가 시 신규 kept 곡선 (10% 구간별)
        news = [int(r["new_unique_kept"]) for r in y]
        n = len(news)
        for k in range(1, 11):
            seg = news[(k - 1) * n // 10: k * n // 10]
            print(f"   시드 {k * 10:>3}% 구간: 시드당 신규 kept 평균 {sum(seg) / max(len(seg), 1):6.0f}")
        # 시드에 없는 빈출 표현 (kept 글 기준, 문서 빈도)
        seed_words = set(w for r in y for w in nouns(r["seed"])) | set(GENERIC)
        df = collections.Counter()
        for r in kept:
            df.update(set(nouns(f"{r['title']} {r['description']}")))
        cand = [(w, c) for w, c in df.most_common(800) if w not in seed_words]
        print("   시드에 없는 빈출 명사(구) 상위 80:", ", ".join(f"{w}({c})" for w, c in cand[:80]))


if __name__ == "__main__":
    main(sys.argv[1:])
