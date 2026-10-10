"""설문 연결: 텍스트 세그먼트를 가장 잘 가르는 상황 코드를 골라 설문 문항 후보로 만든다 (API 호출 없음).
  - 입력: 상황 2개 이상 글(LCA 적합 표본)과 배정 세그먼트, 상황 상위 22개 이진 지표
  - 앞으로 선택(forward selection): 다항 로지스틱 5겹 CV의 균형 정확도가 가장 많이 오르는 코드를 하나씩 추가(최대 12개)
  - 기준선: 최다 세그먼트(균형 정확도 = 1/k), 22개 전부 사용
  - 각 코드를 현재 설문 문항(있으면)이나 새 문항 제안에 대응

  python run/07_4_survey_items.py
출력: data/output/tables/extra/s3_survey_items.csv, s3_survey_meta.json
주의: 글 단위 '언급' 지표로 고른 후보다. 설문 응답자로 다시 학습·검증해야 한다(PLAN 8.1).
"""
import importlib
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, accuracy_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
A = importlib.import_module("06_analyze")
X1 = importlib.import_module("07_extra")
MAX_ITEMS = 12

# 상황 코드 → 설문 문항 (현행 설문 기준. 신규 = 새로 넣어야 하는 문항 제안)
SURVEY = {
    "S_PARK_TIGHT": ("B2-1 ①", "현행", "옆 차가 가까워 문을 열기 힘든 칸에 차를 댄 날"),
    "S_LONG_DRIVE": ("B2-1 ②", "현행", "고속도로를 1시간 넘게 달린 날"),
    "S_WEATHER": ("B2-1 ③", "현행", "타기 전 차 안이 너무 덥거나 추웠던 날"),
    "S_INCAR_REST": ("B2-1 ④", "현행", "차 안에서 10분 넘게 기다린 날"),
    "S_CHILD": ("B2-1 ⑥", "현행", "고등학생 이하 아이를 태워 데려다주거나 데려온 날"),
    "S_SECURITY": ("B2-3", "현행", "인적 드문 곳·밤늦게 세워 두고 마음이 쓰인 빈도"),
    "S_FAMILY_SHARE": ("B1-5b", "현행", "이 차를 주 1회 이상 운전하는 사람(본인 외)"),
    "S_AFTERMARKET": ("C-1a", "현행", "차를 받은 뒤 돈 내고 직접 단 것"),
    "S_COST": ("C-2a", "현행(간접)", "선택 사양·등급 결정 경험 — 돈 계산 상황을 직접 묻지는 않음"),
    "S_REPLACE": ("B1-7 + C-11", "현행", "보유 계획 2년 안 + 다음 차 행동"),
    "S_ADAS": ("C-2b ② / D1-2c", "현행(간접)", "주행 보조 아쉬움·지불 방식"),
    "S_PERSONAL": ("D1-1", "현행", "조명·화면을 바꾸고 싶다는 생각 빈도"),
    "S_EV": ("B1-4", "현행", "연료(전기·하이브리드)"),
    "S_REMOTE_CTRL": ("신규", "신규", "지난 3개월 앱으로 원격 시동·공조·잠금을 할 때: 쓴 적 없음 / 늘 됨 / 가끔 안 됨 / 자주 안 됨"),
    "S_CARGO": ("신규(B2-1 ⑧)", "신규", "트렁크를 가득 채우거나 큰 짐(유모차·골프백 등)을 실은 날"),
    "S_LEISURE": ("신규(B2-1 ⑨)", "신규", "캠핑·차박·레저로 차를 쓴 날"),
    "S_MAINTAIN": ("A1-3 ② (4050만)", "부분", "정비 정보 탐색 — 전 연령 문항 필요: 지난 3개월 정비·소모품·방전 문제를 겪은 적"),
    "S_NAV": ("신규", "신규", "순정 내비 대신 휴대폰 내비(티맵 등)를 주로 쓴다"),
    "S_PARK_SKILL": ("신규", "신규", "평행·기계식·후진 주차가 부담돼 다른 곳을 찾은 적"),
    "S_PARK_REMOTE": ("C-12", "현행(간접)", "좁은 칸에서 주로 한 대처"),
    "S_ENTERTAIN": ("C1-5", "현행(간접)", "차량 화면 앱·콘텐츠 설치 경험"),
    "S_PET": ("신규", "신규", "반려동물을 태운 날"),
    "S_HANDOVER": ("C-7b", "현행", "차를 넘길 때 한 일"),
}


def cv_scores(X, y, cols):
    clf = LogisticRegression(max_iter=300, class_weight="balanced")
    pred = cross_val_predict(clf, X[:, cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=A.SEED))
    return balanced_accuracy_score(y, pred), accuracy_score(y, pred), pred


def main():
    df = A.load()
    seg = pd.read_csv(A.OUT / "post_segment.csv")
    df = df.merge(seg, on="post_id")
    codes = [c for c, _ in Counter(x for xs in df.situation for x in set(xs)).most_common(A.TOP_CODES)]
    X = np.array([[int(c in xs) for c in codes] for xs in df.situation])
    m = (X.sum(1) >= 2) & (df.segment.values >= 0)
    X, y = X[m], df.segment.values[m]
    full_bal, full_acc, _ = cv_scores(X, y, list(range(len(codes))))
    chosen, rows = [], []
    for step in range(MAX_ITEMS):
        best = max((c for c in range(len(codes)) if c not in chosen), key=lambda c: cv_scores(X, y, chosen + [c])[0])
        chosen.append(best)
        bal, acc, pred = cv_scores(X, y, chosen)
        code = codes[best]
        item, status, text = SURVEY.get(code, ("?", "?", ""))
        # 이 코드가 가장 많이 가리키는 세그먼트(코드 보유 시 세그먼트 비율의 lift)
        share = pd.Series(y[X[:, best] == 1]).value_counts(normalize=True)
        base = pd.Series(y).value_counts(normalize=True)
        top_seg = (share / base).idxmax()
        rows.append({"rank": step + 1, "code": code, "name": X1.A_SIT.get(code, code), "balanced_acc": round(bal, 3), "acc": round(acc, 3),
                     "points_to_segment": int(top_seg), "lift": round(float((share / base).max()), 2),
                     "survey_item": item, "status": status, "item_text": text})
        print(f"{step + 1:>2} {code:<16} 균형정확도 {bal:.3f} → 세그 {top_seg} · {item} ({status})")
    # 세그먼트별 재현율(최종 12개)
    per = {int(s): round(float((pred[y == s] == s).mean()), 3) for s in np.unique(y)}
    meta = {"n": int(len(y)), "k": int(len(np.unique(y))), "chance_balanced": round(1 / len(np.unique(y)), 3),
            "full22_balanced": round(full_bal, 3), "full22_acc": round(full_acc, 3), "recall_by_segment_12": per}
    out = pd.DataFrame(rows)
    out.to_csv(X1.OUT / "s3_survey_items.csv", index=False, encoding="utf-8-sig")
    (X1.OUT / "s3_survey_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    import yaml
    cb = yaml.safe_load((A.ROOT / "config/codebook_v2.yaml").read_text(encoding="utf-8"))
    X1.A_SIT.update({k: v.split(":")[0] for k, v in cb["families"]["situation"]["values"].items()})
    main()
