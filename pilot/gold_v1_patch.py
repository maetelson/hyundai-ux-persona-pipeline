"""정답 v0 → v1: 코드북 v1 규칙에 맞춰 바뀌는 건만 덮어쓴다. 근거 인용 검증 통과 시에만 저장.
  python pilot/gold_v1_patch.py   → data/gold/gold_v1_claude.jsonl
"""
import json
from pathlib import Path

from gold_append import errors

ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "data/gold/gold_v0_claude.jsonl"
V1 = ROOT / "data/gold/gold_v1_claude.jsonl"

EXCL = {"layer": "EXCLUDE", "situation": [], "journey": [], "attitude": [], "evidence": {}}
PATCH = {
    # FoD 판매 대상 기능(원격주차) 요구·불만은 출고 옵션이어도 A_FOD
    "g061": {"layer": "A_FOD", "attitude": ["T_WANT"], "evidence": {"S_PARK_REMOTE": "스마트원격주차 위젯이 왜 없을까요?", "T_WANT": "스마트원격주차 위젯이 왜 없을까요?"}},
    "g133": {"layer": "A_FOD", "journey": ["J_CHECK"], "attitude": ["T_WANT"], "evidence": {"S_PARK_REMOTE": "주차보조로 앞 뒤 차량 이동이 가능하더군요", "J_CHECK": "왜 마이현대 어플에 주차보조 기능이 추가되지 않는지 궁금합니다", "T_WANT": "왜 같은 옵션을 갖고있는 차량간에 어플 사용에 있어 차등을 두는지"}},
    "g201": {"layer": "A_FOD", "attitude": ["T_WANT"], "evidence": {"S_PARK_REMOTE": "아이오닉9 앱으로도 원격주차 가능하게 업데이트 요청 드립니다", "T_WANT": "아이오닉9 앱으로도 원격주차 가능하게 업데이트 요청 드립니다"}},
    "g202": {"layer": "A_FOD", "journey": ["J_CHECK"], "evidence": {"S_PARK_REMOTE": "마이현대 어플로 원격스마트주차보조", "J_CHECK": "싼타페mx5도 같은소프트웨어인데 업데이트. 얘정이실가요?"}},
    "g082": {"layer": "A_FOD", "attitude": ["T_WANT"], "evidence": {"S_PARK_REMOTE": "차량 전,후진 기능을 앱에서도 사용 할수 있도록 해주세요", "T_WANT": "차량 전,후진 기능을 앱에서도 사용 할수 있도록 해주세요"}},
    "g184": {"layer": "A_FOD", "journey": ["J_USE"], "attitude": ["T_DISAPPOINT"], "evidence": {"S_PARK_REMOTE": "원격스마트주차보조 기능이 안됩니다", "J_USE": "처음 시동걸때 자주 발생하네요", "T_DISAPPOINT": "원격스마트주차보조 기능이 안됩니다"}},
    "g242": {"layer": "A_FOD", "journey": ["J_USE"], "attitude": ["T_DISAPPOINT"], "evidence": {"S_PARK_REMOTE": "지하주차장에서 사용하려하는데 실패했다는 문구만 나오고", "J_USE": "지하주차장에서 사용하려하는데 실패했다는 문구만 나오고", "T_DISAPPOINT": "실패했다는 문구만 나오고"}},
    "g207": {"layer": "A_FOD", "attitude": ["T_WANT"], "evidence": {"S_PARK_TIGHT": "주차할때 공간이 협소할때 종종 원격주차를 했는데", "S_PARK_REMOTE": "주차할때 공간이 협소할때 종종 원격주차를 했는데", "S_REMOTE_CTRL": "휴대폰에서는 컨트롤이 안되쥬????", "T_WANT": "휴대폰에서는 컨트롤이 안되쥬????"}},
    # 스토어 언급 → A_FOD
    "g027": {"layer": "A_FOD", "journey": ["J_USE"], "attitude": ["T_SATISFIED"], "evidence": {"S_REMOTE_CTRL": "몇초이상 누르라는 점은 좀 불편 한것 같습니다", "J_USE": "어플내 스토어 사용등 시인성이 좋습니다", "T_SATISFIED": "어플내 스토어 사용등 시인성이 좋습니다"}, "note": "v1: 스토어 언급 → A"},
    "g118": {"layer": "A_FOD", "situation": [], "journey": ["J_DISCOVER"], "attitude": ["T_SATISFIED"], "evidence": {"J_DISCOVER": "덤으로^^ 기아 스토어 몰까지", "T_SATISFIED": "기아 앱 정말 넘버원 별 5개입니다"}, "note": "v1: 스토어 언급 → A"},
    # 커넥티드 앱 사용 경험 → B
    "g042": {"layer": "B_SITUATION", "situation": ["S_REMOTE_CTRL"], "evidence": {"S_REMOTE_CTRL": "기존앱보다 직관성떨어지고 UI 무지 불편"}, "note": "v1: 앱 UI 경험 → B"},
    "g187": {"layer": "B_SITUATION", "situation": ["S_REMOTE_CTRL"], "evidence": {"S_REMOTE_CTRL": "여러 앱을 하나로 통합하여 서비스이용이 많이 편해지고"}, "note": "v1: 앱 사용 경험 → B"},
    # 칼럼이라도 작성자 판단이 있으면 A
    "g055": {"layer": "A_FOD", "situation": ["S_ADAS"], "attitude": ["T_CALCULATE"], "evidence": {"S_ADAS": "지금 FSD는 누구에게 필요한가", "T_CALCULATE": "수백만 원짜리 일시불 대신 월 구독제를 활용하는 편이"}, "note": "v1: 칼럼+작성자 판단 → A"},
    "g299": {"layer": "A_FOD", "situation": ["S_NAV"], "attitude": ["T_DISTRUST"], "evidence": {"S_NAV": "내비게이션 업데이트 종료되는", "T_DISTRUST": "구독서비스제도가 확대 정착되리라고 봅니다 절판은 싫어요!"}, "note": "v1: 정보글+작성자 반감 → A"},
    # 판매 목적 홍보 → EXCLUDE
    **{i: {**EXCL, "note": "v1: 판매 목적 홍보 → EXCLUDE"} for i in ("g006", "g051", "g104", "g116", "g192", "g206", "g234", "g239", "g264")},
    # 새 코드: S_NAV, S_MAINTAIN
    "g110": {"situation": ["S_NAV"], "evidence": {"S_NAV": "몇 천만원짜리 내비게이션이 몇만 원짜리보다 못해요?"}, "confidence": "high"},
    "g044": {"situation": ["S_MAINTAIN"], "evidence": {"S_MAINTAIN": "소모품관리에 항목을 추가도 할수 잇게 해주세요"}, "confidence": "high"},
    "g154": {"situation": ["S_MAINTAIN", "S_COST"], "evidence": {"S_MAINTAIN": "항균필터 교체시기는 언제일까요?", "S_COST": "블루멤버스 포인트로 결재했어요"}, "confidence": "high"},
    "g222": {"situation": ["S_MAINTAIN", "S_COST"], "evidence": {"S_MAINTAIN": "블루링크 블루투스 마이크 다 정상작동하는걸 보고", "S_COST": "데크(70,000원)"}, "confidence": "high"},
    "g297": {"situation": ["S_MAINTAIN"], "evidence": {"S_MAINTAIN": "급감속, 급브레이크, 급과속한 시간이 나왔는데"}, "confidence": "high"},
    "g089": {"situation": ["S_EV", "S_MAINTAIN", "S_SECURITY"], "evidence": {"S_EV": "첫 전기차로 중고 코나 SX2 EV를 운용해보게 되었습니다만", "S_MAINTAIN": "블루링크 12V 배터리 항목은 정상으로 확인되어", "S_SECURITY": "블랙박스 저전압 차단이"}},
    # 새 코드: J_TRIAL
    "g113": {"journey": ["J_TRIAL"], "evidence": {"S_ADAS": "모델3 퍼포먼스 EAP 후기", "S_PARK_REMOTE": "스마트 서먼 (SMART SUMMON ?) 자동 차량호출 후기", "J_TRIAL": "구독 5천원에 하루동안 EAP 활용 후", "T_DISAPPOINT": "저는 이 기능보단 차라리 오토파일럿이낫습니다"}},
    "g204": {"journey": ["J_TRIAL", "J_PURCHASE"], "evidence": {"S_ADAS": "구독으로 fsd 써보고 있는데", "S_REPLACE": "중고 22모y 1주일차라", "J_TRIAL": "구독으로 fsd 써보고 있는데", "J_PURCHASE": "오늘이 마지막 구매일이라 일시불 구매하려고 합니다"}},
    "g276": {"journey": ["J_TRIAL"], "evidence": {"S_ADAS": "FSD Lite를 구독해서 써봤는데", "S_REPLACE": "모델 Y로 바꿀까 고민중인데", "J_TRIAL": "FSD Lite를 구독해서 써봤는데", "T_DISAPPOINT": "전 좀 느리고 답답해서 잘 안쓰게 되더라구요"}},
    "g149": {"journey": ["J_TRIAL", "J_SETTLE", "J_PURCHASE"], "evidence": {"S_ADAS": "FSD Lite 4일 사용하고 환불했습니다", "J_TRIAL": "FSD Lite 4일 사용하고 환불했습니다", "J_SETTLE": "FSD Lite 4일 사용하고 환불했습니다", "J_PURCHASE": "구독하는 방법을 택하기로 했습니다", "T_CALCULATE": "가격이 FSD와 같은거 실화 ??"}},
    "g188": {"journey": ["J_DISCOVER", "J_TRIAL", "J_PURCHASE"], "evidence": {"S_PERSONAL": "하늘색으로 시원한 느낌이 나는 아르헨티나 테마", "J_DISCOVER": "월드컵 테마가 나왔다고해서 기아 커넥트샵 사용도 해볼겸", "J_TRIAL": "저처럼 커넥트샵 처음 사용해보시는 분들이라면 연습겸", "J_PURCHASE": "아르헨티나 테마를 다운받았습니다", "T_SATISFIED": "잘 적용했다는 생각이 들었습니다", "T_DISAPPOINT": "약간 아쉽긴합니다"}},
}

if __name__ == "__main__":
    labels = {r["id"]: r for r in map(json.loads, V0.open(encoding="utf-8"))}
    bad = 0
    for i, patch in PATCH.items():
        new = {**labels[i], **patch, "id": i}
        errs = errors(new)
        if errs:
            bad += 1
            print(f"  ✗ {i}: " + " | ".join(errs))
        else:
            labels[i] = new
    V1.write_text("".join(json.dumps(labels[k], ensure_ascii=False) + "\n" for k in sorted(labels)), encoding="utf-8")
    print(f"v1 저장 {len(labels)}건, 수정 {len(PATCH) - bad}건, 오류 {bad}건")
