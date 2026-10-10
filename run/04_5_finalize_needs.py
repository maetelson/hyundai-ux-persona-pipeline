"""니즈 계층 초안 정리 → config/need_hierarchy_v1.yaml (사람 검토 대상).

규칙: 구성원 3건 미만 니즈는 제외하고 기록만 남긴다 / 해시태그 중복은 뒤쪽에 코드 꼬리표를 붙인다.
  python run/04_5_finalize_needs.py
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MIN_MEMBERS = 3

if __name__ == "__main__":
    draft = yaml.safe_load((ROOT / "config/need_hierarchy_draft.yaml").read_text(encoding="utf-8"))
    seen, dropped, out = {}, [], {}
    for code, v in draft.items():
        keep = []
        for n in v["needs"]:
            if n["n_members"] < MIN_MEMBERS:
                dropped.append(f"{n['id']} {n['hashtag']} ({n['n_members']}건)")
                continue
            if n["hashtag"] in seen:
                n["hashtag"] = f"{n['hashtag']}_{code.split('_', 1)[1].lower()}"
            seen[n["hashtag"]] = code
            keep.append(n)
        out[code] = {**v, "needs": keep}
    header = ("# 니즈 계층 v1 (2026-10-10, 오픈 코딩 2,000+보충 → Opus 도출 → 규칙 정리). 사람 검토 필요.\n"
              "# 1차 = situation 대분류, 2차 = situation 코드, 3차 = needs[].id (#hashtag)\n"
              f"# 제외(구성원 {MIN_MEMBERS}건 미만): " + "; ".join(dropped) + "\n")
    (ROOT / "config/need_hierarchy_v1.yaml").write_text(header + yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"3차 니즈 {sum(len(v['needs']) for v in out.values())}개, 제외 {len(dropped)}개: {dropped}")
