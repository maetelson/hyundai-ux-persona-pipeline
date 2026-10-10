# 결과물 구조화 계획: 롯데마트형 페르소나 산출 (2026-10-10)

> 목표: 롯데마트 보고서(2026-03-30)와 같은 형태의 결과물(퍼널 → 코드 체계 → 인사이트 → 페르소나 요약 → 카드 → 아이디어)을 **표(테이블) → 렌더링** 구조로 재현 가능하게 만든다.
> 비교 대상: 이 레포의 현재 자산, `maetelson/persona-pipeline`의 모듈.

---

## 0. 데이터 사용 가능 범위 (전제)

| 데이터 | 건수 | 분석 사용 | 근거 |
|---|---|---|---|
| 네이버 검색 API(블로그·카페 스니펫) | 554,435 (규칙 통과 고유 363,764) | **불가** (허락·라이선스 전까지) | 검색 API 특약 2.3: 복사·저장·캐싱 금지, AI 입력 금지(가공·파생물 포함). 2.4: 서버 보관은 이력 조회 목적 최대 21일. 연구 예외 없음 |
| 기아 커넥트 스토어 리뷰 | 2,982 | 조건부 (기아 이용약관 확인 후) | 공개 리뷰 페이지 |
| App Store RSS (기아·마이현대) | 600 | 조건부 (Apple 피드, 상대적 안전) | 공식 RSS |
| Google Play (3앱) | 8,508 | 조건부 (비공식 스크래퍼, 회색지대) | — |

→ **Run A**: 지금 쓸 수 있는 리뷰 약 1.2만 건으로 구조 전체를 끝까지 한 번 돌린다.
→ **Run B**: 허락받은 데이터나 라이선스 데이터(네이버, 텍스톰·썸트렌드 등)나 커뮤니티 데이터가 확보되면 **같은 표 구조**에 넣어 다시 돌린다.
구조는 데이터 출처와 무관하게 설계한다.

---

## 1. 롯데마트 결과물 해부 → 필요한 표

| 롯데마트 페이지 | 화면 요소 | 그 화면을 만드는 표 |
|---|---|---|
| p3 수집·분석 프로세스 | 4단 퍼널 수치 | `funnel` |
| p4~5 크롤링 설계·판정 기준 | 키워드 목록, 유효/가비지 예시, 플랫폼별 유효 건수 | `seed_registry`, `layer_rules`(코드북), `source_distribution` |
| p6 분석 프로세스 | 니즈 트랙 / 데모 트랙 다이어그램 | (정적) |
| p7, p27~29 데모 클루 | 클루 우선순위, 키워드 → 추정값 표 | `demo_clue_rules`(설정), `post_demo` |
| p8 코드 체계 | 대분류 10개 × 72코드 | `need_hierarchy` |
| p10 인사이트 요약 | 헤드라인 + 근거 수치 1줄 × 6 | `insights` |
| p11 페르소나 도출 | 페르소나별 코드 네트워크(버블·선) | `theme_edges`, `theme_nodes` |
| p12 페르소나 요약 | 이름·비중·한 줄 프로필·#해시태그 | `persona_summary` |
| p13 페르소나 카드 | 분석 수·비중 / INSIGHT 2줄 / 인구통계 상위 2 / 교차언급 상위 5 / 핵심 특징 3 / 니즈 상세(#, %, 건수, 설명, 인용 6) | `persona_summary`, `persona_profile`, `persona_cooccurrence`, `persona_features`, `persona_needs`, `persona_quotes` |
| p14~24 파일럿 아이디어 | 아이디어 제목 + 연결 #해시태그 + 개념 + 구체안 3 | `ideas`(+ `opportunities`) |
| p25 Next step | CRM 결합 | (설문 연결 계획) |

---

## 2. 현재 자산 · persona-pipeline · 할 일 비교

| 결과물 요소 | 이 레포 현재 | persona-pipeline | 할 일 |
|---|---|---|---|
| 수집·시드 | ✅ 시드 뱅크 9종, 수집기 3종, 시드 수율 | `collectors/*`, `config/seeds`, `seed_validation` | 유지. 리뷰 수집기를 `RawRecord` 스키마로 통일 |
| 정규화 | ❌ 소스마다 다른 JSON | `normalizers/*`, `02_normalize_all` | **이식**: 공통 `posts` 표 |
| 정제·필터 | 🔺 규칙 판정(kept)만 | `invalid_filter`, `dedupe`, `03_filter_valid`, `03_5_prefilter_relevance` | **이식 + 신규**: MinHash, 정보성 필터 |
| 에피소드 분할 | ❌ | `episodes/builder`, `04_build_episodes` | 리뷰는 짧아서 Run A에선 생략. Run B에서 이식 |
| 코드북 | ✅ v2 (layer·situation·journey·attitude·life_stage) | `codebook.yaml`, `labeling_policy.yaml` | 정의·선택 조건·unknown 조건·예시를 repo 형식으로 보강 |
| 니즈 계층(#해시태그) | ❌ | ❌ (axis discovery가 유사) | **신규**: 오픈 코딩 → 결과 문장 군집 → `need_hierarchy` |
| LLM 라벨링 실행기 | 🔺 `model_eval.py`(평가용) | `llm_labeler`(배치), `labelability`, `batch_builder/merge` | **이식**: 배치 실행·병합 + 근거 검증 |
| 라벨 품질 | ✅ 정답 300 + 근거 검증 + 모델 비교 | `labeling/quality`, `unknown_reasons`, `audit` | **이식**: unknown 원인, 코드별 커버리지 |
| 데모 클루 | ❌ | `record_access.get_record_demo` (단순) | **신규**: `demo_clue_rules.yaml` + 추출기(본문 > 메타(차종·상품) > 채널, 출처 열 분리) |
| 테마 네트워크 | ❌ | `cooccurrence.py` | **이식 + 수정**: PMI·lift, Leiden |
| 세그먼트 | ❌ | `bottleneck_clustering`(BI 전용), `persona_axes` | **신규**: LCA·k-modes + 안정성 |
| 페르소나 요약·프로필 | ❌ | `persona_service._build_persona_summary_df / _axes / _pains / _cooccurrence` | **이식(구조만)**: 시트 생성 로직을 우리 코드로 교체 |
| 대표 인용 | ❌ | `example_selection` (점수·중복 제거·출처 다양성) | **이식**: 롯데마트 "인용 6개" 공급 |
| 이름·메시지 | ❌ | `persona_messaging.yaml`, `_apply_persona_name_policy` | **이식**: 해시태그 이름 규칙 + 인사이트 필드 |
| 품질 게이트·등급 | 🔺 PLAN에 정의만 | `quality_status`, `pipeline_thresholds`, readiness tier | **이식**: 게이트 판정 → `quality_checks` |
| 워크북 | ❌ | `xlsx_exporter`(시트 검증, 서식, README 시트) | **이식**: 시트 목록만 교체 |
| 카드·덱 렌더링 | ❌ | ❌ | **신규**: 표 → HTML 카드(→ PDF/PPT) |
| 스냅샷 비교 | ❌ | `17_analysis_snapshot --compare-latest` | Run A ↔ Run B 비교에 이식 |
| 테스트 | ❌ | 63개 | 각 단계마다 최소 1개 회귀 테스트 |

범례: ✅ 있음 · 🔺 일부 · ❌ 없음

---

## 3. 데이터 모델 (모든 결과물의 원천)

> 카드와 덱은 이 표들을 렌더링만 한다. 손으로 쓰는 문장은 `insights`와 `ideas`의 검토 필드뿐이다.

### 3.1 기반 표

| 표 | 키 | 주요 열 |
|---|---|---|
| `posts` | post_id | source, source_tier, url, created_at, author_hash, text, meta(차종·상품·앱·카페명), seed, seed_code |
| `funnel` | stage × source | n_in, n_out, drop_reason_top3 |
| `post_labels` | post_id | layer, situation[], journey[], attitude[], life_stage, gender, outcome_statement, intensity, frequency, wtp_signal, substitute, sentiment, actor, evidence{code: quote}, confidence, model, codebook_version |
| `post_demo` | post_id × dimension | value, basis(본문/메타/채널), clue_quote |
| `need_hierarchy` | need_l3 | need_l2(situation 코드), need_l1(대분류), hashtag(#…), definition, example_outcomes |
| `post_needs` | post_id × need_l3 | score |

### 3.2 분석 표

| 표 | 키 | 주요 열 | 롯데마트 대응 |
|---|---|---|---|
| `theme_nodes` / `theme_edges` | code / code_a × code_b | prevalence / cooc_n, pmi, lift, community | p11 |
| `segments` | segment_id | method, k, size_n, size_pct, stability_jaccard, top_source_share, status(승격/후보) | p12 비중 |
| `post_segment` | post_id | segment_id, prob | — |
| `persona_profile` | segment × dimension × value | share, n, basis, gate_pass | p13 인구통계 상위 2 |
| `persona_needs` | segment × need_l3 | share_in_segment, n, rank, description | p13 니즈 상세 |
| `persona_cooccurrence` | segment × need_a × need_b | pct_of_segment, n | p13 교차언급 막대 |
| `persona_features` | segment × rank | metric_text("메뉴지옥 50% + 워킹맘 28% 33% 교차"), interpretation("→ …"), source_tables | p13 핵심 특징 |
| `persona_quotes` | segment × need_l3 × rank | post_id, quote, quality_score, source | p13 인용 6개 |
| `journey_barriers` | segment × journey × attitude | n, pct, ci_low, ci_high, hypothesis_flag | (FoD 추가) |
| `opportunities` | segment × need_l3 | text_signal_score, n, intensity_mean, neg_pct, substitute_pct | p14 근거 |
| `persona_summary` | segment | name, hashtag_name, one_liner, insight_1, insight_2, size_n, size_pct, readiness_tier, evidence_tier | p12, p13 머리 |
| `insights` | insight_id | headline, evidence_line, metric_refs, reviewer_ok | p10 |
| `ideas` | idea_id | segment, linked_needs[], title, concept, specifics[3], touchpoint(앱/IVI), pricing_form, experiment | p14~24 |
| `quality_checks` | gate | value, threshold, status | (repo) |

---

## 4. 단계와 파일 배치 (persona-pipeline 구조 차용)

```
run/
  01_collect.py           # 시드 뱅크 수집 (기존 pilot/naver_search.py, collect_keyless.py 이전)
  02_normalize.py         # → data/stage/posts.parquet        [repo normalizers 이식]
  03_filter.py            # 정제·MinHash·정보성 → posts_valid   [repo invalid_filter 이식]
  04_open_code.py         # 표본 결과 문장 → 군집 → need_hierarchy 초안 (사람 검토)
  05_label.py             # 배치 LLM → post_labels, 근거 검증   [repo llm_labeler 배치 이식]
  05_5_demo.py            # 데모 클루 → post_demo
  06_analyze.py           # theme_*, segments, persona_* , journey_barriers, opportunities
  07_export_xlsx.py       # 워크북                             [repo xlsx_exporter 이식]
  08_render_cards.py      # persona_summary 등 → HTML 카드·요약표·인사이트 장
config/
  codebook_v3.yaml, demo_clue_rules.yaml, need_hierarchy.yaml, persona_naming.yaml, thresholds.yaml
data/stage/*.parquet, data/output/persona_workbook.xlsx, data/output/cards/*.html
```

---

## 5. 카드 생성 규칙 (롯데마트 p13 재현)

| 카드 영역 | 계산 | 표시 규칙 |
|---|---|---|
| 분석 수·비중 | `segments.size_n`, `size_pct` | Run B부터 설문 비중을 나란히 표기 |
| INSIGHT 2줄 | LLM 초안 → 사람 확정. 문장마다 `persona_features`·`persona_needs` 행 ID 연결 | 근거 ID가 없으면 삭제 |
| 인구통계 상위 2 | `persona_profile`에서 dimension별 상위 2(본문 단서) | 게이트 미달은 "판단 불가". Run A는 차종·EV·상품·앱을 대체 차원으로 쓰고 그렇게 표기 |
| 교차언급 상위 5 | `persona_cooccurrence`에서 pct 상위 5쌍 | 쌍 건수 ≥ 20 |
| 핵심 특징 3 | `persona_features`: 규칙으로 후보 생성(최대 교차, 특이 lift 차원, 계절·전환) → 사람이 3개 선택 | "수치 → 해석" 형식 |
| 니즈 상세 | `persona_needs` 상위 3~5 + `persona_quotes` 5~6개 | 인용은 원문 그대로, 출처 다양성 보장 |
| #해시태그 이름 | `need_hierarchy.hashtag` + 페르소나 이름 규칙(`persona_naming.yaml`: #상황+정체성) | 중복 이름 방지(repo `_dedupe_persona_names`) |

인사이트 장(p10)은 `insights`에서 6~8개를 쓴다. 헤드라인은 한 문장이고, 근거 줄에는 `metric_refs` 수치를 자동으로 채운다.

---

## 6. Run A 기대치 (리뷰 약 1.2만 건)

| 항목 | 예상 | 대응 |
|---|---|---|
| 층 구성 | 기아 리뷰 = A층 100%, 앱 리뷰 = A/B 혼합 | A층 저니·태도 중심 결과 |
| 세그먼트 | 3~5개 (예: 테마 감성 소비, 원격주차 실용, 회생제동 EV 운전, 유료 커넥티드 불만, 체험 후 결정) | 안정성 게이트로 확정 |
| 인구통계 | 본문 생애단계 단서 약 2% → 대부분 판단 불가 | 차종·EV·상품·앱 메타를 대체 프로필로 쓰고, 그렇다고 명시 |
| 출처 편향 | 기아 리뷰는 구매자·이벤트 편중, 테마 79% | 출처 지배 게이트, "구매자 관점" 경계 문구 |
| 결과물 | 롯데마트형 전체 구조 1벌(카드 3~5장 + 요약 + 인사이트 + 아이디어) | 구조 검증용 → Run B에서 확장 |

---

## 7. 실행 순서

1. 데이터 정리: 네이버 데이터 격리 또는 삭제 결정. 기아·Google·Apple 약관 확인
2. `02_normalize` → `posts` (리뷰 1.2만)
3. `04_open_code` → `need_hierarchy` 초안 → 팀 검토
4. 정답 세트를 리뷰 기준으로 재구성(네이버 180건 제외 → 리뷰에서 보충) → 모델 재측정
5. `05_label`, `05_5_demo`
6. `06_analyze` (theme → segments → persona_* → barriers → opportunities)
7. `07_export_xlsx`, `08_render_cards` → 카드·요약표·인사이트 초안
8. 팀 리뷰 → `insights`·`ideas` 확정 → Run A 결과물
9. Run B 데이터 확보 시 1~8 반복, 스냅샷 비교
