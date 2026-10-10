# hyundai-ux-persona-pipeline

**온라인 텍스트로 만든 FoD(Features on Demand) 니즈·순간 페르소나**
DECK × 현대자동차 UX 프로젝트

공개 리뷰와 커뮤니티 글 약 37만 건을 모았고, 시간 창과 할당 표본을 거쳐 11만 건을 LLM으로 라벨링했습니다. 그 결과로 상황·니즈 기반 세그먼트 6개를 도출했습니다. 이 결과는 후속 설문(n≈1,500)의 문항 설계와 세그먼트 연결에 쓰입니다.

> ⚠️ 원자료(`data/`)는 저장소에 없습니다. 네이버 검색 결과는 이용 약관상 저장·AI 입력이 제한되어 있어 팀 결정으로 **2026-10-30까지 삭제**합니다([데이터 처리 결정](docs/PLAN.md)). 보고서(`report.html`)에는 원문 인용이 들어 있으므로 외부에 공개하지 않습니다.

---

## 주요 결과

| 세그먼트 | 이름 | 텍스트 비중 | 대표 상황 코드 | FoD 직접 언급 |
|---|---|---|---|---|
| 1 | 커넥티드 관리자 `#먹통원격좌절러` | 34.7% | S_REMOTE_CTRL, S_MAINTAIN, S_PERSONAL | 26.5% |
| 3 | 구매 결정자 `#옵션고민첫차족` | 32.2% | S_COST, S_REPLACE, S_ADAS | 25.6% |
| 2 | 아이 동승 부모 `#카시트두개공간맘` | 12.7% | S_CHILD, S_LONG_DRIVE, S_FAMILY_SHARE | 5.2% |
| 4 | 공간 활용족 `#캠핑짐테트리스` | 9.6% | S_CARGO, S_LEISURE, S_PET | 0.4% |
| 0 | 사제 보충파 `#신차꾸밈합리파` | 6.6% | S_AFTERMARKET, S_NAV, S_SECURITY | 4.3% |
| 5 | 문콕 방어자 `#범인찾기억울러` | 4.2% | S_PARK_TIGHT, S_PARK_REMOTE, S_HANDOVER | 11.7% |

**신뢰도 요약**

| 지표 | 값 |
|---|---|
| 세그먼트 수 k | 6 (안정성 규칙 v2, 아래 참고) |
| 부트스트랩 Jaccard 안정성 (세그먼트별) | 0.66 / 0.74 / 0.92 / 0.81 / 0.72 / 0.76 |
| LCA vs KMeans ARI | 0.43 |
| 혼합 vs 엄격 시간 창 ARI | 0.91 |
| 테마 네트워크 모듈성 Q | 0.54 (노드 22, 커뮤니티 4) |
| 라벨링 모델 (Haiku 5.5 medium) 상황 코드 F1 | 0.84 (임시 정답 300건 기준) |

비중은 **온라인 글에서의 비중**입니다. 인구 비율이 아니며, 인구 비율은 설문으로 추정합니다.

---

## 파이프라인

```
수집 ─▶ 정규화·중복 제거 ─▶ 시간 창 ─▶ 할당 표본 ─▶ 오픈 코딩 ─▶ LLM 라벨링 ─▶ 분석 ─▶ 보고서
pilot/   02_normalize    02_5_time_window  03_sample  04_open_code  05_label     06_*     08_report
                                                       04_5_finalize 05_5_demo
```

| 단계 | 스크립트 | 방법 | 산출 |
|---|---|---|---|
| 수집 | `pilot/naver_search.py`, `pilot/collect_keyless.py` | NAVER API HUB 검색(blog, cafearticle), 소스별 시드 뱅크, 규칙 기반 관련성 판정, 저수율 조기 중단. 기아 커넥트 스토어·앱 리뷰 | `data/raw/` |
| 정규화 | `run/02_normalize.py` | URL·본문 해시 중복 제거, 15자 미만 제외 | `posts.parquet` 370,775건 |
| 시간 창 | `run/02_5_time_window.py` | 2021-10-10 이후. 날짜 없는 카페 글은 최신순 결과와 카페별 글번호 기준선으로 판정(혼합). 엄격 모드는 민감도 분석용 | `date_status` |
| 표본 | `run/03_sample.py` | A층(FoD 직접)·리뷰는 전수, 생애단계·상황 코드별 상한, 채널 30% 상한 | `corpus.parquet` 110,374건 |
| 오픈 코딩 | `run/04_open_code.py`, `04_5_finalize_needs.py` | 표본 2,000건 → 결과 문장 → 상황 코드별 니즈 귀납(TnT-LLM 방식, Opus 5.5) → 뒤 절반으로 포화 확인 | `config/need_hierarchy_v1.yaml` (니즈 102개) |
| 라벨링 | `run/05_label.py` | Message Batches API, Haiku 5.5 medium, 10건 묶음, 구조화 출력, 근거 인용을 원문 부분 문자열로 검증 | `post_labels.parquet` |
| 인구 단서 | `run/05_5_demo.py` | 규칙 기반 생애단계·성별 단서(본문/메타/채널 근거 분리) | `post_demo.parquet` |
| 분석 | `run/06_1_select_k.py`, `run/06_analyze.py` | Wilson CI 유병률, PMI 네트워크 + Louvain, LCA(stepmix), 부트스트랩 안정성, KMeans 교차 확인 | `data/output/tables/` |
| 추가 분석 | `run/07_extra.py` | 가격 앵커(금액 파싱), 스토어 상품 성과(Wilson CI), 대체재 경쟁 지도(lift·log-odds), 텍스트 Kano 추정(부트스트랩 안정성). API 호출 없음 | `data/output/tables/extra/` |
| 보고서 | `run/08_report.py` | 단일 HTML. 최종 정리본(롯데마트 AI 페르소나 구조 참고) + 방법론 M1–M8(방법·변수·신뢰도·한계) | `data/output/report.html` |

### 코드북 (`config/codebook_v2.yaml`)

- **layer**: `A_FOD`(FoD 직접) / `B_SITUATION`(운전 상황) / `INFO` / `EXCLUDE`
- **상황** 23개(`S_*`), **저니** 10개(`J_*`), **태도** 10개, **생애단계** 6개, 성별
- **세부 속성**: 결과, 강도(0–3), 빈도, 지불 의향 신호, 대체 수단, 감성, 행위자
- 라벨마다 근거 인용(evidence)이 필수입니다. 원문에 없는 인용은 코드째 제거합니다.

---

## 재현

### 환경

Python 3.13에서 테스트했습니다.

```bash
pip install anthropic==1.12.1 pandas==3.0.2 pyarrow==23.0.1 numpy==2.4.4 scikit-learn==1.9.1 stepmix==3.0.0 networkx==3.7 kiwipiepy==0.24.0 requests==2.34.2 PyYAML==6.0.3 google-play-scraper==1.2.7
```

```bash
cp .env.example .env
```

`.env`에 들어갈 키: `NAVER_HUB_CLIENT_ID`, `NAVER_HUB_CLIENT_SECRET`, `NAVER_DAILY_BUDGET`, `ANTHROPIC_API_KEY`

### 실행 순서

```bash
python pilot/collect_keyless.py
python pilot/naver_search.py collect config/seeds/naver_cafe.yaml config/seeds/naver_blog.yaml config/seeds/naver_situation.yaml config/seeds/naver_lifestage.yaml
python run/02_normalize.py
python run/03_sample.py
python run/02_5_time_window.py hybrid
python run/04_open_code.py statements
python run/04_open_code.py induce all
python run/04_5_finalize_needs.py
python run/05_label.py submit all
python run/05_label.py collect
python run/05_5_demo.py
python run/06_1_select_k.py
python run/06_analyze.py
python run/06_analyze.py strict
python run/07_extra.py
python run/08_report.py
```

- `05_label.py collect`는 배치가 끝난 뒤 실행합니다. `status`로 진행 상태를 확인할 수 있습니다.
- 수집기는 일일 호출 장부(`data/naver_calls.json`)와 이어받기를 지원합니다.

### 비용과 시간 (실측, 2026-10)

| 항목 | 비용 |
|---|---|
| 라벨링 11만 건 (배치 50% 할인) | 약 $21 |
| 오픈 코딩 | 약 $6.5 |
| 모델 비교 시험 | 약 $5.5 |
| NAVER API | 무료 한도 안 (일 25,000건) |

---

## 검증과 설계 결정

**모델 선택** (`pilot/model_eval.py`, 임시 정답 300건, 코드북 v0)

| 모델 | 상황 F1 | 저니 F1 | 1천 건당 $ |
|---|---|---|---|
| Haiku 5.5 medium, 10건 묶음 | 0.84 | 0.64 | 0.32 |
| Sonnet 5.5 low | 0.86 | 0.64 | 2.67 |
| Opus 5.5 low | 0.88 | 0.70 | 5.10 |

Sonnet보다 F1이 0.02 낮지만 비용이 1/8이라서 Haiku medium을 골랐습니다. 정답은 Opus가 만든 초벌 라벨이라 Opus에 유리한 편향이 있을 수 있습니다. 팀 검수(`data/gold/gold_v1_review_focus.csv`)는 아직 하지 않았습니다.

**k 선택 규칙 변경 (공개)**

- BIC는 k가 커질수록 계속 낮아져서 기준으로 쓸 수 없었습니다. 그래서 부트스트랩 안정성으로 k를 골랐습니다.
- 처음 규칙(v1)은 k=4를 골랐지만 페르소나로 쓰기에 너무 거칠었습니다.
- 그래서 규칙을 v2로 바꿨습니다. v2는 평균 ≥0.8이고 최소 ≥0.7인 k 가운데 가장 큰 k를 고르며, 그 결과가 k=6입니다.
- 이 변경은 결과를 본 뒤에 한 결정이라서 보고서 M6에 적어 두었습니다. 비교 수치는 `k_selection.csv`에 있습니다.

| k | BIC | 안정성 평균 | 안정성 최소 |
|---|---|---|---|
| 4 | 163,262 | 0.93 | 0.90 |
| 5 | 161,376 | 0.78 | 0.50 |
| **6** | 159,809 | 0.86 | 0.75 |
| 7 | 158,509 | 0.69 | 0.33 |

**ODI를 쓰지 않은 이유**

텍스트에는 중요도와 만족도를 따로 재는 문항이 없습니다. 그래서 보고서의 기회 점수는 "텍스트 기회 신호 지수"이고 ODI가 아닙니다. ODI는 설문 단계에서 계산합니다.

---

## 한계

- **표본 편향.** 온라인에 글을 쓰는 사람만 담깁니다. 카페 글이 많고(세그먼트별 60–98%), 전기차·수입차 커뮤니티 쪽으로 쏠려 있습니다. 채널 상한으로 일부만 보정했습니다.
- **인구 정보 부족.** 나이와 성별은 본문 단서로만 추정할 수 있습니다. 세그먼트의 연령 구성은 설문으로 확인해야 합니다.
- **카페 날짜 부재.** 카페 글은 작성일이 없어 대리 지표로 시간 창을 판정했습니다(`cafe_unknown` 포함). 엄격 모드와의 ARI 0.91로 영향이 작은 것은 확인했습니다.
- **세그먼트 경계.** 일부 세그먼트(0, 4)는 안정성이 0.7 근처라 경계가 흐릴 수 있습니다.
- **미처리 배치.** 라벨링 배치 요청 13건(약 130건)이 실패했고 다시 돌리지 않았습니다.

---

## 저장소 구조

```
config/          코드북, 니즈 계층, 관련성 규칙, 시드 뱅크, 보고서 서술
  seeds/         소스별 시드 뱅크 (FoD, 상황, 생애단계, 벤치마크, 대체재, 이벤트)
docs/
  DESIGN.md      소스·수집·라벨링 설계와 파일럿 결과
  PLAN.md        단계별 계획, 진행표, 외부 검토 반영, 데이터 처리 결정
  OUTPUT_PLAN.md 롯데마트 페르소나 구조 ↔ 산출 표 대응, 데이터 모델
pilot/           수집기, 정답 세트, 모델 비교 시험
run/             본 파이프라인 (번호 순서대로 실행)
data/            (git 제외) raw / stage / gold / eval / output
```

## 다음 단계

- 팀 검수: `need_hierarchy_v1.yaml`, `report_narrative.yaml`, 정답 세트
- 설문(n≈1,500, FoD 이용자 300명 이상 보강)
  - 상황 문항으로 설문 LCA를 돌리고, 텍스트 세그먼트와 연결합니다.
  - 실제 응답자로만 골든 질문을 학습합니다.
  - ODI·Kano·MaxDiff를 측정하고, 사후 층화 가중을 적용합니다.
- 네이버 원자료 삭제 (2026-10-30)

## 참고

- 출력 구조: 롯데마트 AI 페르소나 보고서
- 선행 파이프라인: [maetelson/persona-pipeline](https://github.com/maetelson/persona-pipeline)
- 니즈 귀납: Wan et al., *TnT-LLM: Text Mining at Scale with Large Language Models* (KDD 2024)
- 잠재 계층 분석: Morin et al., *StepMix: A Python Package for Pseudo-Likelihood Estimation of Generalized Mixture Models with External Variables* (JSS)
- 안정성: Hennig, *Cluster-wise assessment of cluster stability* (CSDA 2007)
