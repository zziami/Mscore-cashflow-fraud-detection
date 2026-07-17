# Fraud-detection-mscore

# 자금유출 기반 회계분식 탐지를 위한 한국형 변수 확장 및 모델 검증 - 단독프로젝트 (우수상 수상)

---

**비상장 외부감사대상 제조기업을 대상으로, Beneish M-Score 기반 재무왜곡 변수에 한국형 자금유출 변수(가지급금·단기대여금·단기차입금·예수금·이자수익)를 결합해, 감사보고서상 적정의견에도 은폐된 자금유출형 분식 리스크를 정량 탐지하는 머신러닝 프로젝트입니다.**

## 1. Data Collection

- 출처: TS2000(비상장 외감 제조법인 재무데이터), DART(기업개황·업종코드), 금융감독원(회계감리제재 기업), 대법원(법인회생사건 공고)
- 수집기간: 2018 ~ 2024 (新외감법 2020년 사업연도 본격 적용 전후 비교를 위해 2018년부터 수집)
- 규모: Row Data 54,173개사(246,954행) → 감리·회생폐지 분리 후 54,091개사 → 산출불가 기업 제거 후 37,365개사(178,051행)

## 2. EDA (Target Labeling & Data Quality)

- Y 레이블 정의: 감사의견코드 기반, 비적정 기업(Y=1)을 한정계열(관리)과 거절·부적정(사망)으로 통합 정의 — 도메인 관찰상 한정기업이 거절로 전이하는 경향이 있어 조기개입 목적상 통합
- Y 레이블 비율: M-Score 계산 및 최종 필터링 후 34,702개사(135,397행) 중 Y=1 10,311행, 비율 7.6%
- 결측·이상치 처리: 분모0·음수·결측 13개 조건 합집합으로 기업 단위 통째 제거(16,726개사 제거), 왜도(Skewness) 기반 차등 윈저라이징(제거 대신 정보 보존 원칙)
- 변수 타당성: 상관관계 전반적으로 낮은 수준, 전 변수 VIF < 10 확인 → 다중공선성 문제 미발견

## 3. Preprocessing & Feature Engineering

- 전처리 파이프라인(8단계): Train/Val/Test 분할(70/15/15, 기업 단위) → 기초통계·결측치·이상치 진단 → Train 데이터 통계 처리 → 원저라이징 → Scaling(StandardScaler) → SMOTE(ratio=1:2, Train only) → Case1~4 생성 → Val/Test는 Train 기준 transform only
- 누수 방지 원칙: SMOTE·Scaling 모두 Train fit → Val/Test transform only, StratifiedGroupKFold(기업 단위 그룹 유지)로 동일 기업 Train/Test 혼재 방지
- 변수 한계 식별: 가지급금 단독 임계값 분석 시 상위 10% 구간에서도 부실발생률 0% → 단일변수로는 탐지 불가 확인
- Feature 구성:
  * TypeA: Beneish M-Score 8개(DSRI·GMI·AQI·SGI·DEPI·SGAI·LVGI·TATA), 글로벌 기준 베이스라인
  * TypeB: TypeA + 한국형 변수 5개(Z1 내부자채권·Z2 단기대여금·Z3 단기차입금·Z4 예수금·Z5 이자수익), 자금유출 구조 반영
- 통계적 입증: Fisher's Exact Test로 Z1×Z5 결합 조건의 유의성 확증(OR 2.52, p<0.0001) — 가지급금 단독은 무관하나 이자수익과 결합 시 분식과 유의하게 연결

## 4. Model & Algorithms

- 모델 구성: 통계(Logistic Regression, 베이스라인) / ML(Random Forest·XGBoost·LGBM, 트리 기반 안정성) / DL(MLP, 단순구조 균형성능) / Tabular DL(TabNet·FT-Transformer, 정형데이터 특화 여부 검증)
- 평가지표: 핵심 Recall(분식 미탐지 최소화, Cost-Asymmetric Problem 고려), 보조 F2 → AUC-PR → MCC → Precision, Accuracy는 양성비율 7.6% 불균형 구조상 제외
- 실험 흐름(7단계): 데이터구성 → 기업단위분할 → 전처리(Case1~4) → 모델학습 → GridSearch → 평가 및 최종선택 → 외부검증
- 그리드서치 결과(TypeB, Threshold 0.5, Test): XGBoost(Case4) Recall 0.8426 1위, LGBM(Case3) 0.7280, MLP(Case3) 0.7482, FT-Transformer(Case1) 0.7274, LR(Case2) 0.7450, RF(Case1) 0.7092, TabNet(Case2) 0.5621 최하위(연속형 8~13개 변수 환경에서 Sparse Masking의 이점 발휘 여지 부족)
- 최종 모델: XGBoost(TypeB Case4, SMOTE+Scaling) 메인 + MLP(TypeB Case3, Scaling) 서브 복합운용

## 5. Report (외부검증 기준, Threshold 0.5)

| 지표 | 대법원 회생폐지(N=53) | 금융감독원 감리제재(N=13) |
|---|---|---|
| XGBoost Recall | 0.9811 (52사) | 0.8462 (11사) |
| MLP Recall | 0.4717 (25사) | 0.6154 (8사) |
| McNemar p-value | 0.0000 (유의) | 0.2500 (비유의, 표본 소규모) |

- 외부검증 방법: StratifiedGroupKFold로 확보한 학습데이터와 외부 검증 표본의 독립성 유지, McNemar 검정으로 TypeA vs TypeB 알고리즘 간 탐지력 차이 검증
- SHAP 분석: XGBoost·MLP 두 모델 모두 한국형 변수(Z1 내부자채권, Z5 이자수익)가 SHAP·Feature Importance 상위 공통 점유 → 단일 모델의 우연이 아닌 변수 자체의 실질적 설명력 확인
- 실무 활용: 거래소코드+회계연도 기반 RAG 자동 리포트 생성(LangChain 연동), 위험등급 3단계(정상/의심/고위험) 분류 대시보드

## 6. Review

- 한계: 감리제재 외부검증 표본(N=13) 통계적 검정력 부족 — 통계 검증보다 분식 탐지 방향성 확인에 의의. 회생폐지(N=53)는 McNemar 유의(p<0.0001)로 통계적 검증력 확보
- TabNet 한계: 연속형 재무비율 환경에서 Sparse Mask 미작동 확인(데이터특성·모델아키텍처 정합성 부족), 변수 8~13개로는 자동 변수선택 구조의 이점이 발휘될 여지 부족
- 향후 과제: 업종 확장(제조업 외 외감법인), TabNet 강점 발현 가능성 검증(범주형 변수 다수 혼합 환경), FN 블라인드 스팟 원인 규명(한국형 변수 개별 재무제표 정밀 대조)
