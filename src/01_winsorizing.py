# 전처리 작업중 윈저라이징 왜도 진단 -  코드 일부분 발췌
# ============================================================
# STEP 08. EDA(왜도 진단) + 윈저라이징
# ============================================================
# [목적]
# 1. Train 기준 왜도(Skewness) 진단
#    - |Skew| >= 3 → 심각 → 1%/99% 윈저라이징
#    - 1 <= |Skew| < 3 → 중간 → 2%/98% 윈저라이징
#    - |Skew| < 1 → 정상 → 윈저라이징 생략
#
# 2. 변수별 개별 윈저라이징 강도 적용 (일괄 1% 금지)
#    - 분식징후는 극단값에서 나타날 수 있음
#    - 극단값을 완전히 제거하지 않고 최소한으로 눌러줌
#    - 데이터 손실 최소화 원칙
#
# 3. Train fit only → Val/Test transform
#    - Train 데이터의 분위수(lower/upper)만으로 경계값 계산
#    - Val/Test는 Train 기준 경계값으로 clip만 적용
#    - 데이터 누수 완전 차단
#
# [적용 변수]
# Model A: DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI (8개)
# Model B: + z1_내부자채권, z2_단기대여금, z3_단기차입금, z4_예수금, z5_이자수익 (5개)
# 총 13개
# ============================================================

from scipy.stats import skew, kurtosis

# 변수별 윈저라이징 강도 (왜도 기반 사전 설정)
# |Skew| >= 3 → 1%/99%, 1~3 → 2%/98%
WINSOR_CONFIG = {
    "DSRI":         (0.01, 0.99),
    "GMI":          (0.02, 0.98),
    "AQI":          (0.01, 0.99),
    "SGI":          (0.02, 0.98),
    "DEPI":         (0.01, 0.99),
    "SGAI":         (0.02, 0.98),
    "TATA":         (0.02, 0.98),
    "LVGI":         (0.01, 0.99),
    "z1_내부자채권": (0.02, 0.98),
    "z2_단기대여금": (0.02, 0.98),
    "z3_단기차입금": (0.01, 0.99),
    "z4_예수금":     (0.02, 0.98),
    "z5_이자수익":   (0.01, 0.99),
}

# 1. 왜도(Skewness) 진단 — Train 기준
eda_results = []
for var in VARS_B:
    vals = train_df[var].dropna()
    sk = skew(vals)
    ku = kurtosis(vals)

    if abs(sk) >= 3:
        severity = "심각"
    elif abs(sk) >= 1:
        severity = "중간"
    else:
        severity = "정상"

    lo_pct, hi_pct = WINSOR_CONFIG[var]
    eda_results.append({
        "변수": var, "skewness": round(sk, 3), "kurtosis": round(ku, 3),
        "심각도": severity, "윈저강도": f"{lo_pct*100:.0f}%/{hi_pct*100:.0f}%"
    })

# 2. 윈저라이징 적용 (Train fit → Val/Test transform)
winsor_bounds = {}
for var in VARS_B:
    lo_pct, hi_pct = WINSOR_CONFIG[var]
    lo = float(train_df[var].quantile(lo_pct))
    hi = float(train_df[var].quantile(hi_pct))
    winsor_bounds[var] = (lo, hi)

    train_df[var] = train_df[var].clip(lower=lo, upper=hi)
    val_df[var]   = val_df[var].clip(lower=lo, upper=hi)
    test_df[var]  = test_df[var].clip(lower=lo, upper=hi)
