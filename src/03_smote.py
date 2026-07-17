# # ============================================================
# # STEP 11. SMOTE 비율 탐색 (0.3 / 0.5 / 1.0)
# # ============================================================
# # [목적]
# # Y=1(소수) 비율을 어느 정도까지 증강할지 탐색
# # 0.3 → 0.5 → 1.0 순차 실행, 각각 별도 파일로 저장
# # 기본파라미터 모델링(12번)에서 성능(Recall/Confusion Matrix) 비교 후 확정
# #
# # [누수 방지]
# # Train에만 적용 / Val·Test 원본 유지 (SMOTE 대상 아님)
# #



from imblearn.over_sampling import SMOTE

# STEP 03. SMOTE 비율 탐색 (0.3 / 0.5 / 1.0)
# 기본파라미터 모델링에서 Recall 비교 후 0.5(=1:2)로 최종 확정
RATIOS = [0.3, 0.5, 1.0]
RANDOM_STATE = 42

results = {}
for ratio in RATIOS:
    smote = SMOTE(sampling_strategy=ratio, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(train_df[VARS_B], train_df[TARGET])
    results[ratio] = (X_res, y_res)
    print(f"ratio={ratio} → Y=1 {sum(y_res==1):,} / Y=0 {sum(y_res==0):,}")

# 비교 결과 ratio=0.5 확정 → StratifiedGroupKFold 각 fold Train에만 최종 적용
smote_final = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)
X_train_res, y_train_res = smote_final.fit_resample(X_train, y_train)

