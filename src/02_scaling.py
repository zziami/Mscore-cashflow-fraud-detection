#============================================================
# STEP 10. StandardScaler 스케일링 
# ============================================================
# [목적]
# 윈저라이징된 변수들의 수치 범위를 통일
# 평균 0, 표준편차 1로 표준화
#
#
# [누수 방지]
# Train fit only → Val/Test transform only


from sklearn.preprocessing import StandardScaler

# STEP 09. StandardScaler (Train fit only → Val/Test transform)
scaler = StandardScaler()
scaler.fit(train_df[VARS_B])

train_df[VARS_B] = scaler.transform(train_df[VARS_B])
val_df[VARS_B]   = scaler.transform(val_df[VARS_B])
test_df[VARS_B]  = scaler.transform(test_df[VARS_B])
