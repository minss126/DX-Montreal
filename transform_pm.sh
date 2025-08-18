#!/bin/bash

# ===================================================================
# 파라미터 설정
# ===================================================================
# t와 epsilon 값 배열을 정의합니다.
t_values=(3)
eps_values=(1.0 2.0 3.0 4.0 5.0)
# 사용자의 요청에 따라 모든 데이터셋의 레이블 컬럼을 'label'로 통일합니다.
label_col="label"

# ANSI 색상 코드를 정의하여 터미널 출력을 꾸밉니다.
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # 색상 없음

# ===================================================================
# 데이터셋별 변환 작업
# ===================================================================

# --- 수치형 레이블을 가진 데이터셋 (Numerical Labels) ---
echo -e "${CYAN}수치형 레이블(Numerical Label) 데이터셋 변환을 시작합니다...${NC}"

:<<END
# 1. 로그 변환이 필요한 수치형 데이터셋
numerical_log_datasets=("OnlineNewsPopularity" "Beijing_housing" "cal_housing")
for dataset in "${numerical_log_datasets[@]}"; do
    for t in "${t_values[@]}"; do
        for eps in "${eps_values[@]}"; do
            command="python pm_transform_train_test_batch_label.py --t $t --eps $eps --transform_label_numerical True --transform_label_categorical False --transform_label_log True --csv_path data/${dataset}.csv --label_col $label_col"
            echo -e "${GREEN}Executing: $command${NC}"
            eval "$command"
        done
    done
done
END

# 2. 로그 변환이 필요 없는 수치형 데이터셋

numerical_datasets=("CASP" "elevators")
for dataset in "${numerical_datasets[@]}"; do
    for t in "${t_values[@]}"; do
        for eps in "${eps_values[@]}"; do
            command="python pm_transform_train_test_batch_label.py --t $t --eps $eps --transform_label_numerical True --transform_label_categorical False --transform_label_log False --csv_path data/${dataset}.csv --label_col $label_col"
            echo -e "${GREEN}Executing: $command${NC}"
            eval "$command"
        done
    done
done


# --- 범주형 레이블을 가진 데이터셋 (Categorical Labels) ---
echo -e "${CYAN}범주형 레이블(Categorical Label) 데이터셋 변환을 시작합니다...${NC}"

categorical_datasets=("gamma" "credit" "wine" "shuttle")
for dataset in "${categorical_datasets[@]}"; do
    for t in "${t_values[@]}"; do
        for eps in "${eps_values[@]}"; do
            command="python pm_transform_train_test_batch_label.py --t $t --eps $eps --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/${dataset}.csv --label_col $label_col"
            echo -e "${GREEN}Executing: $command${NC}"
            eval "$command"
        done
    done
done

# ===================================================================
# 최종 완료 메시지
# ===================================================================
echo -e "${CYAN}모든 작업이 완료되었습니다.${NC}"