#!/bin/bash

# ===================================================================
# 파라미터 설정
# ===================================================================
t_values=(3)
eps_values=(1.0 2.0 3.0 4.0 5.0)
label_col="label"
output_dir_base="transformed_data_batch_label"

# ANSI 색상 코드를 정의하여 터미널 출력을 꾸밉니다.
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # 색상 없음

# ===================================================================
# 데이터셋별 변환 작업
# ===================================================================
:<< END
# --- 수치형 레이블을 가진 데이터셋 (Numerical Labels) ---
echo -e "${CYAN}수치형 레이블(Numerical Label) 데이터셋 변환을 시작합니다...${NC}"

# 1. 로그 변환이 필요 없는 수치형 데이터셋 (CASP)
# QM 스크립트처럼 label_index 옵션을 바꿔가며 실행
numerical_datasets=("wine")
for dataset in "${numerical_datasets[@]}"; do
    for t in "${t_values[@]}"; do
        for eps in "${eps_values[@]}"; do
            # label_index를 True/False로 반복 실행
            for label_index_flag in "True" "False"; do
                
                # label_index 값에 따라 하위 폴더 이름 결정
                sub_dir="inverse_linear"
                if [[ "$label_index_flag" == "True" ]]; then
                    sub_dir="inverse_index"
                fi
                
                # 최종 출력 경로 설정
                current_output_dir="${output_dir_base}/${sub_dir}"

                command="python pm_transform_train_test_batch_label.py --t $t --eps $eps --transform_label_numerical True --transform_label_categorical False --transform_label_log False --csv_path data/${dataset}.csv --label_col $label_col --output_dir ${current_output_dir} --label_index ${label_index_flag}"
                echo -e "${GREEN}Executing: $command${NC}"
                eval "$command"
            done
        done
    done
done
END

# --- 범주형 레이블을 가진 데이터셋 (Categorical Labels) ---
echo -e "${CYAN}범주형 레이블(Categorical Label) 데이터셋 변환을 시작합니다...${NC}"

categorical_datasets=("gamma" "iris")
for dataset in "${categorical_datasets[@]}"; do
    for t in "${t_values[@]}"; do
        for eps in "${eps_values[@]}"; do
            # 범주형은 고정된 'pm' 폴더에 저장
            current_output_dir="${output_dir_base}/${dataset}/pm"
            command="python pm_transform_train_test_batch_label.py --t $t --eps $eps --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/${dataset}.csv --label_col $label_col --output_dir ${output_dir_base}"
            echo -e "${GREEN}Executing: $command${NC}"
            eval "$command"
        done
    done
done

# ===================================================================
# 최종 완료 메시지
# ===================================================================
echo -e "${CYAN}모든 작업이 완료되었습니다.${NC}"
