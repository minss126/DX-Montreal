#!/usr/bin/env bash
set -e # 에러 발생 시 즉시 종료

# ─────────────────────────────────────────────
# 데이터셋별 설정
# ─────────────────────────────────────────────
declare -A LABEL_N_MAP=( \
  ["elevators"]=7 \
  ["CASP"]=7 \
  ["credit"]=2 \
  ["gamma"]=2 \
  ["wine"]=7 \
  ["shuttle"]=7
)

DATASETS=("elevators" "CASP" "credit" "gamma" "wine" "shuttle")

SEEDS_LIST="0 1 2 3 4 5 6 7 8 9"

# ─────────────────────────────────────────────
# 실험 루프 - 수치형 레이블 변환 있음
# ─────────────────────────────────────────────
echo "⚙️ 변환 시작"

TRANSFORM_SCRIPT="qm_transform_train_test_batch_label.py"
LABEL_COL="label"
OUTPUT_DIR="transformed_data_batch_label"

# 값 리스트
EPS_VALUES="1.0 2.0 3.0 4.0 5.0"
#LABEL_EPS_VALUES="2.0 4.0"
N_VALUES="7"

for DATASET in "${DATASETS[@]}"; do
  CSV_PATH="data/${DATASET}.csv"

  echo "🔄 실험 시작: ${DATASET}"

  # 데이터셋에 따라 transform_label_log 설정
  TRANSFORM_LABEL_LOG_FLAG="False"
  if [[ "${DATASET}" == "OnlineNewsPopularity" || "${DATASET}" == "Beijing_housing" ]]; then
    TRANSFORM_LABEL_LOG_FLAG="True"
  fi

  # 각 데이터셋에 대한 label_N 값 설정
  CURRENT_FIXED_LABEL_N=""

  for N_VAL in $N_VALUES; do
    # label_N은 데이터셋에 따라 설정된 고정 값 사용
    if [[ "${DATASET}" == "wine" || "${DATASET}" == "shuttle" ]]; then
      CURRENT_FIXED_LABEL_N=7
    elif [[ "${DATASET}" == "gamma" || "${DATASET}" == "credit" ]]; then
      CURRENT_FIXED_LABEL_N=2
    else
      CURRENT_FIXED_LABEL_N=${N_VAL}
    fi
    LABEL_N_CURRENT="${CURRENT_FIXED_LABEL_N}"

    # 1. eps == label_eps 인 경우 처리 (예: (1,1), (2,2), ..., (5,5))
    for EPS_EQ in $EPS_VALUES; do
      echo "▶️ dataset=${DATASET}, N=${N_VAL}, eps=${EPS_EQ}, label_eps=${EPS_EQ}, label_N=${LABEL_N_CURRENT}, transform_label_numerical=True, transform_label_log=${TRANSFORM_LABEL_LOG_FLAG} (eps == label_eps)"
      python ${TRANSFORM_SCRIPT} \
        --csv_path "${CSV_PATH}" \
        --label_col "${LABEL_COL}" \
        --output_dir "${OUTPUT_DIR}" \
        --eps "${EPS_EQ}" \
        --N "${N_VAL}" \
        --label_N "${LABEL_N_CURRENT}" \
        --label_epsilon "${EPS_EQ}" \
        --obj worst \
        --with_categorical False \
        --transform_label_numerical True \
        --transform_label_categorical False \
        --test_size 0.2 \
        --random_state 42 \
        --transform_label_log "${TRANSFORM_LABEL_LOG_FLAG}"
    done
:<<END
    # 2. eps != label_eps 인 경우 처리 (모든 조합 중 eps==label_eps 제외)
    for EPS in $EPS_VALUES; do
      for LABEL_EPS in $LABEL_EPS_VALUES; do
        echo "▶️ dataset=${DATASET}, N=${N_VAL}, eps=${EPS}, label_eps=${LABEL_EPS}, label_N=${LABEL_N_CURRENT}, transform_label_numerical=True, transform_label_log=${TRANSFORM_LABEL_LOG_FLAG} (eps != label_eps)"
        python ${TRANSFORM_SCRIPT} \
          --csv_path "${CSV_PATH}" \
          --label_col "${LABEL_COL}" \
          --output_dir "${OUTPUT_DIR}" \
          --eps "${EPS}" \
          --N "${N_VAL}" \
          --label_N "${LABEL_N_CURRENT}" \
          --label_epsilon "${LABEL_EPS}" \
          --obj avg \
          --with_categorical False \
          --transform_label_numerical True \
          --transform_label_categorical False \
          --test_size 0.2 \
          --random_state 42 \
          --transform_label_log "${TRANSFORM_LABEL_LOG_FLAG}"
      done
    done
END

  done # N_VAL 루프 종료
done

echo "✨ 전체 실험 완료!"
