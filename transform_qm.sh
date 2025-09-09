#!/usr/bin/env bash
set -e # 에러 발생 시 즉시 종료

# ─────────────────────────────────────────────
# 데이터셋별 설정
# ─────────────────────────────────────────────
DATASETS=("elevators" "CASP" "credit" "gamma" "wine" "shuttle")
# ▼▼▼ [추가] 회귀 데이터셋을 별도로 정의합니다 ▼▼▼
REGRESSION_DATASETS=("elevators" "CASP")

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
N_VALUES="7"

for DATASET in "${DATASETS[@]}"; do
  CSV_PATH="data/${DATASET}.csv"

  echo "🔄 실험 시작: ${DATASET}"

  # 데이터셋에 따라 transform_label_log 설정
  TRANSFORM_LABEL_LOG_FLAG="False"
  if [[ "${DATASET}" == "OnlineNewsPopularity" || "${DATASET}" == "Beijing_housing" ]]; then
    TRANSFORM_LABEL_LOG_FLAG="True"
  fi

  for N_VAL in $N_VALUES; do
    # label_N은 데이터셋에 따라 설정된 고정 값 사용
    if [[ "${DATASET}" == "wine" || "${DATASET}" == "shuttle" ]]; then
      LABEL_N_CURRENT=7
    elif [[ "${DATASET}" == "gamma" || "${DATASET}" == "credit" ]]; then
      LABEL_N_CURRENT=2
    else
      LABEL_N_CURRENT=${N_VAL}
    fi

    # 1. eps == label_eps 인 경우 처리
    for EPS_EQ in $EPS_VALUES; do

      # ▼▼▼ [수정] 데이터셋이 회귀용인지 아닌지에 따라 분기 처리 ▼▼▼
      if [[ " ${REGRESSION_DATASETS[@]} " =~ " ${DATASET} " ]]; then
        # ✅ 회귀 데이터셋인 경우: label_index를 True/False로 반복 실행
        echo "  -> 회귀 데이터셋으로 감지. label_index 옵션을 모두 테스트합니다."
        for LABEL_INDEX_FLAG in "True" "False"; do
        
          SUB_DIR="inverse_linear"
          if [[ "${LABEL_INDEX_FLAG}" == "True" ]]; then
            SUB_DIR="inverse_index"
          fi
          CURRENT_OUTPUT_DIR="${OUTPUT_DIR}/${SUB_DIR}"

          echo "▶️ dataset=${DATASET}, eps=${EPS_EQ}, label_index=${LABEL_INDEX_FLAG}"
          python ${TRANSFORM_SCRIPT} \
            --csv_path "${CSV_PATH}" \
            --label_col "${LABEL_COL}" \
            --output_dir "${CURRENT_OUTPUT_DIR}" \
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
            --transform_label_log "${TRANSFORM_LABEL_LOG_FLAG}" \
            --label_index "${LABEL_INDEX_FLAG}" # <-- label_index 옵션 추가
        done
      else
        # ⚠️ 회귀 데이터셋이 아닌 경우: 기존 방식대로 한 번만 실행
        echo "▶️ dataset=${DATASET}, eps=${EPS_EQ}"
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
      fi
      # ▲▲▲ 분기 처리 종료 ▲▲▲
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
