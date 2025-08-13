#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status.

# ─────────────────────────────────────────────
# Unified Model Experiments
# ─────────────────────────────────────────────

# Define common parameters for all experiments
COMMON_SEEDS=(0 1 2 3 4 5 6 7 8 9)
EPS_VALUES=(1.0 3.0 5.0)
N_LIST=(7)
LEARNING_RATE=(0.01 0.05 0.1)
EPOCHS=(10 1000)
BATCH_SIZES=(512 4096 -1) # -1 for full-batch mode
REG_LAMBDA=0.1
RESULT_ROOT_DIR="results_unified"
LDP_DATA_ROOT_DIR="transformed_data_batch_label"

echo "Starting All Machine Learning Experiments..."

# =====================================================
#  1. Linear Regression Experiments
# =====================================================
echo ""
echo "#######################################"
echo "# Running Linear Regression Experiments #"
echo "#######################################"
LINEAR_CSV_PATHS=(
    "data/CASP.csv"
    "data/OnlineNewsPopularity.csv"
)

for CSV_PATH in "${LINEAR_CSV_PATHS[@]}"; do
    DATASET_NAME=$(basename "${CSV_PATH%.*}")
    TRANSFORM_LABEL_LOG_FLAG="False"
    if [[ "${DATASET_NAME}" == "OnlineNewsPopularity" ]]; then
        TRANSFORM_LABEL_LOG_FLAG="True"
    fi

    for CURRENT_N in "${N_LIST[@]}"; do
        for BATCH in "${BATCH_SIZES[@]}"; do
            for LR in "${LEARNING_RATE[@]}"; do
                for EPOCH in "${EPOCHS[@]}"; do
                    for EPS in "${EPS_VALUES[@]}"; do
                        echo "Processing: Linear | ${DATASET_NAME} | N=${CURRENT_N} | eps=${EPS} | bs=${BATCH} | lr=${LR} | epoch=${EPOCH}"
                        python model.py \
                            --model_type "linear" \
                            --csv_path "${CSV_PATH}" \
                            --N "${CURRENT_N}" --label_N 15 \
                            --seeds "${COMMON_SEEDS[@]}" \
                            --eps "${EPS}" --label_eps "${EPS}" \
                            --learning_rate "${LR}" --epochs "${EPOCH}" \
                            --regularization_lambda "${REG_LAMBDA}" \
                            --batch_size "${BATCH}" \
                            --transform_label_log "${TRANSFORM_LABEL_LOG_FLAG}" \
                            --output_dir "${LDP_DATA_ROOT_DIR}" \
                            --result_dir "${RESULT_ROOT_DIR}" \
                            --total_result_dir "${RESULT_ROOT_DIR}"
                    done
                done
            done
        done
    done
done


# =====================================================
#  2. Logistic Regression (Binary) Experiments
# =====================================================
echo ""
echo "###########################################"
echo "# Running Logistic Regression Experiments #"
echo "###########################################"
LOGISTIC_CSV_PATHS=(
    "data/gamma.csv"
    "data/credit.csv"
)

for CSV_PATH in "${LOGISTIC_CSV_PATHS[@]}"; do
    DATASET_NAME=$(basename "${CSV_PATH%.*}")
    for CURRENT_N in "${N_LIST[@]}"; do
        for BATCH in "${BATCH_SIZES[@]}"; do
            for LR in "${LEARNING_RATE[@]}"; do
                for EPOCH in "${EPOCHS[@]}"; do
                    for EPS in "${EPS_VALUES[@]}"; do
                        echo "Processing: Logistic | ${DATASET_NAME} | N=${CURRENT_N} | eps=${EPS} | bs=${BATCH} | lr=${LR} | epoch=${EPOCH}"
                        python model.py \
                            --model_type "logistic" \
                            --csv_path "${CSV_PATH}" \
                            --N "${CURRENT_N}" --label_N 2 \
                            --seeds "${COMMON_SEEDS[@]}" \
                            --eps "${EPS}" --label_eps "${EPS}" \
                            --learning_rate "${LR}" --epochs "${EPOCH}" \
                            --regularization_lambda "${REG_LAMBDA}" \
                            --batch_size "${BATCH}" \
                            --output_dir "${LDP_DATA_ROOT_DIR}" \
                            --result_dir "${RESULT_ROOT_DIR}" \
                            --total_result_dir "${RESULT_ROOT_DIR}"
                    done
                done
            done
        done
    done
done


# =====================================================
#  3. Multi-class Logistic Regression Experiments
# =====================================================
echo ""
echo "######################################################"
echo "# Running Multi-class Logistic Regression Experiments #"
echo "######################################################"
MULTI_CSV_PATHS=(
    "data/wine.csv"
    "data/shuttle.csv"
)

for CSV_PATH in "${MULTI_CSV_PATHS[@]}"; do
    DATASET_NAME=$(basename "${CSV_PATH%.*}")
    for CURRENT_N in "${N_LIST[@]}"; do
        for BATCH in "${BATCH_SIZES[@]}"; do
            for LR in "${LEARNING_RATE[@]}"; do
                for EPOCH in "${EPOCHS[@]}"; do
                    for EPS in "${EPS_VALUES[@]}"; do
                        echo "Processing: Multi-Logistic | ${DATASET_NAME} | N=${CURRENT_N} | eps=${EPS} | bs=${BATCH} | lr=${LR} | epoch=${EPOCH}"
                        python model.py \
                            --model_type "logistic_multi" \
                            --csv_path "${CSV_PATH}" \
                            --N "${CURRENT_N}" --label_N 7 \
                            --seeds "${COMMON_SEEDS[@]}" \
                            --eps "${EPS}" --label_eps "${EPS}" \
                            --learning_rate "${LR}" --epochs "${EPOCH}" \
                            --regularization_lambda "${REG_LAMBDA}" \
                            --batch_size "${BATCH}" \
                            --output_dir "${LDP_DATA_ROOT_DIR}" \
                            --result_dir "${RESULT_ROOT_DIR}" \
                            --total_result_dir "${RESULT_ROOT_DIR}"
                    done
                done
            done
        done
    done
done

echo ""
echo "All Machine Learning Experiments Completed."