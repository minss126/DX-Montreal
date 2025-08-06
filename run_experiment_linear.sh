#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status.

# ─────────────────────────────────────────────
# Linear regression
# ─────────────────────────────────────────────

# Define common parameters for all experiments
COMMON_SEEDS=(0 1 2 3 4 5 6 7 8 9)
EPS_VALUES=(1.0 2.0 3.0 4.0 5.0)
LABEL_EPS_VALUES=(1.0 3.0 5.0)
LEARNING_RATE=0.05
EPOCHS=1000
REG_LAMBDA=0.1
RESULT_ROOT_DIR="results_structured3"

# Define N values
N_LIST=(2 3 4 7 15 31)

echo "Starting Machine Learning Experiments..."

# Define the single label privacy scenario
declare -A LABEL_PRIVACY_SCENARIOS
LABEL_PRIVACY_SCENARIOS[label]="transformed_data_batch_label"

# --- Linear Regression Experiments (linear.py) ---
echo "Running Linear Regression Experiments (linear.py)"
echo "-----------------------------------------------------"
LINEAR_CSV_PATHS=(
    "data/elevators.csv"
)
CURRENT_TRANSFORMED_BASE_DIR="transformed_data_batch_label"

for CSV_PATH in "${LINEAR_CSV_PATHS[@]}"; do
    DATASET_NAME=$(basename "${CSV_PATH%.*}")
    for CURRENT_N in "${N_LIST[@]}"; do
        # N and label_N are always the same
        CURRENT_LABEL_N="${CURRENT_N}"
        # 2. Handle all other cases, including when eps=3.0 (with all label_eps values)
        for EPS in "${EPS_VALUES[@]}"; do
            for LABEL_EPS in "${LABEL_EPS_VALUES[@]}"; do
                echo "Processing dataset: ${DATASET_NAME}, N=${CURRENT_N}, eps=${EPS}, label_eps=${LABEL_EPS} (Linear Regression, Scenario: ${SCENARIO})"
                TOTAL_RESULT_CSV_PATH="${RESULT_ROOT_DIR}/${DATASET_NAME}/results_summary.csv"

                python linear.py \
                    --csv_path "${CSV_PATH}" \
                    --N "${CURRENT_N}" \
                    --label_N "${CURRENT_LABEL_N}" \
                    --seeds "${COMMON_SEEDS[@]}" \
                    --eps "${EPS}" \
                    --label_eps "${LABEL_EPS}" \
                    --learning_rate "${LEARNING_RATE}" \
                    --epochs "${EPOCHS}" \
                    --regularization_lambda "${REG_LAMBDA}" \
                    --output_dir "${CURRENT_TRANSFORMED_BASE_DIR}/${DATASET_NAME}" \
                    --result_dir "${RESULT_ROOT_DIR}" \
                    --total_result_csv "${TOTAL_RESULT_CSV_PATH}"
            done
        done
    done
done

echo "All Machine Learning Experiments Completed."