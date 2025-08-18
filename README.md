# DX-Montreal

---

## 🔹 1. LDP Transformation

### 1.1 `transform_pm.sh`
- **데이터셋**
  - Numerical datasets → Linear regression (CASP, elevators)
  - Categorical datasets → Logistic regression (gamma, credit), Multi-logistic regression (wine, shuttle)
- **저장 위치**
  - `pm_transform_train_test_batch_label.py`의 `output_dir` 인자로 변경 가능
- **실험 파라미터**
  - `eps_values` → epsilon 값 조정

**실행 방법**
```bash
bash transform_pm.sh
```

---

### 1.2 `transform_qm.sh`
- **데이터셋**: `DATASETS` 안의 데이터셋 이름 변경
- **저장 위치**: `OUTPUT_DIR` 수정
- **실험 파라미터**
  - `EPS_VALUES` → epsilon 값 조정
  - `N_VALUES` → N 값 조정

**실행 방법**
```bash
bash transform_qm.sh
```

---

## 🔹 2. Regression

### 2.1 `run_experiments.sh`
- **수정 가능한 파라미터**
  - `learning rate`, `epoch`, `batch size`, `epsilon`, `N`
- **저장 위치**: `RESULT_ROOT_DIR` 수정
- **데이터 경로**
  - 회귀 방법에 따라 아래 리스트에서 변경
    - `LINEAR_CSV_PATHS` → Linear regression datasets
    - `LOGISTIC_CSV_PATHS` → Logistic regression datasets
    - `MULTI_CSV_PATHS` → Multi-logistic regression datasets

**실행 방법**
```bash
bash run_experiments.sh
```

---

## 📂 Folder Structure (예시)
```
project/
 ┣ data/                         # 원본 데이터셋
 ┣ transformed_data_batch_label/ # 변환된 데이터셋
 ┣ result_unified/               # 회귀 결과 폴더
 ┣ transform_pm.sh
 ┣ transform_qm.sh
 ┣ run_experiments.sh
 ┣ pm_transform_train_test_batch_label.py
 ┣ qm_transform_train_test_batch_label.py
 ┣ model.py
 ┣ pm.py
 ┣ make_mechanism_avg.py
 ┣ make_mechanism_worst.py

 ┗ README.md
```

---

## 📜 License
MIT License
