import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import traceback
import warnings
from collections import Counter

np.random.seed(42)

# ===================================================================
# HE(Homomorphic Encryption) 연산 플레이스홀더 (더미 함수)
# ===================================================================
def he_dot(a_enc: np.ndarray, b_enc: np.ndarray) -> np.ndarray:
    """HE 기반의 벡터 내적 연산을 위한 더미 함수."""
    return a_enc @ b_enc

def he_decrypt_vector(v_enc: np.ndarray) -> np.ndarray:
    """HE 기반의 벡터 복호화를 위한 더미 함수."""
    return v_enc

def _print_dist(tag: str, arr: np.ndarray):
    """배열의 값 분포를 출력합니다 (주로 레이블 값)."""
    arr = np.asarray(arr).flatten().tolist()
    cnt = Counter(int(x) for x in arr)
    print(f"{tag} 분포: {dict(cnt)}")

# ===================================================================
# 로지스틱 회귀 실험 클래스
# ===================================================================
class LogisticExperiment:
    """LDP(Local Differential Privacy) 및 HE(Homomorphic Encryption) 기반 로지스틱 회귀 실험 클래스"""

    def __init__(self, config: argparse.Namespace):
        """실험 설정 초기화"""
        self.config = config
        self.output_dir = Path(config.output_dir)
        np.random.seed(config.seed) # 각 실험 인스턴스에 대한 시드 설정

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """시그모이드 함수 (이진 분류용, 수치적 안정성을 위한 클리핑 포함)"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """소프트맥스 함수 (다중 분류용, 수치적 안정성을 위한 클리핑 포함)"""
        z = np.clip(z, -500, 500)
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def _train_gd(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        경사 하강법(Gradient Descent)을 사용하여 로지스틱 회귀 모델을 학습합니다.
        이진 분류 (y가 1차원 & 2개 클래스) 또는 다중 클래스 분류 (softmax 기반)를 지원합니다.
        """
        n_samples = len(y)
        cfg = self.config

        # ---------- 이진 분류 (Binary Classification) ----------
        if y.ndim == 1 and len(np.unique(y)) == 2:
            beta = np.zeros(X.shape[1]) # 가중치 초기화
            for _ in range(cfg.epochs):
                p = self._sigmoid(X @ beta) # 예측 확률
                
                # 비정규화된 기울기
                grad_unregularized = X.T @ (p - y) / n_samples
                # L2 정규화 항 기울기 (편향(bias)에는 적용 안 함)
                grad_regularization_term = cfg.regularization_lambda * beta
                grad_regularization_term[0] = 0 # 편향 항은 정규화하지 않음
                
                grad = grad_unregularized + grad_regularization_term
                beta -= cfg.learning_rate * grad # 가중치 업데이트
            return beta

        # ---------- 다중 클래스 분류 (Multiclass Classification) ----------
        # 레이블이 정수형이면 원-핫 인코딩
        if y.ndim == 1:
            K = len(np.unique(y)) # 클래스 개수
            y_onehot = np.zeros((n_samples, K))
            y_onehot[np.arange(n_samples), y.astype(int)] = 1
        else: # 이미 원-핫 인코딩되어 있음
            y_onehot = y
            K = y_onehot.shape[1]

        beta = np.zeros((X.shape[1], K)) # 가중치 초기화 (특징 수 + 1, 클래스 수)
        for _ in range(cfg.epochs):
            p = self._softmax(X @ beta)              # 예측 확률 (n_samples, K)
            
            # 비정규화된 기울기
            grad_unregularized = X.T @ (p - y_onehot) / n_samples # (특징 수, K)
            
            # L2 정규화 항 기울기 (편향(bias)에는 적용 안 함)
            grad_regularization_term = (cfg.regularization_lambda * beta)
            grad_regularization_term[0, :] = 0 # 편향 항은 정규화하지 않음

            grad = grad_unregularized + grad_regularization_term
            
            beta -= cfg.learning_rate * grad # 가중치 업데이트
        return beta


    def _impute(self, df_train: pd.DataFrame, df_apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """결측치를 훈련 세트의 평균으로 대체(Imputation)합니다."""
        # 주의: 훈련 세트의 평균으로 테스트 세트의 결측치도 대체해야 데이터 누수 방지
        train_mean = df_train.mean()
        return df_train.fillna(train_mean), df_apply.fillna(train_mean)

    def _scale(self, df_train: pd.DataFrame, df_apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        데이터를 StandardScaler와 유사하게 스케일링합니다 (평균 0, 표준편차 1).
        훈련 세트에서 학습된 스케일러를 테스트 세트에 적용합니다.
        """
        scaler = StandardScaler()
        df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns, index=df_train.index)
        df_apply_scaled = pd.DataFrame(scaler.transform(df_apply), columns=df_apply.columns, index=df_apply.index)
        # 스케일링 후 발생할 수 있는 NaN 값은 0으로 대체 (예: 모든 값이 동일한 컬럼)
        return df_train_scaled.fillna(0), df_apply_scaled.fillna(0)

    def run(self) -> pd.DataFrame | None:
        """
        로지스틱 회귀 실험의 전체 파이프라인을 실행합니다.
        데이터 로드, 전처리, 모델 학습, 평가를 포함합니다.
        """
        try:
            cfg = self.config
            
            # 결과 파일명에 사용될 LDP 및 레이블 LDP 파라미터 문자열 생성
            targ_str = f"{cfg.eps:.1f}"
            base_filename = (f"{cfg.dataset}_Eps{targ_str}_N{cfg.N}_avg_"
                             f"Leps{cfg.label_eps}_LN{cfg.label_N}_seed{cfg.seed}")

            # 1) 원본 데이터 로드
            orig = pd.read_csv(cfg.csv_path)
            # print('Original label counts:', orig[cfg.label_col].value_counts().to_dict()) # 디버깅용
            
            # 결측치가 40% 이상인 숫자형 컬럼 제거
            num_cols = orig.select_dtypes(include='number')
            drop_cols = num_cols.columns[num_cols.isna().mean() > 0.4]
            orig_full = orig.drop(columns=drop_cols).select_dtypes(include='number')

            # 2) LDP(Local Differential Privacy) 적용된 데이터 로드
            train_ldp = pd.read_csv(self.output_dir / f"{base_filename}_train.csv", index_col=0).select_dtypes(include='number')
            test_ldp = pd.read_csv(self.output_dir / f"{base_filename}_test.csv", index_col=0).select_dtypes(include='number')
            
            # 레이블 컬럼을 제외한 특징 컬럼 목록 생성
            feats = [c for c in train_ldp.columns if c != cfg.label_col]

            # 3) 원본 데이터셋을 LDP 데이터셋의 인덱스에 맞춰 분할
            full_tr = orig_full.loc[train_ldp.index].reset_index(drop=True)
            full_te = orig_full.loc[test_ldp.index].reset_index(drop=True)
            
            # 4) 결측치 대체 (Imputation)
            full_tr_imp, full_te_imp = self._impute(full_tr, full_te)
            ldp_tr_imp,  ldp_te_imp  = self._impute(train_ldp, test_ldp)

            # 5) 데이터 스케일링 (Scaling)
            full_tr_s, full_te_s = self._scale(full_tr_imp[feats], full_te_imp[feats])
            ldp_tr_s,  ldp_te_s  = self._scale(ldp_tr_imp[feats], ldp_te_imp[feats])

            # 상수항(bias) 추가 함수
            def add_bias(X):
                """입력 행렬 X에 상수항(1)을 추가합니다."""
                return np.hstack([np.ones((X.shape[0], 1)), X])

            X_full_tr = add_bias(full_tr_s.values)
            X_full_te = add_bias(full_te_s.values)
            X_ldp_tr  = add_bias(ldp_tr_s.values)
            X_ldp_te  = add_bias(ldp_te_s.values)
            
            # 레이블 데이터 추출
            y_full_tr = full_tr_imp[cfg.label_col].values
            y_ldp_tr  = ldp_tr_imp[cfg.label_col].values
            y_full_te = full_te_imp[cfg.label_col].values
            y_ldp_te  = ldp_te_imp[cfg.label_col].values

            # 레이블을 0부터 K-1까지의 정수로 통일 (Ordinal Encoding)
            # 훈련 세트를 기준으로 인코더를 학습하고, 테스트 세트에 적용
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
            y_full_tr = oe.fit_transform(y_full_tr.reshape(-1,1)).ravel()
            y_full_te = oe.transform(y_full_te.reshape(-1,1)).ravel()

            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
            y_ldp_tr = oe.fit_transform(y_ldp_tr.reshape(-1,1)).ravel()
            y_ldp_te = oe.transform(y_ldp_te.reshape(-1,1)).ravel()

            num_classes = len(np.sort(np.unique(y_full_tr)))

            # 6) 모델 학습
            beta_full = self._train_gd(X_full_tr, y_full_tr) # 원본 데이터로 학습
            beta_ldp  = self._train_gd(X_ldp_tr,  y_ldp_tr)  # LDP 데이터로 학습

            data = {} # 결과 저장용 딕셔너리

            # ---------- 이진 분류 모델 평가 ----------
            if num_classes == 2:
                pred_full = self._sigmoid(X_full_te @ beta_full)
                pred_ldp  = self._sigmoid(X_ldp_te  @ beta_ldp)

                # 예측 클래스 (임계값 0.5 기준)
                yhat = {
                    'Full': (pred_full >= 0.5).astype(int),
                    'LDP':  (pred_ldp  >= 0.5).astype(int)
                }
                # 예측 확률
                proba = {'Full': pred_full, 'LDP': pred_ldp}

                # 각 모델(Full, LDP)에 대한 성능 지표 계산
                for m in yhat:
                    y_true_m = y_full_te if m=='Full' else y_ldp_te # 해당 모델의 실제 레이블
                    data[m] = {
                        'Accuracy':  accuracy_score(y_true_m, yhat[m]),
                        'AUC':       roc_auc_score(y_true_m, proba[m]),
                        'Precision': precision_score(y_true_m, yhat[m], zero_division=0),
                        'Recall':    recall_score(y_true_m, yhat[m]),
                        'F1':        f1_score(y_true_m, yhat[m])
                    }

            # ---------- 다중 클래스 분류 모델 평가 ----------
            else:
                pred_full = self._softmax(X_full_te @ beta_full)   # (샘플 수, 클래스 수)
                pred_ldp  = self._softmax(X_ldp_te  @ beta_ldp)

                # 예측 클래스 (가장 높은 확률을 가진 클래스)
                yhat = {
                    'Full': np.argmax(pred_full, axis=1),
                    'LDP':  np.argmax(pred_ldp,  axis=1)
                }
                # 예측 확률 (각 클래스에 대한 확률)
                proba = {'Full': pred_full, 'LDP': pred_ldp}

                # 각 모델(Full, LDP)에 대한 성능 지표 계산
                for m in yhat:
                    y_true = y_full_te if m == 'Full' else y_ldp_te
                    labels  = np.unique(y_true) # 실제 레이블에 존재하는 클래스

                    # ----- 클래스별 One-vs-Rest AUC 계산 -----
                    auc_per_cls = []
                    for c in labels:
                        y_bin   = (y_true == c).astype(int)        # 현재 클래스를 1, 나머지를 0으로
                        score_c = proba[m][:, c]                   # 해당 클래스의 예측 확률
                        
                        # 이진 분류를 위한 ROC AUC 계산 시 양성/음성 샘플이 모두 존재해야 함
                        if y_bin.min() == y_bin.max(): # 한 종류의 레이블만 존재하는 경우
                            auc_val = float('nan')
                            warnings.warn(f"AUC for class {c} ({m}): only one label present → NaN.")
                        else:
                            auc_val = roc_auc_score(y_bin, score_c)
                        auc_per_cls.append(auc_val)

                    # 매크로 평균 AUC (NaN 값은 무시)
                    auc_macro = (np.nanmean(auc_per_cls)
                                 if not all(np.isnan(auc_per_cls)) else float('nan'))

                    # ----- 메트릭 딕셔너리 생성 -----
                    metrics_dict = {
                        'Accuracy':  accuracy_score(y_true, yhat[m]),
                        'Precision': precision_score(y_true, yhat[m],
                                                     average='macro', # 매크로 평균 정밀도
                                                     zero_division=0),
                        'Recall':    recall_score(y_true, yhat[m],
                                                     average='macro'), # 매크로 평균 재현율
                        'F1':        f1_score(y_true, yhat[m],
                                                     average='macro'), # 매크로 평균 F1-Score
                        'AUC_macro': round(auc_macro, 3)
                    }
                    # 클래스별 AUC 값을 결과 딕셔너리에 추가
                    for cls, auc_val in zip(labels, auc_per_cls):
                        metrics_dict[f'AUC_c{cls}'] = round(auc_val, 3)

                    data[m] = metrics_dict

            # 결과를 DataFrame으로 변환 후 소수점 셋째 자리까지 반올림
            df_scores = pd.DataFrame(data).T.round(3)
            
            return df_scores # 튜닝을 위해 결과를 반환
        except Exception as e:
            print('Error during run():', e)
            traceback.print_exc()
            return None # 오류 발생 시 None 반환

# ===================================================================
# 결과 파일 관리 유틸리티
# ===================================================================
def get_total_features(csv_path: Path, label_col: str) -> int:
    """데이터셋의 최종 피처 개수 계산"""
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include='number')
    miss_cols = num_cols.columns[num_cols.isna().mean() > 0.4]
    final_cols = num_cols.drop(columns=miss_cols)
    return len([col for col in final_cols.columns if col != label_col])

def _get_last_meta_info(file_path: Path) -> tuple[str | None, int | None, float | None]:
    """
    결과 파일에서 마지막으로 기록된 메타 정보를 읽어옵니다.
    데이터셋 이름, N 값, 정규화 람다 값을 반환합니다.
    """
    last_dataset = None
    last_N = None
    last_lambda_reg = None
    
    if not file_path.exists():
        return None, None, None
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # 파일 끝에서부터 역순으로 탐색하여 마지막 메타 블록을 찾음
            for i in reversed(range(len(lines))):
                line = lines[i].strip()
                if line.startswith("N:"):
                    try:
                        last_N = int(line.split(":")[1].strip())
                    except ValueError:
                        last_N = None
                elif line.startswith("Dataset:"):
                    last_dataset = line.split(":")[1].strip()
                elif line.startswith("Regularization Lambda:"):
                    try:
                        last_lambda_reg = float(line.split(":")[1].strip())
                    except ValueError:
                        last_lambda_reg = None
                elif line.startswith("="*50): # 메타 블록의 시작 구분자
                    # 필요한 모든 메타 정보를 찾았으면 반환
                    if last_dataset is not None and last_N is not None and last_lambda_reg is not None:
                        return last_dataset, last_N, last_lambda_reg
                    else:
                        # 이 블록에서 모두 찾지 못했다면 다음 블록을 탐색 (초기화)
                        last_dataset, last_N, last_lambda_reg = None, None, None
                # 데이터 라인이나 eps 라인을 만나면 마지막 완전한 메타 블록을 이미 통과했음을 의미
                if (line.startswith("Model,") or line.strip().startswith("Full,") or 
                    line.strip().startswith("LDP,") or line.startswith("eps (per-feature epsilon):")):
                    return None, None, None # 유효한 메타 블록 이전에 데이터 시작
    except Exception as e:
        print(f"Error reading last meta info from {file_path}: {e}")
    return None, None, None # 찾지 못했거나 오류 발생 시 None 반환

def manage_result_file_header(result_file: Path, config: argparse.Namespace, total_features: int):
    """실험 결과 파일의 헤더를 관리 (필요시 작성)"""
    current_dataset_name = Path(config.csv_path).stem
    current_N = config.N
    current_lambda_reg = config.regularization_lambda

    # 파일의 마지막 메타 정보와 현재 실행의 메타 정보가 다를 경우 새 블록 작성
    last_dataset, last_N, last_lambda_reg = _get_last_meta_info(result_file)
    
    should_write_new_meta_block = True
    if (last_dataset == current_dataset_name and 
        last_N == current_N and 
        last_lambda_reg == current_lambda_reg):
        should_write_new_meta_block = False
    
    # 필요한 경우, 현재 스크립트 실행을 위한 초기 메타 정보 및 CSV 헤더 작성
    if should_write_new_meta_block:
        header_content = (
            f"\n{'='*50}\n"
            f"Dataset: {current_dataset_name}\n"
            f"Total Features: {total_features}\n"
            f"N: {current_N} / label_N: {config.label_N}\n"
            f"Regularization Lambda: {config.regularization_lambda}\n"
            f"--- Classification Results by Epsilon ---\n"
            f"Model,Accuracy,Precision,Recall,F1,AUC_macro,AUC_c0,AUC_c1,AUC_c2,AUC_c3,AUC_c4,AUC_c5\n"
        )
        with result_file.open('a') as f: # 항상 파일에 추가 모드로 열기
            f.write(header_content)

# ===================================================================
# 메인 실행 로직
# ===================================================================
def main(args: argparse.Namespace):
    """스크립트의 메인 실행 로직"""
    
    result_file_path = Path(args.result_csv)
    os.makedirs(result_file_path.parent, exist_ok=True) # 결과 파일 디렉토리 생성
    
    # 데이터셋의 전체 피처 수 계산
    total_features = get_total_features(Path(args.csv_path), args.label_col)

    # 결과 파일 헤더 관리
    manage_result_file_header(result_file_path, args, total_features)

    seeds_to_run = args.seeds
    eps_list = args.eps_list
    label_eps_list = args.label_eps_list

    # 각 특징별 epsilon (ep)과 레이블 epsilon (label_ep) 조합에 대해 실험 실행
    for ep in eps_list:
        args.eps = ep # 현재 루프의 특징 ε 설정
        for label_ep in label_eps_list:
            args.label_eps = label_ep
            print(f"\n--- eps(특징): {ep}, eps(레이블): {label_ep} 실행 중 ({len(seeds_to_run)}개 시드 평균) ---")

            # 결과 파일에 현재 엡실론 조합 헤더 추가
            with result_file_path.open('a') as f:
                f.write(f"eps(per-feature): {ep} / eps(label): {label_ep}\n")

            try:
                df_list = []
                for seed in seeds_to_run:
                    # config 객체에 현재 시드와 데이터셋 이름을 추가
                    current_config = args
                    current_config.seed = seed
                    current_config.dataset = Path(args.csv_path).stem # dataset 이름 설정
                    
                    # LogisticExperiment 인스턴스 생성 및 실행
                    exp = LogisticExperiment(current_config)
                    res = exp.run() # 실험 실행 및 결과 반환
                    if res is not None:
                        df_list.append(res) # 성공적인 결과만 리스트에 추가
                
                if not df_list: # 실행 결과가 없으면 다음 엡실론으로 건너뛰기
                    print(f"경고: eps {ep}, label_eps {label_ep} 에 대해 유효한 결과가 없습니다. 다음으로 건너뜁니다.")
                    continue

                # 여러 시드의 결과를 평균하여 최종 결과 계산
                concat = pd.concat(df_list)
                mean_df = (concat.groupby(concat.index)
                           .mean(numeric_only = True) # 숫자형 컬럼만 평균
                           .round(3)
                           .reset_index()
                           .rename(columns={'index': 'Model'}))
                
                # 결과 CSV에 쓸 컬럼 목록 정의 (AUC_c0~c5까지 포함)
                cols_to_write = [
                        'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC_macro',
                        'AUC_c0', 'AUC_c1', 'AUC_c2', 'AUC_c3', 'AUC_c4', 'AUC_c5'
                    ]
                
                # 결과 DataFrame에 없는 AUC 클래스 컬럼은 NaN으로 추가하여 일관된 형식 유지
                for col in cols_to_write:
                        if col not in mean_df.columns:
                            mean_df[col] = np.nan

                # 결과를 CSV 파일에 추가 (헤더 없이, 인덱스 없이)
                mean_df[cols_to_write].to_csv(result_file_path,
                                              mode='a', header=False, index=False)
                
            except Exception as ex:
                print(f'Error in __main__ loop for epsilon(feature) {ep}, epsilon(label) {label_ep}:', ex)
                traceback.print_exc()

    print("\n✅ 모든 작업이 성공적으로 완료되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LDP & HE Linear Regression GD Comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- 실험 설정 인자 ---
    parser.add_argument('--eps_list', type=float, nargs='+', default=[1.0,2.0,3.0,4.0,5.0], help='공백으로 구분된 피처 epsilon 목록')
    parser.add_argument('--label_eps_list', type=float, nargs='+', default=[1.0,3.0,5.0], help='공백으로 구분된 레이블 epsilon 목록')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='실행할 랜덤 시드 목록 (공백으로 구분)')
    
    # --- 데이터 및 경로 인자 ---
    parser.add_argument('--csv_path', type=str, default='data/gamma.csv', help='입력 원본 데이터 CSV 경로')
    parser.add_argument('--label_col', type=str, default='label', help='레이블 칼럼 이름')
    parser.add_argument('--output_dir', type=str, default='transformed_data_batch_label', help='변환된 데이터가 있는 디렉토리')
    parser.add_argument('--result_csv', type=str, default='results/results_logistic_multi_label.csv', help='최종 결과 저장 CSV 경로')
    
    # --- 모델 하이퍼파라미터 ---
    parser.add_argument('--learning_rate', type=float, default=0.05, help='경사 하강법 학습률')
    parser.add_argument('--epochs', type=int, default=1000, help='학습 에포크 수')
    parser.add_argument('--regularization_lambda', type=float, default=0.1, help='L2 정규화 강도')

    # --- LDP 파라미터 ---
    parser.add_argument('--N', type=int, default=7, help='피처 LDP 메커니즘의 해상도')
    parser.add_argument('--label_N', type=int, default=7, help='레이블 LDP 메커니즘의 해상도')

    args = parser.parse_args()
    
    # 변환된 데이터가 저장된 하위 디렉토리 경로 설정
    args.output_dir = Path(args.output_dir) / Path(args.csv_path).stem
    
    main(args)