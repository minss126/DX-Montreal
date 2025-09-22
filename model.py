import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score, mean_squared_error,
                             mean_absolute_error, roc_curve)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
import traceback
import warnings
import json
import logging
import pickle

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# ===================================================================
# HE 연산 플레이스홀더 (Placeholder for Homomorphic Encryption Operations)
# ===================================================================
def he_dot(a_enc: np.ndarray, b_enc: np.ndarray) -> np.ndarray:
    """ Placeholder for HE dot product """
    return a_enc @ b_enc

def he_decrypt_vector(v_enc: np.ndarray) -> np.ndarray:
    """ Placeholder for HE vector decryption """
    return v_enc

# ===================================================================
# 통합 회귀 실험 클래스 (Unified Regression Experiment Class)
# ===================================================================
class UnifiedExperiment:

    def __init__(self, config: argparse.Namespace):
        self.config = config
        np.random.seed(config.seed)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    # model.py 파일의 UnifiedExperiment 클래스 내부

    def _train_gd(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ 모델 타입에 따라 적절한 경사 하강법으로 모델을 학습시킵니다. (마지막 불완전 배치 제외) """
        cfg = self.config
        n_samples = len(y)

        if cfg.batch_size <= 0:
            batch_size = n_samples
            if cfg.seed == 0:
                logging.info(f"Batch size <= 0. Switched to full-batch mode (batch_size={n_samples}).")
        else:
            batch_size = cfg.batch_size

        # --- Linear Regression (Mini-Batch GD) ---
        if cfg.model_type == 'linear':
            beta = np.zeros(X.shape[1])
            beta[0] = np.mean(y)
            for _ in range(cfg.epochs):
                permutation = np.random.permutation(n_samples)
                X_shuffled, y_shuffled = X[permutation], y[permutation]
                
                # 마지막 불완전한 배치를 버리기 위해 n_samples // batch_size 로 몫만 취함
                n_batches = n_samples // batch_size
                for i in range(n_batches):
                    start = i * batch_size
                    end = start + batch_size
                    X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]
                    
                    residual = X_batch @ beta - y_batch
                    # n_batch_samples 대신 batch_size 사용
                    grad_unregularized = 2 * X_batch.T @ residual / batch_size
                    grad_regularization = 2 * cfg.regularization_lambda * beta
                    grad_regularization[0] = 0
                    
                    total_grad = grad_unregularized + grad_regularization

                    grad_norm = np.linalg.norm(total_grad)
                    if grad_norm > cfg.gradient_clip_val:
                        total_grad = total_grad * cfg.gradient_clip_val / grad_norm

                    beta -= cfg.learning_rate * total_grad
                    beta = np.clip(beta, -cfg.parameter_clip_val, cfg.parameter_clip_val)
            return beta

        # --- Logistic Regression (Binary Classification) ---
        elif cfg.model_type == 'logistic':
            beta = np.zeros(X.shape[1])
            p = np.mean(y)
            p = np.clip(p, 1e-9, 1 - 1e-9) # log(0) 또는 0으로 나누기 방지를 위한 클리핑
            beta[0] = np.log(p / (1 - p))
            for _ in range(cfg.epochs):
                permutation = np.random.permutation(n_samples)
                X_shuffled, y_shuffled = X[permutation], y[permutation]
                
                # 마지막 불완전한 배치를 버리기 위해 n_samples // batch_size 로 몫만 취함
                n_batches = n_samples // batch_size
                for i in range(n_batches):
                    start = i * batch_size
                    end = start + batch_size
                    X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]

                    p = self._sigmoid(X_batch @ beta)
                    # n_batch_samples 대신 batch_size 사용
                    grad_unregularized = X_batch.T @ (p - y_batch) / batch_size
                    grad_regularization = cfg.regularization_lambda * beta
                    grad_regularization[0] = 0
                    
                    total_grad = grad_unregularized + grad_regularization

                    grad_norm = np.linalg.norm(total_grad)
                    if grad_norm > cfg.gradient_clip_val:
                        total_grad = total_grad * cfg.gradient_clip_val / grad_norm

                    beta -= cfg.learning_rate * total_grad
                    beta = np.clip(beta, -cfg.parameter_clip_val, cfg.parameter_clip_val)
            return beta

        # --- Multi-class Logistic Regression ---
        elif cfg.model_type == 'logistic_multi':
            if y.ndim == 1:
                unique_classes = np.unique(y)
                class_map = {val: i for i, val in enumerate(unique_classes)}
                y_mapped = np.array([class_map[v] for v in y])
                K = len(unique_classes)
                y_onehot = np.zeros((n_samples, K))
                y_onehot[np.arange(n_samples), y_mapped.astype(int)] = 1
            else:
                y_onehot = y
                K = y_onehot.shape[1]

            beta = np.zeros((X.shape[1], K))
            # y_onehot의 열별 평균 = 각 클래스의 비율(빈도)
            class_proportions = np.mean(y_onehot, axis=0)
            class_proportions = np.clip(class_proportions, 1e-9, 1 - 1e-9) # 수치 안정성 확보
            beta[0, :] = np.log(class_proportions) # beta의 첫 행이 모든 클래스의 편향에 해당
            for _ in range(cfg.epochs):
                permutation = np.random.permutation(n_samples)
                X_shuffled, y_onehot_shuffled = X[permutation], y_onehot[permutation]

                # 마지막 불완전한 배치를 버리기 위해 n_samples // batch_size 로 몫만 취함
                n_batches = n_samples // batch_size
                for i in range(n_batches):
                    start = i * batch_size
                    end = start + batch_size
                    X_batch, y_batch_onehot = X_shuffled[start:end], y_onehot_shuffled[start:end]
                
                    p = self._softmax(X_batch @ beta)
                    # n_batch_samples 대신 batch_size 사용
                    grad_unregularized = X_batch.T @ (p - y_batch_onehot) / batch_size
                    grad_regularization_term = (cfg.regularization_lambda * beta)
                    grad_regularization_term[0, :] = 0
                    
                    total_grad = grad_unregularized + grad_regularization_term

                    grad_norm = np.linalg.norm(total_grad)
                    if grad_norm > cfg.gradient_clip_val:
                        total_grad = total_grad * cfg.gradient_clip_val / grad_norm

                    beta -= cfg.learning_rate * total_grad
                    beta = np.clip(beta, -cfg.parameter_clip_val, cfg.parameter_clip_val)
            return beta
            
        else:
            raise ValueError(f"Unknown model_type: {cfg.model_type}")

    def _impute(self, df_train: pd.DataFrame, df_apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Train 데이터의 평균으로 결측치 대체 """
        train_mean = df_train.mean()
        return df_train.fillna(train_mean), df_apply.fillna(train_mean)

    def _scale(self, df_train: pd.DataFrame, df_apply: pd.DataFrame, mechanism_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ 
        피처 스케일링을 적용합니다.
        - 'pm': MinMaxScaler만 적용
        - 그 외: StandardScaler -> MinMaxScaler 순차 적용
        """
        # --- 2. MinMaxScaler 설정 (공통) ---
        n_val = self.config.N
        feature_min = -(n_val - 1) / 2
        feature_max = (n_val - 1) / 2
        minmax_scaler = MinMaxScaler(feature_range=(feature_min, feature_max))

        # 원본 데이터프레임의 컬럼과 인덱스 정보 유지
        train_columns, train_index = df_train.columns, df_train.index
        apply_columns, apply_index = df_apply.columns, df_apply.index

        # --- 1. mechanism_type에 따라 스케일링 분기 ---
        if mechanism_type == 'pm':
            logging.info(f"Applying MinMaxScaler ONLY for PM. N={n_val}, range=({feature_min}, {feature_max})")
            
            train_final_scaled_np = minmax_scaler.fit_transform(df_train)
            apply_final_scaled_np = minmax_scaler.transform(df_apply)
        else:
            logging.info(f"Applying StandardScaler -> MinMaxScaler for '{mechanism_type}'. N={n_val}, range=({feature_min}, {feature_max})")

            # StandardScaler 먼저 적용
            std_scaler = StandardScaler()
            train_std_scaled_np = std_scaler.fit_transform(df_train)
            apply_std_scaled_np = std_scaler.transform(df_apply)
            
            # 그 결과에 MinMaxScaler 적용
            train_final_scaled_np = minmax_scaler.fit_transform(train_std_scaled_np)
            apply_final_scaled_np = minmax_scaler.transform(apply_std_scaled_np)

        # --- 최종 결과를 다시 DataFrame으로 변환 ---
        train_scaled = pd.DataFrame(train_final_scaled_np, columns=train_columns, index=train_index)
        apply_scaled = pd.DataFrame(apply_final_scaled_np, columns=apply_columns, index=apply_index)

        return train_scaled.fillna(0), apply_scaled.fillna(0)
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """ 최적의 분류 임계값 (Youden's J statistic) """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        if len(thresholds) == 0: return 0.5
        youden_j = tpr - fpr
        return thresholds[np.argmax(youden_j)]

    # UnifiedExperiment 클래스 내부의 run 함수
    def run(self) -> pd.DataFrame:
        try:
            cfg = self.config
            # --- 1. 데이터 로드 및 분할 ---
            orig = pd.read_csv(cfg.csv_path)
            num_cols = orig.select_dtypes(include='number')
            drop_cols = num_cols.columns[num_cols.isna().mean() > 0.4]
            orig_full = orig.drop(columns=drop_cols).select_dtypes(include='number')
            
            # LDP 데이터 중 하나를 기준으로 train/test 인덱스를 가져옴
            # (어떤 메커니즘이든 인덱스는 동일해야 함)
            first_mechanism_key = cfg.mechanisms[0]
            if first_mechanism_key.startswith('pm'):
                mechanism_name = 'pm'
                t_val = int(first_mechanism_key.split('_t')[-1])
                base_filename = f"{Path(cfg.csv_path).stem}_Eps{cfg.eps:.1f}_t{t_val}_Leps{cfg.label_eps:.1f}_seed{cfg.seed}"
            else: # qm
                mechanism_name = 'qm'
                base_filename = f"{Path(cfg.csv_path).stem}_Eps{cfg.eps:.1f}_N{cfg.N}_{cfg.obj}_Leps{cfg.label_eps:.1f}_LN{cfg.label_N}_seed{cfg.seed}"
            
            base_output_dir = Path(cfg.output_dir)
            sub_dir = ""
            # 선형 회귀 모델일 때만 inversion_mode를 적용
            if cfg.model_type == 'linear':
                if cfg.inversion_mode == 'index':
                    sub_dir = "inverse_index"
                elif cfg.inversion_mode == 'linear':
                    sub_dir = "inverse_linear"

            # 최종 데이터 경로 설정
            output_dir = base_output_dir / sub_dir / Path(cfg.csv_path).stem / mechanism_name

            train_indices = pd.read_csv(output_dir / f"{base_filename}_train.csv", index_col=0).index
            test_indices = pd.read_csv(output_dir / f"{base_filename}_test.csv", index_col=0).index

            train_orig = orig_full.loc[train_indices].reset_index(drop=True)
            test_orig = orig_full.loc[test_indices].reset_index(drop=True)

            # --- 2. 원본(Original) 데이터 전처리 및 모델 학습 ---
            logging.info("--- Processing Original Data ---")
            feats = [c for c in train_orig.columns if c != cfg.label_col]
            train_orig_imp, test_orig_imp = self._impute(train_orig, test_orig)
            train_orig_scaled, test_orig_scaled = self._scale(train_orig_imp[feats], test_orig_imp[feats], mechanism_type='original')
            add_bias = lambda X: np.hstack([np.ones((X.shape[0], 1)), X])
            X_orig_train, X_orig_test = add_bias(train_orig_scaled.values), add_bias(test_orig_scaled.values)
            y_orig_train, y_orig_test = train_orig_imp[cfg.label_col].values, test_orig_imp[cfg.label_col].values

            # --- 모델별 원본 레이블 전처리 ---
            if cfg.model_type == 'linear' and cfg.transform_label_log:
                y_orig_train, y_orig_test = np.log1p(y_orig_train), np.log1p(y_orig_test)
            elif cfg.model_type == 'logistic_multi':
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
                y_orig_train = oe.fit_transform(y_orig_train.reshape(-1, 1)).ravel()
                y_orig_test = oe.transform(y_orig_test.reshape(-1, 1)).ravel()

            beta_orig = self._train_gd(X_orig_train, y_orig_train)
            
            # --- 3. LDP 메커니즘 순회 처리 ---
            ldp_results_to_evaluate = {}
            label_inverse_meta = {} # [추가] 역변환 메타데이터를 저장할 딕셔너리
            for mechanism_key in cfg.mechanisms:
                logging.info(f"--- Processing LDP mechanism: {mechanism_key} ---")
                
                # 메커니즘 키 파싱 및 파일명 생성
                t_val = None
                if mechanism_key.startswith('pm'):
                    mechanism_name = 'pm'
                    t_val = int(mechanism_key.split('_t')[-1])
                    # PM 파일명 형식
                    base_filename = f"{Path(cfg.csv_path).stem}_Eps{cfg.eps:.1f}_t{t_val}_Leps{cfg.label_eps:.1f}_seed{cfg.seed}"
                else: # qm
                    mechanism_name = 'qm'
                    # QM 파일명 형식
                    base_filename = f"{Path(cfg.csv_path).stem}_Eps{cfg.eps:.1f}_N{cfg.N}_{cfg.obj}_Leps{cfg.label_eps:.1f}_LN{cfg.label_N}_seed{cfg.seed}"

                # 현재 메커니즘에 맞는 최종 데이터 경로를 루프 안에서 설정
                current_output_dir = base_output_dir / sub_dir / Path(cfg.csv_path).stem / mechanism_name

                # 이름 형식을 만드는 로직을 더 명확하게 수정합니다.
                if 'pm' in mechanism_key:
                    # 'pm_t2'를 'PM'과 '2'로 분리
                    parts = mechanism_key.split('_t')
                    mech_name = parts[0].upper() # 'PM'
                    t_val = parts[1] # '2'
                    # f-string을 사용해 'LDP-PM(T2)' 형태로 조합
                    ldp_name = f"LDP-{mech_name}(T{t_val})"
                else:
                    # 'qm' -> 'LDP-QM'
                    ldp_name = f"LDP-{mechanism_key.upper()}"

                # 메타데이터 파일 로드 시도
                metadata_path = current_output_dir / f"{base_filename}_metadata.pkl"
                if metadata_path.exists():
                    with open(metadata_path, 'rb') as f:
                        label_inverse_meta[ldp_name] = pickle.load(f)
                    logging.info(f"Loaded label inversion metadata for '{ldp_name}'")
                
                # LDP 데이터 로드 및 전처리
                train_ldp = pd.read_csv(current_output_dir / f"{base_filename}_train.csv", index_col=0)
                test_ldp = pd.read_csv(current_output_dir / f"{base_filename}_test.csv", index_col=0)
                train_ldp_imp, test_ldp_imp = self._impute(train_ldp, test_ldp)
                if mechanism_name == 'pm':
                    train_ldp_scaled, test_ldp_scaled = self._scale(train_ldp_imp[feats], test_ldp_imp[feats], mechanism_type=mechanism_name)
                    X_ldp_train, X_ldp_test = add_bias(train_ldp_scaled.values), add_bias(test_ldp_scaled.values)
                else:  # QM은 스케일링하지 않음
                    logging.info(f"Skipping scaling for {mechanism_name.upper()} mechanism.")
                    X_ldp_train, X_ldp_test = add_bias(train_ldp_imp[feats].values), add_bias(test_ldp_imp[feats].values)
                y_ldp_train, y_ldp_test = train_ldp_imp[cfg.label_col].values, test_ldp_imp[cfg.label_col].values
                
                # --- ★ 모델별 LDP 레이블 전처리 ★ ---
                if cfg.model_type == 'linear' and cfg.transform_label_log:
                    # 메타데이터가 존재하면(즉, label_index=True이면) log 변환을 건너뜀
                    if ldp_name not in label_inverse_meta:
                        y_ldp_train = np.log1p(y_ldp_train)

                if cfg.model_type == 'logistic_multi':
                    oe_ldp = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
                    y_ldp_train = oe_ldp.fit_transform(y_ldp_train.reshape(-1, 1)).ravel()
                    y_ldp_test = oe_ldp.transform(y_ldp_test.reshape(-1, 1)).ravel()

                beta_ldp = self._train_gd(X_ldp_train, y_ldp_train)
                
                ldp_results_to_evaluate[ldp_name] = {'beta': beta_ldp, 'X_test': X_ldp_test, 'y_test': y_ldp_test}

            # --- 4. 모든 결과 통합 평가 ---
            final_scores = {}
            if cfg.model_type == 'linear':
                # 원본 평가
                pred_orig = X_orig_test @ beta_orig
                if cfg.transform_label_log:
                    y_orig_test, pred_orig = np.expm1(y_orig_test), np.expm1(pred_orig)
                final_scores['Full'] = {'RMSE': np.sqrt(mean_squared_error(y_orig_test, pred_orig)), 'MAE': mean_absolute_error(y_orig_test, pred_orig)}
                
                # LDP 평가
                for name, data in ldp_results_to_evaluate.items():
                    try:
                        # 루프 내에서 올바른 변수를 사용하도록 수정
                        pred_ldp = data['X_test'] @ data['beta']
                        y_ldp_test_local = data['y_test']

                        # [추가] 예측값 역변환 로직
                        if name in label_inverse_meta:
                            logging.info(f"Applying label inversion for '{name}' before evaluation.")
                            meta = label_inverse_meta[name]
                            idx_min, idx_max = meta['ldp_index_min'], meta['ldp_index_max']
                            val_min, val_max = meta['original_label_min'], meta['original_label_max']

                            # 예측값이 인덱스 범위를 벗어나지 않도록 클리핑
                            pred_ldp = np.clip(pred_ldp, idx_min, idx_max)
                            
                            if (idx_max - idx_min) > 1e-9: # 분모가 0이 되는 것 방지
                                # 선형 보간을 통해 원본 스케일로 복원
                                normalized_pos = (pred_ldp - idx_min) / (idx_max - idx_min)
                                pred_ldp = normalized_pos * (val_max - val_min) + val_min
                            else: # min, max가 같은 경우
                                pred_ldp = np.full_like(pred_ldp, val_min)

                        if cfg.transform_label_log:
                            y_ldp_test_local = np.expm1(y_ldp_test_local)
                            pred_ldp = np.expm1(pred_ldp)
                        
                        rmse = np.sqrt(mean_squared_error(y_orig_test, pred_ldp))
                        mae = mean_absolute_error(y_orig_test, pred_ldp)
                        final_scores[name] = {'RMSE': rmse, 'MAE': mae}
                    except (ValueError, OverflowError) as e:
                        logging.warning(f"Could not calculate metrics for '{name}' due to error: {e}. Recording as NaN.")
                        final_scores[name] = {'RMSE': np.nan, 'MAE': np.nan}

            elif cfg.model_type == 'logistic':
                # 원본 평가
                prob_orig = self._sigmoid(X_orig_test @ beta_orig)
                thr_orig = self._find_optimal_threshold(y_orig_test, prob_orig)
                pred_orig_class = (prob_orig >= thr_orig).astype(int)
                final_scores['Full'] = {'Accuracy': accuracy_score(y_orig_test, pred_orig_class), 'AUC': roc_auc_score(y_orig_test, prob_orig)}
                
                # LDP 평가
                for name, data in ldp_results_to_evaluate.items():
                    try:
                        # 루프 내에서 올바른 변수를 사용하도록 수정
                        y_ldp_test_local = data['y_test']
                        prob_ldp = self._sigmoid(data['X_test'] @ data['beta'])
                        thr_ldp = self._find_optimal_threshold(y_orig_test, prob_ldp)
                        pred_ldp_class = (prob_ldp >= thr_ldp).astype(int)
                        
                        accuracy = accuracy_score(y_orig_test, pred_ldp_class)
                        auc = roc_auc_score(y_orig_test, prob_ldp)
                        final_scores[name] = {'Accuracy': accuracy, 'AUC': auc}
                    except (ValueError, OverflowError) as e:
                        logging.warning(f"Could not calculate metrics for '{name}' due to error: {e}. Recording as NaN.")
                        final_scores[name] = {'Accuracy': np.nan, 'AUC': np.nan}


            elif cfg.model_type == 'logistic_multi':
                # --- 원본 평가 (Full) ---
                pred_orig_proba = self._softmax(X_orig_test @ beta_orig)
                y_true_full = y_orig_test.astype(int, copy=False)
                mask_full = (y_true_full != -1)  # OrdinalEncoder의 미지 클래스(-1) 제거

                if np.any(mask_full):
                    y_t = y_true_full[mask_full]
                    p   = pred_orig_proba[mask_full]
                    y_hat = np.argmax(p, axis=1)

                    # 클래스별 AUC → 매크로 평균
                    K = p.shape[1]
                    auc_list = []
                    for c in range(K):
                        y_bin = (y_t == c).astype(int)
                        if y_bin.min() == y_bin.max():  # 양/음이 모두 있어야 AUC 정의 가능
                            auc_list.append(float('nan'))
                        else:
                            auc_list.append(roc_auc_score(y_bin, p[:, c]))
                    auc_macro = np.nanmean(auc_list) if not np.all(np.isnan(auc_list)) else float('nan')

                    final_scores['Full'] = {
                        'Accuracy': accuracy_score(y_t, y_hat),
                        'AUC_macro': auc_macro
                    }
                else:
                    final_scores['Full'] = {'Accuracy': float('nan'), 'AUC_macro': float('nan')}

                # --- LDP 평가 ---
                for name, data in ldp_results_to_evaluate.items():
                    try:
                        proba = self._softmax(data['X_test'] @ data['beta'])
                        y_true_orig = y_orig_test.astype(int, copy=False)
                        mask = np.ones_like(y_true_orig, dtype=bool)

                        if np.any(mask):
                            y_t = y_true_orig[mask]
                            p   = proba[mask]
                            y_hat = np.argmax(p, axis=1)

                            K = p.shape[1]
                            auc_list = []
                            for c in range(K):
                                y_bin = (y_t == c).astype(int)
                                if y_bin.min() == y_bin.max():
                                    auc_list.append(float('nan'))
                                else:
                                    auc_list.append(roc_auc_score(y_bin, p[:, c]))
                            auc_macro = np.nanmean(auc_list) if not np.all(np.isnan(auc_list)) else float('nan')

                            final_scores[name] = {
                                'Accuracy': accuracy_score(y_t, y_hat),
                                'AUC_macro': auc_macro
                            }
                        else:
                            final_scores[name] = {'Accuracy': float('nan'), 'AUC_macro': float('nan')}
                    
                    except (ValueError, OverflowError) as e:
                        logging.warning(
                            f"Could not calculate metrics for '{name}' due to error: {e}. Recording as NaN."
                        )
                        final_scores[name] = {'Accuracy': np.nan, 'AUC_macro': np.nan}

            df_scores = pd.DataFrame(final_scores).T.round(3)
            logging.info("Model Performance:\n%s", df_scores)
            return df_scores
        
        except Exception as e:
            logging.error(f'실험 실행 중 오류 발생: {e}')
            logging.error(traceback.format_exc())
            return None

# ===================================================================
# 유틸리티 및 메인 실행 로직 (Utility and Main Execution Logic)
# ===================================================================
def get_total_features(csv_path: Path, label_col: str) -> int:
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include='number')
    miss_cols = num_cols.columns[num_cols.isna().mean() > 0.4]
    final_cols = num_cols.drop(columns=miss_cols)
    return len([col for col in final_cols.columns if col != label_col])

def main(args: argparse.Namespace):
    # 1. 통합 결과 파일에 공통 헤더 작성
    base_result_dir = Path(args.total_result_dir)
    dataset_name= Path(args.csv_path).stem
    total_result_file = base_result_dir / dataset_name / 'result_summary.csv'
    total_result_file.parent.mkdir(parents=True, exist_ok=True)
    total_features = get_total_features(Path(args.csv_path), args.label_col)

    with open(total_result_file, 'a', newline='') as f:
        f.write("\n" + "="*50 + "\n")
        # 모든 메커니즘에 적용되는 공통 파라미터 정보만 기록
        f.write(f"[batch: {args.batch_size} / lr: {args.learning_rate} / epochs: {args.epochs}]\n"
                f"eps: {args.eps} / label_eps: {args.label_eps} / N: {args.N} / label_N: {args.label_N}\n")

    # 2. exp_name 생성
    exp_name = (f"eps{args.eps}_label-eps{args.label_eps}_N{args.N}_LN{args.label_N}_"
                f"lr{args.learning_rate}_bs{args.batch_size}_ep{args.epochs}")

    # 3. 결과 디렉터리 설정
    exp_dir = Path(args.result_dir) / dataset_name / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_file = exp_dir / 'run.log'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    logging.info(f"\n--- Running Model: {args.model_type.upper()} for eps={args.eps}, label_eps={args.label_eps} (Avg over {len(args.seeds)} seeds) ---")
    logging.info(f"Results will be saved in: {exp_dir}")

    metadata = vars(args).copy()
    metadata['total_features'] = total_features
    serializable_metadata = {k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()}
    with open(exp_dir / 'metadata.json', 'w') as f:
        json.dump(serializable_metadata, f, indent=4)

    all_seed_results = []
    for seed in args.seeds:
        logging.info(f"--- Running seed: {seed} ---")
        current_config = argparse.Namespace(**metadata)
        current_config.eps = args.eps
        current_config.label_eps = args.label_eps
        current_config.seed = seed
        
        exp = UnifiedExperiment(current_config)
        result_df = exp.run()

        if result_df is not None:
            result_path = exp_dir / f'results_seed_{seed}.pkl'
            result_df.to_pickle(result_path)
            logging.info(f"Saved seed {seed} results to {result_path}")
            all_seed_results.append(result_df)

    if not all_seed_results:
        logging.warning("No results were generated for this configuration.")
        return

    mean_df = (pd.concat(all_seed_results).groupby(level=0).mean().round(3)
               .reset_index().rename(columns={'index': 'Data'}))
    
    logging.info("--- Averaged Results ---")
    logging.info("\n%s", mean_df.to_string())

    mean_df.to_pickle(exp_dir / 'results_mean.pkl')
    mean_df.to_csv(exp_dir / 'results_mean.csv', index=False)
    
    with open(total_result_file, 'a', newline='') as f:
        header_line = ",".join(mean_df.columns)
        f.write(f"\n{header_line}\n")
        mean_df.to_csv(f, header=False, index=False, lineterminator='\n')

    print("\n✅ All tasks completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unified LDP & HE Regression Model Comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- 핵심 실험 설정 (Core Experiment Settings) ---
    parser.add_argument('--model_type', type=str, default='logistic',
                        choices=['linear', 'logistic', 'logistic_multi'], help='모델 종류')
    parser.add_argument('--inversion_mode', type=str, default='index', choices=['index', 'linear'],
                        help='[회귀 모델 전용] inverse_index 또는 inverse_linear 폴더 중 선택')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9],
                        help='실행할 랜덤 시드 목록 (공백으로 구분)')
    parser.add_argument('--transform_label_log', type=str2bool, default=False)
    parser.add_argument('--obj', type=str, default='avg', choices=['avg', 'worst'])
    
    # --- 데이터 및 경로 (Data and Paths) ---
    parser.add_argument('--csv_path', type=str, default='data/gamma.csv', help='입력 원본 데이터 CSV 경로')
    parser.add_argument('--label_col', type=str, default='label', help='레이블 칼럼 이름')
    parser.add_argument('--output_dir', type=str, default='transformed_data_batch_label', help='LDP 변환 데이터가 있는 루트 디렉토리')
    parser.add_argument('--result_dir', type=str, default='results_unified', help='구조화된 결과가 저장될 루트 디렉토리')
    parser.add_argument('--total_result_dir', type=str, default='results_unified', help='통합 결과 요약 CSV 경로')
    
    # --- 모델 하이퍼파라미터 (Model Hyperparameters) ---
    parser.add_argument('--learning_rate', type=float, default=0.05, help='경사 하강법 학습률')
    parser.add_argument('--epochs', type=int, default=10, help='학습 에포크 수')
    parser.add_argument('--regularization_lambda', type=float, default=0.1, help='L2 정규화 강도')
    parser.add_argument('--batch_size', type=int, default=512, help='경사 하강법 배치 크기')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='그래디언트 클리핑 임계값 (L2 norm)')
    parser.add_argument('--parameter_clip_val', type=float, default=1.0, help='파라미터(가중치) 클리핑 절대값')

    # --- LDP 파라미터 (LDP Parameters) ---
    parser.add_argument('--mechanisms', type=str, nargs='+', default=['qm', 'pm_t3'],
                        help="비교할 LDP 메커니즘 목록. 예: qm pm_t2 pm_t3")
    #parser.add_argument('--t', type=int, help='LDP parameter for PM. t=2: PM, t=3: PM_sub')
    parser.add_argument('--eps', type=float, default=3.0, help='피처 epsilon 값')
    parser.add_argument('--label_eps', type=float, default=3.0, help='레이블 epsilon 값')
    parser.add_argument('--N', type=int, default=7, help='피처 LDP 메커니즘의 해상도')
    parser.add_argument('--label_N', type=int, default=2, help='레이블 LDP 메커니즘의 해상도')

    args = parser.parse_args()

    main(args)
