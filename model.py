import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score, mean_squared_error,
                             mean_absolute_error, roc_curve)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import traceback
import warnings
import json
import logging

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

        base_output_dir = Path(config.output_dir)
        dataset_name = Path(config.csv_path).stem
        self.output_dir = base_output_dir / dataset_name / config.mechanism
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
        """ 모델 타입에 따라 적절한 경사 하강법으로 모델을 학습시킵니다. """
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
            for _ in range(cfg.epochs):
                permutation = np.random.permutation(n_samples)
                X_shuffled, y_shuffled = X[permutation], y[permutation]
                for i in range(0, n_samples, batch_size):
                    X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                    n_batch_samples = len(y_batch)
                    if n_batch_samples == 0: continue
                    
                    residual = X_batch @ beta - y_batch
                    grad_unregularized = 2 * X_batch.T @ residual / n_batch_samples
                    grad_regularization = 2 * cfg.regularization_lambda * beta
                    grad_regularization[0] = 0
                    
                    total_grad = grad_unregularized + grad_regularization
                    beta -= cfg.learning_rate * total_grad
            return beta

        # --- Logistic Regression (Binary Classification) ---
        elif cfg.model_type == 'logistic':
            beta = np.zeros(X.shape[1])
            for _ in range(cfg.epochs):
                # 데이터 셔플링
                permutation = np.random.permutation(n_samples)
                X_shuffled, y_shuffled = X[permutation], y[permutation]
                
                # 미니배치 루프
                for i in range(0, n_samples, batch_size):
                    X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                    n_batch_samples = len(y_batch)
                    if n_batch_samples == 0: continue

                    # 배치 단위로 그래디언트 계산
                    p = self._sigmoid(X_batch @ beta)
                    grad_unregularized = X_batch.T @ (p - y_batch) / n_batch_samples
                    grad_regularization = cfg.regularization_lambda * beta
                    grad_regularization[0] = 0
                    
                    total_grad = grad_unregularized + grad_regularization
                    beta -= cfg.learning_rate * total_grad
            return beta

        # --- Multi-class Logistic Regression ---
        elif cfg.model_type == 'logistic_multi':
            if y.ndim == 1:
                K = len(np.unique(y))
                y_onehot = np.zeros((n_samples, K))
                y_onehot[np.arange(n_samples), y.astype(int)] = 1
            else:
                y_onehot = y
                K = y_onehot.shape[1]

            beta = np.zeros((X.shape[1], K))
            for _ in range(cfg.epochs):
                # 데이터 셔플링
                permutation = np.random.permutation(n_samples)
                X_shuffled, y_onehot_shuffled = X[permutation], y_onehot[permutation]

                # 미니배치 루프
                for i in range(0, n_samples, batch_size):
                    X_batch, y_batch_onehot = X_shuffled[i:i+batch_size], y_onehot_shuffled[i:i+batch_size]
                    n_batch_samples = len(y_batch_onehot)
                    if n_batch_samples == 0: continue
                
                    # 배치 단위로 그래디언트 계산
                    p = self._softmax(X_batch @ beta)
                    grad_unregularized = X_batch.T @ (p - y_batch_onehot) / n_batch_samples
                    grad_regularization_term = (cfg.regularization_lambda * beta)
                    grad_regularization_term[0, :] = 0
                    
                    total_grad = grad_unregularized + grad_regularization_term
                    beta -= cfg.learning_rate * total_grad
            return beta
            
        else:
            raise ValueError(f"Unknown model_type: {cfg.model_type}")

    def _impute(self, df_train: pd.DataFrame, df_apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Train 데이터의 평균으로 결측치 대체 """
        train_mean = df_train.mean()
        return df_train.fillna(train_mean), df_apply.fillna(train_mean)

    def _scale(self, df_train: pd.DataFrame, df_apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ 피처 스케일링 : StandardScaler """
        scaler = StandardScaler()
        train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns, index=df_train.index)
        apply_scaled = pd.DataFrame(scaler.transform(df_apply), columns=df_apply.columns, index=df_apply.index)
        return train_scaled.fillna(0), apply_scaled.fillna(0)
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """ 최적의 분류 임계값 (Youden's J statistic) """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        if len(thresholds) == 0: return 0.5
        youden_j = tpr - fpr
        return thresholds[np.argmax(youden_j)]

    def run(self) -> pd.DataFrame:
        try:
            # --- 1. 데이터 로드 및 전처리 ---
            cfg = self.config
            base_filename_stem = Path(cfg.csv_path).stem
            if cfg.mechanism == 'qm':
                base_filename = (f"{base_filename_stem}_Eps{cfg.eps:.1f}_N{cfg.N}_avg_"
                                f"Leps{cfg.label_eps:.1f}_LN{cfg.label_N}_seed{cfg.seed}")
            elif cfg.mechanism == 'pm':
                base_filename = (f"{base_filename_stem}_Eps{cfg.eps:.1f}_t{cfg.t}_"
                                f"Leps{cfg.label_eps:.1f}_seed{cfg.seed}")

            orig = pd.read_csv(cfg.csv_path)
            num_cols = orig.select_dtypes(include='number')
            drop_cols = num_cols.columns[num_cols.isna().mean() > 0.4]
            orig_full = orig.drop(columns=drop_cols).select_dtypes(include='number')

            train_ldp = pd.read_csv(self.output_dir / f"{base_filename}_train.csv", index_col=0).select_dtypes(include='number')
            test_ldp = pd.read_csv(self.output_dir / f"{base_filename}_test.csv", index_col=0).select_dtypes(include='number')
            
            train_orig = orig_full.loc[train_ldp.index].reset_index(drop=True)
            test_orig = orig_full.loc[test_ldp.index].reset_index(drop=True)

            feats = [c for c in train_ldp.columns if c != cfg.label_col]
            train_orig_imp, test_orig_imp = self._impute(train_orig, test_orig)
            train_ldp_imp, test_ldp_imp = self._impute(train_ldp, test_ldp)
            train_orig_scaled, test_orig_scaled = self._scale(train_orig_imp[feats], test_orig_imp[feats])
            train_ldp_scaled, test_ldp_scaled = self._scale(train_ldp_imp[feats], test_ldp_imp[feats])

            add_bias = lambda X: np.hstack([np.ones((X.shape[0], 1)), X])
            X_orig_train, X_orig_test = add_bias(train_orig_scaled.values), add_bias(test_orig_scaled.values)
            X_ldp_train, X_ldp_test = add_bias(train_ldp_scaled.values), add_bias(test_ldp_scaled.values)
            y_orig_train, y_orig_test = train_orig_imp[cfg.label_col].values, test_orig_imp[cfg.label_col].values
            y_ldp_train, y_ldp_test = train_ldp_imp[cfg.label_col].values, test_ldp_imp[cfg.label_col].values

            # 모델별 레이블 전처리
            if cfg.model_type == 'linear' and cfg.transform_label_log:
                y_orig_train, y_orig_test = np.log1p(y_orig_train), np.log1p(y_orig_test)
            elif cfg.model_type == 'logistic_multi':
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
                y_orig_train = oe.fit_transform(y_orig_train.reshape(-1,1)).ravel()
                y_orig_test = oe.transform(y_orig_test.reshape(-1,1)).ravel()
                
                oe_ldp = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
                y_ldp_train = oe_ldp.fit_transform(y_ldp_train.reshape(-1,1)).ravel()
                y_ldp_test = oe_ldp.transform(y_ldp_test.reshape(-1,1)).ravel()

            # --- 2. 모델 학습 ---
            beta_orig = self._train_gd(X_orig_train, y_orig_train)
            beta_ldp = self._train_gd(X_ldp_train, y_ldp_train)

            # --- 3. 예측 및 평가 ---
            data = {}
            if cfg.model_type == 'linear':
                pred_orig = X_orig_test @ beta_orig
                pred_ldp = X_ldp_test @ beta_ldp
                if cfg.transform_label_log:
                    y_orig_test = np.expm1(y_orig_test)
                    pred_orig, pred_ldp = np.expm1(pred_orig), np.expm1(pred_ldp)
                
                metrics_orig = {
                    'MSE': mean_squared_error(y_orig_test, pred_orig),
                    'RMSE': np.sqrt(mean_squared_error(y_orig_test, pred_orig)),
                    'MAE': mean_absolute_error(y_orig_test, pred_orig)
                }
                metrics_ldp = {
                    'MSE': mean_squared_error(y_ldp_test, pred_ldp),
                    'RMSE': np.sqrt(mean_squared_error(y_ldp_test, pred_ldp)),
                    'MAE': mean_absolute_error(y_ldp_test, pred_ldp)
                }
                data['Full'] = metrics_orig
                data['LDP'] = metrics_ldp
                df_scores = pd.DataFrame(data).T.round(3)

            elif cfg.model_type == 'logistic':
                prob_orig = self._sigmoid(X_orig_test @ beta_orig)
                prob_ldp = self._sigmoid(X_ldp_test @ beta_ldp)
                thr_orig = self._find_optimal_threshold(y_orig_test, prob_orig)
                thr_ldp = self._find_optimal_threshold(y_ldp_test, prob_ldp)
                pred_orig_class = (prob_orig >= thr_orig).astype(int)
                pred_ldp_class = (prob_ldp >= thr_ldp).astype(int)
                
                data['Full'] = {'Accuracy': accuracy_score(y_orig_test, pred_orig_class), 'AUC': roc_auc_score(y_orig_test, prob_orig)}
                data['LDP'] = {'Accuracy': accuracy_score(y_ldp_test, pred_ldp_class), 'AUC': roc_auc_score(y_ldp_test, prob_ldp)}
                df_scores = pd.DataFrame(data).T.round(3)

            elif cfg.model_type == 'logistic_multi':
                pred_orig_proba = self._softmax(X_orig_test @ beta_orig)
                pred_ldp_proba = self._softmax(X_ldp_test @ beta_ldp)
                pred_orig_class = np.argmax(pred_orig_proba, axis=1)
                pred_ldp_class = np.argmax(pred_ldp_proba, axis=1)

                for m, y_true, proba, y_hat in [('Full', y_orig_test, pred_orig_proba, pred_orig_class), 
                                               ('LDP', y_ldp_test, pred_ldp_proba, pred_ldp_class)]:
                    labels = np.unique(y_true)
                    auc_per_cls = []
                    for c in labels:
                        y_bin = (y_true == c).astype(int)
                        score_c = proba[:, c]
                        if len(np.unique(y_bin)) < 2:
                            auc_val = float('nan')
                        else:
                            auc_val = roc_auc_score(y_bin, score_c)
                        auc_per_cls.append(auc_val)
                    
                    auc_macro = np.nanmean(auc_per_cls) if not all(np.isnan(auc_per_cls)) else float('nan')
                    data[m] = {'Accuracy': accuracy_score(y_true, y_hat), 'AUC_macro': auc_macro}
                df_scores = pd.DataFrame(data).T.round(3)

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
    # 1. 기존 CSV 결과 파일 준비 (헤더 작성)
    base_result_dir = Path(args.total_result_dir)
    dataset_name= Path(args.csv_path).stem
    total_result_file = base_result_dir / dataset_name / 'result_summary.csv'
    total_result_file.parent.mkdir(parents=True, exist_ok=True)
    total_features = get_total_features(Path(args.csv_path), args.label_col)

    if args.mechanism == 'qm':
        with open(total_result_file, 'a', newline='') as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"[batch: {args.batch_size} / lr: {args.learning_rate} / epochs: {args.epochs}]\n"
                    f"eps: {args.eps} / label_eps: {args.label_eps} / N: {args.N} / label_N: {args.label_N} \n")

        exp_name = (f"eps{args.eps}_label-eps{args.label_eps}_N{args.N}_LN{args.label_N}_"
                    f"lr{args.learning_rate}_bs{args.batch_size}_ep{args.epochs}")
    elif args.mechanism == 'pm':
        with open(total_result_file, 'a', newline='') as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"[batch: {args.batch_size} / lr: {args.learning_rate} / epochs: {args.epochs}]\n"
                    f"eps: {args.eps} / label_eps: {args.label_eps} / t: {args.t} \n")

        exp_name = (f"eps{args.eps}_label-eps{args.label_eps}_t{args.t}_"
                    f"lr{args.learning_rate}_bs{args.batch_size}_ep{args.epochs}")
    exp_dir = Path(args.result_dir) / Path(args.csv_path).stem / exp_name
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
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9],
                        help='실행할 랜덤 시드 목록 (공백으로 구분)')
    parser.add_argument('--transform_label_log', type=str2bool, default=False)

    
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

    # --- LDP 파라미터 (LDP Parameters) ---
    parser.add_argument('--mechanism', help='[qm, pm, duchi, to]')
    parser.add_argument('--t', type=int, help='LDP parameter for PM. t=2: PM, t=3: PM_sub')
    parser.add_argument('--eps', type=float, default=3.0, help='피처 epsilon 값')
    parser.add_argument('--label_eps', type=float, default=3.0, help='레이블 epsilon 값')
    parser.add_argument('--N', type=int, default=7, help='피처 LDP 메커니즘의 해상도')
    parser.add_argument('--label_N', type=int, default=2, help='레이블 LDP 메커니즘의 해상도')

    args = parser.parse_args()
    main(args)