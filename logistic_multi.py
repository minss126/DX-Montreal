import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import traceback
import warnings
import json
import logging

# ===================================================================
# HE 연산 플레이스홀더
# ===================================================================
def he_dot(a_enc: np.ndarray, b_enc: np.ndarray) -> np.ndarray:
    return a_enc @ b_enc

def he_decrypt_vector(v_enc: np.ndarray) -> np.ndarray:
    return v_enc

'''
def _print_dist(tag: str, arr: np.ndarray):
    arr = np.asarray(arr).flatten().tolist()
    cnt = Counter(int(x) for x in arr)
    print(f"{tag} 분포: {dict(cnt)}")
'''

# ===================================================================
# 로지스틱 회귀 실험 클래스
# ===================================================================
class LogisticExperiment:

    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.output_dir = Path(config.output_dir)
        np.random.seed(config.seed)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def _train_gd(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples = len(y)
        cfg = self.config

        # 원-핫 인코딩
        if y.ndim == 1: # 정수형 레이블 (원-핫 인코딩 X)
            K = len(np.unique(y))
            y_onehot = np.zeros((n_samples, K))
            y_onehot[np.arange(n_samples), y.astype(int)] = 1
        else: # 원-핫 인코딩되어 있음
            y_onehot = y
            K = y_onehot.shape[1]

        beta = np.zeros((X.shape[1], K))
        for _ in range(cfg.epochs):
            p = self._softmax(X @ beta)
            
            # L2 정규화 (절편(bias) 항은 제외)
            grad_unregularized = X.T @ (p - y_onehot) / n_samples
            grad_regularization_term = (cfg.regularization_lambda * beta)
            grad_regularization_term[0, :] = 0

            total_grad = grad_unregularized + grad_regularization_term
            beta -= cfg.learning_rate * total_grad

        return beta


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

    def run(self) -> pd.DataFrame:
        try:
            # --- 1. 데이터 로드 및 전처리 ---
            cfg = self.config
            base_filename_stem = Path(cfg.csv_path).stem
            base_filename = (f"{base_filename_stem}_Eps{cfg.eps[0]:.1f}_N{cfg.N}_avg_"
                             f"Leps{cfg.label_eps[0]:.1f}_LN{cfg.label_N}_seed{cfg.seed}")
            
            # 원본 데이터 로드
            orig = pd.read_csv(cfg.csv_path)
            num_cols = orig.select_dtypes(include='number')
            drop_cols = num_cols.columns[num_cols.isna().mean() > 0.4]
            orig_full = orig.drop(columns=drop_cols).select_dtypes(include='number')
            # print('Original label counts:', orig_full[cfg.label_col].value_counts().to_dict()) # 디버깅용

            # LDP 변환 데이터 로드
            train_ldp = pd.read_csv(self.output_dir / f"{base_filename}_train.csv", index_col=0).select_dtypes(include='number')
            test_ldp = pd.read_csv(self.output_dir / f"{base_filename}_test.csv", index_col=0).select_dtypes(include='number')
            
            # 원본 데이터를 LDP 데이터와 동일한 인덱스로 정렬
            train_orig = orig_full.loc[train_ldp.index].reset_index(drop=True)
            test_orig = orig_full.loc[test_ldp.index].reset_index(drop=True)

            # 결측치 처리 및 스케일링
            feats = [c for c in train_ldp.columns if c != cfg.label_col]
            train_orig_imp, test_orig_imp = self._impute(train_orig, test_orig)
            train_ldp_imp, test_ldp_imp = self._impute(train_ldp, test_ldp)
            train_orig_scaled, test_orig_scaled = self._scale(train_orig_imp[feats], test_orig_imp[feats])
            train_ldp_scaled, test_ldp_scaled = self._scale(train_ldp_imp[feats], test_ldp_imp[feats])

            # 절편(bias) 항 추가 및 데이터/레이블 분리
            add_bias = lambda X: np.hstack([np.ones((X.shape[0], 1)), X])
            X_orig_train, X_orig_test = add_bias(train_orig_scaled.values), add_bias(test_orig_scaled.values)
            X_ldp_train, X_ldp_test = add_bias(train_ldp_scaled.values), add_bias(test_ldp_scaled.values)
            y_orig_train, y_orig_test = train_orig_imp[cfg.label_col].values, test_orig_imp[cfg.label_col].values
            y_ldp_train, y_ldp_test = train_ldp_imp[cfg.label_col].values, test_ldp_imp[cfg.label_col].values

            # 레이블을 0부터 K-1까지의 정수로 통일 (Ordinal Encoding)
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
            y_orig_train = oe.fit_transform(y_orig_train.reshape(-1,1)).ravel()
            y_orig_test = oe.transform(y_orig_test.reshape(-1,1)).ravel()

            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
            y_ldp_train = oe.fit_transform(y_ldp_train.reshape(-1,1)).ravel()
            y_ldp_test = oe.transform(y_ldp_test.reshape(-1,1)).ravel()

            # --- 2. 모델 학습 ---
            beta_orig = self._train_gd(X_orig_train, y_orig_train)
            beta_ldp  = self._train_gd(X_ldp_train,  y_ldp_train)

            # --- 3. 예측 및 평가 ---
            data = {} # 결과 저장용 딕셔너리
            
            pred_orig = self._softmax(X_orig_test @ beta_orig)
            pred_ldp  = self._softmax(X_ldp_test @ beta_ldp)

            # 예측 클래스 (가장 높은 확률을 가진 클래스)
            yhat = {
                'Full': np.argmax(pred_orig, axis=1),
                'LDP':  np.argmax(pred_ldp,  axis=1)
            }
            # 예측 확률 (각 클래스에 대한 확률)
            proba = {'Full': pred_orig, 'LDP': pred_ldp}

            # 각 모델에 대한 성능 지표 계산
            for m in yhat:
                y_true = y_orig_test if m == 'Full' else y_ldp_test
                labels  = np.unique(y_true)

                # ----- 클래스별 One-vs-Rest AUC 계산 -----
                auc_per_cls = []
                for c in labels:
                    y_bin   = (y_true == c).astype(int)
                    score_c = proba[m][:, c]
                    
                    if y_bin.min() == y_bin.max(): # 한 종류의 레이블만 존재
                        auc_val = float('nan')
                        warnings.warn(f"AUC for class {c} ({m}): only one label present → NaN.")
                    else:
                        auc_val = roc_auc_score(y_bin, score_c)
                    auc_per_cls.append(auc_val)

                # 매크로 평균 AUC
                auc_macro = (np.nanmean(auc_per_cls)
                                if not all(np.isnan(auc_per_cls)) else float('nan'))

                # ----- 메트릭 딕셔너리 생성 -----
                metrics_dict = {
                    'Accuracy':  accuracy_score(y_true, yhat[m]),
                    'AUC_macro': round(auc_macro, 3)
                }

                data[m] = metrics_dict

            df_scores = pd.DataFrame(data).T.round(3)
            logging.info("Model Performance (Regression):\n%s", df_scores)
            return df_scores
        
        except Exception as e:
            logging.error(f'실험 실행 중 오류 발생: {e}')
            logging.error(traceback.format_exc())
            return None

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

# ===================================================================
# 메인 실행 로직
# ===================================================================
def main(args: argparse.Namespace):
    # 1. 기존 CSV 결과 파일 준비 (헤더 작성)
    total_result_file = Path(args.total_result_csv)
    total_result_file.parent.mkdir(parents=True, exist_ok=True)
    total_features = get_total_features(Path(args.csv_path), args.label_col)

    with open(total_result_file, 'a', newline='') as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"N: {args.N} / label_N: {args.label_N} / eps: {args.eps[0]} / label_eps: {args.label_eps[0]}\n")

    # 2. label 변환 여부 별로 각 Epsilon 조합에 대해 실험 반복
    # 2.1. 현재 실험 설정을 기반으로 결과 폴더 경로 생성
    exp_name = (f"eps{args.eps[0]}_label-eps{args.label_eps[0]}_N{args.N}_LN{args.label_N}_"
                f"lr{args.learning_rate}_reg{args.regularization_lambda}")
    exp_dir = Path(args.result_dir) / Path(args.csv_path).stem / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 2.2. 로거 설정 (실험 폴더에 run.log 파일 생성)
    log_file = exp_dir / 'run.log'
    # 이전 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"\n--- Running for eps={args.eps}, label_eps={args.label_eps} (Avg over {len(args.seeds)} seeds) ---")
    logging.info(f"Results will be saved in: {exp_dir}")

    # 2.3. 메타데이터 저장 (metadata.json)
    # seed는 개별 실행마다 다르므로, 메타데이터에서는 리스트로 전체를 보여줌
    metadata = vars(args).copy()
    metadata['total_features'] = total_features
    
    serializable_metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata.items()
    }

    with open(exp_dir / 'metadata.json', 'w') as f:
        json.dump(serializable_metadata, f, indent=4)

    # 2.4. 각 시드에 대해 개별 실험 실행
    for seed in args.seeds:
        logging.info(f"--- Running seed: {seed} ---")
        current_config = argparse.Namespace(**metadata)
        current_config.eps = args.eps
        current_config.label_eps = args.label_eps
        current_config.seed = seed
        
        exp = LogisticExperiment(current_config)
        result_df = exp.run()

        # 2.5. 개별 시드 결과 저장 (results_seed_X.pkl)
        if result_df is not None:
            result_path = exp_dir / f'results_seed_{seed}.pkl'
            result_df.to_pickle(result_path)
            logging.info(f"Saved seed {seed} results to {result_path}")
    
    # 2.6. 모든 시드 결과 취합 및 평균 계산
    all_seed_results = []
    for pkl_file in exp_dir.glob("results_seed_*.pkl"):
        df = pd.read_pickle(pkl_file)
        all_seed_results.append(df)

    if not all_seed_results:
        logging.warning("해당 epsilon 설정에 대한 결과가 없습니다.")
    
    mean_df = (pd.concat(all_seed_results)
            .groupby(level=0)
            .mean()
            .round(3)
            .reset_index()
            .rename(columns={'index': 'Model'}))
    
    logging.info("--- Averaged Results ---")
    logging.info("\n%s", mean_df.to_string())

    # 2.7. 평균 결과를 pkl 및 csv로 저장
    mean_df.to_pickle(exp_dir / 'results_mean.pkl')
    mean_df.to_csv(exp_dir / 'results_mean.csv', index=False)
    
    # 2.8. (유지보수) 기존의 통합 CSV 파일에 평균 결과 추가
    with open(total_result_file, 'a', newline='') as f:
        f.write(f"\nModel, Accuracy, AUC_macro\n")
        mean_df.to_csv(f, header=False, index=False, lineterminator='\n')

    print("\n✅ 모든 작업이 성공적으로 완료되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LDP & HE Multi Logistic Regression GD Comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- 실험 설정 ---
    parser.add_argument('--eps', type=float, nargs='+', default=3.0, help='공백으로 구분된 피처 epsilon 목록')
    parser.add_argument('--label_eps', type=float, nargs='+', default=3.0, help='공백으로 구분된 레이블 epsilon 목록')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='실행할 랜덤 시드 목록 (공백으로 구분)')
    
    # --- 데이터 및 경로 ---
    parser.add_argument('--csv_path', type=str, default='data/wine.csv', help='입력 원본 데이터 CSV 경로')
    parser.add_argument('--label_col', type=str, default='label', help='레이블 칼럼 이름')
    parser.add_argument('--output_dir', type=str, default='transformed_data_batch_label', help='LDP 변환 데이터가 있는 루트 디렉토리')
    parser.add_argument('--result_dir', type=str, default='results_structured', help='구조화된 결과가 저장될 루트 디렉토리')
    parser.add_argument('--total_result_csv', type=str, default='results/results_logistic_multi_label_summary.csv', help='(유지보수용) 통합 결과 요약 CSV 경로')
    
    # --- 모델 하이퍼파라미터 ---
    parser.add_argument('--learning_rate', type=float, default=0.05, help='경사 하강법 학습률')
    parser.add_argument('--epochs', type=int, default=1000, help='학습 에포크 수')
    parser.add_argument('--regularization_lambda', type=float, default=0.1, help='L2 정규화 강도')

    # --- LDP 파라미터 ---
    parser.add_argument('--N', type=int, default=7, help='피처 LDP 메커니즘 해상도')
    parser.add_argument('--label_N', type=int, default=7, help='레이블 LDP 메커니즘 해상도')

    args = parser.parse_args()
    #args.output_dir = Path(args.output_dir) / Path(args.csv_path).stem
    
    main(args)
