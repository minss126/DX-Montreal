import argparse
import traceback
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ===================================================================
# HE 연산 플레이스홀더
# ===================================================================
# HE(동형암호) 기반 훈련은 구현되지 않았으므로, 함수는 플레이스홀더로 유지합니다.
def he_dot(a_enc: np.ndarray, b_enc: np.ndarray) -> np.ndarray:
    return a_enc @ b_enc

def he_decrypt_vector(v_enc: np.ndarray) -> np.ndarray:
    return v_enc

# ===================================================================
# 선형 회귀 실험 클래스
# ===================================================================
class LinearExperiment:
    """LDP로 보호된 데이터셋을 사용하여 선형 회귀 모델을 학습하고 평가하는 실험을 관리합니다."""

    def __init__(self, config: argparse.Namespace):
        """
        실험 설정을 초기화합니다.
        
        Args:
            config (argparse.Namespace): 스크립트 실행 시 전달된 모든 인자.
        """
        self.config = config
        self.output_dir = Path(config.output_dir)

        # 재현성을 위한 시드 설정
        np.random.seed(config.seed)

    def _train_gd(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        L2 정규화(Ridge)를 적용한 경사 하강법으로 선형 회귀 모델을 학습합니다.
        β_{t+1} = β_t − α · [(2/n) · Xᵀ (Xβ_t − y) + 2λβ_t]
        """
        beta = np.zeros(X.shape[1])
        n_samples = len(y)
        
        for _ in range(self.config.epochs):
            residual = X @ beta - y
            grad_unregularized = 2 * X.T @ residual / n_samples
            
            # L2 정규화 (절편(bias) 항은 정규화에서 제외)
            grad_regularization = 2 * self.config.regularization_lambda * beta
            grad_regularization[0] = 0  # 절편 항의 정규화 기울기를 0으로 설정
            
            total_grad = grad_unregularized + grad_regularization
            beta -= self.config.learning_rate * total_grad
            
        return beta

    def _impute(self, df_train: pd.DataFrame, df_apply: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train 데이터의 평균으로 결측치 대체."""
        train_mean = df_train.mean()
        return df_train.fillna(train_mean), df_apply.fillna(train_mean)

    def _scale(self, df_train: pd.DataFrame, df_apply: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """StandardScaler로 피처 스케일링 (평균 0, 표준편차 1)."""
        scaler = StandardScaler()
        df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns, index=df_train.index)
        df_apply_scaled = pd.DataFrame(scaler.transform(df_apply), columns=df_apply.columns, index=df_apply.index)
        return df_train_scaled.fillna(0), df_apply_scaled.fillna(0)

    def run(self) -> pd.DataFrame:
        """실험의 전체 과정을 실행하고 결과를 반환합니다."""
        try:
            # --- 1. 데이터 로드 및 전처리 ---
            cfg = self.config
            base_filename = (f"{Path(cfg.csv_path).stem}_Eps{cfg.eps:.1f}_N{cfg.N}_avg_"
                             f"Leps{cfg.label_eps}_LN{cfg.label_N}_seed{cfg.seed}")

            # 원본 데이터 로드 및 정제
            orig_df = pd.read_csv(cfg.csv_path)
            num_cols = orig_df.select_dtypes(include='number')
            miss_cols = num_cols.columns[num_cols.isna().mean() > 0.4]
            orig_full_df = orig_df.drop(columns=miss_cols).select_dtypes(include='number')

            # LDP 변환 데이터 로드
            train_ldp_df = pd.read_csv(self.output_dir / f"{base_filename}_train.csv", index_col=0).select_dtypes(include='number')
            test_ldp_df = pd.read_csv(self.output_dir / f"{base_filename}_test.csv", index_col=0).select_dtypes(include='number')
            
            # 원본 데이터를 LDP 데이터와 동일한 인덱스로 정렬
            train_orig_df = orig_full_df.loc[train_ldp_df.index].reset_index(drop=True)
            test_orig_df = orig_full_df.loc[test_ldp_df.index].reset_index(drop=True)
            
            # 결측치 처리 및 스케일링
            feats = [c for c in train_ldp_df.columns if c != cfg.label_col]
            train_orig_imp, test_orig_imp = self._impute(train_orig_df, test_orig_df)
            train_ldp_imp, test_ldp_imp = self._impute(train_ldp_df, test_ldp_df)

            train_orig_scaled, test_orig_scaled = self._scale(train_orig_imp[feats], test_orig_imp[feats])
            train_ldp_scaled, test_ldp_scaled = self._scale(train_ldp_imp[feats], test_ldp_imp[feats])

            # 절편(bias) 항 추가 및 데이터/레이블 분리
            add_bias = lambda X: np.hstack([np.ones((X.shape[0], 1)), X])
            X_orig_train, X_orig_test = add_bias(train_orig_scaled.values), add_bias(test_orig_scaled.values)
            X_ldp_train, X_ldp_test = add_bias(train_ldp_scaled.values), add_bias(test_ldp_scaled.values)
            
            y_orig_train, y_orig_test = train_orig_imp[cfg.label_col].values, test_orig_imp[cfg.label_col].values
            y_ldp_train, y_ldp_test = train_ldp_imp[cfg.label_col].values, test_ldp_imp[cfg.label_col].values
            
            # --- 2. 모델 학습 ---
            beta_orig = self._train_gd(X_orig_train, y_orig_train)
            beta_ldp = self._train_gd(X_ldp_train, y_ldp_train)

            # --- 3. 예측 및 평가 ---
            pred_orig = X_orig_test @ beta_orig
            pred_ldp = X_ldp_test @ beta_ldp

            def get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
                return {
                    'MSE': mean_squared_error(y_true, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'MAE': mean_absolute_error(y_true, y_pred),
                    'R2': r2_score(y_true, y_pred)
                }

            results = {
                'Full': get_regression_metrics(y_orig_test, pred_orig),
                'LDP': get_regression_metrics(y_ldp_test, pred_ldp) # LDP 모델은 원본 테스트 레이블로 평가
            }
            df_scores = pd.DataFrame(results).T.round(4)
            print("Model Performance (Regression):\n", df_scores)
            return df_scores
        
        except Exception as e:
            print(f'실험 실행 중 오류 발생: {e}')
            traceback.print_exc()
            return None

# ===================================================================
# 결과 파일 관리 유틸리티
# ===================================================================
def get_total_features(csv_path: Path, label_col: str) -> int:
    """데이터셋에서 최종적으로 사용될 피처의 개수를 계산합니다."""
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include='number')
    miss_cols = num_cols.columns[num_cols.isna().mean() > 0.4]
    final_cols = num_cols.drop(columns=miss_cols)
    feature_cols = [col for col in final_cols.columns if col != label_col]
    return len(feature_cols)

def write_metadata_header(result_file: Path, config: argparse.Namespace, total_features: int):
    """실험 결과 파일에 메타데이터 헤더를 작성합니다."""
    with open(result_file, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"Dataset: {Path(config.csv_path).stem}\n")
        f.write(f"Total Features: {total_features}\n")
        f.write(f"N: {config.N} / label_N: {config.label_N}\n")
        f.write(f"Regularization Lambda: {config.regularization_lambda}\n")
        f.write("--- Regression Results by Epsilon ---\n")
        f.write("Model,MSE,RMSE,MAE,R2\n")

# ===================================================================
# 메인 실행 로직
# ===================================================================
def main(args: argparse.Namespace):
    """스크립트의 메인 실행 로직."""
    result_file = Path(args.result_csv)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋의 전체 피처 수 계산
    total_features = get_total_features(Path(args.csv_path), args.label_col)
    
    # 메타데이터 헤더 작성
    write_metadata_header(result_file, args, total_features)
    
    seeds_to_run = args.seeds
    eps_list = args.eps_list
    label_eps_list = args.label_eps_list

    # 각 Epsilon 조합에 대해 실험 반복
    for eps in eps_list:
        for label_eps in label_eps_list:
            print(f"\n--- Running for eps={eps}, label_eps={label_eps} (Avg over {len(seeds_to_run)} seeds) ---")
            
            with open(result_file, 'a') as f:
                f.write(f"eps(per-feature): {eps} / eps(label): {label_eps}\n")

            try:
                all_seed_results = []
                for seed in seeds_to_run:
                    # 현재 루프의 파라미터로 config 업데이트
                    current_config = args
                    current_config.eps = eps
                    current_config.label_eps = label_eps
                    current_config.seed = seed
                    
                    exp = LinearExperiment(current_config)
                    result_df = exp.run()
                    if result_df is not None:
                        all_seed_results.append(result_df)
                
                if not all_seed_results:
                    print("해당 epsilon 설정에 대한 결과가 없습니다.")
                    continue

                # 여러 시드의 결과 평균 계산
                mean_df = (pd.concat(all_seed_results)
                           .groupby(level=0)
                           .mean()
                           .round(3)
                           .reset_index()
                           .rename(columns={'index': 'Model'}))
                
                # 평균 결과 저장
                mean_df.to_csv(result_file, mode='a', header=False, index=False)

            except Exception as e:
                print(f'Epsilon {eps}, Label Epsilon {label_eps} 루프에서 오류 발생:', e)
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
    parser.add_argument('--csv_path', type=str, default='data/elevators.csv', help='입력 원본 데이터 CSV 경로')
    parser.add_argument('--label_col', type=str, default='label', help='레이블 칼럼 이름')
    parser.add_argument('--output_dir', type=str, default='transformed_data_batch_label', help='변환된 데이터가 있는 디렉토리')
    parser.add_argument('--result_csv', type=str, default='results/results_linear_label.csv', help='최종 결과 저장 CSV 경로')
    
    # --- 모델 하이퍼파라미터 ---
    parser.add_argument('--learning_rate', type=float, default=0.05, help='경사 하강법 학습률')
    parser.add_argument('--epochs', type=int, default=1000, help='학습 에포크 수')
    parser.add_argument('--regularization_lambda', type=float, default=0.1, help='L2 정규화 강도')

    # --- LDP 파라미터 ---
    parser.add_argument('--N', type=int, default=31, help='피처 LDP 메커니즘의 해상도')
    parser.add_argument('--label_N', type=int, default=15, help='레이블 LDP 메커니즘의 해상도')

    args = parser.parse_args()
    
    # 변환된 데이터가 저장된 하위 디렉토리 경로 설정
    args.output_dir = Path(args.output_dir) / Path(args.csv_path).stem
    
    main(args)