import argparse
import traceback
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, roc_curve)
from sklearn.preprocessing import StandardScaler

# ===================================================================
# HE 연산 플레이스홀더
# ===================================================================
# 동형암호(HE) 기반 훈련은 현재 구현되지 않은 플레이스홀더입니다.
def he_dot(a_enc: np.ndarray, b_enc: np.ndarray) -> np.ndarray:
    return a_enc @ b_enc

def he_decrypt_vector(v_enc: np.ndarray) -> np.ndarray:
    return v_enc

# ===================================================================
# 로지스틱 회귀 실험 클래스
# ===================================================================
class LogisticExperiment:
    """LDP로 보호된 데이터셋으로 로지스틱 회귀 모델을 학습하고 평가합니다."""

    def __init__(self, config: argparse.Namespace):
        """실험 설정 초기화"""
        self.config = config
        self.output_dir = Path(config.output_dir)
        np.random.seed(config.seed)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """시그모이드 함수 (수치적 안정성을 위한 클리핑 포함)"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _train_gd(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """L2 정규화를 적용한 경사 하강법으로 모델 학습"""
        beta = np.zeros(X.shape[1])
        n_samples = len(y)
        cfg = self.config
        
        for _ in range(cfg.epochs):
            p = self._sigmoid(X @ beta)
            
            # 그래디언트 계산 (정규화 항 포함)
            grad_unregularized = X.T @ (p - y) / n_samples
            grad_regularization = cfg.regularization_lambda * beta
            grad_regularization[0] = 0  # 절편(bias) 항은 정규화에서 제외
            
            total_grad = grad_unregularized + grad_regularization
            beta -= cfg.learning_rate * total_grad
            
        return beta

    def _impute(self, df_train: pd.DataFrame, df_apply: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train 데이터의 평균으로 결측치 대체"""
        train_mean = df_train.mean()
        return df_train.fillna(train_mean), df_apply.fillna(train_mean)

    def _scale(self, df_train: pd.DataFrame, df_apply: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """StandardScaler로 피처 스케일링 (평균 0, 표준편차 1)"""
        scaler = StandardScaler()
        train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns, index=df_train.index)
        apply_scaled = pd.DataFrame(scaler.transform(df_apply), columns=df_apply.columns, index=df_apply.index)
        return train_scaled.fillna(0), apply_scaled.fillna(0)
        
    def _find_optimal_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Youden's J statistic을 이용한 최적의 분류 임계값 탐색"""
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden_j = tpr - fpr
        return thresholds[np.argmax(youden_j)]

    def run(self) -> pd.DataFrame:
        """실험 전체 과정 실행 및 결과 반환"""
        try:
            # --- 1. 데이터 준비 ---
            cfg = self.config
            base_filename = (f"{Path(cfg.csv_path).stem}_Eps{cfg.eps:.1f}_N{cfg.N}_avg_"
                             f"Leps{cfg.label_eps}_LN{cfg.label_N}_seed{cfg.seed}")

            # 원본 및 LDP 데이터 로드
            orig_df = pd.read_csv(cfg.csv_path).select_dtypes(include='number')
            train_ldp_df = pd.read_csv(self.output_dir / f"{base_filename}_train.csv", index_col=0).select_dtypes(include='number')
            test_ldp_df = pd.read_csv(self.output_dir / f"{base_filename}_test.csv", index_col=0).select_dtypes(include='number')
            
            # 원본 데이터를 LDP 데이터와 동일한 인덱스로 정렬
            train_orig_df = orig_df.loc[train_ldp_df.index].reset_index(drop=True)
            test_orig_df = orig_df.loc[test_ldp_df.index].reset_index(drop=True)
            
            # 결측치 처리 및 스케일링
            feats = [c for c in train_ldp_df.columns if c != cfg.label_col]
            train_orig_imp, test_orig_imp = self.impute(train_orig_df, test_orig_df)
            train_ldp_imp, test_ldp_imp = self.impute(train_ldp_df, test_ldp_df)
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
            prob_orig = self._sigmoid(X_orig_test @ beta_orig)
            prob_ldp = self._sigmoid(X_ldp_test @ beta_ldp)
            
            # 최적 임계값으로 예측값 이진화
            thr_orig = self._find_optimal_threshold(y_orig_test, prob_orig)
            thr_ldp = self._find_optimal_threshold(y_ldp_test, prob_ldp)
            pred_orig_class = (prob_orig >= thr_orig).astype(int)
            pred_ldp_class = (prob_ldp >= thr_ldp).astype(int)
            
            def get_classification_metrics(y_true: np.ndarray, y_pred_class: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
                if len(np.unique(y_true)) < 2:  # 레이블이 한 종류일 경우
                    return {'Accuracy': accuracy_score(y_true, y_pred_class), 'AUC': np.nan, 'Precision': np.nan, 'Recall': np.nan, 'F1': np.nan}
                return {
                    'Accuracy': accuracy_score(y_true, y_pred_class),
                    'AUC': roc_auc_score(y_true, y_pred_prob),
                    'Precision': precision_score(y_true, y_pred_class, zero_division=0),
                    'Recall': recall_score(y_true, y_pred_class, zero_division=0),
                    'F1': f1_score(y_true, y_pred_class, zero_division=0)
                }

            results = {
                'Full': get_classification_metrics(y_orig_test, pred_orig_class, prob_orig),
                'LDP': get_classification_metrics(y_ldp_test, pred_ldp_class, prob_ldp)
            }
            df_scores = pd.DataFrame(results).T.round(3)
            print("Model Performance (Classification):\n", df_scores)
            return df_scores
        
        except Exception as e:
            print(f'실험 실행 중 오류 발생: {e}')
            traceback.print_exc()
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

def manage_result_file_header(result_file: Path, config: argparse.Namespace):
    """실험 결과 파일의 헤더를 관리 (필요시 작성)"""
    header_content = (
        f"\n{'='*50}\n"
        f"Dataset: {Path(config.csv_path).stem}\n"
        f"Total Features: {get_total_features(Path(config.csv_path), config.label_col)}\n"
        f"N: {config.N} / label_N: {config.label_N}\n"
        f"Regularization Lambda: {config.regularization_lambda}\n"
        f"--- Classification Results by Epsilon ---\n"
        f"Model,Accuracy,AUC,Precision,Recall,F1\n"
    )
    
    # 파일이 없거나 내용이 비어있으면 새로 작성
    if not result_file.exists() or result_file.stat().st_size == 0:
        result_file.write_text(header_content)
    else:
        # 파일 내용은 있지만, 현재 데이터셋에 대한 헤더가 없으면 추가
        content = result_file.read_text()
        if f"Dataset: {Path(config.csv_path).stem}" not in content:
            with result_file.open('a') as f:
                f.write(header_content)

# ===================================================================
# 메인 실행 로직
# ===================================================================
def main(args: argparse.Namespace):
    """스크립트의 메인 실행 로직"""
    result_file = Path(args.result_csv)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 결과 파일 헤더 관리
    manage_result_file_header(result_file, args)
    
    seeds_to_run = args.seeds
    eps_list = args.eps_list
    label_eps_list = args.label_eps_list

    # 각 Epsilon 조합에 대해 실험 반복
    for eps in eps_list:
        for label_eps in label_eps_list:
            print(f"\n--- Running for eps={eps}, label_eps={label_eps} (Avg over {len(seeds_to_run)} seeds) ---")
            
            with result_file.open('a') as f:
                f.write(f"eps(per-feature): {eps} / eps(label): {label_eps}\n")

            try:
                all_seed_results = []
                for seed in seeds_to_run:
                    # 현재 루프의 파라미터로 config 업데이트
                    current_config = args
                    current_config.eps = eps
                    current_config.label_eps = label_eps
                    current_config.seed = seed
                    
                    exp = LogisticExperiment(current_config)
                    result_df = exp.run()
                    if result_df is not None:
                        all_seed_results.append(result_df)
                
                if not all_seed_results: continue

                # 여러 시드의 결과 평균 계산 및 저장
                mean_df = (pd.concat(all_seed_results)
                           .groupby(level=0).mean().round(3)
                           .reset_index().rename(columns={'index': 'Model'}))
                mean_df.to_csv(result_file, mode='a', header=False, index=False)

            except Exception as e:
                print(f'Epsilon {eps}, Label Epsilon {label_eps} 루프에서 오류 발생:', e)

    print("\n✅ 모든 작업이 성공적으로 완료되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LDP & HE Logistic Regression GD Comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- 실험 설정 ---
    parser.add_argument('--eps_list', type=float, nargs='+', default=[1.0,2.0,3.0,4.0,5.0], help='공백으로 구분된 피처 epsilon 목록')
    parser.add_argument('--label_eps_list', type=float, nargs='+', default=[1.0,3.0,5.0], help='공백으로 구분된 레이블 epsilon 목록')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='실행할 랜덤 시드 목록 (공백으로 구분)')
    
    # --- 데이터 및 경로 ---
    parser.add_argument('--csv_path', type=str, default='data/gamma.csv', help='입력 원본 데이터 CSV 경로')
    parser.add_argument('--label_col', type=str, default='label', help='레이블 칼럼 이름')
    parser.add_argument('--output_dir', type=str, default='transformed_data_batch_label', help='변환된 데이터가 있는 디렉토리')
    parser.add_argument('--result_csv', type=str, default='results/results_logistic_label.csv', help='최종 결과 저장 CSV 경로')
    
    # --- 모델 하이퍼파라미터 ---
    parser.add_argument('--learning_rate', type=float, default=0.05, help='경사 하강법 학습률')
    parser.add_argument('--epochs', type=int, default=1000, help='학습 에포크 수')
    parser.add_argument('--regularization_lambda', type=float, default=0.1, help='L2 정규화 강도')

    # --- LDP 파라미터 ---
    parser.add_argument('--N', type=int, default=31, help='피처 LDP 메커니즘 해상도')
    parser.add_argument('--label_N', type=int, default=2, help='레이블 LDP 메커니즘 해상도')

    args = parser.parse_args()
    
    # 변환된 데이터가 저장된 하위 디렉토리 경로 설정
    args.output_dir = Path(args.output_dir) / Path(args.csv_path).stem
    
    main(args)