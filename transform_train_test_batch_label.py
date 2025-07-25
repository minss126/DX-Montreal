import make_mechanism_avg
import make_mechanism_worst

import numpy as np
import pandas as pd
import pickle, os, argparse, time, warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ===================================================================
# LDP 메커니즘 로더 클래스 (Numerical)
# ===================================================================
class LDPMechanism:
    def __init__(self, pkl_path):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"지정된 경로에 pkl 파일이 없습니다: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.a_values = data['a_values']
        self.p_matrix = data['p_matrix']
        self.metadata = data['metadata']
        self.n_param = self.metadata['n_param']
        self.epsilon = self.metadata['epsilon']
        self.N = self.metadata['N_total_a_points']
        
        self.x_values = self.a_values @ self.p_matrix
        if not np.all(np.diff(self.x_values) >= -1e-9):
            print("[Warning] x_values가 단조 증가하지 않아 정렬합니다.")
            sort_indices = np.argsort(self.x_values)
            self.x_values = self.x_values[sort_indices]
            self.p_matrix = self.p_matrix[:, sort_indices]

        if self.metadata['use_a0']:
            self.a_indices = np.arange(-self.n_param, self.n_param + 1)
        else:
            self.a_indices = np.concatenate([np.arange(-self.n_param, 0), np.arange(1, self.n_param + 1)])
        
        # ✨ 역변환을 위한 인덱스-값 맵
        self.a_index_to_value_map = {idx: val for idx, val in zip(self.a_indices, self.a_values)}
            
        print(f"Numerical LDP 메커니즘 로드 완료 (epsilon={self.epsilon}, N={self.N}).")
    
    def map_indices_to_values(self, indices):
        """ ✨ [역변환용] LDP 메커니즘이 출력한 정수 인덱스를 [-1, 1] 범위의 값으로 변환합니다. """
        return np.vectorize(self.a_index_to_value_map.get)(indices)

    def get_prob_vectors_batch(self, x_scaled_batch):
        x_scaled_batch = np.asarray(x_scaled_batch)
        signs = np.sign(x_scaled_batch)
        x_abs_batch = np.abs(x_scaled_batch)
        x_clipped_batch = np.clip(x_abs_batch, self.x_values[0], self.x_values[-1])
        j_batch = np.searchsorted(self.x_values, x_clipped_batch, side='left')
        j_batch[j_batch == 0] = 1
        j_batch[j_batch >= len(self.x_values)] = len(self.x_values) - 1
        x_j_batch = self.x_values[j_batch]
        x_jm1_batch = self.x_values[j_batch - 1]
        p_j_batch = self.p_matrix[:, j_batch]
        p_jm1_batch = self.p_matrix[:, j_batch - 1]
        dx_batch = x_j_batch - x_jm1_batch
        dx_batch[np.abs(dx_batch) < 1e-9] = 1.0 
        slope_batch = (p_j_batch - p_jm1_batch) / dx_batch
        prob_vectors = slope_batch * (x_clipped_batch - x_j_batch) + p_j_batch
        prob_vectors[prob_vectors < 0] = 0
        prob_sums = np.sum(prob_vectors, axis=0)
        prob_sums[prob_sums < 1e-9] = 1.0
        prob_vectors /= prob_sums
        neg_indices = np.where(signs < 0)[0]
        if len(neg_indices) > 0:
            prob_vectors[:, neg_indices] = prob_vectors[::-1, neg_indices]
        return prob_vectors

    def perturb_batch_to_indices(self, x_scaled_batch):
        prob_vectors = self.get_prob_vectors_batch(x_scaled_batch)
        cumulative_probs = np.cumsum(prob_vectors.T, axis=1)
        random_values = np.random.rand(len(x_scaled_batch), 1)
        internal_indices = (random_values < cumulative_probs).argmax(axis=1)
        return self.a_indices[internal_indices]

    def perturb_to_index_deterministic(self, x_scaled):
        prob_vector = self.get_prob_vectors_batch(np.array([x_scaled])).flatten()
        best_internal_idx = np.argmax(prob_vector)
        return self.a_indices[best_internal_idx]

# ===================================================================
# 범주형 데이터용 Direct Encoding (Randomized Response) 클래스
# ===================================================================
class CategoricalDE:
    def __init__(self, epsilon, num_categories):
        self.epsilon = epsilon
        self.k = num_categories
        exp_eps = np.exp(epsilon)
        self.p = exp_eps / (exp_eps + self.k - 1)
        print(f"Categorical DE 메커니즘 생성 (epsilon={epsilon:.4f}, k={self.k}, p={self.p:.4f})")

    def perturb_batch(self, true_indices):
        true_indices = np.asarray(true_indices)
        valid_mask = ~np.isnan(true_indices)
        if not np.any(valid_mask): return true_indices
        
        valid_indices = true_indices[valid_mask].astype(int)
        perturbed_valid_indices = np.copy(valid_indices)
        
        n_samples = len(valid_indices)
        change_mask = np.random.rand(n_samples) > self.p
        n_to_change = np.sum(change_mask)
        if n_to_change > 0:
            random_choices = np.random.randint(0, self.k, size=n_to_change)
            perturbed_valid_indices[change_mask] = random_choices
            
        final_perturbed_indices = np.full(len(true_indices), np.nan)
        final_perturbed_indices[valid_mask] = perturbed_valid_indices
        return final_perturbed_indices

# ===================================================================
# 메커니즘 로딩/생성 헬퍼 함수
# ===================================================================
def load_or_create_mechanism(obj_type, epsilon, N, usage="피처"):
    print(f"{usage}용 Numerical 메커니즘 준비 중 (Objective: {obj_type}, Epsilon: {epsilon:.4f}, N: {N})...")
    dir_path = 'results_a1_optimized' if obj_type == 'avg' else 'results_worst_case'
    pkl_path = os.path.join(dir_path, f'opt_results_{"worst_case_" if obj_type == "worst" else ""}eps{epsilon:.4f}_N{N}.pkl')
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    try:
        mechanism = LDPMechanism(pkl_path)
    except FileNotFoundError:
        print(f"-> '{pkl_path}' 파일 없음. 최적화 스크립트를 실행합니다.")
        if obj_type == 'avg': make_mechanism_avg.optimize(epsilon, N)
        else: make_mechanism_worst.optimize(epsilon, N)
        mechanism = LDPMechanism(pkl_path)
    print(f"-> {usage}용 Numerical 메커니즘 준비 완료.")
    return mechanism

# ===================================================================
# 데이터셋 변환을 담당하는 클래스
# ===================================================================
class DatasetTransformer:
    def __init__(self, args):
        self.args = args
        raw_df = pd.read_csv(args.csv_path)
        
        if args.label_col not in raw_df.columns:
            raise ValueError(f"지정한 레이블 칼럼 '{args.label_col}'이 데이터셋에 없습니다.")

        miss = raw_df.isna().mean()
        drop_cols = [c for c, m in miss.items() if m > 0.5 and c != args.label_col]
        if drop_cols:
            print(f"[Info] 50% 이상 결측치 칼럼 제외: {drop_cols}")
            raw_df.drop(columns=drop_cols, inplace=True)
        
        self.numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = raw_df.select_dtypes(exclude=np.number).columns.tolist()
        
        if args.label_col in self.numeric_cols: self.numeric_cols.remove(args.label_col)
        if args.label_col in self.categorical_cols: self.categorical_cols.remove(args.label_col)
        
        print(f"[Info] 수치형 피처: {self.numeric_cols}")
        print(f"[Info] 범주형 피처: {self.categorical_cols}")

        if not args.with_categorical and self.categorical_cols:
            print("[Info] 범주형 피처를 사용하지 않고 제외합니다.")
            self.df = raw_df.drop(columns=self.categorical_cols)
            self.categorical_cols = []
        else:
            self.df = raw_df

        self.train_df, self.test_df = None, None
        
        # 역변환에 필요한 값들을 저장할 변수
        self.scalers = {}
        self.feature_mechanism_numerical = None
        self.label_mechanism_numerical = None
        
        self.ldp_index_min = None
        self.ldp_index_max = None
        self.original_label_min = None
        self.original_label_max = None
        
    def split_data(self):
        print(f"\n원본 데이터를 Train({1-self.args.test_size:.0%})/Test({self.args.test_size:.0%}) 세트로 분할합니다.")
        self.train_df, self.test_df = train_test_split(self.df, test_size=self.args.test_size, random_state=self.args.random_state)
        print(f"Train: {len(self.train_df)}개, Test: {len(self.test_df)}개")

    def _fit_scalers_and_encoders(self):
        print("\nTrain 데이터로 스케일러와 인코더를 학습합니다...")
        # 피처 스케일러
        for col in self.numeric_cols:
            self.scalers[col] = MinMaxScaler(feature_range=(-1, 1)).fit(self.train_df[[col]])
        
        # 수치형 레이블 변환 준비
        if self.args.transform_label_numerical:
            label_eps = self.args.label_epsilon if self.args.label_epsilon is not None else self.args.eps
            label_n = self.args.label_N if self.args.label_N is not None else self.args.N
            
            print(f"[Info] 수치형 레이블 변환을 위해 LDP 메커니즘과 역변환 파라미터를 준비합니다 (epsilon={label_eps:.4f}, N={label_n}).")
            
            # 1. LDP 입력용 스케일러 ([-1, 1]로 변환)
            self.scalers[self.args.label_col] = MinMaxScaler(feature_range=(-1, 1)).fit(self.train_df[[self.args.label_col]])
            self.label_mechanism_numerical = load_or_create_mechanism(self.args.obj, label_eps, label_n, usage="레이블")

            # 2. ✨ [핵심] 역변환(선형 보간)에 사용할 min/max 값 저장
            self.ldp_index_min = self.label_mechanism_numerical.a_indices.min()
            self.ldp_index_max = self.label_mechanism_numerical.a_indices.max()
            self.original_label_min = self.train_df[self.args.label_col].min()
            self.original_label_max = self.train_df[self.args.label_col].max()

            print(f"[Info] 역변환 파라미터 설정:")
            print(f"  - LDP 인덱스 범위: [{self.ldp_index_min}, {self.ldp_index_max}]")
            print(f"  - 원본 레이블 범위: [{self.original_label_min}, {self.original_label_max}]")

    def _transform_train_batch(self):
        print("\nTrain 데이터셋 변환 중 (Stochastic & Batch)...")
        num_sam = len(self.train_df)
        final_df = pd.DataFrame(index=self.train_df.index)

        # 피처 변환 (간소화된 버전)
        if self.numeric_cols:
            print(f"  - {len(self.numeric_cols)}개의 수치형 피처 변환...")
            if not self.feature_mechanism_numerical:
                self.feature_mechanism_numerical = load_or_create_mechanism(self.args.obj, self.args.eps, self.args.N, usage="피처")
            scaled_data = np.array([self.scalers[col].transform(self.train_df[[col]]).flatten() for col in self.numeric_cols]).T
            perturbed_indices = self.feature_mechanism_numerical.perturb_batch_to_indices(scaled_data.flatten()).reshape(num_sam, -1)
            feature_df = pd.DataFrame(perturbed_indices, index=self.train_df.index, columns=self.numeric_cols)
            final_df = pd.concat([final_df, feature_df], axis=1)

        # --- 레이블 변환 ---
        if self.args.transform_label_numerical:
            print(f"  - ✅ 수치형 레이블 '{self.args.label_col}' 변환 및 선형 보간 역변환...")
            
            # 1. 원본 -> LDP 인덱스
            scaled_label = self.scalers[self.args.label_col].transform(self.train_df[[self.args.label_col]]).flatten()
            perturbed_indices = self.label_mechanism_numerical.perturb_batch_to_indices(scaled_label)
            
            # 2. ✨ [새로운 역변환] LDP 인덱스 -> 원본 범위로 선형 보간
            idx_min, idx_max = self.ldp_index_min, self.ldp_index_max
            val_min, val_max = self.original_label_min, self.original_label_max

            # 분모가 0이 되는 경우 방지
            if idx_max == idx_min:
                inversed_values = np.full_like(perturbed_indices, val_min, dtype=float)
            else:
                # 선형 보간 공식
                normalized_pos = (perturbed_indices - idx_min) / (idx_max - idx_min)
                inversed_values = normalized_pos * (val_max - val_min) + val_min
            
            final_df[self.args.label_col] = inversed_values
            print_debug_info(self.train_df, final_df, self.args.label_col)
        else:
            print(f"  - ⚠️ 레이블 변환 플래그가 설정되지 않아 원본 값을 사용합니다.")
            final_df[self.args.label_col] = self.train_df[self.args.label_col].values

        return final_df


    def _transform_test_deterministic(self):
        print("\nTest 데이터셋 변환 중 (Deterministic)...")
        transformed_features = []
        
        if self.numeric_cols:
            print(f"  - {len(self.numeric_cols)}개의 수치형 피처 변환...")
            if not self.feature_mechanism_numerical: self.feature_mechanism_numerical = load_or_create_mechanism(self.args.obj, self.args.eps, self.args.N, usage="피처")
            
            perturbed_data = np.zeros((len(self.test_df), len(self.numeric_cols)))
            for i, col in enumerate(self.numeric_cols):
                scaled_vals = self.scalers[col].transform(self.test_df[[col]])
                perturbed_vals = [self.feature_mechanism_numerical.perturb_to_index_deterministic(val) for val in tqdm(scaled_vals.flatten(), desc=f'  -> Det. Perturbing {col}', leave=False)]
                perturbed_data[:, i] = perturbed_vals
            transformed_features.append(pd.DataFrame(perturbed_data, index=self.test_df.index, columns=self.numeric_cols))
            
        if self.args.with_categorical and self.categorical_cols:
            print(f"  - {len(self.categorical_cols)}개의 범주형 피처 변환 (One-Hot)...")
            ohe_features = self.feature_ohe.transform(self.test_df[self.categorical_cols])
            ohe_df = pd.DataFrame(ohe_features, index=self.test_df.index, columns=self.feature_ohe.get_feature_names_out())
            transformed_features.append(ohe_df)

        final_df = pd.concat(transformed_features, axis=1) if transformed_features else pd.DataFrame(index=self.test_df.index)
        
        print("  - Test 레이블은 원본 값을 사용합니다.")
        final_df[self.args.label_col] = self.test_df[self.args.label_col].values
        
        return final_df

    def transform(self):
        self.split_data()
        self._fit_scalers_and_encoders()
        perturbed_train_df = self._transform_train_batch()
        perturbed_test_df = self._transform_test_deterministic()
        print("\n데이터셋 변환 완료.")
        return perturbed_train_df, perturbed_test_df
    
    def save(self, df, output_dir, file_suffix):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        base_filename = os.path.splitext(os.path.basename(self.args.csv_path))[0]
        cat_suffix = "_withCat" if self.args.with_categorical else ""
        
        label_eps_suffix = ""
        if (self.args.transform_label_numerical or self.args.transform_label_categorical):
             label_eps = self.args.label_epsilon if self.args.label_epsilon is not None else self.args.eps
             label_eps_suffix = f"_Leps{label_eps}"
        
        label_n_suffix = ""
        if self.args.transform_label_numerical:
            label_n = self.args.label_N if self.args.label_N is not None else self.args.N
            label_n_suffix = f"_LN{label_n}"

        ## 파일 이름에 seed 값 추가
        output_csv_filename = f"{base_filename}_Eps{self.args.eps}_N{self.args.N}_{self.args.obj}{cat_suffix}{label_eps_suffix}{label_n_suffix}_seed{self.args.seed}{file_suffix}.csv"
        output_csv_path = os.path.join(output_dir, output_csv_filename)
        df.to_csv(output_csv_path, index=True)
        print(f"'{output_csv_path}'에 저장되었습니다.")

def print_debug_info(train_df, final_df, label_col):
    """디버깅을 위해 원본과 변환된 레이블을 출력하는 헬퍼 함수"""
    print("  - [디버그] 레이블 변환 결과 (처음 5개):")
    print(f"    - 원본: {train_df[label_col].values[:5]}")
    print(f"    - 변환: {final_df[label_col].values[:5]}")

# ===================================================================
# 메인 실행 블록
# ===================================================================
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="LDP 메커니즘으로 CSV 데이터셋을 배치 변환합니다."
    )

    # ── 새 인자: 에피실론 리스트 ───────────────────────
    parser.add_argument('--eps_list', type=float, nargs='+', default=[1.0,2.0,3.0,4.0,5.0],
                        help='공백으로 구분된 피처 epsilon 목록')
    parser.add_argument('--label_eps_list', type=float, nargs='+', default=[1.0,3.0,5.0],
                        help='공백으로 구분된 레이블 epsilon 목록')

    # 기존 인자 (일부 생략 없이 그대로 유지)
    parser.add_argument('--N', type=int, default=31)
    parser.add_argument('--label_N', type=int, default=15)
    parser.add_argument('--label_epsilon', type=float, default=None)
    parser.add_argument('--obj', type=str, default='avg', choices=['avg', 'worst'])
    parser.add_argument('--csv_path', type=str, default='data/wine.csv')
    parser.add_argument('--label_col', type=str, default='label')
    parser.add_argument('--output_dir', type=str, default='transformed_data_batch_label')
    parser.add_argument('--with_categorical', type=str2bool, default=False)
    parser.add_argument('--transform_label_numerical', type=str2bool, default=True)
    parser.add_argument('--transform_label_categorical', type=str2bool, default=False)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)

    # 다중 시드 지원
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=str, default='0')

    args = parser.parse_args()

    seeds_to_run = args.seeds
    eps_list = args.eps_list
    label_eps_list = args.label_eps_list

    dataset_name = Path(args.csv_path).stem
    args.output_dir = Path(args.output_dir) / dataset_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for current_seed in seeds_to_run:
        args.seed = current_seed
        np.random.seed(current_seed)
        print(f"\n--- Seed {current_seed} 시작 ---")

        for eps in eps_list:
            for label_eps in label_eps_list:
                args.eps = eps
                args.label_epsilon = label_eps

                print(f"\n=== Transform: ε={eps}, label ε={label_eps} ===")
                transformer = DatasetTransformer(args)

                start = time.time()
                train_df, test_df = transformer.transform()
                print(f"    -> 변환 완료: {time.time() - start:.2f} 초")

                transformer.save(train_df, args.output_dir, "_train")
                transformer.save(test_df,   args.output_dir, "_test")

    print("\n✨ 모든 변환 작업이 완료되었습니다.")