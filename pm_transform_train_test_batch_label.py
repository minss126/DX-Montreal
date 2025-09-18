import make_mechanism_avg
import make_mechanism_worst

import numpy as np
import pandas as pd
import pickle, os, argparse, time, warnings
# StandardScaler 추가
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

import pm

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ===================================================================
# 범주형 데이터용 Direct Encoding (Randomized Response) 클래스
# ===================================================================
class CategoricalDE:
    """
    고정된 카테고리 목록을 받아 차분 프라이버시를 적용하는 가장 표준적인 클래스.
    """
    def __init__(self, epsilon, categories):
        self.categories = np.asarray(categories)
        self.k = len(self.categories)
        self.epsilon = epsilon
        self.label_map = {label: i for i, label in enumerate(self.categories)}
        self.inverse_map = {i: label for i, label in enumerate(self.categories)}
        exp_eps = np.exp(self.epsilon)
        self.p = exp_eps / (exp_eps + self.k - 1)

    def perturb_batch(self, true_labels):
        original_labels = np.asarray(true_labels)
        mapped_indices = np.array([self.label_map[label] for label in original_labels.flatten()])
        change_mask = np.random.rand(len(mapped_indices)) > self.p
        n_to_change = np.sum(change_mask)
        random_indices = np.random.randint(0, self.k, size=n_to_change)
        perturbed_indices = np.copy(mapped_indices)
        perturbed_indices[change_mask] = random_indices
        final_labels = np.array([self.inverse_map[idx] for idx in perturbed_indices])
        return final_labels.reshape(original_labels.shape)

# ===================================================================
# 데이터셋 변환을 담당하는 클래스
# ===================================================================
class DatasetTransformer:
    def __init__(self, args):
        self.args = args
        raw_df = pd.read_csv(args.csv_path)
        if args.label_col not in raw_df.columns:
            raise ValueError(f"지정한 레이블 칼럼 '{args.label_col}'이 데이터셋에 없습니다.")
        if self.args.transform_label_log:
            raw_df[args.label_col] = np.log1p(raw_df[args.label_col])
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
        self.preprocessors = {}
        self.feature_mechanism_numerical = None
        self.label_mechanism_numerical = None
        self.label_mechanism_categorical = None
        
        self.ldp_val_min = None
        self.ldp_val_max = None
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
            std_scaler = StandardScaler().fit(self.train_df[[col]])
            train_std = std_scaler.transform(self.train_df[[col]])
            minmax_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_std)
            self.preprocessors[col] = {'std_scaler': std_scaler, 'minmax_scaler': minmax_scaler}
        
        if self.args.transform_label_numerical:            
            label_eps = self.args.label_epsilon if self.args.label_epsilon is not None else self.args.eps
            print(f"[Info] 수치형 레이블 변환을 위해 LDP 메커니즘과 역변환 파라미터를 준비합니다.")
            
            label_std_scaler = StandardScaler().fit(self.train_df[[self.args.label_col]])
            train_label_std = label_std_scaler.transform(self.train_df[[self.args.label_col]])
            label_minmax_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_label_std)
            self.preprocessors[self.args.label_col] = {'std_scaler': label_std_scaler, 'minmax_scaler': label_minmax_scaler}
            
            self.label_mechanism_numerical = pm.PM(label_eps, t=self.args.t)
            
            self.ldp_val_min = -self.label_mechanism_numerical.A
            self.ldp_val_max = self.label_mechanism_numerical.A
            self.original_label_min = self.train_df[self.args.label_col].min()
            self.original_label_max = self.train_df[self.args.label_col].max()

            print(f"[Info] 역변환 파라미터 설정:")
            print(f"  - LDP 출력값 범위: [{self.ldp_val_min:.4f}, {self.ldp_val_max:.4f}]")
            print(f"  - 원본 레이블 범위: [{self.original_label_min:.4f}, {self.original_label_max:.4f}]")

        elif self.args.transform_label_categorical:
            label_eps = self.args.label_epsilon if self.args.label_epsilon is not None else self.args.eps
            self.label_mechanism_categorical = CategoricalDE(label_eps, np.unique(self.train_df[[self.args.label_col]]))

    def _transform_train_batch(self):
        print("\nTrain 데이터셋 변환 중 (Stochastic & Batch)...")
        num_sam = len(self.train_df)
        final_df = pd.DataFrame(index=self.train_df.index)

        if self.numeric_cols:
            print(f"  - {len(self.numeric_cols)}개의 수치형 피처 변환...")
            if not self.feature_mechanism_numerical:
                self.feature_mechanism_numerical = pm.PM(self.args.eps, t=self.args.t)
            
            scaled_data_list = []
            for col in self.numeric_cols:
                train_std = self.preprocessors[col]['std_scaler'].transform(self.train_df[[col]])
                train_minmax = self.preprocessors[col]['minmax_scaler'].transform(train_std)
                scaled_data_list.append(train_minmax.flatten())
            scaled_data = np.array(scaled_data_list).T
            
            perturbed_vals = self.feature_mechanism_numerical.PM_batch(scaled_data.flatten()).reshape(num_sam, -1)
            feature_df = pd.DataFrame(perturbed_vals, index=self.train_df.index, columns=self.numeric_cols)
            final_df = pd.concat([final_df, feature_df], axis=1)

        if self.args.transform_label_numerical:
            # 1. 원본 -> LDP 인덱스 (This part is common)
            label_std = self.preprocessors[self.args.label_col]['std_scaler'].transform(self.train_df[[self.args.label_col]])
            scaled_label = self.preprocessors[self.args.label_col]['minmax_scaler'].transform(label_std).flatten()
            perturbed_vals = self.label_mechanism_numerical.PM_batch(scaled_label)
            
            # 2. Check the new --label_index flag
            if self.args.label_index:
                print(f"  - ✅ 수치형 레이블 '{self.args.label_col}'을 LDP 출력값(인덱스처럼 사용)으로 변환...")
                final_df[self.args.label_col] = perturbed_vals
            else:
                print(f"  - ✅ 수치형 레이블 '{self.args.label_col}' 변환 및 선형 보간 역변환...")
                pm_min, pm_max = self.ldp_val_min, self.ldp_val_max
                val_min, val_max = self.original_label_min, self.original_label_max
                
                if pm_max == pm_min:
                    inversed_values = np.full_like(perturbed_vals, val_min, dtype=float)
                else:
                    normalized_pos = (perturbed_vals - pm_min) / (pm_max - pm_min)
                    inversed_values = normalized_pos * (val_max - val_min) + val_min
                
                final_df[self.args.label_col] = inversed_values
            
            print_debug_info(self.train_df, final_df, self.args.label_col)
        elif self.args.transform_label_categorical:
            perturbed_indices = self.label_mechanism_categorical.perturb_batch(self.train_df[[self.args.label_col]])
            final_df[self.args.label_col] = perturbed_indices
        else:
            print(f"  - ⚠️ 레이블 변환 플래그가 설정되지 않아 원본 값을 사용합니다.")
            final_df[self.args.label_col] = self.train_df[self.args.label_col].values

        return final_df

    def _transform_test_deterministic(self):
        print("\nTest 데이터셋 변환 중 (Deterministic)...")
        transformed_features = []
        
        if self.numeric_cols:
            print(f"  - {len(self.numeric_cols)}개의 수치형 피처 변환...")
            if not self.feature_mechanism_numerical: self.feature_mechanism_numerical = pm.PM(self.args.eps, t=self.args.t)
            
            perturbed_data = np.zeros((len(self.test_df), len(self.numeric_cols)))
            for i, col in enumerate(self.numeric_cols):
                test_std = self.preprocessors[col]['std_scaler'].transform(self.test_df[[col]])
                scaled_vals = self.preprocessors[col]['minmax_scaler'].transform(test_std)
                
                perturbed_vals = [self.feature_mechanism_numerical.PM_batch_deterministic(val) for val in tqdm(scaled_vals.flatten(), desc=f'  -> Det. Perturbing {col}', leave=False)]
                perturbed_data[:, i] = perturbed_vals
            transformed_features.append(pd.DataFrame(perturbed_data, index=self.test_df.index, columns=self.numeric_cols))
            
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
    
    def _get_base_filename(self):
        """ model.py와 호환되는 기본 파일 이름을 생성합니다. """
        base_filename = Path(self.args.csv_path).stem
        label_eps = self.args.label_epsilon if self.args.label_epsilon is not None else self.args.eps
        
        return (f"{base_filename}_Eps{self.args.eps:.1f}_t{self.args.t}_"
            f"Leps{label_eps:.1f}_seed{self.args.seed}")

    def save_label_inversion_metadata(self, output_dir):
        """
        레이블이 LDP 출력값으로 변환된 경우, 역변환에 필요한 파라미터를 .pkl 파일로 저장합니다.
        """
        if self.args.transform_label_numerical and self.args.label_index:
            metadata_to_save = {
                "original_label_min": self.original_label_min,
                "original_label_max": self.original_label_max,
                "ldp_val_min": self.ldp_val_min,
                "ldp_val_max": self.ldp_val_max
            }
            base_name = self._get_base_filename()
            metadata_filepath = os.path.join(output_dir, f"{base_name}_metadata.pkl")
            with open(metadata_filepath, 'wb') as f:
                pickle.dump(metadata_to_save, f)
            print(f"레이블 역변환 메타데이터 저장: {metadata_filepath}")

    def save(self, df, output_dir, file_suffix):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        base_name = self._get_base_filename()
        output_csv_filename = f"{base_name}{file_suffix}.csv"
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
    
    parser.add_argument('--eps', type=float, default=5.0)
    parser.add_argument('--label_epsilon', type=float, default=None)
    parser.add_argument('--t', type=int, default=3)
    parser.add_argument('--csv_path', type=str, default='data/elevators.csv')
    parser.add_argument('--label_col', type=str, default='label')
    parser.add_argument('--output_dir', type=str, default='transformed_data_batch_label_mod')
    parser.add_argument('--with_categorical', type=str2bool, default=False)
    parser.add_argument('--transform_label_numerical', type=str2bool, default=True)
    parser.add_argument('--transform_label_categorical', type=str2bool, default=False)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--transform_label_log', type=str2bool, default=True)
    parser.add_argument('--label_index', type=str2bool, default=False, help='LDP 적용된 레이블을 역변환하지 않고 그대로 사용할지 여부')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4,5,6,7,8,9')

    args = parser.parse_args()
    
    seeds_to_run = [int(s) for s in args.seeds.split(',')]

    dataset_name = Path(args.csv_path).stem
    args.output_dir = Path(args.output_dir) / dataset_name / 'pm'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for current_seed in seeds_to_run:
        args.seed = current_seed
        np.random.seed(current_seed)
        print(f"\n--- Seed {current_seed} 시작 ---")

        print(f"\n=== Transform: ε={args.eps}, label ε={args.label_epsilon} ===")
        transformer = DatasetTransformer(args)

        start = time.time()
        train_df, test_df = transformer.transform()
        print(f"    -> 변환 완료: {time.time() - start:.2f} 초")

        transformer.save_label_inversion_metadata(args.output_dir)
        
        transformer.save(train_df, args.output_dir, "_train")
        transformer.save(test_df,   args.output_dir, "_test")

    print("\n✨ 모든 변환 작업이 완료되었습니다.")
