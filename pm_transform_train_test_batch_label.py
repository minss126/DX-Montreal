import make_mechanism_avg
import make_mechanism_worst

import numpy as np
import pandas as pd
import pickle, os, argparse, time, warnings
from sklearn.preprocessing import MinMaxScaler
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
        # np.unique()로 추출한 실제 카테고리 목록
        self.categories = np.asarray(categories)
        self.k = len(self.categories)
        self.epsilon = epsilon
        
        # 내부적으로 매핑/역매핑을 위한 딕셔너리를 자동 생성
        self.label_map = {label: i for i, label in enumerate(self.categories)}
        self.inverse_map = {i: label for i, label in enumerate(self.categories)}

        # p 값은 k가 고정되므로 한 번만 계산되어 안정적임
        exp_eps = np.exp(self.epsilon)
        self.p = exp_eps / (exp_eps + self.k - 1)

    def perturb_batch(self, true_labels):
        original_labels = np.asarray(true_labels)
        
        # 1. 레이블 -> 0-인덱스로 매핑
        # self.categories에 없는 값은 안전하게 처리되지 않으므로, 입력 데이터는 항상 
        # self.categories 내에 있는 값이라고 가정함.
        mapped_indices = np.array([self.label_map[label] for label in original_labels.flatten()])
        
        # 2. 노이즈 추가
        change_mask = np.random.rand(len(mapped_indices)) > self.p
        n_to_change = np.sum(change_mask)
        random_indices = np.random.randint(0, self.k, size=n_to_change)
        
        perturbed_indices = np.copy(mapped_indices)
        perturbed_indices[change_mask] = random_indices
        
        # 3. 0-인덱스 -> 원래 레이블로 역매핑
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
        self.scalers = {}
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
            self.scalers[col] = MinMaxScaler(feature_range=(-1, 1)).fit(self.train_df[[col]])
        
        if self.args.transform_label_numerical:            
            label_eps = self.args.label_epsilon if self.args.label_epsilon is not None else self.args.eps
            print(f"[Info] 수치형 레이블 변환을 위해 LDP 메커니즘과 역변환 파라미터를 준비합니다,.")
            
            # 1. LDP 입력용 스케일러 ([-1, 1]로 변환)
            self.scalers[self.args.label_col] = MinMaxScaler(feature_range=(-1, 1)).fit(self.train_df[[self.args.label_col]])
            self.label_mechanism_numerical = pm.PM(label_eps, t=self.args.t)
            
            # 2. ✨ [핵심] 역변환(선형 보간)에 사용할 min/max 값 저장
            self.ldp_val_min = -self.label_mechanism_numerical.A
            self.ldp_val_max = self.label_mechanism_numerical.A
            self.original_label_min = self.train_df[self.args.label_col].min()
            self.original_label_max = self.train_df[self.args.label_col].max()

            print(f"[Info] 역변환 파라미터 설정:")
            print(f"  - LDP 인덱스 범위: [{self.ldp_val_min}, {self.ldp_val_max}]")
            print(f"  - 원본 레이블 범위: [{self.original_label_min}, {self.original_label_max}]")

        elif self.args.transform_label_categorical:
            label_eps = self.args.label_epsilon if self.args.label_epsilon is not None else self.args.eps
            self.label_mechanism_categorical = CategoricalDE(label_eps, np.unique(self.train_df[[self.args.label_col]]))

    def _transform_train_batch(self):
        print("\nTrain 데이터셋 변환 중 (Stochastic & Batch)...")
        num_sam = len(self.train_df)
        final_df = pd.DataFrame(index=self.train_df.index)

        # 피처 변환 (간소화된 버전)
        if self.numeric_cols:
            print(f"  - {len(self.numeric_cols)}개의 수치형 피처 변환...")
            if not self.feature_mechanism_numerical:
                self.feature_mechanism_numerical = pm.PM(self.args.eps, t=self.args.t)
            scaled_data = np.array([self.scalers[col].transform(self.train_df[[col]]).flatten() for col in self.numeric_cols]).T
            perturbed_vals = self.feature_mechanism_numerical.PM_batch(scaled_data.flatten()).reshape(num_sam, -1)
            feature_df = pd.DataFrame(perturbed_vals, index=self.train_df.index, columns=self.numeric_cols)
            final_df = pd.concat([final_df, feature_df], axis=1)

        # --- 레이블 변환 ---
        if self.args.transform_label_numerical:
            print(f"  - ✅ 수치형 레이블 '{self.args.label_col}' 변환 및 선형 보간 역변환...")
            
            # 1. 원본 -> LDP 인덱스
            scaled_label = self.scalers[self.args.label_col].transform(self.train_df[[self.args.label_col]]).flatten()
            perturbed_vals = self.label_mechanism_numerical.PM_batch(scaled_label)
            
            # 2. ✨ [새로운 역변환] LDP 인덱스 -> 원본 범위로 선형 보간
            pm_min, pm_max = self.ldp_val_min, self.ldp_val_max
            val_min, val_max = self.original_label_min, self.original_label_max

                # 선형 보간 공식
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
                scaled_vals = self.scalers[col].transform(self.test_df[[col]])
                perturbed_vals = [self.feature_mechanism_numerical.PM_batch_deterministic(val) for val in tqdm(scaled_vals.flatten(), desc=f'  -> Det. Perturbing {col}', leave=False)]
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

        ## 파일 이름에 seed 값 추가
        output_csv_filename = f"{base_filename}_Eps{self.args.eps}_t{self.args.t}{label_eps_suffix}_seed{self.args.seed}{file_suffix}.csv"
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

# python pm_transform_train_test_batch_label.py --eps 3.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/wine.csv --mode uniform
# python pm_transform_train_test_batch_label.py --eps 3.0 --transform_label_numerical True --transform_label_categorical False --transform_label_log False --mode midpoint

# python pm_transform_train_test_batch_label.py --eps 1.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/shuttle.csv --label_col label

# python pm_transform_train_test_batch_label.py --eps 1.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/gamma.csv --label_col label
# python pm_transform_train_test_batch_label.py --eps 2.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/gamma.csv --label_col label
# python pm_transform_train_test_batch_label.py --eps 3.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/gamma.csv --label_col label
# python pm_transform_train_test_batch_label.py --eps 4.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/gamma.csv --label_col label
# python pm_transform_train_test_batch_label.py --eps 5.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/gamma.csv --label_col label
# python pm_transform_train_test_batch_label.py --eps 1.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/credit.csv --label_col label --t 2 
# python pm_transform_train_test_batch_label.py --eps 2.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/credit.csv --label_col label --t 2 
# python pm_transform_train_test_batch_label.py --eps 3.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/credit.csv --label_col label --t 2 
# python pm_transform_train_test_batch_label.py --eps 4.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/credit.csv --label_col label --t 2 
# python pm_transform_train_test_batch_label.py --eps 5.0 --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/credit.csv --label_col label --t 2 

# python pm_transform_train_test_batch_label.py --eps 1.0 --transform_label_numerical True --transform_label_categorical False --transform_label_log False --csv_path data/CASP.csv --label_col RMSD
# python pm_transform_train_test_batch_label.py --eps 2.0 --transform_label_numerical True --transform_label_categorical False --transform_label_log False --csv_path data/CASP.csv --label_col RMSD
# python pm_transform_train_test_batch_label.py --eps 3.0 --transform_label_numerical True --transform_label_categorical False --transform_label_log False --csv_path data/CASP.csv --label_col RMSD
# python pm_transform_train_test_batch_label.py --eps 4.0 --transform_label_numerical True --transform_label_categorical False --transform_label_log False --csv_path data/CASP.csv --label_col RMSD
# python pm_transform_train_test_batch_label.py --eps 5.0 --transform_label_numerical True --transform_label_categorical False --transform_label_log False --csv_path data/CASP.csv --label_col RMSD

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="LDP 메커니즘으로 CSV 데이터셋을 배치 변환합니다."
    )
    
    # 기존 인자 (일부 생략 없이 그대로 유지)
    parser.add_argument('--eps', type=float, default=5.0)
    parser.add_argument('--label_epsilon', type=float, default=None)
    parser.add_argument('--t', type=int, default=3)
    parser.add_argument('--csv_path', type=str, default='data/elevators.csv')
    parser.add_argument('--label_col', type=str, default='label')
    parser.add_argument('--output_dir', type=str, default='transformed_data_batch_label')
    parser.add_argument('--with_categorical', type=str2bool, default=False)
    parser.add_argument('--transform_label_numerical', type=str2bool, default=True)
    parser.add_argument('--transform_label_categorical', type=str2bool, default=False)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--transform_label_log', type=str2bool, default=True)

    # 다중 시드 지원
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=str, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    args = parser.parse_args()

    seeds_to_run = args.seeds

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

        transformer.save(train_df, args.output_dir, "_train")
        transformer.save(test_df,   args.output_dir, "_test")

    print("\n✨ 모든 변환 작업이 완료되었습니다.")