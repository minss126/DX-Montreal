# 전체 코드를 다시 제공합니다. 이 코드를 복사하여 사용해 주세요.

import numpy as np
from scipy.optimize import minimize, Bounds
import time
import argparse
import pickle
import os
import numba

# ===================================================================
# 간단한 진행 상황 출력을 위한 콜백 클래스
# ===================================================================
class SimpleCallbackLogger:
    def __init__(self, optimizer_instance, disp_interval=10):
        self.iteration = 0
        self.optimizer = optimizer_instance
        self.last_fun_val = np.inf
        self.disp_interval = disp_interval
        print("\n--- Optimization Log ---")
        print("{:>5} | {:>18} | {:>15}".format("Iter", "Objective Value", "Change"))
        print("-" * 45)

    def __call__(self, xk):
        if self.iteration == 0 or (self.iteration + 1) % self.disp_interval == 0:
            fun_val = self.optimizer.objective_func(xk)
            change = self.last_fun_val - fun_val
            print("{:5d} | {:18.6f} | {:15.6e}".format(self.iteration, fun_val, change))
            self.last_fun_val = fun_val
        self.iteration += 1

# ===================================================================
# a_1만 최적화 (정수배 간격), a_0 포함 (N이 홀수일 때)
# ===================================================================
class LDPMechanismOptimizerWithA0_IntegerMultiple:
    def __init__(self, n, epsilon):
        self.n, self.epsilon, self.exp_eps = n, epsilon, np.exp(epsilon)
        # === [수정] === a_1 하나만 최적화 변수
        self.num_a_vars, self.num_p_vars = 1, (2 * n + 1) * (n + 1)
        print(f"Optimizer 생성됨 (a_1만 최적화, 정수배 모드, a_0 포함, n={n})")

    def unpack_vars(self, v):
        # === [수정] === v[0]은 이제 a_1 값
        a_1_val = v[0]
        # === [수정] === a_i = i * a_1 관계로 a_pos 생성
        a_pos = a_1_val * np.arange(1, self.n + 1)
        
        p_v = v[self.num_a_vars:].reshape((self.n + 1, 2 * self.n + 1))
        a_full = np.concatenate([-a_pos[::-1], [0], a_pos])
        return a_full, p_v.T
    
    def get_p_val(self, p, i, j):
        if j < 0:
            return p[self.n - i, -j]
        return p[self.n + i, j]

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _calculate_objective_core(a, p, a_sq, n):
        x = a @ p
        x_j = x[1 : n + 1]
        x_jm1 = x[0 : n]
        dx = x_j - x_jm1
        E_Y2 = a_sq @ p
        E_Y2_j = E_Y2[1 : n + 1]
        E_Y2_jm1 = E_Y2[0 : n]
        avg_variance_proxy = (E_Y2_j + E_Y2_jm1) / 2.0
        integral_of_P = avg_variance_proxy * dx
        integral_of_x_sq = (x_j**3 - x_jm1**3) / 3.0
        total = 0.0
        for i in range(len(dx)):
            if np.abs(dx[i]) > 1e-20:
                total += integral_of_P[i] - integral_of_x_sq[i]
        return total

    def objective_func(self, v):
        a, p = self.unpack_vars(v)
        a_sq = a**2
        return self._calculate_objective_core(a, p, a_sq, self.n)

    def _create_constraints(self):
        n = self.n
        cons = []
        cons.append({'type': 'eq', 'fun': lambda v: np.sum(v[self.num_a_vars:].reshape((n + 1, 2 * n + 1)), axis=1) - 1})
        cons.append({'type': 'eq', 'fun': lambda v: self.unpack_vars(v)[0] @ self.unpack_vars(v)[1][:, n] - 1})
        # a_1 > 0 제약조건 (v[0]은 이제 a_1)
        if n > 0: cons.append({'type': 'ineq', 'fun': lambda v: v[0]})
        cons.append({'type': 'ineq', 'fun': lambda v: np.diff(self.unpack_vars(v)[0] @ self.unpack_vars(v)[1])})
        def p_symmetry_j0(v):
            _, p = self.unpack_vars(v)
            return np.array([self.get_p_val(p, i, 0) - self.get_p_val(p, -i, 0) for i in range(1, n + 1)])
        cons.append({'type': 'eq', 'fun': p_symmetry_j0})
        def ldp_constraints(v):
            _, p = self.unpack_vars(v)
            constraints = []
            for i in range(1, n + 1):
                p_neg_i_n = self.get_p_val(p, -i, n)
                for j in range(-n, n + 1):
                    p_i_j = self.get_p_val(p, i, j)
                    constraints.append(p_i_j - p_neg_i_n)
                    constraints.append(self.exp_eps * p_neg_i_n - p_i_j)
            p_zero_n = self.get_p_val(p, 0, n)
            for j in range(-n, n + 1):
                p_zero_j = self.get_p_val(p, 0, j)
                constraints.append(p_zero_j - p_zero_n)
                constraints.append(self.exp_eps * p_zero_n - p_zero_j)
            return np.array(constraints)
        cons.append({'type': 'ineq', 'fun': ldp_constraints})
        return cons

    def run(self, initial_guess, options):
        lv = [1e-9] * self.num_a_vars + [0.0] * self.num_p_vars
        uv = [np.inf] * self.num_a_vars + [1.0] * self.num_p_vars
        b = Bounds(lv, uv)
        c = self._create_constraints()
        cb = SimpleCallbackLogger(self)
        print(f"\nn={self.n},eps={self.epsilon} 최적화 시작 (a_1만, a_0 포함, options={options})...")
        return minimize(self.objective_func, initial_guess, method='SLSQP', bounds=b, constraints=c, options=options, callback=cb)

# ===================================================================
# a_1만 최적화 (정수배 간격), a_0 제외 (N이 짝수일 때)
# ===================================================================
class LDPMechanismOptimizerNoA0_IntegerMultiple:
    def __init__(self, n, epsilon):
        self.n, self.epsilon, self.exp_eps = n, epsilon, np.exp(epsilon)
        self.num_output_points = 2 * n
        # === [수정] === a_1 하나만 최적화 변수
        self.num_a_vars, self.num_p_vars = 1, self.num_output_points * (n + 1)
        print(f"Optimizer 생성됨 (a_1만 최적화, 정수배 모드, a_0 제외, n={n})")

    def _map_i_to_idx(self, i):
        return i + self.n if i < 0 else i + self.n - 1

    def unpack_vars(self, v):
        # === [수정] === v[0]은 이제 a_1 값
        a_1_val = v[0]
        # === [수정] === a_i = i * a_1 관계로 a_pos 생성
        a_pos = a_1_val * np.arange(1, self.n + 1)

        p_v = v[self.num_a_vars:].reshape((self.n + 1, self.num_output_points))
        a_full = np.concatenate([-a_pos[::-1], a_pos])
        return a_full, p_v.T

    def get_p_val(self, p, i, j):
        if j < 0:
            return p[self._map_i_to_idx(-i), -j]
        return p[self._map_i_to_idx(i), j]

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _calculate_objective_core(a, p, a_sq, n):
        x = a @ p
        x_j = x[1 : n + 1]
        x_jm1 = x[0 : n]
        dx = x_j - x_jm1
        E_Y2 = a_sq @ p
        E_Y2_j = E_Y2[1 : n + 1]
        E_Y2_jm1 = E_Y2[0 : n]
        avg_variance_proxy = (E_Y2_j + E_Y2_jm1) / 2.0
        integral_of_P = avg_variance_proxy * dx
        integral_of_x_sq = (x_j**3 - x_jm1**3) / 3.0
        total = 0.0
        for i in range(len(dx)):
            if np.abs(dx[i]) > 1e-20:
                total += integral_of_P[i] - integral_of_x_sq[i]
        return total

    def objective_func(self, v):
        a, p = self.unpack_vars(v)
        a_sq = a**2
        return self._calculate_objective_core(a, p, a_sq, self.n)

    def _create_constraints(self):
        n = self.n
        cons = []
        cons.append({'type': 'eq', 'fun': lambda v: np.sum(v[self.num_a_vars:].reshape((n + 1, 2 * n)), axis=1) - 1})
        cons.append({'type': 'eq', 'fun': lambda v: self.unpack_vars(v)[0] @ self.unpack_vars(v)[1][:, n] - 1})
        if n > 0: cons.append({'type': 'ineq', 'fun': lambda v: v[0]})

        MIN_X_GAP = 1e-7 # x 값들 사이의 최소 간격
        cons.append({'type': 'ineq', 'fun': lambda v: np.diff(self.unpack_vars(v)[0] @ self.unpack_vars(v)[1]) - MIN_X_GAP})
        def p_symmetry_j0(v):
            _, p = self.unpack_vars(v)
            return np.array([self.get_p_val(p, i, 0) - self.get_p_val(p, -i, 0) for i in range(1, n + 1)])
        cons.append({'type': 'eq', 'fun': p_symmetry_j0})
        def ldp_constraints(v):
            _, p = self.unpack_vars(v)
            constraints = []
            for i in range(1, n + 1):
                p_neg_i_n = self.get_p_val(p, -i, n)
                for j in range(-n, n + 1):
                    p_i_j = self.get_p_val(p, i, j)
                    constraints.append(p_i_j - p_neg_i_n)
                    constraints.append(self.exp_eps * p_neg_i_n - p_i_j)
            return np.array(constraints)
        cons.append({'type': 'ineq', 'fun': ldp_constraints})
        return cons

    def run(self, initial_guess, options):
        num_a, num_p = self.num_a_vars, self.num_p_vars
        lower, upper = [1e-9] * num_a + [0.0] * num_p, [np.inf] * num_a + [1.0] * num_p
        b = Bounds(lower, upper)
        c = self._create_constraints()
        cb = SimpleCallbackLogger(self)
        print(f"\nn={self.n},eps={self.epsilon} 최적화 시작 (a_1만, a_0 제외, options={options})...")
        return minimize(self.objective_func, initial_guess, method='SLSQP', bounds=b, constraints=c, options=options, callback=cb)

# ===================================================================
# 초기값 생성 함수 (a_1을 기준으로 생성하도록 수정)
# ===================================================================
def create_initial_guess_fixed_gap(n, epsilon, use_a0, smoothing_factor=0.1):
    # === [수정] === 전체 로직을 a_1 기준으로 변경
    print(f"n={n}, epsilon={epsilon} PM 초기값 계산 (a_1만 최적화, a_0 {'포함' if use_a0 else '제외'}, smoothing={smoothing_factor})...")
    
    if n == 0:
        raise ValueError("n=0인 경우는 이 스크립트에서 지원하지 않습니다.")

    exp_eps = np.exp(epsilon)
    
    # a_n = n * a_1 이고, 이론적인 a_n ~ (e^e+1)/(e^e-1) 이므로
    # a_1의 초기 추정치를 계산
    if n > 0:
        initial_a_1 = (exp_eps + 1) / ((exp_eps - 1) * n)
    else: # n=0인 경우는 없지만 안전을 위해
        initial_a_1 = (exp_eps + 1) / (exp_eps - 1)
        
    # 초기 p행렬 계산을 위한 최대 출력값 A = a_n = n * a_1
    A = n * initial_a_1

    # 초기 a_i 값들을 a_1의 정수배로 계산
    initial_a_pos = initial_a_1 * np.arange(1, n + 1)

    t = np.exp(epsilon/3)
    p_core, p_tail = exp_eps/(exp_eps+t), t/(exp_eps+t)
    def get_overlap(a, b): return max(0, min(a[1], b[1]) - max(a[0], b[0]))
    
    num_output = 2 * n + 1 if use_a0 else 2 * n
    p_matrix = np.zeros((n + 1, num_output))
    
    i_range = range(-n, n + 1) if use_a0 else list(range(-n,0))+list(range(1,n+1))
    
    for j in range(n + 1):
        x = j / n
        l_x, r_x = (A+1)/2.*x-(A-1)/2., (A+1)/2.*x+(A-1)/2.
        w_c, w_t = r_x - l_x, 2*A - (r_x - l_x)
        pdf_c = p_core/w_c if w_c > 1e-9 else 0
        pdf_t = p_tail/w_t if w_t > 1e-9 else 0
        
        for idx, i in enumerate(i_range):
            if i == -n: s, e = -A, (-n + 0.5) / n * A
            elif i == n: s, e = (n - 0.5) / n * A, A
            elif i == 0 and use_a0: s, e = (-0.5 / n * A, 0.5 / n * A)
            else: s, e = (i - 0.5) / n * A, (i + 0.5) / n * A
            p_matrix[j, idx] = get_overlap([s,e],[-A,l_x])*pdf_t+get_overlap([s,e],[l_x,r_x])*pdf_c+get_overlap([s,e],[r_x,A])*pdf_t
    
        row_sum = np.sum(p_matrix[j, :])
        if row_sum > 1e-9: p_matrix[j, :] /= row_sum

    a_full_initial = np.concatenate([-initial_a_pos[::-1], [0], initial_a_pos]) if use_a0 else np.concatenate([-initial_a_pos[::-1], initial_a_pos])
    x_n_initial = np.dot(p_matrix[n, :], a_full_initial)
    if abs(x_n_initial) > 1e-9:
        scaling_factor = 1.0 / x_n_initial
        # 스케일링 팩터는 a_1에 적용되어야 함
        initial_a_1 *= scaling_factor
        print(f"  [INFO] 초기 a_1 값을 {scaling_factor:.4f}배 스케일링하여 x_n=1 제약 만족.")
    
    uniform = np.full_like(p_matrix, 1./num_output)
    smoothed = (1-smoothing_factor)*p_matrix + smoothing_factor*uniform
    for j in range(n+1):
        row_sum = np.sum(smoothed[j,:])
        if row_sum > 1e-9: smoothed[j,:] /= row_sum
        
    # 최종 초기 추정치 벡터 반환: [a_1, p_1,1, p_1,2, ...]
    return np.concatenate([[initial_a_1], smoothed.flatten()])

# ===================================================================
# 결과 계산, 출력 및 저장 함수
# ===================================================================
def calculate_worst_case_variance(v, optimizer):
    n, a, p = optimizer.n, *optimizer.unpack_vars(v)
    a_sq = a**2
    x = a @ p
    E_Y2 = a_sq @ p
    variances_at_xj = E_Y2 - x**2
    candidate_variances = list(variances_at_xj)
    for j in range(1, n + 1):
        xj, xjm1 = x[j], x[j - 1]
        dx = xj - xjm1
        if abs(dx) < 1e-12: continue
        E_Y2_j, E_Y2_jm1 = E_Y2[j], E_Y2[j-1]
        x_star_num = E_Y2_j - E_Y2_jm1
        x_star_den = 2 * dx
        if abs(x_star_den) < 1e-12: continue
        x_star = x_star_num / x_star_den
        if (x_star >= xjm1 - 1e-9) and (x_star <= xj + 1e-9):
            slope_E_Y2 = x_star_num / dx
            E_Y2_at_x_star = slope_E_Y2 * (x_star - xj) + E_Y2_j
            var_at_x_star = E_Y2_at_x_star - x_star**2
            candidate_variances.append(var_at_x_star)
    return np.max(candidate_variances)

def verify_constraints(v, optimizer):
    """최적화된 결과가 주요 제약조건을 만족하는지 검증합니다."""
    print("\n" + "="*20 + " 최종 결과 검증 시작 " + "="*20)
    
    # 변수 및 파라미터 준비
    _, p_T = optimizer.unpack_vars(v)
    p = p_T.T # (n+1, num_outputs) 형태
    epsilon = optimizer.epsilon
    exp_eps = np.exp(epsilon)
    num_outputs = p.shape[1]
    all_ok = True
    
    # 1. 확률의 합 (Sum-to-One) 검증
    print("\n--- 1. 확률의 합 (Sum-to-One) 검증 ---")
    prob_sums = np.sum(p, axis=1)
    sum_violations = np.where(np.abs(prob_sums - 1.0) > 1e-6)[0]
    
    if len(sum_violations) == 0:
        print("  [성공] 모든 입력(j)에 대한 확률의 합이 1입니다.")
    else:
        all_ok = False
        print("  [실패] 일부 입력(j)에서 확률의 합이 1이 아닙니다:")
        for j_idx in sum_violations:
            print(f"    - 입력 j={j_idx}: 합 = {prob_sums[j_idx]:.8f}")

    # 2. LDP 제약조건 (p* 기준) 검증
    print("\n--- 2. LDP 제약조건 (p* 기준) 검증 ---")
    ldp_violations = []
    for i_idx in range(num_outputs):  # 모든 출력 i에 대해
        output_probs = p[:, i_idx]    # 해당 출력 i에 대한 모든 입력 j의 확률 벡터
        p_star = np.min(output_probs)
        
        # p*가 0에 매우 가까우면, 수치적 오류로 e*p*가 0이 될 수 있음
        # 이 경우, 다른 모든 p(i|j)도 0이어야 함
        upper_bound = exp_eps * p_star

        # 각 p(i|j)가 p* <= p(i|j) <= e^eps * p* 범위에 있는지 확인
        # p(i|j) >= p*는 p_star 정의상 항상 참이므로, 상한만 체크
        for j_idx, p_ij in enumerate(output_probs):
            if p_ij > upper_bound + 1e-9: # 수치적 안정을 위해 작은 톨러런스 추가
                ldp_violations.append({
                    'output_idx': i_idx,
                    'input_j': j_idx,
                    'p_ij': p_ij,
                    'p_star': p_star,
                    'upper_bound': upper_bound
                })
                
    if len(ldp_violations) == 0:
        print("  [성공] 모든 (i, j) 쌍에 대해 LDP 제약조건 (p* <= p(i,j) <= e^eps * p*)을 만족합니다.")
    else:
        all_ok = False
        print(f"  [실패] {len(ldp_violations)}개의 LDP 제약조건 위반 사항 발견:")
        # 너무 많으면 일부만 출력
        for viol in ldp_violations[:5]:
            i_val_str = f"i(idx)={viol['output_idx']}"
            print(f"    - 출력 {i_val_str}, 입력 j={viol['input_j']}: "
                  f"p(i|j) = {viol['p_ij']:.6e} > e^eps * p* = {viol['upper_bound']:.6e} (p*={viol['p_star']:.6e})")
        if len(ldp_violations) > 5:
            print(f"    ... (외 {len(ldp_violations) - 5}개 더 있음)")

    print("\n" + "="*24 + " 검증 종료 " + "="*25)
    return all_ok

def display_and_save_results(result, optimizer, N, output_dir='results_a1_optimized'):
    print("\n" + "#"*20 + f" n = {optimizer.n} 최종 최적화 결과 " + "#"*20)
    worst_case_val = calculate_worst_case_variance(result.x, optimizer)
    final_a, final_p = optimizer.unpack_vars(result.x)
    
    # === [수정] === 최적화된 a_1 값 추출
    final_a1_val = result.x[0]
    
    print(f"\n--- 최종 'a' 값 (a_1으로부터 정수배 생성) ---")
    print(f"a = {np.array2string(final_a, precision=6, max_line_width=120)}")
    print(f"(참고: 최적화된 a_1 = {final_a1_val:.6f})")
    
    print("\n--- 최종 'p' 행렬 (주요 입력 j에 대한 분포) ---")
    indices_to_show = sorted(list(set([0, optimizer.n // 2, optimizer.n])))
    
    if isinstance(optimizer, LDPMechanismOptimizerNoA0_IntegerMultiple):
        i_range = list(range(-optimizer.n, 0)) + list(range(1, optimizer.n + 1))
    else:
        i_range = range(-optimizer.n, optimizer.n + 1)
        
    for j in indices_to_show:
        exp_val = np.dot(final_p[:, j], final_a)
        print(f"\n입력 j = {j} (E[Y|j] = x_{j} = {exp_val:.4f}) 일 때 p(i|j):")
        print("   i   |   p_i,j")
        print("-------|------------")
        for i in i_range:
            prob = optimizer.get_p_val(final_p, i, j)
            if prob > 1e-4: print(f" {i:5d} | {prob:10.6f}")
    print("#" * 58)

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"opt_results_eps{optimizer.epsilon:.4f}_N{N}.pkl")
    
    # === [수정] === results_data 딕셔너리에 final_a1_value 추가
    results_data = {
        'metadata': {'N_total_a_points': N, 'n_param': optimizer.n, 'epsilon': optimizer.epsilon,
                     'use_a0': isinstance(optimizer, LDPMechanismOptimizerWithA0_IntegerMultiple),
                     'optimization_type': 'a1_integer_multiple', 'success': result.success, 
                     'message': result.message, 'final_objective_value': result.fun,
                     'final_a1_value': final_a1_val, # a_1 값 추가
                     'corresponding_worst_case_variance': worst_case_val, 'iterations': result.nit},
        'a_values': final_a, 'p_matrix': final_p, 'scipy_result_object': result }
        
    with open(filename, 'wb') as f: pickle.dump(results_data, f)
    print(f"\n결과가 '{filename}' 파일로 저장되었습니다.")
    print(f"  Success: {result.success}"); print(f"  Message: {result.message}");

    verify_constraints(result.x, optimizer)

    print(f"  Final Objective Value (Avg Var): {result.fun:.6f}")
    print(f"  Corresponding Worst-Case Variance: {worst_case_val:.6f}")

def optimize(epsilon, N, output_dir='results_a1_optimized', ftol=1e-6, maxiter=3000, disp=False):
    my_options = {'ftol': ftol, 'maxiter': maxiter, 'disp': disp}
    
    if N <= 1: raise ValueError("N must be > 1.")
    
    use_a0 = (N % 2 != 0)
    n_param = (N - 1) // 2 if use_a0 else N // 2

    if N % 2 == 0:
        n_param = N // 2
        optimizer = LDPMechanismOptimizerNoA0_IntegerMultiple(n_param, epsilon)
        initial_guess = create_initial_guess_fixed_gap(n_param, epsilon, use_a0=False)
    else:
        n_param = (N - 1) // 2
        optimizer = LDPMechanismOptimizerWithA0_IntegerMultiple(n_param, epsilon)
        initial_guess = create_initial_guess_fixed_gap(n_param, epsilon, use_a0=True)

    result = optimizer.run(initial_guess, options=my_options)
    
    if result.success:
        display_and_save_results(result, optimizer, N, output_dir)
        print("\n모든 과정이 성공적으로 완료되었습니다.")
    else:
        print("\n최적화에 실패했습니다.")
        display_and_save_results(result, optimizer, N, output_dir)
# ===================================================================
# 메인 실행 블록
# ===================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LDP Mechanism Optimizer (a_1 as variable, integer multiples)")
    parser.add_argument('--epsilon', type=float, default=3.0, help='Epsilon for differential privacy')
    parser.add_argument('--N', type=int, default=7, help='Total number of output points (a values)')
    parser.add_argument('--ftol', type=float, default=1e-6, help='Tolerance for termination')
    parser.add_argument('--maxiter', type=int, default=5000, help='Maximum number of iterations')
    args = parser.parse_args()

    my_options = {'ftol': args.ftol, 'maxiter': args.maxiter, 'disp': True}
    start_time = time.time()
    
    if args.N <= 1: raise ValueError("TOTAL_A_POINTS (N) must be > 1 for this script.")
    
    # === [수정] === 새로운 클래스 이름으로 교체
    if args.N % 2 == 0:
        n_param = args.N // 2
        optimizer = LDPMechanismOptimizerNoA0_IntegerMultiple(n_param, args.epsilon)
        initial_guess = create_initial_guess_fixed_gap(n_param, args.epsilon, use_a0=False)
    else:
        n_param = (args.N - 1) // 2
        optimizer = LDPMechanismOptimizerWithA0_IntegerMultiple(n_param, args.epsilon)
        initial_guess = create_initial_guess_fixed_gap(n_param, args.epsilon, use_a0=True)

    result = optimizer.run(initial_guess, options=my_options)
    end_time = time.time()
    print(f"\n--- n = {n_param} finished in {end_time - start_time:.2f} seconds. ---")
    
    if result.success:
        display_and_save_results(result, optimizer, args.N)
        print("\n모든 과정이 성공적으로 완료되었습니다.")
    else:
        print("\n최적화 실패.")
        display_and_save_results(result, optimizer, args.N)