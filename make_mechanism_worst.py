import numpy as np
from scipy.optimize import minimize, Bounds
import time
import argparse
import pickle
import os
import numba # Numba 임포트

# ===================================================================
# 헬퍼 함수: Worst-Case 분산 계산
# ===================================================================

def calculate_worst_case_variance(v, optimizer):
    """
    주어진 변수 벡터 v에 대해 Worst-Case 분산을 계산합니다.
    이 함수는 목적 함수와 결과 분석 모두에서 사용됩니다.
    """
    n, a, p = optimizer.n, *optimizer.unpack_vars(v)
    
    a_sq = a**2
    x = a @ p      # E[Y|j] for j=0,...,n
    E_Y2 = a_sq @ p  # E[Y^2|j] for j=0,...,n
    
    # 후보 1: 그리드 포인트 x_j에서의 분산
    variances_at_xj = E_Y2 - x**2
    candidate_variances = list(variances_at_xj)

    # 후보 2: 구간 내 로컬 최댓값 x_j^*에서의 분산
    for j in range(1, n + 1):
        xj, xjm1 = x[j], x[j - 1]
        dx = xj - xjm1
        
        # dx가 0에 가까우면 구간이 없으므로 로컬 최댓값도 없음
        if abs(dx) < 1e-12:
            continue

        E_Y2_j, E_Y2_jm1 = E_Y2[j], E_Y2[j-1]
        
        # x_j^* 계산
        x_star_num = E_Y2_j - E_Y2_jm1
        x_star_den = 2 * dx
        
        if abs(x_star_den) < 1e-12:
            continue
            
        x_star = x_star_num / x_star_den
        
        # ✨✨✨ 핵심: x_j^*가 유효한 후보인지 (구간 내에 있는지) 확인
        if (x_star >= xjm1 - 1e-9) and (x_star <= xj + 1e-9):
            # Var(Y|x_j^*) 계산 (선형 보간)
            slope_E_Y2 = x_star_num / dx
            E_Y2_at_x_star = slope_E_Y2 * (x_star - xj) + E_Y2_j
            var_at_x_star = E_Y2_at_x_star - x_star**2
            candidate_variances.append(var_at_x_star)
            
    # 모든 후보 중 최댓값을 반환
    return np.max(candidate_variances)

# ===================================================================
# 옵티마이저 클래스 (Worst-Case 목적 함수 적용)
# ===================================================================

class WorstCaseOptimizerWithA0:
    def __init__(self, n, epsilon):
        self.n, self.epsilon, self.exp_eps = n, epsilon, np.exp(epsilon)
        self.num_a_vars, self.num_p_vars = 1, (2 * n + 1) * (n + 1)
        print(f"Worst-Case 옵티마이저 생성됨 (a_1만 최적화, a_0 포함, n={n})")

    def unpack_vars(self, v):
        a_1_val = v[0]
        a_pos = a_1_val * np.arange(1, self.n + 1)
        p_v = v[self.num_a_vars:].reshape((self.n + 1, 2 * self.n + 1))
        a_full = np.concatenate([-a_pos[::-1], [0], a_pos])
        return a_full, p_v.T
    
    def get_p_val(self, p, i, j):
        # p의 모양: (2n+1, n+1), p.T의 모양: (n+1, 2n+1)
        if j < 0:
            return p[self.n - i, -j]
        return p[self.n + i, j]

    def objective_func(self, v):
        # ✨✨✨ [수정] 목적 함수를 worst-case 분산 계산으로 변경
        return calculate_worst_case_variance(v, self)

    def _create_constraints(self):
        n = self.n; cons = []
        MIN_X_GAP = 1e-7 # x 그리드 포인트 사이의 최소 간격

        # 제약조건들은 Average-Case와 동일
        cons.append({'type': 'eq', 'fun': lambda v: np.sum(v[self.num_a_vars:].reshape((n + 1, 2 * n + 1)), axis=1) - 1})
        cons.append({'type': 'eq', 'fun': lambda v: self.unpack_vars(v)[0] @ self.unpack_vars(v)[1][:, n] - 1})
        if n > 0: cons.append({'type': 'ineq', 'fun': lambda v: v[0]})
        
        # ✨✨✨ [수정] x_j < x_{j+1} 제약조건 추가
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
                    constraints.extend([p_i_j - p_neg_i_n, self.exp_eps * p_neg_i_n - p_i_j])
            p_zero_n = self.get_p_val(p, 0, n)
            for j in range(-n, n + 1):
                p_zero_j = self.get_p_val(p, 0, j)
                constraints.extend([p_zero_j - p_zero_n, self.exp_eps * p_zero_n - p_zero_j])
            return np.array(constraints)
        cons.append({'type': 'ineq', 'fun': ldp_constraints})
        return cons

    def run(self, initial_guess, options):
        lv = [1e-9] * self.num_a_vars + [0.0] * self.num_p_vars
        uv = [np.inf] * self.num_a_vars + [1.0] * self.num_p_vars
        b, c = Bounds(lv, uv), self._create_constraints()
        cb = SimpleCallbackLogger(self)
        print(f"\nn={self.n},eps={self.epsilon} Worst-Case 최적화 시작 (a_0 포함)...")
        return minimize(self.objective_func, initial_guess, method='SLSQP', bounds=b, constraints=c, options=options, callback=cb)

class WorstCaseOptimizerNoA0:
    def __init__(self, n, epsilon):
        self.n, self.epsilon, self.exp_eps = n, epsilon, np.exp(epsilon)
        self.num_output_points = 2 * n
        self.num_a_vars, self.num_p_vars = 1, self.num_output_points * (n + 1)
        print(f"Worst-Case 옵티마이저 생성됨 (a_1만 최적화, a_0 제외, n={n})")

    def _map_i_to_idx(self, i):
        return i + self.n if i < 0 else i + self.n - 1

    def unpack_vars(self, v):
        a_1_val = v[0]
        a_pos = a_1_val * np.arange(1, self.n + 1)
        p_v = v[self.num_a_vars:].reshape((self.n + 1, self.num_output_points))
        a_full = np.concatenate([-a_pos[::-1], a_pos])
        return a_full, p_v.T

    def get_p_val(self, p, i, j):
        if j < 0:
            return p[self._map_i_to_idx(-i), -j]
        return p[self._map_i_to_idx(i), j]

    def objective_func(self, v):
        return calculate_worst_case_variance(v, self)

    def _create_constraints(self):
        n = self.n; cons = []
        MIN_X_GAP = 1e-7 # x 그리드 포인트 사이의 최소 간격

        cons.append({'type': 'eq', 'fun': lambda v: np.sum(v[self.num_a_vars:].reshape((n + 1, 2 * n)), axis=1) - 1})
        cons.append({'type': 'eq', 'fun': lambda v: self.unpack_vars(v)[0] @ self.unpack_vars(v)[1][:, n] - 1})
        if n > 0: cons.append({'type': 'ineq', 'fun': lambda v: v[0]})
        
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
                    constraints.extend([p_i_j - p_neg_i_n, self.exp_eps * p_neg_i_n - p_i_j])
            return np.array(constraints)
        cons.append({'type': 'ineq', 'fun': ldp_constraints})
        return cons

    def run(self, initial_guess, options):
        num_a, num_p = self.num_a_vars, self.num_p_vars
        lower, upper = [1e-9] * num_a + [0.0] * num_p, [np.inf] * num_a + [1.0] * num_p
        b, c = Bounds(lower, upper), self._create_constraints()
        cb = SimpleCallbackLogger(self)
        print(f"\nn={self.n},eps={self.epsilon} Worst-Case 최적화 시작 (a_0 제외)...")
        return minimize(self.objective_func, initial_guess, method='SLSQP', bounds=b, constraints=c, options=options, callback=cb)

class SimpleCallbackLogger:
    def __init__(self, optimizer_instance, disp_interval=10):
        self.iteration = 0; self.optimizer = optimizer_instance
        self.last_fun_val = np.inf; self.disp_interval = disp_interval
        print("\n--- Optimization Log ---"); print("{:>5} | {:>18} | {:>15}".format("Iter", "Objective Value", "Change")); print("-" * 45)
    def __call__(self, xk):
        if self.iteration == 0 or (self.iteration + 1) % self.disp_interval == 0:
            fun_val = self.optimizer.objective_func(xk)
            change = self.last_fun_val - fun_val
            print("{:5d} | {:18.6f} | {:15.6e}".format(self.iteration, fun_val, change))
            self.last_fun_val = fun_val
        self.iteration += 1

# --- 초기값 생성 함수 ---
def create_initial_guess(n, epsilon, use_a0, smoothing_factor=0.1):
    print(f"n={n}, epsilon={epsilon} PM 초기값 계산 (a_0 {'포함' if use_a0 else '제외'}, smoothing={smoothing_factor})...")
    if n == 0: raise ValueError("n=0은 지원하지 않습니다.")
    exp_eps = np.exp(epsilon)
    initial_a_1 = (exp_eps + 1) / ((exp_eps - 1) * n)
    A = n * initial_a_1
    initial_a_pos = initial_a_1 * np.arange(1, n + 1)
    t = np.exp(epsilon/3); p_core, p_tail = exp_eps/(exp_eps+t), t/(exp_eps+t)
    get_overlap = lambda a, b: max(0, min(a[1], b[1]) - max(a[0], b[0]))
    num_output = 2 * n + 1 if use_a0 else 2 * n
    p_matrix = np.zeros((n + 1, num_output))
    i_range = range(-n, n + 1) if use_a0 else list(range(-n,0))+list(range(1,n+1))
    for j in range(n + 1):
        x = j / n; l_x, r_x = (A+1)/2.*x-(A-1)/2., (A+1)/2.*x+(A-1)/2.
        w_c, w_t = r_x - l_x, 2*A - (r_x - l_x)
        pdf_c = p_core/w_c if w_c > 1e-9 else 0; pdf_t = p_tail/w_t if w_t > 1e-9 else 0
        for idx, i in enumerate(i_range):
            if i == -n: s, e = -A, (-n + 0.5) / n * A
            elif i == n: s, e = (n - 0.5) / n * A, A
            elif i == 0 and use_a0: s, e = (-0.5 / n * A, 0.5 / n * A)
            else: s, e = (i - 0.5) / n * A, (i + 0.5) / n * A
            p_matrix[j, idx] = get_overlap([s,e],[-A,l_x])*pdf_t+get_overlap([s,e],[l_x,r_x])*pdf_c+get_overlap([s,e],[r_x,A])*pdf_t
        row_sum = np.sum(p_matrix[j, :]);
        if row_sum > 1e-9: p_matrix[j, :] /= row_sum
    a_full = np.concatenate([-initial_a_pos[::-1], [0], initial_a_pos]) if use_a0 else np.concatenate([-initial_a_pos[::-1], initial_a_pos])
    x_n_initial = np.dot(p_matrix[n, :], a_full)
    if abs(x_n_initial) > 1e-9:
        scaling_factor = 1.0 / x_n_initial; initial_a_1 *= scaling_factor
        print(f"  [INFO] 초기 a_1 값을 {scaling_factor:.4f}배 스케일링하여 x_n=1 제약 만족.")
    uniform = np.full_like(p_matrix, 1./num_output); smoothed = (1-smoothing_factor)*p_matrix + smoothing_factor*uniform
    for j in range(n+1):
        row_sum = np.sum(smoothed[j,:]);
        if row_sum > 1e-9: smoothed[j,:] /= row_sum
    return np.concatenate([[initial_a_1], smoothed.flatten()])

def calculate_average_case_variance(v, optimizer):
    n, a, p = optimizer.n, *optimizer.unpack_vars(v)
    x = a @ p; total = 0.0; a_sq = a**2
    for j in range(1, n + 1):
        xj, xjm1 = x[j], x[j - 1]; dx = xj - xjm1
        if abs(dx) < 1e-20: continue
        pj, pjm1 = p[:, j], p[:, j - 1]
        avg_variance_proxy = (np.dot(a_sq, pj) + np.dot(a_sq, pjm1)) / 2.0
        integral_of_P = avg_variance_proxy * dx
        integral_of_x_sq = (xj**3 - xjm1**3) / 3.0
        total += (integral_of_P - integral_of_x_sq)
    return total

def display_and_save_results(result, optimizer, N, output_dir='results_worst_case'):
    print("\n" + "#"*20 + f" n = {optimizer.n} 최종 최적화 결과 " + "#"*20)
    final_a, final_p = optimizer.unpack_vars(result.x)
    final_a1_val = result.x[0]
    final_x = final_a @ final_p
    avg_case_val = calculate_average_case_variance(result.x, optimizer)

    print(f"\n--- 최종 'a' 값 ---"); print(f"a = {np.array2string(final_a, precision=6, max_line_width=120)}")
    print(f"(참고: 최적화된 a_1 = {final_a1_val:.6f})")

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"opt_results_worst_case_eps{optimizer.epsilon:.4f}_N{N}.pkl")
    
    results_data = {
        'metadata': { 'N_total_a_points': N, 'n_param': optimizer.n, 'epsilon': optimizer.epsilon,
                      'use_a0': isinstance(optimizer, WorstCaseOptimizerWithA0),
                      'optimization_type': 'worst_case_variance', 'success': result.success, 
                      'message': result.message, 'final_objective_value': result.fun,
                      'final_a1_value': final_a1_val,
                      'corresponding_average_case_variance': avg_case_val,
                      'iterations': result.nit },
        'a_values': final_a, 'p_matrix': final_p,
        'x_values': final_x, 'scipy_result_object': result }
    
    with open(filename, 'wb') as f: pickle.dump(results_data, f)
    print(f"\n결과가 '{filename}' 파일로 저장되었습니다.")
    print(f"  Success: {result.success}\n  Message: {result.message}");
    print(f"  Final Objective Value (Worst-Case Var): {result.fun:.6f}")
    print(f"  Corresponding Average-Case Variance: {avg_case_val:.6f}")


def optimize(epsilon, N, output_dir='results_worst_case', ftol=1e-6, maxiter=3000, disp=False):
    my_options = {'ftol': ftol, 'maxiter': maxiter, 'disp': disp}
    
    if N <= 1: raise ValueError("N must be > 1.")
    
    use_a0 = (N % 2 != 0)
    n_param = (N - 1) // 2 if use_a0 else N // 2

    if N % 2 == 0:
        n_param = N // 2
        optimizer = WorstCaseOptimizerNoA0(n_param, epsilon)
        initial_guess = create_initial_guess(n_param, epsilon, use_a0=False)
    else:
        n_param = (N - 1) // 2
        optimizer = WorstCaseOptimizerWithA0(n_param, epsilon)
        initial_guess = create_initial_guess(n_param, epsilon, use_a0=True)

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
    parser = argparse.ArgumentParser(description="LDP Mechanism Optimizer for Worst-Case Variance (Numerical Gradients)")
    parser.add_argument('--epsilon', type=float, default=3.0, help='Epsilon for differential privacy')
    parser.add_argument('--N', type=int, default=15, help='Total number of output points (a values)')
    parser.add_argument('--ftol', type=float, default=1e-8, help='Tolerance for termination')
    parser.add_argument('--maxiter', type=int, default=5000, help='Maximum number of iterations')
    args = parser.parse_args()

    my_options = {'ftol': args.ftol, 'maxiter': args.maxiter, 'disp': True}
    start_time = time.time()
    
    if args.N <= 1: raise ValueError("N must be > 1 for this script.")
    
    # ✨✨✨ [수정] Worst-Case 최적화 클래스 사용
    if args.N % 2 == 0:
        n_param = args.N // 2
        optimizer = WorstCaseOptimizerNoA0(n_param, args.epsilon)
        initial_guess = create_initial_guess(n_param, args.epsilon, use_a0=False)
    else:
        n_param = (args.N - 1) // 2
        optimizer = WorstCaseOptimizerWithA0(n_param, args.epsilon)
        initial_guess = create_initial_guess(n_param, args.epsilon, use_a0=True)

    result = optimizer.run(initial_guess, options=my_options)
    
    end_time = time.time()
    print(f"\n--- n = {n_param} finished in {end_time - start_time:.2f} seconds. ---")
    
    if result.success:
        display_and_save_results(result, optimizer, args.N)
        print("\n모든 과정이 성공적으로 완료되었습니다.")
    else:
        print("\n최적화 실패.")
        display_and_save_results(result, optimizer, args.N)