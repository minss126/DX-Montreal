import math
import numpy as np

class PM():
    def __init__(self, eps, range=1.0, t=3):
        self.eps = eps
        self.range = range
        self.exp_t = np.exp(self.eps / t)
        self.A = (np.exp(self.eps) + self.exp_t) * (self.exp_t + 1) / (self.exp_t * (np.exp(self.eps)-1))

    def PM_batch(self, data):
        # print(f'pm: {data}')
        data = data.reshape(-1) / self.range
        noisy_output = np.zeros_like(data)

        u = np.random.random(len(noisy_output))

        l = (np.exp(self.eps) + self.exp_t) * (data * self.exp_t - 1) / (self.exp_t * (np.exp(self.eps) - 1))
        r = (np.exp(self.eps) + self.exp_t) * (data * self.exp_t + 1) / (self.exp_t * (np.exp(self.eps) - 1))

        inner_idx = np.argwhere(u < np.exp(self.eps) / (self.exp_t + np.exp(self.eps))).reshape(-1)
        outer_idx = np.argwhere(u >= np.exp(self.eps) / (self.exp_t + np.exp(self.eps))).reshape(-1)

        inner_y = np.random.random(len(inner_idx))
        # print(inner_y.shape)
        # print((r[inner_idx]-l[inner_idx]).shape)
        inner_y = (r[inner_idx]-l[inner_idx])*inner_y + l[inner_idx]
        # print(inner_y.shape)
        noisy_output[inner_idx] = inner_y

        length_l = np.abs(l[outer_idx] + self.A)
        legnth_r = np.abs(self.A - r[outer_idx])
        interval_l = length_l / (length_l + legnth_r)
        interval_random = np.random.random(len(outer_idx))
        left_idx = outer_idx[interval_random < interval_l]
        right_idx = outer_idx[interval_random >= interval_l]
        
        left_y = np.random.random(len(left_idx))
        left_y = (l[left_idx] + self.A) * left_y - self.A
        noisy_output[left_idx] = left_y

        right_y = np.random.random(len(right_idx))
        right_y = (self.A - r[right_idx]) * right_y + r[right_idx]
        noisy_output[right_idx] = right_y

        return noisy_output * self.range
    
    def PM_batch_deterministic(self, data, mode='midpoint'):
        data = np.asarray(data).reshape(-1) / self.range
        
        l = (np.exp(self.eps) + self.exp_t) * (data * self.exp_t - 1) / (self.exp_t * (np.exp(self.eps) - 1))
        r = (np.exp(self.eps) + self.exp_t) * (data * self.exp_t + 1) / (self.exp_t * (np.exp(self.eps) - 1))

        noisy_output = l + (r - l) * np.random.random(len(data))

        return noisy_output[0] * self.range