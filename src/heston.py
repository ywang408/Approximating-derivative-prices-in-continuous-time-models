import pandas as pd
from sympy.stats import Normal, density
from scipy.stats import norm
import numpy as np
from sympy import diff, lambdify
from sympy import symbols, log, sqrt

kappa = 0.1456
theta = 0.5172
sig = 0.5786
r = 0
K = 1000
rho = -0.0243
v0 = 0.5172
T = 1 / 12
# symbolic vars for KM's method implementation
s, v, t = symbols('s v t')
sigma = np.sqrt(v0)
N = Normal('x', 0, 1)
d1 = (log(s / K) + (r + sigma ** 2 / 2) * t) / (sigma * sqrt(t))
Gamma = density(N)(d1) / (sigma * s * sqrt(t))
delta0 = 0.5 * (v - sigma ** 2) * s ** 2 * Gamma
fft_a = np.array([57.8425, 62.3711, 67.1005, 72.0291, 77.1553, 82.4766, 87.9903, 93.6933,
                  99.5822, 105.6532, 111.9021])


def bs(S, K, T, r, sigma):
    N = norm.cdf
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r * T) * N(d2)


def L(f):
    """infinitesimal generator"""
    ft = diff(f, t)
    fs = diff(f, s)
    fs2 = diff(fs, s)
    fv = diff(f, v)
    fv2 = diff(fv, v)
    fvs = diff(fv, s)
    tmp1 = -ft + r * s * fs + kappa * (theta - v) * fv
    tmp2 = (v * s ** 2 * fs2 + sig ** 2 * v * fv2) / 2
    tmp3 = rho * sig * v * s * fvs
    return tmp1 + tmp2 + tmp3 - r * f


delta = [delta0]  # symbolic equations of delta terms
delta_func = []  # used to save functions of delta
for _ in range(1, 5):
    delta.append(L(delta[-1]))
for i in range(0, 5):
    delta_func.append(lambdify([s, v, t], delta[i]))
print("done")
S = np.arange(950, 1051, 10)
V = np.arange(0.1, 1.2, 0.1)
bs_list, w = [], [[], [], [], [], []]

# compute panel a
for i in S:
    bs_res = bs(i, K, T, r, sigma)
    w0_approx = bs_res + T * delta_func[0](i, v0, T)
    w1_approx = w0_approx + T ** 2 / 2 * delta_func[1](i, v0, T)
    w2_approx = w1_approx + T ** 3 / 6 * delta_func[2](i, v0, T)
    w3_approx = w2_approx + T ** 4 / 24 * delta_func[3](i, v0, T)
    w4_approx = w3_approx + T ** 5 / 120 * delta_func[4](i, v0, T)
    bs_list.append(bs_res)
    w[0].append(w0_approx)
    w[1].append(w1_approx)
    w[2].append(w2_approx)
    w[3].append(w3_approx)
    w[4].append(w4_approx)
df = pd.DataFrame(
    {"S": S, "FFT": fft_a, "bs":bs_list, "N=0": w[0], "N=1": w[1], "N=2": w[2],
     "N=3": w[3], "N=4": w[4]})
print(df)
df.to_csv("panel_a.csv")
