import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def find_peaks_window(x, y, window=1):
    peak_x = []
    peak_vals = []
    for i in range(len(y)):
        if i < window:
            continue
        if i > len(y) - window:
            continue
        left_side = y[i-window:i]
        right_side = y[i+1:i+1+window]
        if all(left_side < y[i]) and all(right_side < y[i]):
            peak_x.append(x[i])
            peak_vals.append(y[i])
    return np.array(peak_x), np.array(peak_vals)

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

def decide_peaks(t, ys):
    wind = 3
    limit = 0
    while True and limit < 10: 
        noisy_peak_x, noisy_peak_y = find_peaks_window(t, ys, window=wind)
        if len(noisy_peak_y) <= 10:
            return noisy_peak_x, noisy_peak_y
        wind += 1
        limit += 1

# not used but possibly could be if wanting to change threshold levels eslwhere
def local_max_peaks(t, ys, peaks, threshold=2):
    y_spl = UnivariateSpline(t,ys,s=0.1,k=4)
    y_spl_2d = y_spl.derivative(n=1)
    first_ddx = y_spl_2d(t)
    new_peaks = []
    for i in range(len(peaks)):
        # print(peaks[i])
        left = first_ddx[peaks[i] - threshold:peaks[i]]
        # print(left)
        right = first_ddx[peaks[i]: peaks[i] + threshold]
        # print(right)
        if np.average(left) > 0 and np.average(right) < 0: 
            new_peaks.append(peaks[i])

    return new_peaks

def get_peaks(ti, ys):
    noisy_peak_x, noisy_peak_y = decide_peaks(ti, ys)
    noisy_peak_x = np.insert(noisy_peak_x, 0, ti[0])
    noisy_peak_y = np.insert(noisy_peak_y, 0, ys[0])
    noisy_peak_x = np.insert(noisy_peak_x, len(noisy_peak_x), ti[-1] + (ti[1] - ti[0]))
    noisy_peak_y = np.insert(noisy_peak_y, len(noisy_peak_y), 0)
    return noisy_peak_x, noisy_peak_y

# write down the reason for each parameter 
# and docs 
def get_monoExp_init_param(ti, ys):
    m_guess = np.max(ys)
    decay_percentage = 0.1
    t_guess = -np.log(decay_percentage) / (ti[-1] - ti[0])
    b_guess = np.min(ys)
    return (m_guess, t_guess, b_guess)

def calc_r2(noisy_peak_x, noisy_peak_y, m, t, b):
    squaredDiffs = np.square(noisy_peak_y - monoExp(noisy_peak_x, m, t, b))
    squaredDiffsFromMean = np.square(noisy_peak_y - np.mean(noisy_peak_y))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    return rSquared

def fit_T2_star(ti, ys):
    # this is a special characteristic of t2_star
    ys = np.absolute(ys)
    noisy_peak_x, noisy_peak_y = get_peaks(ti, ys)
    # perform the fit
    p0 = get_monoExp_init_param(ti, ys)

    try:
        params, cv = scipy.optimize.curve_fit(monoExp, noisy_peak_x, noisy_peak_y, p0=p0)

    except RuntimeError:
        return 0
    except TypeError: 
        return 0 
    m, t, b = params

    rSquared = calc_r2(noisy_peak_x, noisy_peak_y, m, t, b)
    return rSquared, m, t, b, monoExp(ti, m, t, b)


def fit_T2(ti, ys):
    noisy_peak_x, noisy_peak_y = get_peaks(ti, ys)
    # perform the fit
    p0 = get_monoExp_init_param(ti, ys)

    try:
        params, cv = scipy.optimize.curve_fit(monoExp, noisy_peak_x, noisy_peak_y, p0=p0)

    except RuntimeError:
        return 0
    except TypeError: 
        return 0 
    
    m, t, b = params

    rSquared = calc_r2(noisy_peak_x, noisy_peak_y, m, t, b)
    return rSquared, m, t, b, monoExp(ti, m, t, b)


def fit_T1(ti, ys):
    # perform the fit
    p0 = get_monoExp_init_param(ti, ys)

    try:
        params, cv = scipy.optimize.curve_fit(monoExp, ti, ys, p0=p0)

    except RuntimeError:
        return 0
    except TypeError: 
        return 0 
    m, t, b = params

    rSquared = calc_r2(ti, ys, m, t, b)
    return rSquared, m, t, b, monoExp(ti, m, t, b)