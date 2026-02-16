# -*- coding: utf-8 -*-
"""
Adaptive Parallel MiniRocket
Representation-level + MiniRocket-level parallelism
Timing + Accuracy + F1 + Dataset/Combo timing
"""

import os
import warnings
warnings.filterwarnings("ignore")
from sktime.datatypes._panel._convert import from_2d_array_to_nested

import numpy as np
import pandas as pd
from time import time
from itertools import combinations
import multiprocessing

from joblib import Parallel, delayed

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

from scipy.fft import rfft
from scipy.signal import hilbert
from scipy.fftpack import dct
from pywt import wavedec

from sktime.transformations.panel.rocket import MiniRocket

from experiment_minirocket_combos import load_resample_indices
from utils import prep_data


FS = 1.0
NUM_KERNELS = 512
N_FOLDS = 30
TOTAL_CORES = multiprocessing.cpu_count()

RESAMPLE_ROOT = 'dsets_ucr_ts_uv/PythonResampleIndices'
OUT_DIR = 'results/minirocket_adaptive_parallel'
os.makedirs(OUT_DIR, exist_ok=True)


def disable_threads():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def fix_length(x, L):
    x = np.asarray(x)
    if len(x) > L:
        return x[:L]
    if len(x) < L:
        return np.pad(x, (0, L - len(x)))
    return x

TRANSFORMS = {
    'TIME': lambda x, fs: x,
    'DT1': lambda x, fs: np.diff(x, n=1),
    'DT2': lambda x, fs: np.diff(x, n=2),
    'HLB': lambda x, fs: np.abs(hilbert(x)),
    'DWT': lambda x, fs: fix_length(wavedec(x, 'dmey', level=2)[0], len(x)),
    'FFT': lambda x, fs: np.abs(rfft(x)),
    'DCT': lambda x, fs: dct(x, type=1),
}


def make_df(X, transform_fn, fs=1.0):
    n_samples, L = X.shape
    X_new = np.zeros((n_samples, L), dtype=np.float32)
    for i in range(n_samples):
        xi = np.array(X[i], copy=True)
        X_new[i, :] = fix_length(transform_fn(xi, fs), L)
    return X_new

def run_single_transform(name, Xtr, Xte, fs, rocket_jobs):

    disable_threads()  # MiniRocket kendi n_jobs ile paralel
    fn = TRANSFORMS[name]

    t0 = time()

    Xtr_feat = make_df(Xtr, fn, fs)
    Xte_feat = make_df(Xte, fn, fs)
    Xtr_feat = from_2d_array_to_nested(Xtr_feat)
    Xte_feat = from_2d_array_to_nested(Xte_feat)

    rocket = MiniRocket(
        num_kernels=NUM_KERNELS,
        n_jobs=rocket_jobs,
        random_state=0
    )

    Ztr = rocket.fit_transform(Xtr_feat)
    Zte = rocket.transform(Xte_feat)

    rocket_time = time() - t0
    return Ztr, Zte, rocket_time


def run_parallel_representations(transform_combo, Xtr, Xte):
    n_rep = len(transform_combo)
    rep_jobs = min(n_rep, TOTAL_CORES)
    rocket_jobs = max(1, TOTAL_CORES // rep_jobs)

    results = Parallel(
        n_jobs=rep_jobs,
        backend="loky"
    )(
        delayed(run_single_transform)(name, Xtr, Xte, FS, rocket_jobs)
        for name in transform_combo
    )

    Ztr = np.hstack([r[0] for r in results])
    Zte = np.hstack([r[1] for r in results])
    rep_times = {transform_combo[i]: results[i][2] for i in range(len(transform_combo))}

    return Ztr, Zte, rep_times


def run_dataset_combo(X, y, dset_name, transform_combo):

    combo_t0 = time()  # ⏱️ COMBO TIMER

    accs, f1s = [], []

    rocket_times = {t: [] for t in transform_combo}
    clf_fit_times, clf_pred_times = [], []

    for fold in range(N_FOLDS):

        idx_tr, idx_te = load_resample_indices(
            RESAMPLE_ROOT, dset_name, fold
        )

        Xtr, Xte = X[idx_tr], X[idx_te]
        ytr, yte = y[idx_tr], y[idx_te]

        Ztr, Zte, rep_times = run_parallel_representations(
            transform_combo, Xtr, Xte
        )

        for t in rep_times:
            rocket_times[t].append(rep_times[t])

        clf = RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10)
        )

        t0 = time()
        clf.fit(Ztr, ytr)
        clf_fit_times.append(time() - t0)

        t0 = time()
        pred = clf.predict(Zte)
        clf_pred_times.append(time() - t0)

        accs.append(accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, average='macro'))

    combo_total_time = time() - combo_t0
    print(f" combo total time = {combo_total_time}")
    return {
        'dataset': dset_name,
        'transforms': '+'.join(transform_combo),

        'acc_mean': np.mean(accs),
        'acc_std': np.std(accs),
        'f1_mean': np.mean(f1s),

        'rocket_time_mean': np.mean(
            [np.mean(rocket_times[t]) for t in rocket_times]
        ),

        'clf_fit_time_mean': np.mean(clf_fit_times),
        'clf_pred_time_mean': np.mean(clf_pred_times),

        'combo_total_time': combo_total_time,
    }


BASE_TRANSFORMS = ['TIME', 'DWT', 'FFT', 'DCT', 'HLB', 'DT1', 'DT2',]

def build_combinations(transforms, max_len=3):
    combos = []
    for r in range(1, max_len + 1):
        combos.extend(combinations(transforms, r))
    return combos

TRANSFORM_COMBOS = build_combinations(BASE_TRANSFORMS, max_len=8)

ALL_RESULTS = []



df_ref = pd.read_csv('dsets_ucr_ts_uv/dsets_devset_selected50_ucr142.csv')
t_all = time()
DATASET_TIME_ROWS = []
for dset_name in df_ref['dataset'][0:50]:
    dataset_t0 = time()  # ⏱️ DATASET TIMER
    print(f"Dataset : {dset_name}")
    dset, _ = prep_data(
        dset_name,
        repo='ucr_drive',
        orig_split=True
    )

    X_train, X_test, y_train, y_test = dset
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    dataset_rows = []

    for combo in TRANSFORM_COMBOS:
        print(f"  Combo: {combo}")
        row = run_dataset_combo(X, y, dset_name, combo)
        dataset_rows.append(row)
        ALL_RESULTS.append(row)

    dataset_total_time = time() - dataset_t0

    # 🔥 dataset süresini o dataset'in tüm combo satırlarına yaz
    for r in dataset_rows:
        r['dataset_total_time'] = dataset_total_time

pd.DataFrame(ALL_RESULTS).to_csv(
    f"{OUT_DIR}/ALL_RESULTS.csv",
    index=False
)

print(f"\n=== ALL FINISHED in {time() - t_all:.2f}s ===")
