#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:41:47 2026

@author: andrew
"""

import numpy as np
from scipy.optimize import minimize


def fit_Poisson_individual(Y,X,Z,C):
    n_a, n_t, d = X.shape
    _, _, d_beta = Z.shape
    
    p = n_a * d + d_beta
    
    # contiguous memory for speed
    X = np.ascontiguousarray(X)
    Z = np.ascontiguousarray(Z)
    Y = np.ascontiguousarray(Y)
    C = np.ascontiguousarray(C)
    
    # flatten
    X2 = X.reshape(-1, d)
    Z2 = Z.reshape(-1, d_beta)
    
    Y2 = Y.reshape(-1)
    C2 = C.reshape(-1)
    
    # area index for each observation
    area_idx = np.repeat(np.arange(n_a), n_t)
    
    
    # =========================================================
    # NEGATIVE LOG-LIKELIHOOD
    # =========================================================
    
    def nll(params):
    
        B = params[:n_a * d].reshape(n_a, d)
        gamma = params[n_a * d:]
    
        eta = C2 + Z2 @ gamma
        eta += np.sum(X2 * B[area_idx], axis=1)
    
        # numerical stability
        eta = np.clip(eta, -20, 20)
    
        mu = np.exp(eta)
    
        return -(Y2 @ eta - np.sum(mu))
    
    
    # =========================================================
    # GRADIENT
    # =========================================================
    
    def grad_nll(params):
    
        B = params[:n_a * d].reshape(n_a, d)
        gamma = params[n_a * d:]
    
        eta = C2 + Z2 @ gamma
        eta += np.sum(X2 * B[area_idx], axis=1)
    
        eta = np.clip(eta, -20, 20)
    
        mu = np.exp(eta)
    
        residual = mu - Y2
    
        # gradient wrt shared coefficients
        grad_gamma = Z2.T @ residual
    
        # gradient wrt area coefficients
        grad_B = np.zeros((n_a, d))
    
        for a in range(n_a):
    
            mask = (area_idx == a)
    
            grad_B[a] = X2[mask].T @ residual[mask]
    
        return np.concatenate([
            grad_B.ravel(),
            grad_gamma
        ])
    
    
    # =========================================================
    # HESSIAN-VECTOR PRODUCT
    # =========================================================
    
    def hessp(params, v):
    
        B = params[:n_a * d].reshape(n_a, d)
        gamma = params[n_a * d:]
    
        v_B = v[:n_a * d].reshape(n_a, d)
        v_gamma = v[n_a * d:]
    
        eta = C2 + Z2 @ gamma
        eta += np.sum(X2 * B[area_idx], axis=1)
    
        eta = np.clip(eta, -20, 20)
    
        mu = np.exp(eta)
    
        # compute A v
        Av = Z2 @ v_gamma
        Av += np.sum(X2 * v_B[area_idx], axis=1)
    
        # apply W
        WAv = mu * Av
    
        # gamma block
        Hv_gamma = Z2.T @ WAv
    
        # B block
        Hv_B = np.zeros((n_a, d))
    
        for a in range(n_a):
    
            mask = (area_idx == a)
    
            Hv_B[a] = X2[mask].T @ WAv[mask]
    
        return np.concatenate([
            Hv_B.ravel(),
            Hv_gamma
        ])
    
    
    # =========================================================
    # SMART INITIALIZATION
    # =========================================================
    
    base_x0 = np.zeros(p)
    
    B0 = np.zeros((n_a, d))
    
    for a in range(n_a):
    
        y_mean = np.mean(Y[a])
    
        y_mean = max(y_mean, 1e-6)
    
        c_mean = np.mean(C[a])
    
        x_mean = np.mean(X[a, :, 0])
    
        x_mean = max(np.abs(x_mean), 1e-6)
    
        # initialize area intercept coefficient
        B0[a, 0] = (np.log(y_mean) - c_mean) / x_mean
    
    base_x0[:n_a * d] = B0.ravel()
    
    
    # =========================================================
    # MULTIPLE STARTS
    # =========================================================
    
    # =========================================================
    # MANY RANDOMIZED STARTS AROUND SMART INITIALIZATION
    # =========================================================
    
    starts = []
    
    scales = [
        0.0,
        1e-4,
        5e-4,
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
        2e-1,
        5e-1,
        1e-0,
        1.1
    ]
    
    n_random_per_scale = 35
    
    prng = np.random.default_rng(12345)
    
    for scale in scales:
    
        for _ in range(n_random_per_scale):
    
            x0 = base_x0.copy()
    
            x0 += scale * prng.standard_normal(p)
    
            starts.append(x0)
    
    print(f"Total starts: {len(starts)}")
    
    
    # =========================================================
    # RUN OPTIMIZATION
    # =========================================================
    
    # results = []
    
    # for i, x0 in enumerate(starts):
    
    #     print("\n===================================")
    #     print(f"START {i+1}/{len(starts)}")
    #     print("===================================")
    
    #     res = minimize(
    #         nll,
    #         x0,
    #         jac=grad_nll,
    #         hessp=hessp,
    #         method="Newton-CG",
    #         options={
    #             "maxiter": 200,
    #             "xtol": 1e-6,
    #         }
    #     )
    
    #     print("success :", res.success)
    #     print("message :", res.message)
    #     print("fun     :", res.fun)
    #     print("nit     :", res.nit)
    
    #     results.append(res)
    
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1)(delayed(minimize)(nll,x0,jac=grad_nll,hessp=hessp,method="Newton-CG",options={    "maxiter": 200,    "xtol": 1e-1}) for x0 in starts)
    
    
    # =========================================================
    # CHOOSE BEST RESULT
    # =========================================================
    
    successful = [r for r in results if np.isfinite(r.fun)]
    
    if len(successful) == 0:
        raise RuntimeError("No successful optimizations")
    
    best_res = min(successful, key=lambda r: r.fun)
    
    print("\n===================================")
    print("BEST RESULT")
    print("===================================")
    
    print("success :", best_res.success)
    print("fun     :", best_res.fun)
    print("nit     :", best_res.nit)
    
    
    return best_res