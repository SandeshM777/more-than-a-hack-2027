"""Logistic regression fitted by Newton's method (IRLS), in plain numpy.

Small on purpose: a handful of features, a closed-form update, no randomness, so the fitted
weights are the same on every machine and every run. That is what the byte-for-byte rule needs.
"""

import numpy as np


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def fit(X, y, l2=1.0, intercept=True, center=True, iters=50, tol=1e-9):
    """Weights for P(y=1 | x). Returns a dict that predict() understands.

    center=False keeps zero meaning zero, which the antisymmetric team features rely on: with no
    intercept and no centring, swapping home and away flips the prediction exactly.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = X.mean(0) if center else np.zeros(X.shape[1])
    scale = X.std(0)
    scale[scale < 1e-9] = 1.0
    Z = (X - mean) / scale
    if intercept:
        Z = np.hstack([Z, np.ones((len(Z), 1))])
    penalty = np.full(Z.shape[1], l2)
    if intercept:
        penalty[-1] = 0.0
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = sigmoid(Z @ w)
        grad = Z.T @ (p - y) + penalty * w
        hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < tol:
            break
    return {"mean": mean.tolist(), "scale": scale.tolist(), "w": w.tolist(),
            "intercept": bool(intercept)}


def predict(model, X):
    """P(y=1 | x). The dot product is an explicit elementwise sum rather than a BLAS matmul, so
    the summation order (and so every bit of the result) does not depend on the machine."""
    Z = (np.asarray(X, dtype=np.float64) - np.array(model["mean"])) / np.array(model["scale"])
    w = np.array(model["w"])
    if model["intercept"]:
        z = (Z * w[:-1]).sum(axis=1) + w[-1]
    else:
        z = (Z * w).sum(axis=1)
    return sigmoid(z)
