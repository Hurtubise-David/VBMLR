# vbmlr/regression.py
import os
import numpy as np
from .io_utils import ensure_dir, read_features_file_cpp_robust

def _lambda_xi(xi):
    xi = np.asarray(xi, dtype=np.float64)
    out = np.empty_like(xi)
    small = xi < 1e-9
    out[small] = 1.0 / 8.0
    xs = xi[~small]
    out[~small] = np.tanh(xs / 2.0) / (4.0 * xs)
    return out

def vb_logistic_fit(X, y, alpha=1.0, max_iter=200, tol=1e-6):
    """
    Variational Bayesian logistic regression (Jaakkola bound).
    Return w = [w0..w7, bias] (9 params).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    N, D = X.shape
    assert D == 8, "need 8 features"
    Xb = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # bias
    Db = D + 1

    mu = np.zeros(Db, dtype=np.float64)
    Sigma = np.eye(Db, dtype=np.float64) / alpha
    xi = np.ones(N, dtype=np.float64)

    I = np.eye(Db, dtype=np.float64)

    for _ in range(max_iter):
        lam = _lambda_xi(xi)
        XL = Xb * (np.sqrt(2.0 * lam)[:, None])
        A = (alpha * I) + (XL.T @ XL)
        Sigma_new = np.linalg.inv(A)

        t = (y - 0.5)
        mu_new = Sigma_new @ (Xb.T @ t)

        S = Sigma_new + np.outer(mu_new, mu_new)
        xi_new = np.sqrt(np.einsum("nd,dd,nd->n", Xb, S, Xb) + 1e-12)

        dm = np.linalg.norm(mu_new - mu) / (np.linalg.norm(mu) + 1e-9)
        mu, Sigma, xi = mu_new, Sigma_new, xi_new
        if dm < tol:
            break

    return mu  # (9,)

def train_vbmlr_all_pairs(dataset_dir: str, out_dir: str, nb_classes=9, min_samples=10, status_cb=None):
    """
    Read features/class{c}.txt (8D) and create disc/disc_{j}_{i}.txt
    """
    ensure_dir(out_dir)

    Xc = {}
    for c in range(1, nb_classes + 1):
        p = os.path.join(dataset_dir, "features", f"class{c}.txt")
        X = read_features_file_cpp_robust(p, expected_dim=8)
        if X is None or X.shape[0] < min_samples:
            raise RuntimeError(f"Not enough samples for class{c}: {p}")
        Xc[c] = X

    total = nb_classes * (nb_classes - 1)
    k = 0

    for j in range(1, nb_classes + 1):
        for i in range(1, nb_classes + 1):
            if i == j:
                continue
            k += 1
            if status_cb:
                status_cb(f"Training disc_{j}_{i}.txt ({k}/{total})")

            Xj = Xc[j]
            Xi = Xc[i]

            X = np.vstack([Xj, Xi])
            y = np.hstack([np.zeros(Xj.shape[0]), np.ones(Xi.shape[0])])

            w = vb_logistic_fit(X, y, alpha=1.0, max_iter=200, tol=1e-6)

            outp = os.path.join(out_dir, f"disc_{j}_{i}.txt")
            with open(outp, "w", encoding="utf-8") as f:
                f.write(",".join([f"{float(v):.12g}" for v in w]) + "\n")
