__all__ = ["pcp"]

import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


def PCP(M, mu=None, lam=None, delta=1e-7):

    m, n = M.shape
    if not lam:
        lam = 1.0 / np.sqrt(max(m, n))  # follow the suggestion of paper

    if not mu:
        mu = m * n / (4 * np.linalg.norm(M, ord=1)
                      )  # follow the suggestion of paper

    S = np.ones((m, n))  #initialization
    Y = np.zeros((m, n))

    while True:
        u, d, v = np.linalg.svd(M - S - (Y / mu),
                                full_matrices=False)  # SVD decomposition

        d = np.sign(d) * np.maximum(np.abs(d) - (1.0 / mu), 0)  #sh
        rank = np.sum(d > 0.0)

        u, d, v = u[:, :rank], d[:rank], v[:rank, :]
        L = np.dot(u, np.dot(np.diag(d), v))

        S = np.sign(M - L + (Y / mu)) * np.maximum(
            np.abs(M - L + (Y / mu)) - (lam / mu), 0)

        Y = Y + mu * (M - L - S)

        if np.linalg.norm(M - L - S,
                          ord="fro") <= delta * np.linalg.norm(M, ord="fro"):
            break

    return L, S