from __future__ import annotations

import numpy as np


def closest_approach(
    r0: np.ndarray, v0: np.ndarray, T: float
) -> tuple[float, float]:
    """Return (t_star, d_min) for linear relative motion over [0, T].

    Args:
        r0: Initial relative position, shape (3,).
        v0: Relative velocity, shape (3,).
        T: Time window length (non-negative).

    Returns:
        t_star: Time of closest approach in [0, T].
        d_min: Minimum distance over the window.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if r0.shape != (3,) or v0.shape != (3,):
        raise ValueError("r0 and v0 must be shape (3,)")
    if T < 0:
        raise ValueError("T must be non-negative")

    v2 = float(np.dot(v0, v0))
    if v2 == 0.0:
        t_star = 0.0
    else:
        t_star = float(np.clip(-np.dot(r0, v0) / v2, 0.0, T))

    r_star = r0 + v0 * t_star
    d_min = float(np.linalg.norm(r_star))
    return t_star, d_min


def event_indicator(r0: np.ndarray, v0: np.ndarray, T: float, d0: float) -> int:
    """Return 1 if closest approach distance is below threshold d0.

    Args:
        r0: Initial relative position, shape (3,).
        v0: Relative velocity, shape (3,).
        T: Time window length (non-negative).
        d0: Event threshold distance (positive).

    Returns:
        1 if d_min < d0, else 0.
    """
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    _, d_min = closest_approach(r0, v0, T)
    return int(d_min < d0)


def closest_approach_batch(
    r0: np.ndarray, v0: np.ndarray, T: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized closest approach for arrays of r0, v0.

    Args:
        r0: Initial relative positions, shape (n, 3).
        v0: Relative velocities, shape (n, 3).
        T: Time window length (non-negative).

    Returns:
        t_star: Array of closest-approach times, shape (n,).
        d_min: Array of minimum distances, shape (n,).
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if r0.ndim != 2 or v0.ndim != 2 or r0.shape[1] != 3 or v0.shape[1] != 3:
        raise ValueError("r0 and v0 must be shape (n, 3)")
    if T < 0:
        raise ValueError("T must be non-negative")

    v2 = np.sum(v0**2, axis=1)
    dot = np.sum(r0 * v0, axis=1)
    t_star = np.zeros_like(v2)
    mask = v2 > 0.0
    t_star[mask] = np.clip(-dot[mask] / v2[mask], 0.0, T)
    r_star = r0 + v0 * t_star[:, None]
    d_min = np.linalg.norm(r_star, axis=1)
    return t_star, d_min
