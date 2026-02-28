from __future__ import annotations

import numpy as np


def quantize_fixed(
    x: np.ndarray,
    bits: int,
    frac_bits: int | None = None,
    clip: float | None = None,
) -> np.ndarray:
    """Quantize values to signed fixed-point representation.

    The quantized step is `2**(-frac_bits)`. If `frac_bits` is not provided,
    we use `bits // 2` as a balanced default for integer/fraction precision.

    Args:
        x: Input scalar/array values.
        bits: Total signed fixed-point bit width.
        frac_bits: Fractional bits. Defaults to `bits // 2`.
        clip: Optional symmetric clip bound before quantization. If omitted,
            clip is inferred from data magnitude and then constrained by the
            representable range.

    Returns:
        Quantized values with dtype float.
    """
    if bits <= 1:
        raise ValueError("bits must be > 1 for signed fixed-point")

    if frac_bits is None:
        frac_bits = bits // 2
    if frac_bits < 0 or frac_bits >= bits:
        raise ValueError("frac_bits must satisfy 0 <= frac_bits < bits")

    arr = np.asarray(x, dtype=float)
    step = 2.0 ** (-frac_bits)

    int_min = -(2 ** (bits - 1))
    int_max = (2 ** (bits - 1)) - 1
    representable_abs = int_max * step

    if clip is None:
        if arr.size == 0:
            clip_eff = representable_abs
        else:
            clip_eff = float(np.nanmax(np.abs(arr)))
            if not np.isfinite(clip_eff) or clip_eff <= 0:
                clip_eff = representable_abs
    else:
        clip_eff = float(clip)

    clip_eff = min(max(clip_eff, step), representable_abs)

    clipped = np.clip(arr, -clip_eff, clip_eff)
    q_int = np.rint(clipped / step)
    q_int = np.clip(q_int, int_min, int_max)
    out = q_int * step

    if np.isscalar(x):
        return float(out)
    return out.astype(float)


def quantized_event_indicator_xy(
    samples: np.ndarray,
    d0: float,
    bits: int,
    frac_bits: int | None = None,
    clip: float | None = None,
) -> tuple[np.ndarray, float]:
    """Evaluate event indicators after fixed-point quantization.

    Quantization is applied consistently to `(x, y)` and `d0`, then event is
    evaluated via squared-distance compare:
      `x_q**2 + y_q**2 < d0_q**2`

    Returns:
        indicators: 0/1 array, shape (n,)
        d0_q: quantized threshold
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must be shape (n, 2)")
    if d0 <= 0:
        raise ValueError("d0 must be positive")

    if clip is None:
        sample_abs = float(np.nanmax(np.abs(samples))) if samples.size > 0 else 0.0
        clip_eff = max(sample_abs, float(abs(d0)))
    else:
        clip_eff = float(clip)

    x_q = quantize_fixed(samples[:, 0], bits=bits, frac_bits=frac_bits, clip=clip_eff)
    y_q = quantize_fixed(samples[:, 1], bits=bits, frac_bits=frac_bits, clip=clip_eff)
    d0_q = float(
        quantize_fixed(np.array([d0]), bits=bits, frac_bits=frac_bits, clip=clip_eff)[0]
    )

    s_q = x_q * x_q + y_q * y_q
    indicators = (s_q < (d0_q * d0_q)).astype(int)
    return indicators, d0_q
