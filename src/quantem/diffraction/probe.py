from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, binary_opening

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.utils.imaging_utils import cross_correlation_shift


def get_probe_size(
    dp: NDArray,
    thresh_lower: float = 0.01,
    thresh_upper: float = 0.99,
    n_thresholds: int = 100,
) -> tuple[float, float, float]:
    """
    Estimate the center and radius of the direct beam in a diffraction pattern.

    The algorithm sweeps a series of intensity thresholds across the pattern,
    converts each binary mask area into an equivalent circular radius, and
    identifies the plateau region where the radius is stable. The radius is
    taken as the mean over that plateau, and the center is the intensity-
    weighted center-of-mass within the corresponding mask.

    Adapted from py4DSTEM ``get_probe_size``.

    Parameters
    ----------
    dp
        A 2D diffraction pattern (e.g. position-averaged or aligned).
    thresh_lower
        Lower bound of the threshold sweep, as a fraction of the maximum
        intensity. Must be in [0, 1).
    thresh_upper
        Upper bound of the threshold sweep, as a fraction of the maximum
        intensity. Must be in (0, 1].
    n_thresholds
        Number of thresholds to evaluate between *thresh_lower* and
        *thresh_upper*.

    Returns
    -------
    radius
        Estimated radius of the central disk in pixels.
    center_row
        Row coordinate of the disk center (subpixel).
    center_col
        Column coordinate of the disk center (subpixel).
    """
    dp = np.asarray(dp, dtype=np.float64)
    dp_max = np.max(dp)
    if dp_max == 0:
        raise ValueError("Diffraction pattern is all zeros; cannot estimate probe.")

    thresh_vals = np.linspace(thresh_lower, thresh_upper, n_thresholds)
    r_vals = np.sqrt(np.array([np.sum(dp > dp_max * t) for t in thresh_vals]) / np.pi)

    # Identify the stable plateau via the derivative of r(threshold)
    dr = np.gradient(r_vals)
    plateau_mask = (dr <= 0) & (dr >= 2.0 * np.median(dr))

    if not np.any(plateau_mask):
        # Fallback: use the middle half of thresholds
        quarter = n_thresholds // 4
        plateau_mask = np.zeros(n_thresholds, dtype=bool)
        plateau_mask[quarter : 3 * quarter] = True

    radius = float(np.mean(r_vals[plateau_mask]))

    # Center-of-mass within the best-matching mask
    best_thresh = float(np.mean(thresh_vals[plateau_mask]))
    mask = dp > dp_max * best_thresh
    rows, cols = np.indices(dp.shape)
    masked_dp = dp * mask
    total = masked_dp.sum()
    center_row = float(np.sum(rows * masked_dp) / total)
    center_col = float(np.sum(cols * masked_dp) / total)

    return radius, center_row, center_col


def get_vacuum_probe(
    data: NDArray | Dataset4dstem,
    mask_realspace: NDArray | None = None,
    align: bool = True,
) -> NDArray:
    """
    Compute a vacuum probe by averaging diffraction patterns from selected
    scan positions, optionally with iterative cross-correlation alignment.

    Adapted from py4DSTEM ``DataCube.get_vacuum_probe``.

    Parameters
    ----------
    data
        Either a 4D numpy array ``(scan_y, scan_x, ky, kx)`` or a
        :class:`Dataset4dstem`.
    mask_realspace
        Boolean array ``(scan_y, scan_x)`` where ``True`` marks vacuum
        positions.  If ``None``, all positions are averaged.
    align
        If ``True``, iteratively align each diffraction pattern to the
        running average via cross-correlation before accumulating.

    Returns
    -------
    NDArray
        2D averaged (and optionally aligned) diffraction pattern.
    """
    if isinstance(data, Dataset4dstem):
        arr = data.array
    else:
        arr = np.asarray(data)

    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D array (scan_y, scan_x, ky, kx), got shape {arr.shape}.")

    if mask_realspace is not None:
        mask_realspace = np.asarray(mask_realspace, dtype=bool)
        positions = np.argwhere(mask_realspace)
    else:
        sy, sx = arr.shape[:2]
        positions = np.argwhere(np.ones((sy, sx), dtype=bool))

    if len(positions) == 0:
        raise ValueError("No scan positions selected.")

    if not align:
        return np.mean(arr[positions[:, 0], positions[:, 1]], axis=0)

    # Iterative aligned averaging
    probe = arr[positions[0, 0], positions[0, 1]].astype(np.float64).copy()
    for n, (ry, rx) in enumerate(positions[1:], start=2):
        curr_dp = arr[ry, rx].astype(np.float64)
        shift = cross_correlation_shift(probe, curr_dp, upsample_factor=10)
        shifted_dp = _shift_array(curr_dp, shift)
        probe = probe * (n - 1) / n + shifted_dp / n

    return probe


def _shift_array(arr: NDArray, shift: tuple[float, float]) -> NDArray:
    """Shift a 2D array by (row_shift, col_shift) using Fourier interpolation."""
    row_shift, col_shift = shift
    ny, nx = arr.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    phase = np.exp(-2j * np.pi * (row_shift * fy[:, None] + col_shift * fx[None, :]))
    return np.real(np.fft.ifft2(np.fft.fft2(arr) * phase))


def zero_probe_vacuum(
    probe: NDArray,
    radius: float | None = None,
    center_row: float | None = None,
    center_col: float | None = None,
    expansion: float = 1.5,
    opening: int = 3,
    dilation: int = 1,
) -> NDArray:
    """
    Smoothly zero the vacuum region outside the probe disk.

    A binary mask is created from the probe intensity, cleaned with
    morphological opening and dilation, and then a sinusoidal (cosine-
    squared) decay is applied from the mask edge outward so the probe
    tapers smoothly to zero.

    Adapted from py4DSTEM ``Probe.zero_vacuum_sinusoid``.

    Parameters
    ----------
    probe
        2D vacuum-averaged diffraction pattern.
    radius, center_row, center_col
        Probe disk parameters. If ``None``, estimated via
        :func:`get_probe_size`.
    expansion
        Multiplicative factor on *radius* controlling how far out
        from the center the sinusoidal window extends.
    opening
        Size of the structuring element for binary opening (removes
        small bright spots outside the disk).
    dilation
        Number of binary dilation iterations to expand the mask edge
        before applying the window.

    Returns
    -------
    NDArray
        Probe with vacuum smoothly zeroed.
    """
    probe = np.asarray(probe, dtype=np.float64).copy()

    if radius is None or center_row is None or center_col is None:
        r_est, cr_est, cc_est = get_probe_size(probe)
        radius = radius if radius is not None else r_est
        center_row = center_row if center_row is not None else cr_est
        center_col = center_col if center_col is not None else cc_est

    # Threshold mask from the measured disk
    dp_max = np.max(probe)
    mask = probe > dp_max * 0.01

    # Clean the mask
    if opening > 0:
        struct = np.ones((opening, opening), dtype=bool)
        mask = binary_opening(mask, structure=struct)
    if dilation > 0:
        mask = binary_dilation(mask, iterations=dilation)

    # Distance from mask edge (inside = positive, outside = negative concept)
    from scipy.ndimage import distance_transform_edt

    dist_outside = distance_transform_edt(~mask)

    # Sinusoidal window: cosine-squared decay over a range of `expansion * radius`
    # pixels from the mask edge
    decay_width = max(radius * (expansion - 1.0), 1.0)
    window = np.ones_like(probe)
    transition = (dist_outside > 0) & (dist_outside <= decay_width)
    window[transition] = np.cos((np.pi / 2) * dist_outside[transition] / decay_width) ** 2
    window[dist_outside > decay_width] = 0.0

    return probe * window


def get_probe_kernel(
    probe: NDArray,
    mode: str = "sigmoid",
    radius: float | None = None,
    center_row: float | None = None,
    center_col: float | None = None,
    radius_inner: float | None = None,
    radius_outer: float | None = None,
) -> NDArray:
    """
    Generate a cross-correlation kernel from a vacuum probe.

    The kernel is the probe shifted so its center is at the array corner
    (for use with FFT-based cross-correlation). Different modes control
    how the probe is weighted:

    - ``"flat"``: probe values inside the disk, zero outside. Sums to 1.
    - ``"sigmoid"``: probe with a negative trench between *radius_inner*
      and *radius_outer*, creating an edge-sensitive kernel that "locks
      in" on disk edges during template matching. Sums to 0.

    Adapted from py4DSTEM ``Probe.get_kernel``.

    Parameters
    ----------
    probe
        2D vacuum probe (full diffraction-pattern size).
    mode
        ``"flat"`` or ``"sigmoid"``.
    radius, center_row, center_col
        Probe disk parameters. If ``None``, estimated via
        :func:`get_probe_size`.
    radius_inner
        Inner radius of the sigmoid trench. Defaults to *radius*.
    radius_outer
        Outer radius of the sigmoid trench. Defaults to ``2 * radius``.

    Returns
    -------
    NDArray
        2D kernel, same shape as *probe*, suitable for FFT-based
        cross-correlation.
    """
    probe = np.asarray(probe, dtype=np.float64).copy()

    if radius is None or center_row is None or center_col is None:
        r_est, cr_est, cc_est = get_probe_size(probe)
        radius = radius if radius is not None else r_est
        center_row = center_row if center_row is not None else cr_est
        center_col = center_col if center_col is not None else cc_est

    if radius_inner is None:
        radius_inner = radius
    if radius_outer is None:
        radius_outer = 2.0 * radius

    ny, nx = probe.shape
    rows, cols = np.indices(probe.shape)
    r = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)

    if mode == "flat":
        kernel = probe * (r <= radius_outer).astype(float)
        total = kernel.sum()
        if total > 0:
            kernel /= total
    elif mode == "sigmoid":
        # Logistic sigmoid that transitions from 1 → 0 between radius_inner
        # and radius_outer.  Steepness is set so the sigmoid goes from ~0.95
        # to ~0.05 across that interval.
        r_mid = 0.5 * (radius_inner + radius_outer)
        # k chosen so sigmoid ≈ 0.95 at radius_inner and ≈ 0.05 at radius_outer
        dr = max(radius_outer - radius_inner, 1e-6)
        k = 2.0 * np.log(19.0) / dr  # log(19) ≈ 2.944

        sigmoid = 1.0 / (1.0 + np.exp(k * (r - r_mid)))

        # Positive lobe: probe weighted by sigmoid
        # Negative lobe: a uniform negative ring in the trench region,
        # scaled so the kernel sums to zero.  This ensures the trench
        # has real negative amplitude even when the probe is dim there.
        pos_kernel = probe * sigmoid
        neg_ring = (1.0 - sigmoid) * (r <= radius_outer).astype(float)

        pos_sum = pos_kernel.sum()
        neg_sum = neg_ring.sum()
        if neg_sum > 0:
            neg_ring *= pos_sum / neg_sum

        kernel = pos_kernel - neg_ring
    else:
        raise ValueError(f"Unknown kernel mode {mode!r}. Use 'flat' or 'sigmoid'.")

    # Shift center to the array corner for FFT-based cross-correlation
    shift_r = int(np.round(center_row))
    shift_c = int(np.round(center_col))
    kernel = np.roll(kernel, -shift_r, axis=0)
    kernel = np.roll(kernel, -shift_c, axis=1)

    return kernel


def deconvolve_probe(
    dp: NDArray,
    probe: NDArray,
    gamma: float | None = None,
) -> NDArray:
    """
    Wiener-filter deconvolution to remove beam-convergence broadening.

    In convergent-beam nano-diffraction the observed 2D intensity is a
    convolution of the true scattering intensity with the probe intensity
    distribution F(u):

        I_obs(u) = |q(u)|^2 * F(u)

    The Wiener filter recovers |q(u)|^2 in 2D Fourier space::

        I(r) = I_obs(r) · F*(r) / (|F(r)|^2 + γ)

    where r denotes the Fourier-conjugate (real-space) variable and γ
    is a regularisation parameter that suppresses noise amplification.

    The deconvolution is performed on the **full 2D diffraction pattern**.
    To obtain a corrected 1D radial profile for PDF analysis, azimuthally
    average the returned 2D result.

    Adapted from Hirotsu *et al.*, J. Electron Microsc. **50**, 435 (2001).

    Parameters
    ----------
    dp
        2D observed diffraction pattern (e.g. position-averaged or
        vacuum-masked mean).
    probe
        2D vacuum probe pattern.  Must be the same shape as *dp*,
        or will be zero-padded / cropped to match.
    gamma
        Wiener regularisation parameter.  If ``None``, defaults to
        ``0.08 * max(|F(r)|^2)``, following Hirotsu *et al.*

    Returns
    -------
    NDArray
        Deconvolved 2D diffraction pattern.
    """
    dp = np.asarray(dp, dtype=np.float64)
    probe = np.asarray(probe, dtype=np.float64)

    if dp.ndim != 2 or probe.ndim != 2:
        raise ValueError("Both dp and probe must be 2D arrays.")

    # Match probe shape to dp shape by zero-padding or cropping
    if probe.shape != dp.shape:
        matched = np.zeros_like(dp)
        sy = min(probe.shape[0], dp.shape[0])
        sx = min(probe.shape[1], dp.shape[1])
        matched[:sy, :sx] = probe[:sy, :sx]
        probe = matched

    # Shift probe center to array origin (0,0) for FFT-based deconvolution.
    # The probe center is found via get_probe_size; rolling it to (0,0)
    # avoids the phase ramp that would otherwise corrupt the result.
    _, cr, cc = get_probe_size(probe)
    probe = np.roll(probe, -int(np.round(cr)), axis=0)
    probe = np.roll(probe, -int(np.round(cc)), axis=1)

    # Normalise probe to unit sum
    psf_sum = probe.sum()
    if psf_sum > 0:
        probe = probe / psf_sum

    # 2D Wiener deconvolution
    F_obs = np.fft.fft2(dp)
    F_psf = np.fft.fft2(probe)

    F_psf_sq = np.abs(F_psf) ** 2
    if gamma is None:
        gamma = 0.08 * np.max(F_psf_sq)

    F_deconv = F_obs * np.conj(F_psf) / (F_psf_sq + gamma)
    deconvolved = np.real(np.fft.ifft2(F_deconv))

    return deconvolved


def plot_probe_estimate(
    dp: NDArray,
    radius: float | None = None,
    center_row: float | None = None,
    center_col: float | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """
    Plot the diffraction pattern with the estimated probe disk overlaid.

    Parameters
    ----------
    dp
        A 2D diffraction pattern.
    radius, center_row, center_col
        Probe parameters. If any are ``None``, they are estimated via
        :func:`get_probe_size`.
    **kwargs
        Forwarded to :func:`get_probe_size` when auto-estimating
        (e.g. *thresh_lower*, *thresh_upper*, *n_thresholds*).

    Returns
    -------
    matplotlib.figure.Figure
    """
    dp = np.asarray(dp, dtype=np.float64)

    if radius is None or center_row is None or center_col is None:
        r_est, cr_est, cc_est = get_probe_size(dp, **kwargs)
        radius = radius if radius is not None else r_est
        center_row = center_row if center_row is not None else cr_est
        center_col = center_col if center_col is not None else cc_est

    fig, ax = plt.subplots()
    ax.imshow(
        dp,
        cmap="gray",
        norm=plt.Normalize(vmin=np.percentile(dp, 1), vmax=np.percentile(dp, 99)),
    )
    circle = plt.Circle(
        (center_col, center_row),
        radius,
        edgecolor="r",
        facecolor="none",
        linewidth=1.5,
        label=f"r = {radius:.1f} px",
    )
    ax.add_patch(circle)
    ax.plot(center_col, center_row, "r+", markersize=10)
    ax.set_title("Probe estimate")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def show_kernel(
    kernel: NDArray,
    radius: float | None = None,
) -> plt.Figure:
    """
    Visualize a cross-correlation kernel with radial profile.

    Parameters
    ----------
    kernel
        2D kernel array (center at corner, for FFT cross-correlation).
    radius
        If given, marks the probe radius on the radial profile plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Shift kernel center back to the middle for display
    ny, nx = kernel.shape
    display = np.fft.fftshift(kernel)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 2D view
    vmax = np.max(np.abs(display))
    axes[0].imshow(display, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Kernel (centered)")

    # Radial profile
    cy, cx = ny // 2, nx // 2
    rows, cols = np.indices(display.shape)
    r = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    r_int = r.astype(int)
    max_r = min(cy, cx)
    radial = np.bincount(r_int.ravel(), weights=display.ravel(), minlength=max_r)
    counts = np.bincount(r_int.ravel(), minlength=max_r)
    counts[counts == 0] = 1
    radial_profile = radial[:max_r] / counts[:max_r]

    axes[1].plot(radial_profile)
    axes[1].axhline(0, color="k", linewidth=0.5)
    if radius is not None:
        axes[1].axvline(radius, color="r", linestyle="--", label=f"r = {radius:.1f} px")
        axes[1].legend()
    axes[1].set_xlabel("Radius (px)")
    axes[1].set_ylabel("Mean kernel value")
    axes[1].set_title("Radial profile")

    fig.tight_layout()
    return fig
