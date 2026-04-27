import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem


def _amorphous_ring_model(
    params: torch.Tensor,
    offset_row: torch.Tensor,
    offset_col: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a parametric amorphous ring diffraction model.

    The model consists of a central-beam Gaussian, an asymmetric Gaussian ring
    (with separate inner and outer half-widths), and a constant background.
    The radial coordinate is measured in an elliptical frame parameterized
    directly by semiaxes (a, b) and rotation angle theta.

    Parameters
    ----------
    params : torch.Tensor
        11 model parameters: (I0, I1, sigma0, sigma1, sigma2, c_bkgd,
        row0, col0, a, b, theta) where:
        - I0: central beam peak intensity
        - I1: ring peak intensity
        - sigma0: central beam width in pixels
        - sigma1: ring inner half-width in pixels
        - sigma2: ring outer half-width in pixels
        - c_bkgd: constant background
        - row0, col0: center offsets from nominal center in pixels
        - a: semimajor axis of the ring in pixels
        - b: semiminor axis of the ring in pixels
        - theta: rotation angle of the semimajor axis ``a`` measured from
          the column axis toward the row axis, in radians
    offset_row : torch.Tensor
        Row pixel offsets from the nominal center.
    offset_col : torch.Tensor
        Column pixel offsets from the nominal center.

    Returns
    -------
    torch.Tensor
        Model intensity values at each (offset_row, offset_col) position.
    """
    I0, I1, sigma0, sigma1, sigma2, c_bkgd = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
    )
    row0, col0, a, b, theta = params[6], params[7], params[8], params[9], params[10]

    # Shift coordinates by fitted center offset
    u_col = offset_col - col0
    u_row = offset_row - row0

    # Rotate into ellipse-aligned frame and compute elliptical radius
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    u_a = u_col * cos_t + u_row * sin_t
    u_b = -u_col * sin_t + u_row * cos_t
    # Elliptical radius: distance in units where the ring is at r=1
    r_elliptical = torch.sqrt(torch.clamp((u_a / a) ** 2 + (u_b / b) ** 2, min=1e-12))

    # Central beam Gaussian (in pixel-radius space)
    r_pixels = torch.sqrt(torch.clamp(u_col**2 + u_row**2, min=1e-12))
    central = I0 * torch.exp(-(r_pixels**2) / (2.0 * sigma0**2))

    # Asymmetric ring: deviation from r_elliptical=1, scaled to pixel units
    mean_r = (a + b) / 2.0
    dr_pixels = (r_elliptical - 1.0) * mean_r

    inner_mask = (dr_pixels < 0).float()
    outer_mask = 1.0 - inner_mask
    ring = I1 * (
        inner_mask * torch.exp(-(dr_pixels**2) / (2.0 * sigma1**2))
        + outer_mask * torch.exp(-(dr_pixels**2) / (2.0 * sigma2**2))
    )
    return central + ring + c_bkgd


def fit_elliptical_distortion(
    data: Dataset4dstem | Dataset2d,
    center: tuple[float, float],
    fit_radii: tuple[float, float],
    p0: torch.Tensor | None = None,
    mask: NDArray | torch.Tensor | None = None,
    device: str = "cpu",
    max_iter: int = 500,
    lr: float = 0.5,
) -> dict:
    """Fit elliptical distortion from the mean diffraction pattern.

    Computes the mean DP over all scan positions and fits a parametric model
    (central Gaussian + asymmetric ring + background) where the radial
    coordinate is elliptical. Extracts the ellipse parameters (a, b, theta_deg)
    describing the distortion, which are consistent across the full scan.

    Parameters
    ----------
    data : Dataset4dstem or Dataset2d
        Source diffraction data.
        - ``Dataset4dstem``: averaged over scan positions to produce the mean DP.
        - ``Dataset2d``: used directly (e.g. SAED).
    center : (float, float)
        Approximate center as (row, col).
    fit_radii : (float, float)
        Inner and outer radii of the annular fitting region.
    p0 : Tensor or None
        Optional initial guess for the 11 model parameters.
        If None, parameters are estimated automatically.
    mask : ndarray, Tensor, or None
        Optional boolean mask of shape ``(n_row, n_col)`` (True = exclude pixel).
    device : str
        Torch device.
    max_iter : int
        Maximum optimizer iterations.
    lr : float
        Learning rate for Adam optimizer.

    Returns
    -------
    dict with keys:
        "ellipse_params" : (a, b, theta_deg) usable by polar_transform().
            ``theta_deg`` is measured from the column axis toward the row axis.
        "center" : (row, col) refined center
        "fit_params" : Tensor of 11 fitted parameters
        "cost" : final residual sum of squares
    """
    # Compute mean DP over all scan positions
    arr = data.array
    if arr.ndim == 4:
        mean_dp = arr.mean(axis=(0, 1)).astype(np.float32)
    elif arr.ndim == 2:
        mean_dp = arr.astype(np.float32)
    else:
        raise ValueError(f"Got array with shape {arr.shape}. Expected a 2D or 4D-STEM dataset.")

    dp_t = torch.from_numpy(mean_dp).to(device)
    n_row, n_col = dp_t.shape
    center_row, center_col = float(center[0]), float(center[1])
    r_inner, r_outer = float(fit_radii[0]), float(fit_radii[1])

    # Build coordinate grids relative to center
    row_offsets = torch.arange(n_row, dtype=torch.float32, device=device) - center_row
    col_offsets = torch.arange(n_col, dtype=torch.float32, device=device) - center_col
    offset_row_grid, offset_col_grid = torch.meshgrid(row_offsets, col_offsets, indexing="ij")
    r_grid = torch.sqrt(offset_col_grid**2 + offset_row_grid**2)

    # Select pixels in annular region
    annular_mask = (r_grid >= r_inner) & (r_grid <= r_outer)
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(mask.astype(bool)).to(device)
        else:
            mask_t = mask.bool().to(device)
        annular_mask = annular_mask & ~mask_t

    offset_row_fit = offset_row_grid[annular_mask]
    offset_col_fit = offset_col_grid[annular_mask]
    val_fit = dp_t[annular_mask]

    if offset_row_fit.numel() == 0:
        raise ValueError("No pixels in the fitting annulus. Check center and fit_radii.")

    # Auto-estimate initial parameters if not provided
    if p0 is None:
        # Radial integral (r-weighted) to find ring peak.
        # Weighting by r suppresses the central beam contribution because
        # annular area grows with r, emphasizing the ring signal.
        r_flat = r_grid[annular_mask]
        nbins = max(int(r_outer - r_inner), 10)
        bin_edges = torch.linspace(r_inner, r_outer, nbins + 1, device=device)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        radial_integral = torch.zeros(nbins, device=device)
        for i in range(nbins):
            in_bin = (r_flat >= bin_edges[i]) & (r_flat < bin_edges[i + 1])
            if in_bin.any():
                radial_integral[i] = val_fit[in_bin].sum()
        peak_idx = int(radial_integral.argmax().item())
        peak_idx = max(1, min(peak_idx, nbins - 2))
        R_init = float(bin_centers[peak_idx])

        c_bkgd_init = max(float(val_fit.min()), 0.0)
        I1_init = max(float(val_fit[r_flat.sub(R_init).abs() < 3.0].mean()) - c_bkgd_init, 1.0)
        I0_init = max(float(dp_t[int(center_row), int(center_col)]) - c_bkgd_init, 1.0)
        # All sigmas in pixel units
        sigma0_init = R_init * 0.3
        sigma1_init = R_init * 0.15
        sigma2_init = R_init * 0.2

        p0 = torch.tensor(
            [
                I0_init,
                I1_init,
                sigma0_init,
                sigma1_init,
                sigma2_init,
                c_bkgd_init,
                0.0,  # row0 offset (relative to provided center)
                0.0,  # col0 offset
                R_init,  # a (semimajor axis, init as circle)
                R_init,  # b (semiminor axis, init as circle)
                0.0,  # theta in radians
            ],
            dtype=torch.float32,
            device=device,
        )
    else:
        if isinstance(p0, np.ndarray):
            p0 = torch.from_numpy(p0.astype(np.float32)).to(device)
        p0 = p0.float().to(device)

    # Reparameterize positive quantities (I0, I1, sigma0-2, c_bkgd, a, b)
    # via inverse-softplus so the optimizer works in unconstrained space
    def _inv_softplus(x: float) -> float:
        """Inverse of softplus: log(exp(x) - 1)."""
        if x > 20.0:
            return x
        return float(np.log(np.expm1(x)))

    # Indices of parameters that must be positive
    _POS_IDX = [0, 1, 2, 3, 4, 5, 8, 9]  # I0, I1, sigma0-2, c_bkgd, a, b
    raw = p0.clone()
    for i in _POS_IDX:
        raw[i] = _inv_softplus(max(float(raw[i]), 1e-6))
    raw = raw.requires_grad_(True)

    def _to_physical(raw_params: torch.Tensor) -> torch.Tensor:
        """Map unconstrained raw params to physical params."""
        phys = raw_params.clone()
        for i in _POS_IDX:
            phys[i] = F.softplus(raw_params[i])
        return phys

    optimizer = torch.optim.Adam([raw], lr=lr)
    best_loss = float("inf")
    best_raw = raw.detach().clone()
    for _ in range(max_iter):
        optimizer.zero_grad()
        phys = _to_physical(raw)
        model_vals = _amorphous_ring_model(phys, offset_row_fit, offset_col_fit)
        loss = ((model_vals - val_fit) ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            lv = loss.item()
            if lv < best_loss:
                best_loss = lv
                best_raw = raw.detach().clone()

    # Extract final parameters
    with torch.no_grad():
        final_params = _to_physical(best_raw)
        final_loss = float(
            (
                (_amorphous_ring_model(final_params, offset_row_fit, offset_col_fit) - val_fit)
                ** 2
            ).sum()
        )

    a_fit = float(final_params[8])
    b_fit = float(final_params[9])
    theta_fit_rad = float(final_params[10])
    theta_fit_deg = float(np.degrees(theta_fit_rad))

    # Ensure a >= b (a is semimajor)
    if a_fit < b_fit:
        a_fit, b_fit = b_fit, a_fit
        theta_fit_deg = theta_fit_deg + 90.0
    # Normalize angle to [-90, 90)
    theta_fit_deg = ((theta_fit_deg + 90.0) % 180.0) - 90.0

    # Refined center = nominal center + fitted offset
    refined_row = center_row + float(final_params[6])
    refined_col = center_col + float(final_params[7])

    return {
        "ellipse_params": (abs(a_fit), abs(b_fit), theta_fit_deg),
        "center": (refined_row, refined_col),
        "fit_params": final_params,
        "cost": final_loss,
    }
