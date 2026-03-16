from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from .dataset4dstem import Dataset4dstem

from quantem.core.datastructures.dataset4d import Dataset4d


class Polar4dstem(Dataset4d):
    """4D-STEM dataset in polar coordinates (scan_y, scan_x, phi, r)."""

    def __init__(
        self,
        array: NDArray | Any,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
        origin_array: NDArray | None = None,
        _token: object | None = None,
    ):
        if metadata is None:
            metadata = {}
        mdata_keys_polar = [
            "polar_radial_min",
            "polar_radial_max",
            "polar_radial_step",
            "polar_num_annular_bins",
            "polar_two_fold_rotation_symmetry",
            "polar_ellipse_params",
        ]
        for k in mdata_keys_polar:
            if k not in metadata:
                metadata[k] = None
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            metadata=metadata,
            _token=_token,
        )
        self.origin_array = origin_array

    @classmethod
    def from_array(
        cls,
        array: NDArray | Any,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
    ) -> "Polar4dstem":
        array = np.asarray(array)
        if array.ndim != 4:
            raise ValueError(
                f"Found array with shape: {array.shape}. "
                "Polar4dstem.from_array expects a 4D array."
            )
        if origin is None:
            origin = np.zeros(4, dtype=float)
        if sampling is None:
            sampling = np.ones(4, dtype=float)
        if units is None:
            units = ["pixels", "pixels", "deg", "pixels"]
        if metadata is None:
            metadata = {}
        return cls(
            array=array,
            name=name if name is not None else "Polar 4D-STEM dataset",
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            metadata=metadata,
            _token=cls._token,
        )

    @property
    def n_phi(self) -> int:
        return int(self.array.shape[2])

    @property
    def n_r(self) -> int:
        return int(self.array.shape[3])


def _to_numpy(tensor: torch.Tensor) -> NDArray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def _normalize_coords_for_grid_sample(
    coords_y: torch.Tensor,
    coords_x: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Convert pixel coordinates to normalized [-1, 1] coordinates for grid_sample.
    grid_sample expects x_norm = 2*x/(W-1) - 1, y_norm = 2*y/(H-1) - 1,
    stacked as (..., 2) in [x, y] order.
    """
    x_norm = 2.0 * coords_x / (width - 1) - 1.0
    y_norm = 2.0 * coords_y / (height - 1) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def _precompute_polar_coords(
    ny: int,
    nx: int,
    origin_row: float,
    origin_col: float,
    ellipse_params: tuple[float, float, float] | None,
    num_annular_bins: int,
    radial_min: float,
    radial_max: float | None,
    radial_step: float,
    two_fold_rotation_symmetry: bool,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    origin_row = float(origin_row)
    origin_col = float(origin_col)
    if radial_step <= 0:
        raise ValueError(f"Got radial_step = {radial_step}. radial_step must be > 0.")
    if num_annular_bins < 1:
        raise ValueError("num_annular_bins must be >= 1.")
    # Use the shortest distance from the origin to any image edge so the
    # polar grid never samples outside the image bounds.
    if radial_max is None:
        r_row_pos = origin_row
        r_row_neg = (ny - 1) - origin_row
        r_col_pos = origin_col
        r_col_neg = (nx - 1) - origin_col
        radial_max_eff = float(min(r_row_pos, r_row_neg, r_col_pos, r_col_neg))
    else:
        radial_max_eff = float(radial_max)
    # Guarantee at least one radial bin so downstream code never gets an empty array
    if radial_max_eff <= radial_min:
        radial_max_eff = radial_min + radial_step

    #  create radial bins and phi bins, then create the grid of (phi, r) coordinates
    radial_bins = torch.arange(
        radial_min, radial_max_eff, radial_step, dtype=torch.float32, device=device
    )
    if radial_bins.numel() == 0:
        radial_bins = torch.tensor([radial_min], dtype=torch.float32, device=device)
    phi_range = torch.pi if two_fold_rotation_symmetry else 2.0 * torch.pi
    # Drop the last endpoint because 0 and 2pi (or pi) are the same angle
    phi_bins = torch.linspace(
        0.0, phi_range, num_annular_bins + 1, dtype=torch.float32, device=device
    )[:-1]
    phi_grid, r_grid = torch.meshgrid(phi_bins, radial_bins, indexing="ij")

    # apply ellipse distortion correction if requested
    # TODO: implement method to estimate ellipse_params from data
    if ellipse_params is None:
        x = r_grid * torch.cos(phi_grid)
        y = r_grid * torch.sin(phi_grid)
    else:
        if len(ellipse_params) != 3:
            raise ValueError("ellipse_params must be (a, b, theta_deg).")
        a, b, theta_deg = ellipse_params
        theta = torch.deg2rad(torch.tensor(theta_deg, dtype=torch.float32, device=device))
        # Rotate into the ellipse frame, scale by a/b to undo the distortion,
        # then rotate back so sampling follows the true circular rings
        alpha = phi_grid - theta
        u = (a / b) * r_grid * torch.cos(alpha)
        v_prime = r_grid * torch.sin(alpha)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x = u * cos_t - v_prime * sin_t
        y = u * sin_t + v_prime * cos_t
    coords_y = y + origin_row
    coords_x = x + origin_col
    # convert to normalized coordinates for grid_sample
    grid = _normalize_coords_for_grid_sample(coords_y, coords_x, ny, nx)
    grid = grid.unsqueeze(0)  # (1, n_phi, n_r, 2)
    return grid, phi_bins, radial_bins, radial_max_eff


def auto_origin_id(
    data: "Dataset4dstem",
    *,
    ellipse_params: tuple[float, float, float] | None = None,
    num_annular_bins: int = 180,
    radial_min: float = 0.0,
    radial_max: float | None = None,
    radial_step: float = 1.0,
    two_fold_rotation_symmetry: bool = False,
    device: str = "cpu",
) -> NDArray:
    """
    Automatic diffraction center finding by minimizing the standard deviation
    along the annular direction in the polar transform.

    For each scan position, this routine:
    1) Computes a polar transform at an initial origin (image center, or
       warm-started from the previous scan position).
    2) Evaluates the sum of the standard deviation across angle (phi) over
       a mid-radius band.
    3) Performs a local search over neighboring pixel origins until the
       objective no longer improves.

    Parameters
    ----------
    data : Dataset4dstem
        A 4D-STEM dataset (or 2D diffraction pattern wrapped as 4D).
    ellipse_params : tuple or None
        Ellipse parameters (a, b, theta_deg) for distortion correction.
    num_annular_bins : int
        Number of angular bins for polar transform.
    radial_min : float
        Minimum radius in pixels.
    radial_max : float or None
        Maximum radius in pixels.
    radial_step : float
        Radial step size in pixels.
    two_fold_rotation_symmetry : bool
        If True, use only 0 to pi range for angles.
    device : str
        Torch device for computation ("cpu", "cuda", "cuda:0", etc.).

    Returns
    -------
    origin_array : np.ndarray
        Array of shape (scan_y, scan_x, 2) containing (row, col) origin
        estimates in pixels.
    """
    if len(data.array.shape) == 2:
        ny, nx = data.array.shape
        scan_y, scan_x = 1, 1
    elif len(data.array.shape) == 4:
        scan_y, scan_x, ny, nx = data.array.shape
    else:
        raise ValueError(
            f" Got array with shape {data.array.shape}."
            "To use auto_origin_id, pass a 2D or 4DSTEM dataset."
        )

    origin_array = np.zeros((scan_y, scan_x, 2), dtype=float)
    total_positions = scan_y * scan_x

    # --- Pre-compute phi-dependent quantities for the search grid ---
    # Use fewer angular bins (36) for center-finding — ring asymmetry is
    # well-captured at 10° resolution, and this is ~5x faster than using
    # the full num_annular_bins.
    search_n_phi = 36
    phi_range_val = torch.pi if two_fold_rotation_symmetry else 2.0 * torch.pi
    phi_bins = torch.linspace(
        0.0,
        float(phi_range_val),
        search_n_phi + 1,
        dtype=torch.float32,
        device=device,
    )[:-1]  # (search_n_phi,)
    if ellipse_params is None:
        cos_phi = torch.cos(phi_bins)
        sin_phi = torch.sin(phi_bins)
    else:
        if len(ellipse_params) != 3:
            raise ValueError("ellipse_params must be (a, b, theta_deg).")
        ell_a, ell_b, theta_deg = ellipse_params
        ell_theta = torch.deg2rad(torch.tensor(theta_deg, dtype=torch.float32, device=device))
        ell_alpha = phi_bins - ell_theta
        ell_scale = ell_a / ell_b
        ell_cos_t = torch.cos(ell_theta)
        ell_sin_t = torch.sin(ell_theta)
    # Normalization constants for grid_sample [-1, 1] mapping
    x_norm_scale = 2.0 / (nx - 1)
    y_norm_scale = 2.0 / (ny - 1)

    # ---- Step 1: COM of mean DP gives a robust rough center ----
    array_4d = data.array if data.array.ndim == 4 else data.array[None, None, :, :]
    mean_dp_np = array_4d.mean(axis=(0, 1)).astype(np.float32)
    total_intensity = mean_dp_np.sum()
    yy_grid, xx_grid = np.mgrid[0:ny, 0:nx]
    com_row = int(round(float((yy_grid * mean_dp_np).sum() / total_intensity)))
    com_col = int(round(float((xx_grid * mean_dp_np).sum() / total_intensity)))

    # ---- Step 2: Build a fixed polar grid that is safe for all candidates ----
    # global_margin: half-width of per-position exhaustive search window
    # safe_rmax ensures no candidate's grid extends outside the image,
    # eliminating zero-padding bias and keeping the number of radial bins
    # identical across candidates for fair comparison.
    global_margin = 20
    safe_rmax = float(
        min(
            com_row - global_margin,
            (ny - 1) - (com_row + global_margin),
            com_col - global_margin,
            (nx - 1) - (com_col + global_margin),
        )
    )
    if radial_max is not None:
        safe_rmax = min(safe_rmax, float(radial_max))
    if safe_rmax <= radial_min:
        safe_rmax = radial_min + radial_step
    radial_bins_t = torch.arange(
        radial_min,
        safe_rmax,
        radial_step,
        dtype=torch.float32,
        device=device,
    )
    if radial_bins_t.numel() == 0:
        radial_bins_t = torch.tensor([radial_min], dtype=torch.float32, device=device)
    n_r = radial_bins_t.numel()
    min_r_idx = int(np.floor(0.1 * n_r))
    max_r_idx = int(np.ceil(0.9 * n_r))

    # Build base polar grid offsets (n_phi, n_r) — origin-independent
    if ellipse_params is None:
        base_x = radial_bins_t.unsqueeze(0) * cos_phi.unsqueeze(1)
        base_y = radial_bins_t.unsqueeze(0) * sin_phi.unsqueeze(1)
    else:
        base_x = ell_scale * radial_bins_t.unsqueeze(0) * torch.cos(ell_alpha.unsqueeze(1))
        v_prime = radial_bins_t.unsqueeze(0) * torch.sin(ell_alpha.unsqueeze(1))
        base_x_rot = base_x * ell_cos_t - v_prime * ell_sin_t
        base_y = base_x * ell_sin_t + v_prime * ell_cos_t
        base_x = base_x_rot
    # Pre-normalize the base offsets
    base_x_norm = base_x * x_norm_scale  # (n_phi, n_r)
    base_y_norm = base_y * y_norm_scale

    def _build_grids(center_row: int, center_col: int, margin: int):
        """Build batch of candidate grids for a search window."""
        rows = torch.arange(
            max(0, center_row - margin),
            min(ny, center_row + margin + 1),
            dtype=torch.long,
            device=device,
        )
        cols = torch.arange(
            max(0, center_col - margin),
            min(nx, center_col + margin + 1),
            dtype=torch.long,
            device=device,
        )
        rg, cg = torch.meshgrid(rows, cols, indexing="ij")
        rf, cf = rg.reshape(-1), cg.reshape(-1)
        gx = base_x_norm.unsqueeze(0) + (cf.float() * x_norm_scale - 1.0)[:, None, None]
        gy = base_y_norm.unsqueeze(0) + (rf.float() * y_norm_scale - 1.0)[:, None, None]
        grids = torch.stack([gx, gy], dim=-1)  # (N, n_phi, n_r, 2)
        return rf, cf, grids

    def _batch_scores(dp_batch: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
        """Compute angular-std scores for all candidates on one DP."""
        n = grids.shape[0]
        polars = F.grid_sample(
            dp_batch.expand(n, -1, -1, -1),
            grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        region = polars.squeeze(1)[:, :, min_r_idx:max_r_idx]
        return region.std(dim=1).sum(dim=1)

    # ---- Step 3: Find global center from mean DP ----
    # Coarse search (step=4) over ±global_margin around COM
    coarse_step = 4
    coarse_rows = torch.arange(
        max(0, com_row - global_margin),
        min(ny, com_row + global_margin + 1),
        coarse_step,
        dtype=torch.long,
        device=device,
    )
    coarse_cols = torch.arange(
        max(0, com_col - global_margin),
        min(nx, com_col + global_margin + 1),
        coarse_step,
        dtype=torch.long,
        device=device,
    )
    crg, ccg = torch.meshgrid(coarse_rows, coarse_cols, indexing="ij")
    crf, ccf = crg.reshape(-1), ccg.reshape(-1)
    coarse_gx = base_x_norm.unsqueeze(0) + (ccf.float() * x_norm_scale - 1.0)[:, None, None]
    coarse_gy = base_y_norm.unsqueeze(0) + (crf.float() * y_norm_scale - 1.0)[:, None, None]
    coarse_grids = torch.stack([coarse_gx, coarse_gy], dim=-1)

    mean_dp_t = torch.from_numpy(mean_dp_np).to(device).unsqueeze(0).unsqueeze(0)
    coarse_scores = _batch_scores(mean_dp_t, coarse_grids)
    best_ci = coarse_scores.argmin().item()
    coarse_r = int(crf[best_ci].item())
    coarse_c = int(ccf[best_ci].item())

    # Fine search (step=1) in ±fine_margin around coarse best
    fine_margin = 6
    fine_rf, fine_cf, fine_grids = _build_grids(coarse_r, coarse_c, fine_margin)
    fine_scores = _batch_scores(mean_dp_t, fine_grids)
    best_fi = fine_scores.argmin().item()
    global_row = int(fine_rf[best_fi].item())
    global_col = int(fine_cf[best_fi].item())

    # ---- Step 4: Per-position exhaustive search in ±local_margin ----
    local_margin = 10
    local_rf, local_cf, local_grids = _build_grids(
        global_row,
        global_col,
        local_margin,
    )

    pbar = tqdm(total=total_positions, desc="Finding origin for each scan position")
    for y_pos in range(scan_y):
        row_dps = torch.from_numpy(array_4d[y_pos].astype(np.float32)).to(
            device
        )  # (scan_x, ny, nx)

        for x_pos in range(scan_x):
            dp_batch = row_dps[x_pos].unsqueeze(0).unsqueeze(0)
            scores = _batch_scores(dp_batch, local_grids)
            best_idx = scores.argmin().item()
            origin_array[y_pos, x_pos, 0] = local_rf[best_idx].item()
            origin_array[y_pos, x_pos, 1] = local_cf[best_idx].item()
            pbar.update(1)

    pbar.close()
    return origin_array


def dataset4dstem_polar_transform(
    self: "Dataset4dstem",
    origin_array: NDArray | torch.Tensor | None = None,
    ellipse_params: tuple[float, float, float] | None = None,
    num_annular_bins: int = 180,
    radial_min: float = 0.0,
    radial_max: float | None = None,
    radial_step: float = 1.0,
    two_fold_rotation_symmetry: bool = False,
    name: str | None = None,
    signal_units: str | None = None,
    scan_pos: tuple[int, int] | None = None,
    device: str = "cpu",
) -> Polar4dstem | torch.Tensor:
    if self.array.ndim != 4:
        raise ValueError(
            f"Found array with shape: {self.array.shape}. "
            "polar_transform requires a 4D-STEM dataset (ndim=4)."
        )
    scan_y, scan_x, ny, nx = self.array.shape

    # Standardize origin_array input
    if isinstance(origin_array, torch.Tensor):
        origin_array = _to_numpy(origin_array)
    origin_array = np.asarray(origin_array) if origin_array is not None else None
    if origin_array is None:
        center = np.array([(ny - 1) / 2.0, (nx - 1) / 2.0], dtype=float)
        origins = np.broadcast_to(center, (scan_y, scan_x, 2)).copy()
    elif origin_array.shape == (2,):
        origins = np.empty((scan_y, scan_x, 2), dtype=float)
        origins[...] = origin_array
    elif origin_array.shape == (scan_y, scan_x, 2):
        origins = origin_array
    else:
        raise ValueError(
            f" Got {origin_array.shape}. "
            "origin_array must have shape None, (2,) or (scan_y, scan_x, 2)."
        )

    # If scan_pos is provided, compute polar transform only for that position
    if scan_pos is not None:
        iy, ix = scan_pos
        dp = torch.from_numpy(self.array[iy, ix].astype(np.float32)).to(device)
        r0 = float(origins[iy, ix, 0])
        c0 = float(origins[iy, ix, 1])
        grid, phi_bins, radial_bins, radial_max_eff = _precompute_polar_coords(
            ny=ny,
            nx=nx,
            origin_row=r0,
            origin_col=c0,
            ellipse_params=ellipse_params,
            num_annular_bins=num_annular_bins,
            radial_min=radial_min,
            radial_max=radial_max,
            radial_step=radial_step,
            two_fold_rotation_symmetry=two_fold_rotation_symmetry,
            device=device,
        )
        dp_batch = dp.unsqueeze(0).unsqueeze(0)  # (1, 1, ny, nx)
        polar2d = F.grid_sample(
            dp_batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return polar2d.squeeze(0).squeeze(0)  # (n_phi, n_r)

    # Use the global minimum safe radius across all origins so every scan
    # position maps to the same-size polar grid (required for a uniform 4D output)
    if radial_max is None:
        r_row_pos = origins[:, :, 0]
        r_row_neg = (ny - 1) - origins[:, :, 0]
        r_col_pos = origins[:, :, 1]
        r_col_neg = (nx - 1) - origins[:, :, 1]
        radial_max_eff_array = np.minimum.reduce([r_row_pos, r_row_neg, r_col_pos, r_col_neg])
        radial_max = float(max(radial_max_eff_array.min(), radial_min + radial_step))

    # Compute grid for first position to get output shape
    grid, phi_bins, radial_bins, radial_max_eff = _precompute_polar_coords(
        ny=ny,
        nx=nx,
        origin_row=float(origins[0, 0, 0]),
        origin_col=float(origins[0, 0, 1]),
        ellipse_params=ellipse_params,
        num_annular_bins=num_annular_bins,
        radial_min=radial_min,
        radial_max=radial_max,
        radial_step=radial_step,
        two_fold_rotation_symmetry=two_fold_rotation_symmetry,
        device=device,
    )
    n_phi = phi_bins.numel()
    n_r = radial_bins.numel()
    out = np.empty((scan_y, scan_x, n_phi, n_r), dtype=np.float32)
    for iy in range(scan_y):
        for ix in range(scan_x):
            dp = torch.from_numpy(self.array[iy, ix].astype(np.float32)).to(device)
            r0 = float(origins[iy, ix, 0])
            c0 = float(origins[iy, ix, 1])
            grid, _, _, _ = _precompute_polar_coords(
                ny=ny,
                nx=nx,
                origin_row=r0,
                origin_col=c0,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                device=device,
            )
            dp_batch = dp.unsqueeze(0).unsqueeze(0)
            polar2d = F.grid_sample(
                dp_batch,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            out[iy, ix] = _to_numpy(polar2d.squeeze(0).squeeze(0))

    # Express polar axes in physical units matching the input dataset's calibration
    phi_range = np.pi if two_fold_rotation_symmetry else 2.0 * np.pi
    phi_step_deg = (phi_range / float(n_phi)) * (180.0 / np.pi)
    sampling = np.zeros(4, dtype=float)
    origin = np.zeros(4, dtype=float)
    sampling[0:2] = np.asarray(self.sampling)[0:2]
    sampling[2] = phi_step_deg
    sampling[3] = float(np.asarray(self.sampling)[-1]) * radial_step
    origin[0:2] = np.asarray(self.origin)[0:2]
    origin[2] = 0.0
    origin[3] = radial_min * float(np.asarray(self.sampling)[-1])
    units = [
        self.units[0],
        self.units[1],
        "deg",
        self.units[-1],
    ]
    metadata = dict(self.metadata)
    metadata.update(
        {
            "polar_radial_min": float(radial_min),
            "polar_radial_max": float(radial_max_eff),
            "polar_radial_step": float(radial_step),
            "polar_num_annular_bins": int(n_phi),
            "polar_two_fold_rotation_symmetry": bool(two_fold_rotation_symmetry),
            "polar_ellipse_params": tuple(ellipse_params) if ellipse_params is not None else None,
        }
    )
    return Polar4dstem(
        array=out,
        name=name if name is not None else f"{self.name}_polar",
        origin=origin,
        sampling=sampling,
        units=units,
        signal_units=signal_units if signal_units is not None else self.signal_units,
        metadata=metadata,
        origin_array=origins,
        _token=Polar4dstem._token,
    )
