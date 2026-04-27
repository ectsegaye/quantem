"""Tests for elliptical distortion fitting from amorphous ring patterns."""

import numpy as np
import pytest
import torch

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffraction.ellipticity import (
    _amorphous_ring_model,
    fit_elliptical_distortion,
)
from quantem.diffraction.polar_transform import polar_transform


def _make_synthetic_dataset(
    n_row: int = 256,
    n_col: int = 256,
    center: tuple[float, float] = (128.0, 128.0),
    a: float = 80.0,
    b: float = 60.0,
    theta_deg: float = 30.0,
    I0: float = 1000.0,
    I1: float = 50.0,
    sigma0: float = 20.0,
    sigma1: float = 10.0,
    sigma2: float = 14.0,
    c_bkgd: float = 2.0,
    noise: bool = True,
) -> Dataset4dstem:
    """Generate a synthetic 4D-STEM dataset (1x1 scan) with known elliptical distortion."""
    row_offsets = torch.arange(n_row, dtype=torch.float32) - center[0]
    col_offsets = torch.arange(n_col, dtype=torch.float32) - center[1]
    offset_row, offset_col = torch.meshgrid(row_offsets, col_offsets, indexing="ij")

    theta_rad = np.radians(theta_deg)
    params = torch.tensor(
        [I0, I1, sigma0, sigma1, sigma2, c_bkgd, 0.0, 0.0, a, b, theta_rad],
        dtype=torch.float32,
    )
    dp = _amorphous_ring_model(params, offset_row.reshape(-1), offset_col.reshape(-1))
    dp = dp.reshape(n_row, n_col).numpy()
    if noise:
        dp = np.maximum(dp, 0.0)
        rng = np.random.default_rng(42)
        dp = rng.poisson(dp).astype(np.float32)

    # Wrap as a 1x1 scan 4D-STEM dataset
    arr_4d = dp[np.newaxis, np.newaxis, :, :]
    return Dataset4dstem.from_array(arr_4d, name="synthetic")


class TestAmorphousRingModel:
    """Tests for the _amorphous_ring_model function."""

    def test_circular_ring_peak_at_radius(self):
        """Model should peak at the ring radius for a circular ring."""
        R = 50.0
        params = torch.tensor(
            [0.0, 100.0, 10.0, 5.0, 5.0, 0.0, 0.0, 0.0, R, R, 0.0],
            dtype=torch.float32,
        )
        # Sample along the column axis at various radii
        offset_col = torch.linspace(0, 80, 200)
        offset_row = torch.zeros_like(offset_col)
        vals = _amorphous_ring_model(params, offset_row, offset_col)
        peak_r = float(offset_col[vals.argmax()])
        assert abs(peak_r - R) < 1.0, f"Peak at {peak_r}, expected ~{R}"

    def test_background_far_from_ring(self):
        """Far from the ring and center, model should return ~c_bkgd."""
        R = 50.0
        c_bkgd = 5.0
        params = torch.tensor(
            [100.0, 100.0, 5.0, 3.0, 3.0, c_bkgd, 0.0, 0.0, R, R, 0.0],
            dtype=torch.float32,
        )
        # Very far from center and ring
        offset_row = torch.tensor([200.0])
        offset_col = torch.tensor([200.0])
        val = float(_amorphous_ring_model(params, offset_row, offset_col))
        assert abs(val - c_bkgd) < 1.0, f"Expected ~{c_bkgd}, got {val}"


class TestFitEllipticalDistortion:
    """Tests for fit_elliptical_distortion."""

    def test_recover_known_ellipse(self):
        """Fit should recover known ellipse params from a synthetic pattern."""
        a_true, b_true, theta_true = 80.0, 60.0, 30.0
        center_true = (128.0, 128.0)
        ds = _make_synthetic_dataset(a=a_true, b=b_true, theta_deg=theta_true, center=center_true)
        # fit_radii brackets the ring (mean_r~70) well outside the central beam
        result = fit_elliptical_distortion(
            ds,
            center=center_true,
            fit_radii=(45.0, 100.0),
        )
        a_fit, b_fit, theta_fit = result["ellipse_params"]
        # Check semiaxes within 10% relative error
        assert abs(a_fit - a_true) / a_true < 0.10, f"a: got {a_fit:.2f}, expected {a_true}"
        assert abs(b_fit - b_true) / b_true < 0.10, f"b: got {b_fit:.2f}, expected {b_true}"
        # Angle within 5 degrees
        angle_err = abs(theta_fit - theta_true)
        angle_err = min(angle_err, 180.0 - angle_err)  # handle wrapping
        assert angle_err < 5.0, f"theta: got {theta_fit:.1f}, expected {theta_true}"

    def test_recover_circular(self):
        """For a circular pattern (a==b), fitted a and b should be similar."""
        R = 70.0
        ds = _make_synthetic_dataset(a=R, b=R, theta_deg=0.0, noise=False)
        result = fit_elliptical_distortion(
            ds,
            center=(128.0, 128.0),
            fit_radii=(45.0, 100.0),
        )
        a_fit, b_fit, _ = result["ellipse_params"]
        ratio = a_fit / b_fit
        assert 0.95 < ratio < 1.05, f"Expected a/b ~ 1 for circular, got {ratio:.3f}"

    def test_refined_center(self):
        """Fit should refine center when it's slightly off."""
        center_true = (130.0, 126.0)
        ds = _make_synthetic_dataset(center=center_true, noise=False)
        # Give a slightly wrong center
        result = fit_elliptical_distortion(
            ds,
            center=(128.0, 128.0),
            fit_radii=(45.0, 100.0),
        )
        row_fit, col_fit = result["center"]
        assert abs(row_fit - center_true[0]) < 3.0, (
            f"row: got {row_fit:.1f}, expected {center_true[0]}"
        )
        assert abs(col_fit - center_true[1]) < 3.0, (
            f"col: got {col_fit:.1f}, expected {center_true[1]}"
        )

    def test_mask_excludes_pixels(self):
        """Fit should still work when a mask excludes some pixels."""
        ds = _make_synthetic_dataset(noise=False)
        mask = np.zeros((256, 256), dtype=bool)
        mask[100:120, 100:120] = True  # block a small region
        result = fit_elliptical_distortion(
            ds,
            center=(128.0, 128.0),
            fit_radii=(45.0, 100.0),
            mask=mask,
        )
        assert "ellipse_params" in result
        a, b, theta = result["ellipse_params"]
        assert a > 0 and b > 0

    def test_empty_annulus_raises(self):
        """Should raise if no pixels fall in the annulus."""
        ds = _make_synthetic_dataset()
        with pytest.raises(ValueError, match="No pixels"):
            fit_elliptical_distortion(
                ds,
                center=(128.0, 128.0),
                fit_radii=(200.0, 210.0),  # outside image
            )


def test_ellipse_correction_pipeline():
    """fit_elliptical_distortion -> polar_transform: corrected ring is uniform across phi."""
    ds = _make_synthetic_dataset(a=80.0, b=60.0, theta_deg=30.0, noise=False)
    fit = fit_elliptical_distortion(ds, center=(128.0, 128.0), fit_radii=(45.0, 100.0))
    polar = polar_transform(ds, ellipse_params=fit["ellipse_params"])
    # In the corrected polar the ring sits at a constant radius, so intensity
    # at the peak radius should be ~uniform across phi.
    arr = polar.array[0, 0]  # (n_phi, n_r), single 1x1 scan position
    peak_r = int(arr.mean(axis=0).argmax())
    ring_at_phi = arr[:, peak_r]
    cv = float(ring_at_phi.std() / ring_at_phi.mean())
    assert cv < 0.1, f"Ring varies across phi (cv={cv:.3f}); correction not applied."
