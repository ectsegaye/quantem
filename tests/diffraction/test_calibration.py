import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffraction.calibration import (
    AU_LATTICE_PARAM,
    ReciprocalCalibration,
    _au_q_values,
)

# ============================================================================
# Fixtures
# ============================================================================

KNOWN_PIXEL_SIZE = 0.015  # 1/Å per pixel — ground truth for synthetic data


def _make_polycrystalline_pattern(
    shape: tuple[int, int] = (256, 256),
    pixel_size: float = KNOWN_PIXEL_SIZE,
    n_rings: int = 5,
    ring_width: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Synthetic polycrystalline pattern with rings at the first *n_rings* Au q-values."""
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    q_ref = _au_q_values()
    q_vals = sorted(q_ref.values())[:n_rings]

    pattern = np.zeros(shape, dtype=np.float32)
    for q in q_vals:
        r_ring = q / pixel_size  # convert q to pixel radius
        pattern += 100 * np.exp(-((r - r_ring) ** 2) / (2 * ring_width**2))

    # Central beam
    pattern += 500 * np.exp(-(r**2) / (2 * 5**2))

    # Poisson-like noise
    rng = np.random.default_rng(seed)
    pattern += rng.poisson(2, size=shape).astype(np.float32)

    return pattern


@pytest.fixture
def polycrystalline_pattern():
    return _make_polycrystalline_pattern()


@pytest.fixture
def polycrystalline_dataset2d(polycrystalline_pattern):
    return Dataset2d.from_array(
        array=polycrystalline_pattern,
        name="poly_au_2d",
        origin=(0, 0),
        sampling=(1.0, 1.0),
        units=["pixels", "pixels"],
        signal_units="counts",
    )


@pytest.fixture
def polycrystalline_4dstem(polycrystalline_pattern):
    """4D-STEM dataset where the top-left 2×2 scan positions contain Au rings."""
    scan_y, scan_x = 4, 4
    ny, nx = polycrystalline_pattern.shape
    rng = np.random.default_rng(99)

    arr4d = np.zeros((scan_y, scan_x, ny, nx), dtype=np.float32)
    for iy in range(scan_y):
        for ix in range(scan_x):
            if iy < 2 and ix < 2:
                # Au region — polycrystalline rings
                variation = 1.0 + 0.05 * rng.standard_normal()
                arr4d[iy, ix] = polycrystalline_pattern * variation
            else:
                # Non-Au region — just noise + central beam
                y, x = np.ogrid[:ny, :nx]
                cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
                r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                arr4d[iy, ix] = 500 * np.exp(-(r**2) / (2 * 5**2))
                arr4d[iy, ix] += rng.poisson(2, size=(ny, nx)).astype(np.float32)

    return Dataset4dstem.from_array(
        array=arr4d,
        name="test_4dstem_poly",
        origin=(0, 0, 0, 0),
        sampling=(1.0, 1.0, 1.0, 1.0),
        units=["nm", "nm", "pixels", "pixels"],
        signal_units="counts",
    )


@pytest.fixture
def au_mask():
    """Boolean mask selecting the top-left 2×2 quadrant."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    return mask


# ============================================================================
# Reference data tests
# ============================================================================


class TestReferenceData:
    def test_au_lattice_param(self):
        assert AU_LATTICE_PARAM == pytest.approx(4.0782, abs=0.001)

    def test_au_q_values_ordering(self):
        q = _au_q_values()
        vals = list(q.values())
        assert vals == sorted(vals), "Au reflections should be ordered by increasing q"

    def test_au_111_q_value(self):
        q = _au_q_values()
        expected = np.sqrt(3) / AU_LATTICE_PARAM
        assert q["111"] == pytest.approx(expected, rel=1e-6)


# ============================================================================
# Construction tests
# ============================================================================


class TestReciprocalCalibrationConstruction:
    def test_direct_init_raises(self):
        with pytest.raises(RuntimeError, match="Direct instantiation"):
            ReciprocalCalibration(
                radial_profile=np.zeros(10),
                radial_pixel_positions=np.arange(10),
                mean_dp=np.zeros((10, 10)),
            )

    def test_from_data_dataset2d(self, polycrystalline_dataset2d):
        cal = ReciprocalCalibration.from_data(polycrystalline_dataset2d, find_origin=False)
        assert cal.radial_profile.ndim == 1
        assert len(cal.radial_profile) == len(cal.radial_pixel_positions)
        assert cal.mean_dp.ndim == 2

    def test_from_data_dataset4dstem_with_mask(self, polycrystalline_4dstem, au_mask):
        cal = ReciprocalCalibration.from_data(
            polycrystalline_4dstem, mask_realspace=au_mask, find_origin=False
        )
        assert cal.radial_profile.ndim == 1
        assert cal.mean_dp.ndim == 2
        assert cal.mean_dp.shape == polycrystalline_4dstem.array.shape[2:]

    def test_from_data_dataset4dstem_no_mask(self, polycrystalline_4dstem):
        cal = ReciprocalCalibration.from_data(polycrystalline_4dstem, find_origin=False)
        assert cal.radial_profile.ndim == 1

    def test_from_data_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported data type"):
            ReciprocalCalibration.from_data("not a dataset")


# ============================================================================
# Peak finding tests
# ============================================================================


class TestFindPeaks:
    def test_find_peaks_returns_indices(self, polycrystalline_dataset2d):
        cal = ReciprocalCalibration.from_data(polycrystalline_dataset2d, find_origin=False)
        indices = cal.find_peaks(distance=5, prominence=1.0)
        assert len(indices) > 0
        assert cal.peak_indices is not None
        assert cal.peak_pixel_positions is not None
        assert len(cal.peak_indices) == len(cal.peak_pixel_positions)

    def test_find_peaks_detects_rings(self, polycrystalline_dataset2d):
        """Peaks should be found near the expected ring radii."""
        cal = ReciprocalCalibration.from_data(polycrystalline_dataset2d, find_origin=False)
        cal.find_peaks(distance=5, prominence=1.0)

        q_ref = _au_q_values()
        expected_radii = sorted(q / KNOWN_PIXEL_SIZE for q in list(q_ref.values())[:5])

        # Each expected ring should have a detected peak within ±5 pixels
        for r_exp in expected_radii:
            dists = np.abs(cal.peak_pixel_positions - r_exp)
            assert dists.min() < 5, f"No peak found near expected radius {r_exp:.1f} px"


# ============================================================================
# Calibration tests
# ============================================================================


class TestCalibrate:
    def test_calibrate_recovers_pixel_size(self, polycrystalline_dataset2d):
        cal = ReciprocalCalibration.from_data(polycrystalline_dataset2d, find_origin=False)
        cal.find_peaks(distance=5, prominence=1.0)

        # Select the 5 peaks closest to the expected ring positions
        q_ref = _au_q_values()
        expected_radii = sorted(q / KNOWN_PIXEL_SIZE for q in list(q_ref.values())[:5])

        best_indices = []
        for r_exp in expected_radii:
            dists = np.abs(cal.peak_pixel_positions - r_exp)
            best_indices.append(cal.peak_indices[np.argmin(dists)])

        pixel_size = cal.calibrate(material="Au", peak_indices=np.array(best_indices))
        assert pixel_size == pytest.approx(KNOWN_PIXEL_SIZE, rel=0.05)

    def test_calibrate_stores_results(self, polycrystalline_dataset2d):
        cal = ReciprocalCalibration.from_data(polycrystalline_dataset2d, find_origin=False)
        cal.find_peaks(distance=5, prominence=1.0)

        q_ref = _au_q_values()
        expected_radii = sorted(q / KNOWN_PIXEL_SIZE for q in list(q_ref.values())[:5])
        best_indices = []
        for r_exp in expected_radii:
            dists = np.abs(cal.peak_pixel_positions - r_exp)
            best_indices.append(cal.peak_indices[np.argmin(dists)])

        cal.calibrate(material="Au", peak_indices=np.array(best_indices))
        assert cal.pixel_size is not None
        assert cal.matched_hkl is not None
        assert cal.matched_q is not None
        assert len(cal.matched_hkl) == len(best_indices)

    def test_calibrate_no_peaks_raises(self, polycrystalline_dataset2d):
        cal = ReciprocalCalibration.from_data(polycrystalline_dataset2d, find_origin=False)
        with pytest.raises(RuntimeError, match="No peaks available"):
            cal.calibrate()

    def test_calibrate_unsupported_material_raises(self, polycrystalline_dataset2d):
        cal = ReciprocalCalibration.from_data(polycrystalline_dataset2d, find_origin=False)
        cal.find_peaks(distance=5, prominence=1.0)
        with pytest.raises(NotImplementedError, match="not yet supported"):
            cal.calibrate(material="Si")


# ============================================================================
# Apply tests
# ============================================================================


class TestApply:
    def test_apply_updates_sampling_and_units(self, polycrystalline_4dstem, au_mask):
        cal = ReciprocalCalibration.from_data(
            polycrystalline_4dstem, mask_realspace=au_mask, find_origin=False
        )
        cal.find_peaks(distance=5, prominence=1.0)

        q_ref = _au_q_values()
        expected_radii = sorted(q / KNOWN_PIXEL_SIZE for q in list(q_ref.values())[:5])
        best_indices = []
        for r_exp in expected_radii:
            dists = np.abs(cal.peak_pixel_positions - r_exp)
            best_indices.append(cal.peak_indices[np.argmin(dists)])

        cal.calibrate(material="Au", peak_indices=np.array(best_indices))
        cal.apply(polycrystalline_4dstem)

        assert polycrystalline_4dstem.sampling[2] == pytest.approx(cal.pixel_size, rel=1e-6)
        assert polycrystalline_4dstem.sampling[3] == pytest.approx(cal.pixel_size, rel=1e-6)
        assert polycrystalline_4dstem.units[2] == "1/Angstrom"
        assert polycrystalline_4dstem.units[3] == "1/Angstrom"
        # Real-space axes unchanged
        assert polycrystalline_4dstem.sampling[0] == 1.0
        assert polycrystalline_4dstem.sampling[1] == 1.0
        assert polycrystalline_4dstem.units[0] == "nm"
        assert polycrystalline_4dstem.units[1] == "nm"

    def test_apply_without_calibration_raises(self, polycrystalline_4dstem):
        cal = ReciprocalCalibration.from_data(polycrystalline_4dstem, find_origin=False)
        with pytest.raises(RuntimeError, match="No calibration available"):
            cal.apply(polycrystalline_4dstem)


# ============================================================================
# Plotting tests (smoke tests — just check they don't error)
# ============================================================================


class TestPlotting:
    def test_plot_radial_profile(self, polycrystalline_dataset2d):
        cal = ReciprocalCalibration.from_data(polycrystalline_dataset2d, find_origin=False)
        ax = cal.plot_radial_profile()
        assert ax is not None
        plt.close("all")

    def test_plot_radial_profile_with_peaks(self, polycrystalline_dataset2d):
        cal = ReciprocalCalibration.from_data(polycrystalline_dataset2d, find_origin=False)
        cal.find_peaks(distance=5, prominence=1.0)
        ax = cal.plot_radial_profile()
        assert ax is not None
        plt.close("all")

    def test_plot_calibration_fit(self, polycrystalline_dataset2d):
        cal = ReciprocalCalibration.from_data(polycrystalline_dataset2d, find_origin=False)
        cal.find_peaks(distance=5, prominence=1.0)

        q_ref = _au_q_values()
        expected_radii = sorted(q / KNOWN_PIXEL_SIZE for q in list(q_ref.values())[:5])
        best_indices = []
        for r_exp in expected_radii:
            dists = np.abs(cal.peak_pixel_positions - r_exp)
            best_indices.append(cal.peak_indices[np.argmin(dists)])

        cal.calibrate(material="Au", peak_indices=np.array(best_indices))
        ax = cal.plot_calibration_fit()
        assert ax is not None
        plt.close("all")
