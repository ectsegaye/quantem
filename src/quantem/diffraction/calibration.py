from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.diffraction.polar import PairDistributionFunction
from quantem.diffraction.polar_transform import auto_origin_id, polar_transform

# ---------------------------------------------------------------------------
# Reference material data
# {material: (lattice_param_Å, {hkl_label: h²+k²+l²})}
#   Au -- FCC, allowed when (h,k,l) all same parity.
#   Si -- diamond cubic, allowed when (h,k,l) all odd, OR all even with
#         h+k+l = 4n. The 200, 222, 420 FCC reflections are forbidden.
# ---------------------------------------------------------------------------
MATERIALS: dict[str, tuple[float, dict[str, int]]] = {
    "Au": (
        4.0782,
        {
            "111": 3, "200": 4, "220": 8, "311": 11,
            "222": 12, "400": 16, "331": 19, "420": 20,
        },
    ),
    "Si": (
        5.4307,
        {
            "111": 3, "220": 8, "311": 11, "400": 16,
            "331": 19, "422": 24, "511": 27, "440": 32,
            "531": 35, "620": 40,
        },
    ),
}


def _q_values_for(material: str) -> dict[str, float]:
    """Return q = sqrt(h²+k²+l²) / a for each allowed reflection of `material`."""
    if material not in MATERIALS:
        raise ValueError(
            f"Unknown material '{material}'. Available: {sorted(MATERIALS)}"
        )
    a, reflections = MATERIALS[material]
    return {hkl: np.sqrt(n) / a for hkl, n in reflections.items()}


class ReciprocalCalibration:
    """Calibrate reciprocal-space pixel size from polycrystalline diffraction rings.

    This class determines the pixel-to-inverse-Angstrom conversion factor by
    matching detected ring positions in a radial profile to known d-spacings
    of a reference material (currently Au FCC).

    Construction is enforced through :meth:`from_data`.
    """

    _token = object()

    def __init__(
        self,
        pdf: PairDistributionFunction,
        radial_profile: NDArray,
        radial_pixel_positions: NDArray,
        mean_dp: NDArray,
        input_data: Any | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Direct instantiation of ReciprocalCalibration is not allowed. "
                "Use ReciprocalCalibration.from_data() to instantiate this class."
            )

        self._pdf = pdf  # used for fit_bg
        self.radial_profile = radial_profile  # 1-D, length = n_radial_bins
        self.radial_pixel_positions = radial_pixel_positions  # 1-D pixel indices
        self.mean_dp = mean_dp  # 2-D mean diffraction pattern
        self.input_data = input_data

        # Set after fit_bg() / find_peaks()
        self.background: NDArray | None = None
        self.peak_indices: NDArray | None = None
        self.peak_pixel_positions: NDArray | None = None

        # Set after calibrate()
        self.pixel_size: float | None = None  # 1/Å per pixel
        self.material: str | None = None
        self.matched_hkl: list[str] | None = None
        self.matched_q: NDArray | None = None
        self.matched_pixel_positions: NDArray | None = None

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_data(
        cls,
        data: Dataset4dstem | Dataset2d | Polar4dstem,
        *,
        mask_realspace: NDArray | None = None,
        voltage_kV: float = 200,
        find_origin: bool = True,
        origin_row: float | None = None,
        origin_col: float | None = None,
        radial_min: float = 0.0,
        radial_max: float | None = None,
        radial_step: float = 1.0,
        num_annular_bins: int = 180,
        ellipse_params: tuple[float, float, float] | None = None,
    ) -> "ReciprocalCalibration":
        """Create a ReciprocalCalibration from diffraction data.

        Parameters
        ----------
        data
            A Dataset4dstem, Dataset2d, or Polar4dstem.
        mask_realspace
            Boolean array (scan_y, scan_x) selecting which probe positions
            to include when averaging diffraction patterns.  Only used with
            Dataset4dstem input.
        voltage_kV
            Accelerating voltage in kV (stored for metadata / future use).
        find_origin, origin_row, origin_col
            Origin-finding parameters forwarded to the polar transform.
        radial_min, radial_max, radial_step, num_annular_bins
            Polar-transform parameters.
        ellipse_params
            Elliptical-distortion correction ``(a, b, theta_deg)`` applied
            during origin finding and the polar transform. Must match the
            value passed to the PDF's polar transform so the calibration is
            measured on the same (ellipse-corrected) geometry.
        """
        # --- Polar4dstem input: use directly --------------------------------
        if isinstance(data, Polar4dstem):
            polar = data
            mean_dp = None
        else:
            # --- Dataset2d: wrap as 1×1 4D-STEM ----------------------------
            if isinstance(data, Dataset2d):
                arr2d = data.array
                arr4 = arr2d[None, None, ...]
                data = Dataset4dstem.from_array(
                    array=arr4,
                    name=f"{data.name}_as4dstem"
                    if getattr(data, "name", None)
                    else "cal_4dstem_from_2d",
                    origin=np.concatenate(
                        [np.zeros(2, dtype=float), np.asarray(data.origin, dtype=float)]
                    ),
                    sampling=np.concatenate(
                        [np.ones(2, dtype=float), np.asarray(data.sampling, dtype=float)]
                    ),
                    units=["pixels", "pixels"] + list(data.units),
                    signal_units=data.signal_units,
                )
                mask_realspace = None  # irrelevant for 1×1 scan

            # --- Dataset4dstem: compute mean DP then polar transform --------
            if not isinstance(data, Dataset4dstem):
                raise TypeError(
                    f"Unsupported data type {type(data).__name__}. "
                    "Expected Dataset4dstem, Dataset2d, or Polar4dstem."
                )

            arr4d = data.array  # (scan_y, scan_x, dp_y, dp_x)
            if mask_realspace is not None:
                n_pos = int(mask_realspace.sum())
                print(f"Averaging {n_pos} masked diffraction patterns ...")
                mean_dp = arr4d[mask_realspace].mean(axis=0)
            else:
                n_pos = arr4d.shape[0] * arr4d.shape[1]
                print(f"Averaging all {n_pos} diffraction patterns ...")
                mean_dp = arr4d.mean(axis=(0, 1))

            # Wrap the mean DP as a 1×1 Dataset4dstem for polar transform
            mean_ds = Dataset4dstem.from_array(
                array=mean_dp[None, None, ...],
                name="calibration_mean_dp",
                origin=np.array([0.0, 0.0, data.origin[2], data.origin[3]]),
                sampling=np.array([1.0, 1.0, data.sampling[2], data.sampling[3]]),
                units=["pixels", "pixels"] + list(data.units[2:]),
                signal_units=data.signal_units,
            )

            if find_origin:
                origin_array = auto_origin_id(
                    mean_ds,
                    ellipse_params=ellipse_params,
                    num_annular_bins=num_annular_bins,
                    radial_min=radial_min,
                    radial_max=radial_max,
                    radial_step=radial_step,
                )
            else:
                ny, nx = mean_dp.shape
                if origin_row is None:
                    origin_row = (ny - 1) / 2.0
                if origin_col is None:
                    origin_col = (nx - 1) / 2.0
                origin_array = np.zeros((1, 1, 2), dtype=float)
                origin_array[0, 0, 0] = origin_row
                origin_array[0, 0, 1] = origin_col

            polar = polar_transform(
                mean_ds,
                origin_array=origin_array,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
            )

        # --- Build a PairDistributionFunction for background fitting --------
        # PairDistributionFunction.from_data does not accept a Polar4dstem
        # directly, so construct via the class-level token.
        pdf_input = data if isinstance(data, Dataset4dstem) else None
        pdf = PairDistributionFunction(
            polar=polar,
            input_data=pdf_input,
            _token=PairDistributionFunction._token,
        )
        pdf.calculate_radial_mean()

        # --- Compute radial profile (azimuthal average) ---------------------
        radial_profile = pdf.Ik.detach().cpu().numpy()
        n_r = radial_profile.shape[0]
        radial_pixel_positions = np.arange(n_r, dtype=float)

        if mean_dp is None:
            mean_dp = np.zeros((1, 1), dtype=float)

        return cls(
            pdf=pdf,
            radial_profile=radial_profile,
            radial_pixel_positions=radial_pixel_positions,
            mean_dp=mean_dp,
            input_data=data,
            _token=cls._token,
        )

    # ------------------------------------------------------------------
    # Background estimation (SNIP algorithm)
    # ------------------------------------------------------------------
    def fit_bg(self, m: int | None = None) -> NDArray:
        """Estimate the background using the SNIP algorithm.

        Uses the Statistics-sensitive Non-linear Iterative Peak-clipping
        (SNIP) algorithm following Morhac et al. (1997) as implemented in
        :meth:`PairDistributionFunction.compute_bg_snip`.  This non-parametric
        approach handles the steep central-beam falloff and arbitrary
        background shapes without requiring a parametric model.

        Parameters
        ----------
        m
            Number of SNIP clipping iterations.  Should be roughly half the
            FWHM (in pixels) of the widest peak.  Defaults to 8.

        Returns
        -------
        bg : NDArray
            Estimated background curve, same length as :attr:`radial_profile`.
        """
        if m is None:
            m = 8

        Ik = self.radial_profile.copy()
        m = max(1, min(m, len(Ik) // 2 - 1))

        # LLS (log-log-sqrt) transform
        v = np.log(np.log(np.sqrt(Ik + 1.0) + 1.0) + 1.0)

        # SNIP iterations
        for p in range(1, m + 1):
            left = np.empty_like(v)
            left[p:] = v[:-p]
            left[:p] = v[:p]
            right = np.empty_like(v)
            right[:-p] = v[p:]
            right[-p:] = v[-p:]

            vp = (left + right) / 2.0
            v = np.minimum(v, vp)

        # Inverse LLS transform
        t = np.exp(v)
        bg = (np.exp(t - 1.0) - 1.0) ** 2 - 1.0
        bg = np.clip(bg, 0.0, None)

        self.background = bg
        return bg

    # ------------------------------------------------------------------
    # Peak finding
    # ------------------------------------------------------------------
    def find_peaks(
        self,
        radial_min: int = 0,
        n_peaks: int | None = None,
        distance: int | None = None,
        prominence: float | None = None,
        height: float | None = None,
        m: int | None = None,
    ) -> NDArray:
        """Detect peaks in the background-subtracted radial profile.

        First fits and subtracts the smooth background using :meth:`fit_bg`,
        then runs :func:`scipy.signal.find_peaks` on the residual.

        Parameters
        ----------
        radial_min
            Ignore the profile below this pixel index (skip the central beam).
        n_peaks
            If provided, keep only the first ``n_peaks`` detected peaks
            (the innermost rings). Useful when the detector finds more
            candidates than you want to use for the calibration fit.
        distance, prominence, height
            Passed through to ``scipy.signal.find_peaks``.
        m
            SNIP background clipping iterations, forwarded to :meth:`fit_bg`.
            ``None`` uses the ``fit_bg`` default (8).

        Returns
        -------
        peak_indices : NDArray
            Indices into :attr:`radial_profile` where peaks were found.
        """
        from scipy.signal import find_peaks as _find_peaks

        bg = self.fit_bg(m=m)
        residual = self.radial_profile - bg

        # zero out central beam region
        residual[:radial_min] = 0.0
        self._residual = residual

        kwargs: dict[str, Any] = {}
        if distance is not None:
            kwargs["distance"] = distance
        if prominence is not None:
            kwargs["prominence"] = prominence
        if height is not None:
            kwargs["height"] = height

        indices, _ = _find_peaks(residual, **kwargs)
        indices = indices[indices >= radial_min]
        if n_peaks is not None:
            indices = indices[:n_peaks]

        self.peak_indices = indices
        self.peak_pixel_positions = self.radial_pixel_positions[indices]
        return indices

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(
        self,
        material: str = "Au",
        peak_indices: NDArray | list[int] | None = None,
    ) -> float:
        """Match detected peaks to known ring positions and fit pixel size.

        Performs a zero-intercept least-squares fit: ``q = pixel_size * r_pixels``
        to determine the reciprocal-space pixel size in 1/Å per pixel.

        Parameters
        ----------
        material
            Reference material; must be a key in :data:`MATERIALS`
            (currently ``'Au'`` or ``'Si'``).
        peak_indices
            Indices of peaks (into :attr:`radial_profile`) to use for the
            fit.  If ``None``, uses all peaks found by :meth:`find_peaks`.

        Returns
        -------
        pixel_size : float
            Calibrated pixel size in 1/Å per pixel.
        """
        if peak_indices is not None:
            peak_pix = self.radial_pixel_positions[np.asarray(peak_indices)]
        elif self.peak_pixel_positions is not None:
            peak_pix = self.peak_pixel_positions
        else:
            raise RuntimeError(
                "No peaks available. Call find_peaks() first or provide peak_indices."
            )

        q_ref = _q_values_for(material)
        self.material = material
        hkl_labels = list(q_ref.keys())
        q_vals = np.array(list(q_ref.values()))
        sorted_order = np.argsort(q_vals)
        q_sorted = q_vals[sorted_order]
        hkl_sorted = [hkl_labels[i] for i in sorted_order]

        peak_pix = np.sort(peak_pix.astype(float))
        n_peaks = len(peak_pix)
        n_ref = len(q_sorted)

        # Try all ordered subsets of n_peaks reflections from the n_ref
        # available and find the assignment with the lowest relative fit
        # residual.  Since both peaks and q-values are sorted, we only
        # need to consider combinations (order-preserving).
        from itertools import combinations

        best_rss = np.inf
        best_combo = None

        for combo in combinations(range(n_ref), n_peaks):
            q_sel = q_sorted[list(combo)]
            # Zero-intercept LS: pixel_size = (r · q) / (r · r)
            ps = float(np.dot(peak_pix, q_sel) / np.dot(peak_pix, peak_pix))
            if ps <= 0:
                continue
            # Relative residual sum of squares
            predicted_q = ps * peak_pix
            rss = float(np.sum(((predicted_q - q_sel) / q_sel) ** 2))
            if rss < best_rss:
                best_rss = rss
                best_combo = combo

        if best_combo is None:
            raise RuntimeError("Could not find a valid peak-to-reflection matching.")

        matched_q = q_sorted[list(best_combo)]
        matched_hkl = [hkl_sorted[i] for i in best_combo]
        pixel_size = float(np.dot(peak_pix, matched_q) / np.dot(peak_pix, peak_pix))

        self.pixel_size = pixel_size
        self.matched_hkl = matched_hkl
        self.matched_q = matched_q
        self.matched_pixel_positions = peak_pix

        # Print matching for user verification
        print(f"Calibrated pixel size: {pixel_size:.6f} 1/Å/px")
        print("Peak matching:")
        for r_px, hkl, q in zip(peak_pix, matched_hkl, matched_q):
            q_pred = pixel_size * r_px
            print(f"  r={r_px:.1f} px → {hkl} (q={q:.4f}, predicted={q_pred:.4f})")

        return pixel_size

    # ------------------------------------------------------------------
    # Apply calibration
    # ------------------------------------------------------------------
    def apply(self, dataset: Dataset4dstem) -> None:
        """Update a Dataset4dstem's diffraction-axis sampling and units.

        Parameters
        ----------
        dataset
            The dataset whose last two axes will be updated with the
            calibrated pixel size and ``'1/Angstrom'`` units.
        """
        if self.pixel_size is None:
            raise RuntimeError("No calibration available. Call calibrate() first.")

        new_sampling = list(dataset.sampling)
        new_sampling[2] = self.pixel_size
        new_sampling[3] = self.pixel_size
        dataset.sampling = new_sampling

        new_units = list(dataset.units)
        new_units[2] = "1/Angstrom"
        new_units[3] = "1/Angstrom"
        dataset.units = new_units

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_radial_profile(self) -> plt.Figure:
        """Plot the radial profile, background fit, and residual with peaks."""
        has_bg = self.background is not None
        n_axes = 2 if has_bg else 1

        fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 4))
        if n_axes == 1:
            axes = [axes]

        # --- Left panel: raw profile + background ---
        ax = axes[0]
        ax.plot(self.radial_pixel_positions, self.radial_profile, label="I(r)")
        if has_bg:
            ax.plot(self.radial_pixel_positions, self.background, "--", label="Background")
        if self.peak_pixel_positions is not None:
            ax.plot(
                self.peak_pixel_positions,
                self.radial_profile[self.peak_indices],
                "x",
                color="red",
                markersize=8,
                label="Peaks",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Radial position (pixels)")
        ax.set_ylabel("Mean intensity")
        ax.set_title("Radial profile + background")
        ax.legend()

        # --- Right panel: residual with peaks ---
        if has_bg:
            ax2 = axes[1]
            residual = getattr(self, "_residual", self.radial_profile - self.background)
            ax2.plot(self.radial_pixel_positions, residual, label="I(r) - B(r)")
            if self.peak_pixel_positions is not None:
                ax2.plot(
                    self.peak_pixel_positions,
                    residual[self.peak_indices],
                    "x",
                    color="red",
                    markersize=8,
                    label="Peaks",
                )
            ax2.axhline(0, color="gray", linewidth=0.5)
            ax2.set_xlabel("Radial position (pixels)")
            ax2.set_ylabel("Residual intensity")
            ax2.set_title("Background-subtracted profile")
            ax2.legend()

        fig.tight_layout()
        return fig

    def plot_trial_pixel_sizes(
        self,
        trial_sizes: list[float] | None = None,
        material: str | None = None,
    ) -> plt.Figure:
        """Overlay expected ring positions on the residual for several
        trial pixel sizes so the user can visually identify the correct
        matching.

        Parameters
        ----------
        trial_sizes
            List of pixel sizes (1/Å per pixel) to try.  Defaults to
            three values spanning a plausible range.
        material
            Reference material whose rings are overlaid. Defaults to
            ``self.material`` (set by :meth:`calibrate`) or ``'Au'``.
        """
        if trial_sizes is None:
            trial_sizes = [0.018, 0.026, 0.040]

        if material is None:
            material = self.material or "Au"
        q_ref = _q_values_for(material)
        residual = getattr(self, "_residual", self.radial_profile)
        r_max = float(self.radial_pixel_positions[-1])

        n = len(trial_sizes)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, ps in zip(axes, trial_sizes):
            ax.plot(self.radial_pixel_positions, residual, "b-", linewidth=1)
            if self.peak_pixel_positions is not None:
                ax.plot(
                    self.peak_pixel_positions,
                    residual[self.peak_indices.astype(int)],
                    "rx",
                    markersize=8,
                    zorder=5,
                )
            y_top = max(residual.max() * 1.05, 1.0)
            for hkl, qv in q_ref.items():
                r_exp = qv / ps
                if r_exp < r_max:
                    ax.axvline(r_exp, color="green", alpha=0.5, linestyle="--", linewidth=1)
                    ax.text(
                        r_exp + 0.5,
                        y_top * 0.92,
                        hkl,
                        rotation=90,
                        va="top",
                        fontsize=7,
                        color="green",
                    )
            ax.set_title(f"pixel_size = {ps} 1/\u00c5/px")
            ax.set_xlabel("Radial position (pixels)")
            ax.set_ylabel("Residual intensity")
            ax.set_xlim(0, r_max)
            ax.set_ylim(None, y_top)

        fig.tight_layout()
        return fig

    def plot_calibration_fit(self) -> plt.Figure:
        """Plot the linear calibration fit and residuals.

        Left panel shows matched peak positions vs. known q-values with
        the zero-intercept linear fit.  Right panel shows the residuals
        (measured q - predicted q) for each matched reflection.
        """
        if self.pixel_size is None or self.matched_q is None:
            raise RuntimeError("No calibration available. Call calibrate() first.")

        r = self.matched_pixel_positions
        q_pred = self.pixel_size * r
        residuals = self.matched_q - q_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # --- Left: linear fit ---
        ax1.scatter(r, self.matched_q, zorder=3, label="Matched peaks")
        r_fit = np.linspace(0, r.max() * 1.1, 100)
        ax1.plot(
            r_fit, self.pixel_size * r_fit, "--", label=f"Fit: {self.pixel_size:.6f} 1/\u00c5/px"
        )
        for i, hkl in enumerate(self.matched_hkl):
            ax1.annotate(hkl, (r[i], self.matched_q[i]), textcoords="offset points", xytext=(5, 5))
        ax1.set_xlabel("Radial position (pixels)")
        ax1.set_ylabel("q (1/\u00c5)")
        ax1.set_title("Calibration fit")
        ax1.legend()

        # --- Right: residuals ---
        ax2.stem(r, residuals, markerfmt="o", basefmt="k-")
        for i, hkl in enumerate(self.matched_hkl):
            ax2.annotate(
                hkl, (r[i], residuals[i]), textcoords="offset points", xytext=(5, 5), fontsize=8
            )
        ax2.axhline(0, color="gray", linewidth=0.5)
        ax2.set_xlabel("Radial position (pixels)")
        ax2.set_ylabel("\u0394q (1/\u00c5)")
        ax2.set_title("Residuals (measured - predicted)")

        fig.tight_layout()
        return fig
