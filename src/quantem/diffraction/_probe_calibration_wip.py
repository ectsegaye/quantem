"""Probe-calibration WIP — code archived during merge of polar4dstem.

This file is a CODE ARCHIVE, not importable Python.  It contains methods and
helpers that were added on the ``reciprocal_calibration`` branch (commits
ff4e8e0 + e85421d) before ``polar4dstem`` refactored the surrounding code.

The contents below are extracted verbatim from the HEAD side of the merge
conflicts in ``polar.py`` and ``polar4dstem.py``.  They are wrapped in this
module-level docstring so the file parses, but the code inside is not live —
re-integrating means porting each section back into the (now-refactored)
class structure by hand.

Sections are labelled with their original file and line range.
"""

ARCHIVE = r'''
# ============================================================================
# polar.py lines 288-300
# PairDistributionFunction.radial_bins property
# Originally a member of the PairDistributionFunction class.
# ============================================================================


    @property
    def radial_bins(self) -> Any:
        """
        Radial bin centers in pixel units (convenience alias).
        """
        n = self.polar.shape[3]
        radial_min = float(self.polar.metadata.get("polar_radial_min", 0.0))
        radial_step = float(self.polar.metadata.get("polar_radial_step", 1.0))
        return np.arange(n, dtype=float) * radial_step + radial_min

# ============================================================================
# polar.py lines 630-683
# PairDistributionFunction torch-conversion utilities
# Originally a member of the PairDistributionFunction class.
# ============================================================================

        return mask_bool

    # ------------------------------------------------------------------
    # Torch conversion utilities
    # ------------------------------------------------------------------
    @property
    def polar_tensor(self) -> torch.Tensor:
        """Convert polar array to torch tensor on-the-fly (not cached).

        .. warning::
            This materialises the entire polar array as a float32 tensor.
            Prefer chunked access via ``self.polar.array`` for large datasets.
        """
        return torch.from_numpy(self.polar.array.astype(np.float32)).to(device=self.device)

    def _to_torch(self, arr: NDArray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32)).to(device=self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> NDArray:
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _gaussian_kernel_1d(
        sigma: float, device: str = "cpu", num_sigmas: float = 3.0
    ) -> torch.Tensor:
        """Create 1D Gaussian kernel for torch convolution."""
        radius = int(np.ceil(num_sigmas * sigma))
        support = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (support / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel

    def _gaussian_filter1d_torch(
        self,
        Fk: torch.Tensor,
        sigma: float,
        mode: str = "nearest",
    ) -> torch.Tensor:
        """
        Apply 1D Gaussian filter, replaces scipy.ndimage.gaussian_filter1d.
        """
        kernel = self._gaussian_kernel_1d(sigma, device=self.device)
        padding = len(kernel) // 2
        x = Fk.unsqueeze(0).unsqueeze(0)  # reshape to (batch, channels, length)
        kernel_w = kernel.view(1, 1, -1)
        if mode == "nearest":
            x = F.pad(x, (padding, padding), mode="replicate")
            result = F.conv1d(x, kernel_w)

# ============================================================================
# polar.py lines 1431-2062
# PairDistributionFunction analysis methods (calculate_radial_mean, etc.)
# Originally a member of the PairDistributionFunction class.
# ============================================================================

    # ------------------------------------------------------------------
    # Analysis method stubs
    # ------------------------------------------------------------------

    # TODO: add beamstop mask support (mask diffraction-space pixels before
    #       azimuthal averaging, e.g. to exclude a beam stop shadow)

    def calculate_radial_mean(
        self,
        mask_realspace: NDArray | None = None,
        returnval: bool = False,
        chunk_rows: int = 16,
    ) -> torch.Tensor | None:
        """
        Calculate the radial mean intensity from the Polar4dSTEM dataset.

        The polar array is assumed to have shape (scan_y, scan_x, phi, k).
        This method computes, for each scan position, the mean over the azimuthal
        axis (phi), then averages across scan positions to produce a single 1D
        radial curve. This result is stored in ``self.Ik``.

        If a real-space mask is provided, only the selected scan positions are
        used in the scan-position average.

        The computation streams row-chunks through torch to keep peak
        memory low.  The phi mean is computed *before* mask indexing so the
        largest intermediate per chunk is only (chunk_rows, scan_x, k).

        Parameters
        ----------
        mask_realspace : NDArray or None, optional
            Boolean mask in real space used to select probe positions.
            If ``None``, all probe positions are used.
            Must have shape (scan_y, scan_x) where True means "include".
        returnval : bool, optional
            If True, return the computed 1D radial mean tensor.
        chunk_rows : int, optional
            Number of scan rows to process at a time.

        Returns
        -------
        radial_mean : torch.Tensor or None
            If `returnval=True`, returns the 1D radial mean intensity (Nk,).
            Otherwise returns None.
        """
        polar_np = self.polar.array  # shape: (scan_y, scan_x, phi, k)
        scan_y, scan_x, n_phi, n_k = polar_np.shape

        running_sum = torch.zeros(n_k, device=self.device, dtype=torch.float64)
        n_valid = 0

        for y0 in range(0, scan_y, chunk_rows):
            y1 = min(y0 + chunk_rows, scan_y)
            raw = polar_np[y0:y1]
            chunk = torch.from_numpy(np.ascontiguousarray(raw)).to(self.device)

            # mean over phi first — (chunk, scan_x, phi, k) -> (chunk, scan_x, k)
            phi_mean = chunk.mean(dim=2)

            if mask_realspace is not None:
                mask_chunk = torch.from_numpy(mask_realspace[y0:y1]).to(self.device)
                n_chunk = int(mask_chunk.sum())
                if n_chunk == 0:
                    continue
                # mask on the reduced tensor, then sum
                running_sum += phi_mean[mask_chunk].sum(dim=0)
                n_valid += n_chunk
            else:
                running_sum += phi_mean.sum(dim=(0, 1))
                n_valid += (y1 - y0) * scan_x

        if n_valid == 0:
            raise ValueError(
                "No valid scan positions selected — the real-space mask is "
                "all-False or the dataset is empty."
            )
        self.Ik = (running_sum / n_valid).float()

        if returnval:
            return self.Ik
        else:
            return None

    def fit_bg(
        self,
        Ik: torch.Tensor,
        kmin: float,
        kmax: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fit a smooth background B(k) to a radial intensity curve I(k) using
        PyTorch LBFGS optimizer, with weighting that downweights the low-k
        region and emphasizes higher k.

        The fitted function uses the following form (adopted from py4dstem):
            B(k) = c
                + i0 * exp(-k^2 / (2 s0^2))
                + i1 * exp(-k^4 / (2 s1^4))

        Parameters
        ----------
        Ik
            1D radial intensity tensor (Nk,). Produced by
            :meth:`calculate_radial_mean`.
        kmin, kmax
            k-range (in the same units as the internally constructed `k` grid)
            used to build the low-k weighting mask.

        Returns
        -------
        bg : torch.Tensor
            Fitted background curve B(k), shape (Nk,).
        f : torch.Tensor
            Background minus the constant offset, f(k) = B(k) - c, or functionally
            similar to <f>^2(k)
        """
        k = self._to_torch(np.asarray(self.qq))
        k2 = k**2

        # normalize intensity
        int_mean = Ik.mean()
        Ik_norm = Ik / int_mean
        # initial guesses
        const_bg = float(Ik_norm.min())
        int0 = float(Ik_norm.median()) - const_bg
        sigma0 = float(k.mean())
        # ensure positive values
        const_bg = max(const_bg, 1e-6)
        int0 = max(int0, 1e-6)
        sigma0 = max(sigma0, 1e-6)

        init_vals = torch.tensor(
            [const_bg, int0, sigma0, int0, sigma0],
            device=self.device,
            dtype=torch.float32,
        )
        # Map to unconstrained space via inverse softplus: x = y + log(1 - exp(-y))
        # For numerical stability, clamp init_vals away from zero
        # final values must be positive for a physical model of background scattering
        init_vals = torch.clamp(init_vals, min=1e-6)
        theta = init_vals + torch.log(-torch.expm1(-init_vals))
        theta = theta.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [theta],
            lr=1.0,
            max_iter=20,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        # fitting weights (high-k range is emphasized for better bg estimation)
        # this monotonic model means we don't need parameterized scattering factors
        weights = self._compute_fit_weights(k, kmin, kmax)

        prev_loss = torch.tensor(float("inf"))
        max_outer_iter = 100
        tol = 1e-8
        for step in range(max_outer_iter):
            loss = optimizer.step(lambda: self._closure(optimizer, theta, k2, Ik_norm, weights))
            if torch.abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

        # final params (ensure positivity via softplus)
        with torch.no_grad():
            c = F.softplus(theta[0])
            i0 = F.softplus(theta[1])
            s0 = F.softplus(theta[2])
            i1 = F.softplus(theta[3])
            s1 = F.softplus(theta[4])
            # undo normalization
            c_scaled = c * int_mean
            i0_scaled = i0 * int_mean
            i1_scaled = i1 * int_mean
            # compute bg and the average scattering factor f(k)
            bg = self._scattering_model_torch(k2, c_scaled, i0_scaled, s0, i1_scaled, s1)
            f = bg - c_scaled
        return bg, f

    def calculate_Gr(
        self,
        k_min_fit: float | None = None,
        k_max_fit: float | None = None,
        k_min_window: float | None = None,
        k_max_window: float | None = None,
        k_lowpass: float | None = None,
        k_highpass: float | None = None,
        r_min: float = 0.0,
        r_max: float = 20.0,
        r_step: float = 0.02,
        mask_realspace: NDArray | None = None,
        damp_origin_oscillations: bool = False,
        density: float | None = None,
        r_cut: float = 0.8,
        returnval: bool = False,
    ) -> list[NDArray] | None:
        """
        Calculate the reduced pair distribution function G(r) from a 4D-STEM dataset.

        This routine:
        * Computes the radial mean intensity I(k) from self.polar (optionally
            restricted to a real-space mask).
        * Fits a smooth background B(k) and associated f(k) using :meth:`fit_bg`.
        * Constructs the reduced structure factor F(k) with optional low/highpass filtering.
        * Applies a window in k (low-k sin^2 ramp x Lorch high-k taper).
        * Computes the reduced PDF using a discrete sine transform:
           G(r) = sum_k sin(2*pi*k*r) * F_windowed(k)

        If ``damp_origin_oscillations=True``, :meth:`estimate_density` is called
        and the corrected F(k)/G(r) are stored as ``self.Fk_damped`` and
        ``self.reduced_pdf_damped``. The estimated density is cached in
        ``self.rho0`` so that a subsequent :meth:`calculate_gr` call can reuse it.

        Stored attributes:
        * self.Ik, self.bg, self.Fk, self.Fk_masked
        * self.Sk, self.r, self.reduced_pdf
        * self.rho0, self.Fk_damped, self.reduced_pdf_damped (if damping)

        Parameters
        ----------
        k_min_fit : float, optional
            Minimum k (A^-1) for the background fit.
        k_max_fit : float or None, optional
            Maximum k (A^-1) for the background fit.
        k_min_window : float or None, optional
            Minimum k (A^-1) for the structure-factor Lorch window.
            If None, falls back to ``k_min_fit``.
        k_max_window : float or None, optional
            Maximum k (A^-1) for the structure-factor Lorch window.
            If None, falls back to ``k_max_fit``.
        k_lowpass : float or None, optional
            Low-pass Gaussian filter sigma in k-space.
        k_highpass : float or None, optional
            High-pass Gaussian filter sigma in k-space.
        r_min : float, optional
            Minimum r (A) for the real-space grid.
        r_max : float, optional
            Maximum r (A) for the real-space grid.
        r_step : float, optional
            Step size in r (A) for the real-space grid.
        mask_realspace : NDArray or None, optional
            Boolean real-space mask selecting probe positions.
        damp_origin_oscillations : bool, optional
            If True, run :meth:`estimate_density` and store corrected F(k)/G(r).
        density : float or None, optional
            Known number density (atoms/A^3). If provided together with
            ``damp_origin_oscillations=True``, the S(k)/G(r) correction uses
            this value instead of estimating it.
        r_cut : float, optional
            Minimum radial distance (A) for peak search in density estimation.
            Forwarded to :meth:`estimate_density`.
        returnval : bool, optional
            If True, return ``[r, G(r)]`` as numpy arrays.

        Returns
        -------
        list[np.ndarray] or None
        """
        # clear results from any previous run so stale state doesn't leak
        self.Fk_damped = None
        self.reduced_pdf_damped = None
        self.rho0 = None

        # this is missing a 2pi term that we add back during the pdf calc later
        k_np = np.asarray(self.qq)
        k = self._to_torch(k_np)
        dk = k[1] - k[0]
        # small epsilon to avoid division by very small k values
        k_safe = torch.clamp(k, min=1e-10)
        self.kmax_fit = k_max_fit if k_max_fit is not None else float(k.max())
        self.kmin_fit = k_min_fit if k_min_fit is not None else float(k.min())
        # window range defaults to bg-fit range when not specified
        self.kmin_window = k_min_window if k_min_window is not None else self.kmin_fit
        self.kmax_window = k_max_window if k_max_window is not None else self.kmax_fit

        mask_bool = self._get_mask_bool(mask_realspace)
        # reuse existing radial mean if already computed
        if self.Ik is not None:
            Ik = self.Ik
        else:
            Ik = self.calculate_radial_mean(mask_realspace=mask_bool, returnval=True)
        # background fitting on Ik (uses the wider fit range)
        bg, f = self.fit_bg(Ik, self.kmin_fit, self.kmax_fit)
        # prevent division by near-zero values which cause NaNs at high k
        f_safe = torch.clamp(f, min=1e-10 * f.max())

        # below is the standard definition of F(k) used in PDF analysis, except for missing 2pi factor
        Fk = (Ik - bg) * k_safe / f_safe
        # apply optional frequency filtering for noise reduction
        Fk = self._frequency_filtering(Fk, k_lowpass, k_highpass, dk)
        # Compute Sk from Fk BEFORE applying the 2pi scaling,
        # so that estimate_density corrections are on the same scale
        self.Sk = torch.ones_like(k)
        mask = k > 0
        self.Sk = torch.where(mask, 1.0 + (Fk / k_safe), self.Sk)
        # apply that missing 2pi factor
        Fk = Fk * 2 * torch.pi
        # damp edges with lorch window (uses the narrower window range)
        wk = self._lorch_window(k, self.kmin_window, self.kmax_window)
        Fk_win = Fk * wk

        r = torch.arange(r_min, r_max, r_step, device=self.device, dtype=torch.float32)
        ka, ra = torch.meshgrid(k, r, indexing="ij")
        # compute reduced PDF using discrete sine transform
        reduced_pdf = (
            (2 / torch.pi)
            * dk
            * 2
            * torch.pi
            * torch.sum(
                torch.sin(2 * torch.pi * ra * ka) * Fk_win[:, None],
                dim=0,
            )
        )
        reduced_pdf[0] = 0  # physically must be at 0 when r = 0

        self.Ik = Ik
        self.bg = bg
        self.Fk = Fk
        self.Fk_masked = Fk_win
        self._r = r
        self._reduced_pdf = reduced_pdf

        # optionally damped unphysical oscillations near the origin by iteratively estimating density and correcting F(k)
        if damp_origin_oscillations:
            density_est = self.estimate_density(
                density=density,
                r_cut=r_cut,
                max_iter=40,
                tol_percent=1e-1,
            )
            self.rho0 = density_est[0]
            self.Fk_damped = density_est[1]
            self.reduced_pdf_damped = density_est[2]

        if returnval:
            Gr = (
                self.reduced_pdf_damped
                if self.reduced_pdf_damped is not None
                else self._reduced_pdf
            )
            return [self._to_numpy(self._r), self._to_numpy(Gr)]
        return None

    def calculate_gr(
        self,
        density: float | None = None,
        r_cut: float = 0.8,
        set_pdf_positive: bool = False,
        zero_low_r: bool = False,
        returnval: bool = False,
    ) -> list[NDArray] | None:
        """
        Calculate the pair distribution function g(r) from G(r).

        Requires :meth:`calculate_Gr` to have been run first. The density
        rho0 is determined by (in priority order):

        1. The ``density`` argument, if provided.
        2. ``self.rho0``, if already cached from a prior :meth:`estimate_density` call
           (e.g. via ``calculate_Gr(damp_origin_oscillations=True)``).
        3. A fresh call to :meth:`estimate_density` (result cached in ``self.rho0``).

        The G(r) used is ``self.reduced_pdf_damped`` if it exists (i.e. the user
        chose damping in :meth:`calculate_Gr`), otherwise ``self.reduced_pdf``.

        Parameters
        ----------
        density : float or None, optional
            Number density (atoms/A^3). If None, uses cached or estimated value.
        r_cut : float, optional
            Minimum radial distance (A) for peak search in density estimation.
            Only used when density must be estimated. Forwarded to
            :meth:`estimate_density`.
        set_pdf_positive : bool, optional
            If True, clamp negative g(r) values to 0.
        zero_low_r : bool, optional
            If True, zero g(r) below the first local minimum to the left of the
            primary peak and linearly ramp from 0 to g(r_thresh) at that point.
            This enforces the physical constraint that g(r) ≈ 0 at small r.
        returnval : bool, optional
            If True, return ``[r, g(r)]`` as numpy arrays.

        Returns
        -------
        list[np.ndarray] or None
        """
        if self._reduced_pdf is None or self._r is None:
            raise RuntimeError(
                "Reduced PDF not computed."
                "Run PairDistributionFunction.calculate_Gr() before calculate_gr()."
            )

        # Determine density
        if density is not None:
            rho0 = density
        elif self.rho0 is not None:
            rho0 = self.rho0
            print(f"  Using estimated rho0 = {rho0:.6f} atoms/A^3", flush=True)
        else:
            # the oscillation correction simultaneously produces a density estimate
            # if the user didn't run damping in calculate_Gr, we can still run the density estimation without using the corrected Fk/G(r)
            density_est = self.estimate_density(
                r_cut=r_cut,
                max_iter=40,
                tol_percent=1e-1,
            )
            self.rho0 = density_est[0]
            rho0 = self.rho0
            print(f"  Estimated rho0 = {rho0:.6f} atoms/A^3", flush=True)

        # Use damped G(r) if the user opted into damping, otherwise undamped
        Gr = self.reduced_pdf_damped if self.reduced_pdf_damped is not None else self._reduced_pdf
        Gr = Gr.clone()

        r = self._r

        if zero_low_r:
            # find the primary peak in G(r) and the closest local minimum to
            # its left, then replace G(r) below that point with -4πρ₀r.
            # This enforces g(r) = 0 at small r since:
            #   g(r) = 1 + G(r)/(4πrρ₀) = 1 + (-4πρ₀r)/(4πrρ₀) = 0
            mask_search = r >= r_cut
            r_search = r[mask_search]
            Gr_search = Gr[mask_search]
            ind_max = torch.argmax(Gr_search)
            r_peak = float(r_search[ind_max])

            left = (r > 0) & (r < r_peak)
            if torch.any(left):
                r_left = r[left]
                Gr_left = Gr[left]
                mins_cond = (Gr_left[1:-1] < Gr_left[:-2]) & (Gr_left[1:-1] < Gr_left[2:])
                mins_indices = torch.where(mins_cond)[0] + 1
                if mins_indices.numel() > 0:
                    ind_thresh = mins_indices[-1]
                else:
                    ind_thresh = torch.argmin(Gr_left)
                r_thresh = float(r_left[ind_thresh])

                below = r < r_thresh
                Gr = torch.where(below, -4 * torch.pi * rho0 * r, Gr)

        # g(r) = 1 + G(r) / (4 * pi * r * rho0)
        mask = r > 0
        pdf = torch.where(mask, 1 + Gr / (4 * torch.pi * r * rho0), torch.zeros_like(Gr))

        if set_pdf_positive:  # negative values are unphysical
            pdf = torch.maximum(pdf, torch.zeros_like(pdf))

        self._pdf = pdf
        if returnval:
            return [self._to_numpy(self._r), self._to_numpy(self._pdf)]
        return None

    def estimate_density(
        self,
        density: float | None = None,
        r_cut: float = 0.8,
        max_iter: int = 40,
        tol_percent: float = 1e-4,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """
        Estimate number density rho0 (atoms/A^3) and compute a corrected G(r).

        This method implements an iterative Q-space density estimation by
        Yoshimoto & Omote (2022). It uses the structure factor `self.Sk` and
        the reduced PDF `self.reduced_pdf` to iteratively update rho0 and a
        corrected S(k) so that the implied G(r) is more physically consistent
        at low r.

        If ``density`` is provided, the given value is used as a fixed rho0
        for the S(k)/G(r) correction instead of estimating it iteratively.

        This method requires that :meth:`calculate_Gr` has already been run,
        because it depends on `self.Sk`, `self.reduced_pdf`, `self.r`,
        and the k-window bounds (`self.kmin_fit`, `self.kmin_window`,
        `self.kmax_window`).

        Parameters
        ----------
        density : float or None, optional
            Known number density (atoms/A^3). If provided, used as a fixed
            rho0 — the iterative estimation is skipped and only the S(k)/G(r)
            correction is performed.
        r_cut : float, optional
            Minimum radial distance (A) for the peak search used to determine
            the correction interval. Peaks below this distance are ignored.
        max_iter : int, optional
            Maximum number of Q-space iterations.
        tol_percent : float, optional
            Convergence threshold on the relative change in rho0 (in %),
            as defined in Eq. (12) of Yoshimoto & Omote (2022).

        Returns
        -------
        rho0 : float
            Number density (atoms/A^3), either provided or estimated.
        Fk_win_damped : torch.Tensor
            Windowed corrected reduced structure function used for the transform.
        G_cor : torch.Tensor
            Reduced PDF G(r) with dampened oscillations near origin.
        """
        # we need the non-reduced structure factor (S(k) = 1 + F(k)/k) for the density estimation correction,
        # so we compute it here from the Fk we already have
        if self.Sk is None or self._reduced_pdf is None or self._r is None:
            raise RuntimeError(
                "This method depends on Sk, reduced_pdf, and r from calculate_Gr. "
                "Run PairDistributionFunction.calculate_Gr() before estimate_density()."
            )

        k = self._to_torch(np.asarray(self.qq))
        dk = k[1] - k[0]
        k_fit_mask = (k >= self.kmin_fit) & (k <= self.kmax_window)
        k_fit = k[k_fit_mask]
        ka, ra = torch.meshgrid(k, self._r, indexing="ij")

        # r_cut sets the minimum r for the peak search used to determine the correction interval
        mask_search = self._r >= r_cut
        r_search = self._r[mask_search]
        G_search = self._reduced_pdf[mask_search]
        # find tallest peak and first local minimum to the left of r_peak
        ind_max = torch.argmax(G_search)
        r_max = r_search[ind_max]
        left = self._r < r_max
        if not torch.any(left):
            # If peak is immediately at cutoff, just use cutoff as rmin
            rmin = r_cut
        else:
            r_left = self._r[left]
            G_left = self._reduced_pdf[left]
            mins_cond = (G_left[1:-1] < G_left[:-2]) & (G_left[1:-1] < G_left[2:])
            # fix indexing from slicing with +1
            mins_indices = torch.where(mins_cond)[0] + 1
            # minimum closest to the peak, else global min in left interval
            if mins_indices.numel() > 0:
                rmin = float(r_left[mins_indices[-1]])
            else:
                rmin = float(r_left[torch.argmin(G_left)])

        # Restrict r to [0, rmin] for the correction
        r_mask = (self._r >= 0.0) & (self._r <= rmin)
        r_short = self._r[r_mask]
        k_fit_scaled = k_fit * 2 * torch.pi
        k2d_fit, r2d_fit = torch.meshgrid(k_fit_scaled, r_short, indexing="ij")

        # Iterative refinement of rho0 and S(k) using windowed G(r).
        fixed_density = density is not None
        rho0 = density if fixed_density else 0.0
        rho0_prev = None
        Sk_cor = self.Sk.clone()
        # full Lorch window for final output
        wk = self._lorch_window(k, self.kmin_window, self.kmax_window)
        # windowed G(r) for the iteration
        Fk_win = k * (Sk_cor - 1.0) * wk * 2 * torch.pi
        G_iter = (
            (2.0 / torch.pi)
            * dk
            * 2
            * torch.pi
            * torch.sum(torch.sin(2 * torch.pi * ka * ra) * Fk_win[:, None], dim=0)
        )
        G_iter[0] = 0.0
        G_beta = G_iter[r_mask]

        beta_prev = None
        for j in range(max_iter):
            if j > 0:
                G_beta = G_iter[r_mask]

            alpha, beta = self._compute_alpha_beta(k2d_fit, r2d_fit, G_beta, r_short)

            if not fixed_density:
                rho0 = float(torch.sum(alpha * beta) / torch.sum(alpha**2))
                if rho0_prev is not None:
                    Rj = np.sqrt(((rho0_prev - rho0) ** 2) / (rho0**2)) * 100.0
                    if Rj < tol_percent:
                        break
            else:
                # fixed density: converge on the S(k) correction magnitude
                if beta_prev is not None:
                    delta = float(torch.max(torch.abs(beta - beta_prev)))
                    if delta < tol_percent * 1e-2:
                        break
                beta_prev = beta.clone()

            # Eq. (8): S_cor = S_obs - beta + rho0 * alpha
            Sk_cor[k_fit_mask] = Sk_cor[k_fit_mask] - beta + rho0 * alpha

            # recompute windowed G(r) for the next iteration
            Fk_win = k * (Sk_cor - 1.0) * wk * 2 * torch.pi
            G_iter = (
                (2.0 / torch.pi)
                * dk
                * 2
                * torch.pi
                * torch.sum(torch.sin(2 * torch.pi * ka * ra) * Fk_win[:, None], dim=0)
            )
            G_iter[0] = 0.0
            rho0_prev = rho0

        # final output: windowed G(r)
        Fk_cor = k * (Sk_cor - 1.0)
        Fk_win_damped = Fk_cor * wk * 2 * torch.pi
        G_cor = (
            (2.0 / torch.pi)
            * dk
            * 2
            * torch.pi
            * torch.sum(torch.sin(2 * torch.pi * ka * ra) * Fk_win_damped[:, None], dim=0)
        )
        G_cor[0] = 0.0

        return rho0, Fk_win_damped, G_cor

    # ------------------------------------------------------------------
    # Plotting functions
    # ------------------------------------------------------------------

    PlotName = Literal[
        "radial_mean",
        "background_fits",
        "reduced_sf",
        "reduced_pdf",
        "pdf",
        "oscillation_damping",
    ]


# ============================================================================
# polar.py lines 2081-2396
# PairDistributionFunction plotting methods (plot_pdf_results, etc.)
# Originally a member of the PairDistributionFunction class.
# ============================================================================


    def plot_pdf_results(
        self,
        which: Iterable[PlotName] = ("reduced_pdf",),
        *,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,
        rmax: float | None = None,
        figsize: tuple[float, float] = (6, 4),
        returnfigs: bool = False,
    ):
        """
        Convenience plotting dispatcher.

        Examples
        --------
        pdfc.calculate_Gr(...)
        pdfc.plot(["radial_mean", "background", "reduced_pdf"])
        """
        mapping = {
            "radial_mean": self.plot_radial_mean,
            "background_fits": self.plot_background_fits,
            "reduced_sf": self.plot_reduced_sf,
            "reduced_pdf": self.plot_reduced_pdf,
            "pdf": self.plot_pdf,
            "oscillation_damping": self.plot_oscillation_damping,
        }

        figs = []
        for name in which:
            if name not in mapping:
                raise ValueError(f"Unknown plot '{name}'. Options: {tuple(mapping)}")
            fig = mapping[name](
                qmin=qmin, qmax=qmax, rmin=rmin, rmax=rmax, figsize=figsize, returnfig=returnfigs
            )
            if returnfigs:
                figs.append(fig)

        return figs if returnfigs else None

    def plot_radial_mean(
        self,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,  # accepted for dispatcher compatibility, unused
        rmax: float | None = None,  # accepted for dispatcher compatibility, unused
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting radial mean intensity vs scattering vector.
        """

        if self.Ik is None:
            raise RuntimeError(
                "Radial mean intensity has not been calculated yet."
                "Run PairDistributionFunction.calculate_Gr() or PairDistributionFunction.calculate_radial_mean() before plotting."
            )

        x = np.asarray(self.qq)
        y = self._to_numpy(self.Ik)
        x, y = self._apply_xrange(x, y, qmin, qmax)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Radial Mean Intensity I(k)")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Radial Mean Intensity vs Scattering Vector")
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_background_fits(
        self,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,  # accepted for dispatcher compatibility, unused
        rmax: float | None = None,  # accepted for dispatcher compatibility, unused
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting background fit vs radial mean intensity.
        """
        if self.Ik is None or self.bg is None:
            raise RuntimeError(
                "Radial mean intensity or background has not been calculated yet."
                "Run PairDistributionFunction.calculate_Gr() or both calculate_radial_mean() and calculate_background() before plotting."
            )

        x = np.asarray(self.qq)
        y1 = self._to_numpy(self.Ik)
        x, y1 = self._apply_xrange(x, y1, qmin, qmax)
        x = np.asarray(self.qq)
        y2 = self._to_numpy(self.bg)
        x, y2 = self._apply_xrange(x, y2, qmin, qmax)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y1, label="Radial Mean Intensity I(k)")
        ax.plot(x, y2, label="Background B(k)", linestyle="--")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Radial Mean Intensity and Background Fit")
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_reduced_sf(
        self,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,  # accepted for dispatcher compatibility, unused
        rmax: float | None = None,  # accepted for dispatcher compatibility, unused
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting reduced structure factor F(k).
        """
        if self.Fk_masked is None:
            raise RuntimeError(
                "Reduced structure factor F(k) has not been calculated yet."
                "Run PairDistributionFunction.calculate_Gr() before plotting."
            )

        Fk = getattr(self, "Fk_damped", None)
        if Fk is None:
            Fk = self.Fk_masked

        x = np.asarray(self.qq)
        y = self._to_numpy(Fk)
        x, y = self._apply_xrange(x, y, qmin, qmax)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Reduced Structure Factor F(k)")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Reduced Structure Factor F(k)")
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_reduced_pdf(
        self,
        qmin: float | None = None,  # accepted for dispatcher compatibility, unused
        qmax: float | None = None,  # accepted for dispatcher compatibility, unused
        rmin: float | None = None,
        rmax: float | None = None,
        padding_frac: float = 0.1,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting reduced PDF g(r).
        """
        if self._reduced_pdf is None:
            raise RuntimeError(
                "Reduced PDF has not been calculated yet."
                "Run PairDistributionFunction.calculate_Gr() before plotting."
            )
        Gr = self.reduced_pdf_damped if self.reduced_pdf_damped is not None else self._reduced_pdf

        x = self._to_numpy(self._r)
        y = self._to_numpy(Gr)
        x, y = self._apply_xrange(x, y, rmin, rmax)

        # Find radial value of primary peak and trough for y-limits
        # Filter out NaN and Inf values to avoid plot errors
        valid_mask = np.isfinite(y)
        if np.any(valid_mask):
            y_valid = y[valid_mask]
            y_max = np.max(y_valid)
            y_min = np.min(y_valid)
        else:
            # Fallback if all values are invalid
            y_max = 1.0
            y_min = -1.0
        yrange = y_max - y_min
        pad = padding_frac * yrange

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Reduced Pair Distribution Function G(r)")
        ax.set_xlabel("Radial Distance r (Å)")
        ax.set_ylabel("Reduced Pair Distribution Function G(r)")
        ax.set_ylim(y_min - pad, y_max + pad)
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_pdf(
        self,
        qmin: float | None = None,  # accepted for dispatcher compatibility, unused
        qmax: float | None = None,  # accepted for dispatcher compatibility, unused
        rmin: float | None = None,
        rmax: float | None = None,
        padding_frac: float = 0.1,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting pair distribution function g(r).
        """
        if self._reduced_pdf is None or self._pdf is None:
            raise RuntimeError(
                "PDF has not been calculated yet."
                "Run PairDistributionFunction.calculate_gr() before plotting."
            )

        x = self._to_numpy(self._r)
        y = self._to_numpy(self._pdf)
        x, y = self._apply_xrange(x, y, rmin, rmax)

        # Find radial value of primary peak
        # Filter out NaN and Inf values to avoid plot errors
        valid_mask = np.isfinite(y)
        if np.any(valid_mask):
            y_valid = y[valid_mask]
            y_max = np.max(y_valid)
            y_min = np.min(y_valid)
        else:
            # Fallback if all values are invalid
            y_max = 1.0
            y_min = -1.0
        yrange = y_max - y_min
        pad = padding_frac * yrange

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Pair Distribution Function g(r)")
        ax.set_xlabel("Radial Distance r (Å)")
        ax.set_ylabel("Pair Distribution Function g(r)")
        ax.set_ylim(y_min - pad, y_max + pad)
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_oscillation_damping(
        self,
        qmin: float | None = None,  # accepted for dispatcher compatibility, unused
        qmax: float | None = None,  # accepted for dispatcher compatibility, unused
        rmin: float | None = None,
        rmax: float | None = None,
        padding_frac: float = 0.1,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        if self.Fk_masked is None or self.Fk_damped is None or self.reduced_pdf_damped is None:
            raise RuntimeError(
                "Oscillation damping data not available. "
                "Run calculate_Gr(damp_origin_oscillations=True) first."
            )

        k = np.asarray(self.qq)

        # Convert torch tensors to numpy for plotting
        Fk_masked = self._to_numpy(self.Fk_masked)
        Fk_damped = self._to_numpy(self.Fk_damped)
        r = self._to_numpy(self._r)
        reduced_pdf = self._to_numpy(self._reduced_pdf)
        reduced_pdf_damped = self._to_numpy(self.reduced_pdf_damped)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # F(k)
        axS_top = axes[0, 0]
        axS_res = axes[1, 0]
        axS_top.plot(k, Fk_masked, label="F_obs(k)", color="gray")
        axS_top.plot(k, Fk_damped, label="F_cor(k)", color="red")
        axS_top.set_xlabel("k (A$^{-1}$)")
        axS_top.set_ylabel("F(k)")
        axS_top.legend()

        axS_res.plot(k, Fk_damped - Fk_masked, color="blue")
        axS_res.set_xlabel("k (A$^{-1}$)")
        axS_res.set_ylabel("F_cor - F_obs")

        # G(r)
        axG_top = axes[0, 1]
        axG_res = axes[1, 1]
        axG_top.plot(r, reduced_pdf, label="G_obs(r)", color="gray")
        axG_top.plot(r, reduced_pdf_damped, label="G_cor(r)", color="red")
        axG_top.set_xlabel("r (A)")
        axG_top.set_ylabel("G(r)")
        axG_top.legend()

        axG_res.plot(r, reduced_pdf_damped - reduced_pdf, color="blue")
        axG_res.set_xlabel("r (A)")
        axG_res.set_ylabel("G_cor - G_obs")

        fig.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

# ============================================================================
# polar4dstem.py lines 92-593
# polar4dstem.py module-level helpers and class additions
# ============================================================================



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

'''
