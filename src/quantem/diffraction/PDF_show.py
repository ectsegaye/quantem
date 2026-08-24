import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.ndimage import gaussian_filter


def plot_RDF3b(
    gr,
    dtheta,
    dr,
    theta_slices=None,
    r_slices=None,
    log=False,
    title=None,
    cbar=False,
    aspect=None,
    intensity_range = "ordered",
    vmin=None,
    vmax=None,
    **kwargs,
):
    def rFormatter(x, pos):
        return f"{x*dr:.3g}"

    def thetaFormatter(x, pos):
        return f"{x*dtheta:.3g}"

    # alias
    r_slices = kwargs.pop("r_slice", None) if r_slices is None else r_slices
    theta_slices = (
        kwargs.pop("theta_slice", None) if theta_slices is None else theta_slices
    )

    cmap = kwargs.pop("cmap", "magma")
    origin = kwargs.pop("origin", "lower")
    if aspect is None:
        aspect = "auto"
        aspect_num = 1
    else:
        aspect_num = aspect

    plot_thetas = False if theta_slices is None else True
    plot_rs = False if r_slices is None else True
    if plot_thetas == plot_rs:
        print(f"Theta slices: {theta_slices} -- {plot_thetas}")
        print(f"r slices: {r_slices} -- {plot_rs}")
        print("Currently plots theta slices or r slices, not both (or neither)")
        return

    val_slices = theta_slices if plot_thetas else r_slices
    dind = dtheta if plot_thetas else dr

    if isinstance(val_slices, str):
        if val_slices.lower() not in ["peak", "max"]:
            print(f"Wrong val slices given {val_slices}")
            print("Expected 'peak' or 'max'")
            return
        if plot_rs:
            val_slices = np.unravel_index(np.argmax(gr), gr.shape)[1] * dr
        elif plot_thetas:
            val_slices = np.unravel_index(np.argmax(gr), gr.shape)[0] * dr

    if isinstance(val_slices, (int, float)):
        int_slice = int(round(val_slices / dind))
        ind_slices = np.array([(int_slice, int_slice + 1)])
        val_slices = [val_slices]
    elif isinstance(val_slices, (tuple, list, np.ndarray)):
        if np.ndim(val_slices) == 1:
            int_slices = np.round(np.array(val_slices) / dind).astype("int")
            ind_slices = np.array([[sl, sl + 1] for sl in int_slices])
        elif np.ndim(val_slices) == 2:
            ind_slices = np.array(
                [np.round(np.array(sl) / dind) for sl in val_slices]
            ).astype("int")

    ncols = len(ind_slices)
    figsize = kwargs.pop("figsize", (ncols * 4 * aspect_num, 3))
    fig, axs = plt.subplots(ncols=ncols, nrows=1, figsize=figsize)
    if ncols == 1:
        axs = [axs]
    for i, (ax, slice) in enumerate(zip(axs, ind_slices)):
        if np.ndim(val_slices[i]) == 0:
            slice_title = str(round(val_slices[i], 3))
        elif np.ndim(val_slices[i]) == 1:
            islice = np.round(val_slices[i], 3)
            slice_title = f"{islice[0]} : {islice[1]}"
        if plot_thetas:
            gslice = gr[slice[0] : slice[1]].mean(axis=0)
            yformatter = rFormatter
            ax.set_title(f"$\\theta = {slice_title}^\\circ$")
            if i == 0:
                ax.set_ylabel(f"r2 (A)")
        elif plot_rs:
            gslice = gr[:, slice[0] : slice[1]].mean(axis=1)
            yformatter = thetaFormatter
            ax.set_title(f"r2 = {slice_title} A")
            if i == 0:
                ax.set_ylabel(f"$\\theta$ (deg)")
        if log:
            _mask = gslice > 0.0
            gslice[_mask] = np.log(gslice[_mask])
            gslice[~_mask] = 0

        if intensity_range == "ordered":
            vmin = 0.001 if vmin is None else vmin
            vmax = 0.999 if vmax is None else vmax
            vals = np.sort(gslice.ravel())
            ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
            ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
            ind_vmin = np.max([0, ind_vmin])
            ind_vmax = np.min([len(vals) - 1, ind_vmax])
            vmin_ax = vals[ind_vmin]
            vmax_ax = vals[ind_vmax]
        else:
            vmin_ax, vmax_ax = vmin, vmax

        im = ax.matshow(gslice, cmap=cmap, origin=origin, aspect=aspect, vmin=vmin_ax, vmax=vmax_ax, **kwargs)
        ax.set_xlabel(f"r1 (A)")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(yformatter))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(rFormatter))
        ax.xaxis.tick_bottom()
        ax.tick_params(direction="out")

        if cbar:
            cax = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=str(kwargs.get("cbar_title", "")))

    fig.subplots_adjust(wspace=0.35 if cbar else 0.15)

    if title is not None:
        fig.suptitle(title, y=1.02)

    return

def plot_r1r2_RDF3b(gr, dtheta, dr, log=False, title="", cbar=False, aspect=None, intensity_range="ordered", ax=None, **kwargs):
    "theta_slice and r_slice given in deg and angstroms respectively"

    def rFormatter(x, pos):
        return f"{x*dr:.3g}"

    def thetaFormatter(x, pos):
        return f"{x*dtheta:.3g}"

    cmap = kwargs.pop("cmap", "magma")
    origin = kwargs.pop("origin", "lower")

    gr_r1r2 = np.array([np.diag(gr[i]) for i in range(gr.shape[0])])
    if log:
        _mask = gr_r1r2 > 0.0
        gr_r1r2[_mask] = np.log(gr_r1r2[_mask])
        gr_r1r2[~_mask] = 0

    if aspect is None:
        aspect = gr_r1r2.shape[0] / gr_r1r2.shape[1]

    if ax == None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (5, 5*aspect)))
    else:
        fig = ax.get_figure()

    if intensity_range == "ordered":
        vmin = kwargs.pop("vmin", 0.001)
        vmax = kwargs.pop("vmax", 0.999)
        vals = np.sort(gr_r1r2.ravel())
        ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
        ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
        ind_vmin = np.max([0, ind_vmin])
        ind_vmax = np.min([len(vals) - 1, ind_vmax])
        vmin_ax = vals[ind_vmin]
        vmax_ax = vals[ind_vmax]
    else:
        vmin_ax = kwargs.pop("vmin", None)
        vmax_ax = kwargs.pop("vmax", None)

    im = ax.matshow(gr_r1r2, cmap=cmap, origin=origin, vmin=vmin_ax, vmax=vmax_ax, **kwargs)
    ax.set_ylabel(f"$\\theta$ (deg)")
    ax.set_xlabel(f"r1 = r2 (A)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(thetaFormatter))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(rFormatter))
    ax.set_yticks(np.arange(0, 190, 30) / dtheta)

    ax.xaxis.tick_bottom()
    ax.tick_params(direction="out")

    if cbar:
        ax_divider = make_axes_locatable(ax)
        c_axis = ax_divider.append_axes("right", size="4%", pad="2%")
        fig.colorbar(im, cax=c_axis, format="%g", label=str(kwargs.pop("cbar_title", "")))

    # plt.show()
    return ax

def plot_diagonal_grtruth(
    gr,
    dtheta,
    dr,
    r_search_min: float = 0.5,
    r_max_display: float | None = None,
    r_display_power: int = 0,
    vmax_quantile: float = 0.999,
    r_marks=None,
    markers=None,
    cmap: str = "magma",
    title: str | None = None,
    figsize: tuple[float, float] = (7, 5),
    fontsize: int = 10,
    labelsize: int = 10,
    ax=None,
    cbar: bool = True,
    sigma: int | None = None,
    returnfig: bool = False,
):
    """
    Plot the r = r' diagonal of a ground-truth 3-body PADF, gr(r, r, theta),
    as a 2D map with r on the x-axis (Angstrom) and theta on the y-axis
    (degrees).

    This is the same quantity plotted by PDF_show.plot_r1r2_RDF3b(), but with
    a fix for its main visibility problem: gr's normalization divides by
    r1^2 * r2^2, so bins at very small r (just above the exactly-zeroed r=0
    bin) can take on enormous values that are a normalization artifact, not
    real structure. plot_r1r2_RDF3b computes its percentile color scale over
    the whole diagonal image, so that artifact dominates the scale and washes
    out the genuine, physically meaningful peaks at larger r.

    Following the approach used by padf.py's plot_g2_g3() for an analogous
    problem, the color scale here is instead computed as a quantile taken
    only over r > r_search_min, excluding the small-r artifact region. Since
    gr is non-negative everywhere (histogram counts divided by positive
    normalization factors), a sequential colormap with vmin=0 is used.

    Parameters
    ----------
    gr : ndarray, shape (Ntheta, Nr, Nr)
        Ground-truth 3-body PADF, as returned by PDF3B.pdf3B().
    dtheta, dr : float
        Bin widths (degrees, Angstrom) used to build gr.
    r_search_min : float
        Radii below this are excluded when computing the color scale,
        avoiding the small-r normalization artifact.
    r_max_display : float or None
        Upper limit of the radial axis; defaults to min(8 A, r.max()).
    r_display_power : int
        Display weighting exponent; the map is multiplied by
        r^(2 * r_display_power). 0 plots the raw values.
    vmax_quantile : float
        Quantile (over the r > r_search_min region) used as the color-scale
        maximum.
    r_marks : sequence of float or None
        Radii (e.g. neighbor shell distances) marked with dotted vertical
        lines.
    markers : sequence of (r, theta_deg) or (r, theta_deg, size) or None
        Expected diagonal maxima overlaid as open circles; a third element
        per tuple scales the marker size (relative multiplicity).
    title : str or None
        Figure title; a default is used if None.
    ax : matplotlib Axes or None
        Axes to draw into; a new figure/axes is created if None.
    cbar : bool
        Whether to draw a colorbar.
    returnfig : bool
        Return (fig, ax) instead of calling plt.show().
    """
    gr = np.asarray(gr)

    if sigma is not None:
        gr = gaussian_filter(gr, sigma=sigma, mode="wrap")

    ntheta, nr = gr.shape[0], gr.shape[1]

    r = np.arange(nr) * dr
    theta_deg = np.arange(ntheta) * dtheta

    diag = np.array([np.diag(gr[i]) for i in range(ntheta)])  # (Ntheta, Nr)

    if r_max_display is None:
        r_max_display = float(min(8.0, r[-1]))

    r_mask = r <= r_max_display
    r_sel = r[r_mask]
    diag_sel = diag[:, r_mask]

    weight = r_sel ** (2 * r_display_power)
    diag_disp = diag_sel * weight[None, :]

    artifact_free = r_sel > r_search_min
    vals = diag_disp[:, artifact_free] if np.any(artifact_free) else diag_disp
    vmax = np.quantile(vals, vmax_quantile) if vals.size else diag_disp.max()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    im = ax.pcolormesh(
        r_sel, theta_deg, diag_disp,
        cmap=cmap, vmin=0, vmax=vmax, shading="auto",
    )

    if r_marks is not None:
        for r_mk in r_marks:
            ax.axvline(r_mk, color="w", ls=":", lw=0.8)

    if markers is not None:
        for mk in markers:
            r_mk, t_mk = mk[0], mk[1]
            size = 12.0 * (mk[2] ** 0.5) if len(mk) > 2 else 10.0
            if r_mk <= r_sel[-1]:
                ax.plot(r_mk, t_mk, "o", ms=max(size, 5.0), mfc="none", mec="w", mew=1.0)

    ax.set_yticks(np.arange(0, 181, 30)[1:])
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_xlabel("r = r' (Å)", fontsize=fontsize)
    ax.set_ylabel(r"$\theta$ (degrees)", fontsize=fontsize)
    ax.set_title(title if title is not None else "Ground-truth PADF diagonal", fontsize=10)
    

    if cbar:
        fig.colorbar(im, ax=ax, label="g(r, r, θ)", pad=0.02)

    if returnfig:
        return fig, ax
    plt.show()

def plot_g2_g3_grtruth(
    gr,
    dtheta,
    dr,
    r_min: float | None = None,
    r_max: float | None = None,
    r_display_power: int = 1,
    r_max_display: float | None = None,
    r_search_min: float = 0.5,
    vmax_quantile: float = 0.999,
    r_marks=None,
    markers=None,
    title: str | None = None,
    figsize: tuple[float, float] = (4.8, 4.4),
    returnfig: bool = False,
):
    """
    Stacked 2-body / 3-body correlation summary of a ground-truth 3-body
    PADF (as returned by PDF3B.pdf3B()), analogous to padf.py's
    plot_g2_g3() but built from gr(theta, r2, r1) + bin widths (dtheta, dr)
    instead of a PairAngleDistributionFunction instance.

    Top panel: 3-body correlation map - gr(r, r', theta) integrated over r
    in [r_min, r_max] (symmetrized over both radial axes), plotted as angle
    theta vs r'.

    Bottom panel: 2-body correlation profile - the sin(theta)-weighted mean
    of the r = r' diagonal over the interior angular range (15-165 deg),
        sum_theta gr(r, r, theta) sin(theta) / sum_theta sin(theta).
    Unlike plot_g2_g3's Theta (a correlation function whose true l = 0
    component is removed by the mean subtraction, forcing a sin-weighted
    RMS proxy since a plain integral cancels by Legendre orthogonality),
    gr is a genuine non-negative pair density, so its plain sin-weighted
    mean is itself the angularly-integrated 2-body correlation - no RMS
    trick is needed. It peaks at the radial shells where the PADF has
    angular structure and is used to choose the near-neighbor shell; the
    integration shell [r_min, r_max] is shaded.

    Both panels are weighted by r^(2 * r_display_power) for display. Since
    gr is non-negative, the map uses a sequential colormap with vmin=0
    (rather than plot_g2_g3's diverging RdBu_r), and its color limits are a
    quantile taken over the interior angular range (15-165 deg) and r >
    r_search_min - following plot_diagonal_grtruth's fix for the small-r
    normalization artifact (gr's r1^2 * r2^2 division blows up bins just
    above the exactly-zeroed r=0 bin).

    Parameters
    ----------
    gr : ndarray, shape (Ntheta, Nr, Nr)
        Ground-truth 3-body PADF, as returned by PDF3B.pdf3B().
    dtheta, dr : float
        Bin widths (degrees, Angstrom) used to build gr.
    r_min, r_max : float or None
        Integration shell bounds in Angstrom. If None, the first peak of
        g2 is selected automatically: from the last point below 5% of the
        peak height up to the first local minimum past the peak (same
        shell heuristic as plot_g2_g3).
    r_display_power : int
        Display weighting exponent; each panel is multiplied by
        r^(2 * r_display_power). 0 plots the raw values.
    r_max_display : float or None
        Upper limit of the radial axes; defaults to min(8 A, r.max()).
    r_search_min : float
        Radii below this are ignored both when auto-locating the g2 peak
        and when computing the map's color scale (excludes the small-r
        normalization artifact).
    vmax_quantile : float
        Quantile (over the artifact-free region) used as the map's color
        scale maximum.
    r_marks : sequence of float or None
        Radii (e.g. the 1st/2nd/3rd neighbor shell distances) marked with
        dotted vertical lines on both panels.
    markers : sequence of (r, theta_deg) or (r, theta_deg, size) or None
        Expected 3-body maxima overlaid on the map as open circles - e.g.
        the (shell distance, arm-arm angle) targets from a known
        structure. A third element per tuple, if present, scales the
        marker size (relative multiplicity).
    title : str or None
        Figure title; a shell summary is used if None.
    returnfig : bool
        Return (fig, (ax_g3, ax_g2)) instead of calling plt.show().
    """
    gr = np.asarray(gr)
    ntheta, nr = gr.shape[0], gr.shape[1]

    r = np.arange(nr) * dr
    theta_deg = np.arange(ntheta) * dtheta
    theta = np.deg2rad(theta_deg)

    # pair profile: sin(theta)-weighted mean of the diagonal over the
    # interior angles (gr's l = 0 term is real, unlike the mean-subtracted
    # correlation function plot_g2_g3 was written for)
    diag = np.array([np.diag(gr[i]) for i in range(ntheta)])  # (Ntheta, Nr)
    interior_t = (theta_deg > 15) & (theta_deg < 165)
    w_t = np.abs(np.sin(theta[interior_t]))
    g2 = (diag[interior_t, :] * w_t[:, None]).sum(axis=0) / w_t.sum()

    # auto-select the first g2 peak if no shell was given
    if r_min is None or r_max is None:
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kernel /= kernel.sum()
        smooth = np.convolve(np.where(r < r_search_min, 0.0, g2), kernel, mode="same")
        i_pk = None
        for idx in range(1, smooth.size - 1):
            if smooth[idx] > 0 and smooth[idx] >= smooth[idx - 1] and smooth[idx] > smooth[idx + 1]:
                i_pk = idx
                break
        if i_pk is None:
            i_pk = int(np.argmax(smooth))
        below = np.flatnonzero(smooth[:i_pk] < 0.05 * smooth[i_pk])
        i_lo = int(below[-1]) if below.size else int(np.searchsorted(r, r_search_min))
        i_hi = min(smooth.size - 1, 2 * i_pk - i_lo)
        for idx in range(i_pk + 1, smooth.size - 1):
            if smooth[idx] <= smooth[idx - 1] and smooth[idx] <= smooth[idx + 1]:
                i_hi = idx
                break
        if r_min is None:
            r_min = float(r[i_lo])
        if r_max is None:
            r_max = float(r[i_hi])

    shell = (r >= r_min) & (r <= r_max)
    if not np.any(shell):
        shell[int(np.argmin(np.abs(r - 0.5 * (r_min + r_max))))] = True

    # g3 map: integrate over the shell, symmetrized over both r axes
    g3_map = 0.5 * (gr[:, :, shell].sum(axis=2) + gr[:, shell, :].sum(axis=1))  # (Ntheta, Nr')

    # display weighting and color limits
    if r_max_display is None:
        r_max_display = float(min(8.0, r[-1]))
    weight = r ** (2 * r_display_power)
    g2_disp = g2 * weight
    g3_disp = g3_map * weight[None, :]  # (Ntheta, Nr')
    rs = r <= r_max_display
    interior = (theta_deg > 15) & (theta_deg < 165)
    artifact_free = rs & (r > r_search_min)
    vmax_cols = artifact_free if np.any(artifact_free) else rs
    vmax = np.quantile(g3_disp[np.ix_(interior, vmax_cols)], vmax_quantile)

    fig, (ax_g3, ax_g2) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    im = ax_g3.pcolormesh(
        r[rs], theta_deg, g3_disp[:, rs],
        cmap="magma", vmin=0, vmax=vmax, shading="auto",
    )
    ax_g3.set_ylabel("$\\theta$ (deg)")
    ax_g3.set_yticks(np.arange(0, 181, 45))
    fig.colorbar(im, ax=[ax_g3, ax_g2], label="3-body correlation", pad=0.02)

    if markers is not None:
        for mk in markers:
            r_mk, t_mk = mk[0], mk[1]
            size = 12.0 * (mk[2] ** 0.5) if len(mk) > 2 else 10.0
            if r_mk <= r_max_display:
                ax_g3.plot(r_mk, t_mk, "o", ms=max(size, 5.0), mfc="none",
                           mec="w", mew=1.0)

    ax_g2.plot(r[rs], g2_disp[rs], "k-", lw=1)
    ax_g2.axvspan(r_min, r_max, color="C1", alpha=0.25)
    ax_g2.axhline(0, color="0.85", lw=0.8, zorder=0)

    if r_marks is not None:
        for r_mk in r_marks:
            ax_g3.axvline(r_mk, color="w", ls=":", lw=0.8)
            ax_g2.axvline(r_mk, color="k", ls=":", lw=0.8)
    ax_g2.set_ylabel("2-body corr.")
    ax_g2.set_xlabel("r, r' (Å)")

    ax_g3.set_title(
        title if title is not None
        else f"shell r = {r_min:.2f} - {r_max:.2f}",
        fontsize=10,
    )

    if returnfig:
        return fig, (ax_g3, ax_g2)
    plt.show()
