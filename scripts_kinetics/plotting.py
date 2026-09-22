"""Survival probability plotting with fitted curve overlays."""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import patches as mpatches

try:
    from .fitting import f1_constrained, f2_constrained
except ImportError:
    from fitting import f1_constrained, f2_constrained


HARMONIC_COLORS = [
    "#E76F51", "#F4A261", "#E9C46A", "#2A9D8F",
    "#4C78A8", "#8172B2", "#B07AA1",
]


def harmonic_colors(n):
    """Return exactly n colors spanning the approved harmonic palette."""
    if n < 1:
        return []
    if n == 1:
        return [HARMONIC_COLORS[0]]
    cmap = LinearSegmentedColormap.from_list("residence_harmonic", HARMONIC_COLORS)
    return [cmap(value) for value in np.linspace(0.0, 1.0, n)]


def _natural_key(label):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(label))]


def attribution_order(labels, components, time_ns, method="residue_id",
                      horizon_ns=5.0, manual_order=None,
                      largest_at_bottom=True):
    """Return component indices in bottom-to-top stacking order."""
    labels = list(labels)
    if manual_order:
        requested = [labels.index(label) for label in manual_order if label in labels]
        remaining = [i for i in range(len(labels)) if i not in requested]
        return requested + remaining
    if method == "input":
        return list(range(len(labels)))
    if method == "residue_id":
        return sorted(range(len(labels)), key=lambda i: _natural_key(labels[i]))
    if method == "alphabetical":
        return sorted(range(len(labels)), key=lambda i: labels[i].lower())
    if method == "initial_percentage":
        values = components[:, 0]
    elif method == "integrated_contribution":
        mask = time_ns <= horizon_ns + 1e-12
        values = np.asarray([
            np.trapezoid(curve[mask], time_ns[mask]) for curve in components
        ])
    else:
        raise ValueError(f"Unknown attribution stack order: {method}")
    return list(np.argsort(values)[::-1 if largest_at_bottom else 1])


def plot_attribution(time_ns, regional, components, labels, name, outdir,
                     x_max_plot=5.0, n_bins=25, bin_spacing_factor=0.8,
                     base_fontsize=16, stack_order="residue_id",
                     rank_horizon_ns=5.0, largest_at_bottom=True,
                     color_method="component_identity", palette="harmonic",
                     component_colors=None, manual_order=None, title=""):
    """Plot one condition as additive stacked component contributions."""
    time_ns = np.asarray(time_ns, dtype=float)
    regional = np.asarray(regional, dtype=float)
    components = np.asarray(components, dtype=float)
    labels = list(labels)
    if components.shape != (len(labels), len(time_ns)):
        raise ValueError("components must have shape (n_components, n_time_points)")
    if regional.shape != time_ns.shape:
        raise ValueError("regional must have one value per time point")
    if palette != "harmonic":
        raise ValueError("Only the 'harmonic' attribution palette is currently supported")

    order = attribution_order(
        labels, components, time_ns, method=stack_order,
        horizon_ns=rank_horizon_ns, manual_order=manual_order,
        largest_at_bottom=largest_at_bottom)
    generated = harmonic_colors(len(labels))
    if color_method == "stack_position":
        colors = {index: generated[rank] for rank, index in enumerate(order)}
    elif color_method == "component_identity":
        identity_order = sorted(range(len(labels)), key=lambda i: _natural_key(labels[i]))
        colors = {index: generated[rank] for rank, index in enumerate(identity_order)}
    else:
        raise ValueError(f"Unknown attribution color method: {color_method}")
    for label, color in (component_colors or {}).items():
        if label in labels:
            colors[labels.index(label)] = color

    visible_max = min(float(x_max_plot), float(time_ns[-1]))
    edges = np.linspace(0.0, visible_max, int(n_bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    binned = np.zeros((len(labels), len(centers)))
    for i, curve in enumerate(components):
        for j, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (time_ns >= left) & (time_ns < right if j < len(centers) - 1
                                        else time_ns <= right)
            binned[i, j] = np.nanmean(curve[mask]) if np.any(mask) else np.nan

    total0 = np.nansum(components[:, 0])
    shares = components[:, 0] / total0 if total0 else np.zeros(len(labels))
    os.makedirs(outdir, exist_ok=True)
    plt.rc("font", family="Liberation Sans", size=base_fontsize)
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros_like(centers)
    for index in order:
        ax.bar(
            centers, binned[index], bottom=bottom,
            width=(edges[1] - edges[0]) * bin_spacing_factor,
            color=colors[index], alpha=0.65, edgecolor="none",
            label=f"{labels[index]} ({shares[index]:.1%})")
        bottom += np.nan_to_num(binned[index])
    ax.bar(
        centers, bottom, width=(edges[1] - edges[0]) * bin_spacing_factor,
        fill=False, edgecolor="black", linewidth=2.5, label="Regional total")
    ax.set_xlabel("Time (ns)", fontsize=base_fontsize)
    ax.set_ylabel("Survival contribution", fontsize=base_fontsize)
    if title:
        ax.set_title(title, fontsize=base_fontsize)
    ax.set_xlim(0.0, visible_max)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(labelsize=base_fontsize * 0.9)
    ax.legend(frameon=False, ncol=2, fontsize=base_fontsize * 0.9)
    path = os.path.join(outdir, f"attribution_{name}.svg")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sp_and_fits(tau, S, fit1, fit2, name, outdir, dpi=150, alpha=0.65,
                     x_max_plot=0.5, n_bins=20, bin_spacing_factor=0.8,
                     base_fontsize=16, fit_legend_mode="separate"):
    """Plot SP histogram with stacked component bars and fitted curves.

    fit_legend_mode may be "separate", "on_plot", or "none".
    """
    if fit_legend_mode not in {"separate", "on_plot", "none"}:
        raise ValueError("fit_legend_mode must be 'separate', 'on_plot', or 'none'")
    os.makedirs(outdir, exist_ok=True)
    plt.rc("font", family="Liberation Sans", size=base_fontsize)

    # Apparent residence time label
    app_rt = None
    if fit2 is not None and fit2.get("apparent_res_time") is not None:
        app_rt = fit2["apparent_res_time"]
    elif fit1 is not None and fit1.get("apparent_res_time") is not None:
        app_rt = fit1["apparent_res_time"]

    label_text = "Total" if app_rt is not None else "Simulated result"

    # Histogram bins over visible range
    bins = np.linspace(0, x_max_plot, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_w = bins[1] - bins[0]
    width = bin_w * bin_spacing_factor

    mask = tau <= x_max_plot
    tau_vis, S_vis = tau[mask], S[mask]
    counts, _ = np.histogram(tau_vis, bins=bins)
    hist, _ = np.histogram(tau_vis, bins=bins, weights=S_vis)
    hist = np.divide(hist, counts, out=np.zeros_like(hist), where=counts != 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Raw data bars
    ax.bar(bin_centers, hist, width=width, align="center",
           fill=False, edgecolor="black", linewidth=2.5,
           label=label_text, alpha=alpha)

    # Stacked component bars (bi-exp only)
    if fit2 is not None:
        u_val = fit2["u"]
        c = fit2["c"]
        t1, t2 = fit2["tau1"], fit2["tau2"]
        a1 = u_val * (1.0 - c)
        a2 = (1.0 - u_val) * (1.0 - c)

        p1_vals = a1 * np.exp(-bin_centers / t1)
        p2_vals = a2 * np.exp(-bin_centers / t2)
        pc_vals = np.full_like(bin_centers, c)

        ax.bar(bin_centers, pc_vals, width=width,
               color="lightgray", alpha=0.6, edgecolor="none",
               label="Structural water")
        ax.bar(bin_centers, p2_vals, width=width, bottom=pc_vals,
               color="skyblue", alpha=0.6, edgecolor="none",
               label="Slow water component")
        ax.bar(bin_centers, p1_vals, width=width, bottom=pc_vals + p2_vals,
               color="coral", alpha=0.6, edgecolor="none",
               label="Fast water component")

    # Smooth fitted curves
    tau_smooth = np.linspace(0, x_max_plot, 1000)
    single_label = bi_label = None

    if fit1 is not None:
        tau_fit, c = fit1["tau"], fit1["c"]
        a = fit1["alpha"]
        y_fit = f1_constrained(tau_smooth, tau_fit, c)
        r2 = fit1["r_squared"]
        single_label = (f"$P(t) = {a:.3f} e^{{-t/{tau_fit:.3f}}} + {c:.3f}$\n"
                        f"$R^2 = {r2:.4f}$")
        ax.plot(tau_smooth, y_fit, ":", lw=3.5, color="darkorange")

    if fit2 is not None:
        u_val = fit2["u"]
        c = fit2["c"]
        t1, t2 = fit2["tau1"], fit2["tau2"]
        a1, a2 = fit2["alpha1"], fit2["alpha2"]
        y_fit = f2_constrained(tau_smooth, u_val, c, t1, t2)
        frt = fit2["fitted_res_time"]
        r2 = fit2["r_squared"]
        bi_label = (f"$P(t) = {a1:.3f} e^{{-t/{t1:.3f}}} + {a2:.3f} e^{{-t/{t2:.3f}}} "
                    f"+ {c:.3f}$\n"
                    f"$R^2 = {r2:.4f}$\n"
                    f"Fitted residence time = {frt:.2f} ns")
        ax.plot(tau_smooth, y_fit, "--", lw=3.0, color="darkred")

    ax.set_xlabel("Time (ns)", fontsize=base_fontsize)
    ax.set_ylabel("$P(t)$", fontsize=base_fontsize)
    ax.set_xlim(0, x_max_plot)

    # Legend 1: component patches
    leg1_h, leg1_l = [], []
    leg1_h.append(mpatches.Patch(facecolor="white", edgecolor="black"))
    leg1_l.append(label_text)
    if fit2 is not None:
        leg1_h.append(mpatches.Patch(color="coral"))
        leg1_l.append("Fast component")
        leg1_h.append(mpatches.Patch(color="skyblue"))
        leg1_l.append("Slow component")
        leg1_h.append(mpatches.Patch(color="lightgray"))
        leg1_l.append("Constant component")

    legend1 = ax.legend(leg1_h, leg1_l, loc="upper right",
                        bbox_to_anchor=(0.98, 1.0),
                        fontsize=base_fontsize * 0.9, frameon=False)
    ax.add_artist(legend1)
    eq_h, eq_l = [], []
    if single_label is not None:
        eq_h.append(Line2D([0], [0], linestyle=":", lw=3.5, color="darkorange"))
        eq_l.append(single_label)
    if bi_label is not None:
        eq_h.append(Line2D([0], [0], linestyle="--", lw=3.0, color="darkred"))
        eq_l.append(bi_label)
    if fit_legend_mode == "on_plot" and eq_h:
        legend2 = ax.legend(eq_h, eq_l, loc="center right",
                            bbox_to_anchor=(0.98, 0.52),
                            fontsize=base_fontsize * 0.78, frameon=False,
                            labelspacing=1.0, handlelength=3.0)
        for txt in legend2.get_texts():
            txt.set_linespacing(1.4)
    ax.tick_params(axis="x", labelsize=base_fontsize * 0.9)
    ax.tick_params(axis="y", labelsize=base_fontsize * 0.9)

    fig.savefig(os.path.join(outdir, f"fit_{name}.svg"),
                dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"fit_{name}.png"),
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Legend 2: equation text as separate SVG
    if fit_legend_mode == "separate" and eq_h:
        fig_eq, ax_eq = plt.subplots(figsize=(8, 3))
        ax_eq.axis("off")
        leg = ax_eq.legend(eq_h, eq_l, loc="center",
                           fontsize=base_fontsize * 0.85, frameon=False,
                           labelspacing=1.2, handlelength=3.0)
        for txt in leg.get_texts():
            txt.set_linespacing(1.6)
        fig_eq.savefig(os.path.join(outdir, f"fit_{name}_equations.svg"),
                       dpi=dpi, bbox_inches="tight")
        fig_eq.savefig(os.path.join(outdir, f"fit_{name}_equations.png"),
                       dpi=dpi, bbox_inches="tight")
        plt.close(fig_eq)
