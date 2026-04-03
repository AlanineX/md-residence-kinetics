"""Survival probability plotting with fitted curve overlays."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patches as mpatches

from fitting import f1_constrained, f2_constrained


def plot_sp_and_fits(tau, S, fit1, fit2, name, outdir, dpi=150, alpha=0.65,
                     x_max_plot=0.5, n_bins=20, bin_spacing_factor=0.8,
                     base_fontsize=16):
    """Plot SP histogram with stacked component bars and fitted curves.

    Saves: fit_{name}.svg (main plot) and fit_{name}_equations.svg (legend).
    """
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
    ax.tick_params(axis="x", labelsize=base_fontsize * 0.9)
    ax.tick_params(axis="y", labelsize=base_fontsize * 0.9)

    fig.savefig(os.path.join(outdir, f"fit_{name}.svg"),
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Legend 2: equation text as separate SVG
    eq_h, eq_l = [], []
    if single_label is not None:
        eq_h.append(Line2D([0], [0], linestyle=":", lw=3.5, color="darkorange"))
        eq_l.append(single_label)
    if bi_label is not None:
        eq_h.append(Line2D([0], [0], linestyle="--", lw=3.0, color="darkred"))
        eq_l.append(bi_label)
    if eq_h:
        fig_eq, ax_eq = plt.subplots(figsize=(8, 3))
        ax_eq.axis("off")
        leg = ax_eq.legend(eq_h, eq_l, loc="center",
                           fontsize=base_fontsize * 0.85, frameon=False,
                           labelspacing=1.2, handlelength=3.0)
        for txt in leg.get_texts():
            txt.set_linespacing(1.6)
        fig_eq.savefig(os.path.join(outdir, f"fit_{name}_equations.svg"),
                       dpi=dpi, bbox_inches="tight")
        plt.close(fig_eq)
