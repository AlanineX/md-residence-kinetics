"""Region setup, validation, and IO helpers."""

import os
import csv
import numpy as np
import MDAnalysis as mda

# Canonical RegionSpec lives in kinetics.region (MDA-free).
from kinetics.region import RegionSpec  # re-export for backwards compat
__all__ = ["RegionSpec"]  # everything else exported below is added by name


# ── Universe helpers ─────────────────────────────────────────────────────────

def detect_chain_kw(u):
    try:
        _ = u.residues[0].chainID
        return "chainid"
    except AttributeError:
        pass
    if any(len(r.segid.strip()) > 0 for r in u.residues):
        return "segid"
    return None


def print_protein_info(u, chain_kw):
    prot = u.select_atoms("protein").residues
    if len(prot) == 0:
        print("Protein info: no protein residues found.")
        return

    def _label(res):
        if chain_kw == "chainid":
            return getattr(res, "chainID", "").strip() or "?"
        if chain_kw == "segid":
            return getattr(res, "segid", "").strip() or "?"
        return "?"

    chain_map = {}
    for r in prot:
        chain_map.setdefault(_label(r), []).append(r.resid)

    if chain_kw in ("chainid", "segid"):
        for label in sorted(chain_map.keys()):
            resids = chain_map[label]
            print(f"Protein chain {label}: resid range {min(resids)}-{max(resids)} (n={len(resids)})")
    else:
        resids = [r.resid for r in prot]
        print(f"Protein: resid range {min(resids)}-{max(resids)} (n={len(resids)})")


# ── Selection building ───────────────────────────────────────────────────────

def build_core_selection(config, chain_kw, segids, available_chains=None):
    core = config.get("core", None)
    resid_ranges = config.get("resid_ranges", None)
    chains = config.get("chains", None)

    if chains == "ABCD":
        chains = segids

    if core is None and resid_ranges:
        resid_part = " or ".join(f"resid {r}" for r in resid_ranges)
        chain_part = ""
        if chains:
            if isinstance(chains, list):
                labels = [str(c) for c in chains]
            else:
                chains = str(chains)
                if available_chains is not None and chains in available_chains:
                    labels = [chains]
                else:
                    labels = list(chains)

            if available_chains is not None:
                labels = [c for c in labels if c in available_chains]
                if not labels:
                    print(f"Warning: target {config.get('name', '(unnamed)')} "
                          f"requested {chain_kw}={chains} but none found; ignoring chain filter.")

            if labels:
                if chain_kw == "chainid":
                    cond = " or ".join(f"chainid {c}" for c in labels)
                elif chain_kw == "segid":
                    cond = " or ".join(f"segid {c}" for c in labels)
                else:
                    cond = ""
                if cond:
                    chain_part = f" and ({cond})"

        core = f"(protein and ({resid_part}){chain_part})"
    elif core is None:
        core = "protein"

    return core


# ── Per-residue helpers ──────────────────────────────────────────────────────

def get_protein_chains(u, chain_kw):
    """Return {chain_label: [resid, ...]} sorted by resid, protein only."""
    chains = {}
    for r in u.select_atoms("protein").residues:
        if chain_kw == "segid":
            label = r.segid.strip()
        else:
            label = getattr(r, "chainID", "").strip()
        if not label:
            continue
        chains.setdefault(label, []).append(r.resid)
    for label in chains:
        chains[label].sort()
    return chains


def parse_residue_range(spec):
    """Parse per-residue spec into list of (start, end) 0-based position ranges.

    False/None -> []
    True       -> [(0, 999999)]
    "1:50"     -> [(1, 50)]
    "1:50,100:150" -> [(1, 50), (100, 150)]
    """
    if not spec:
        return []
    if spec is True:
        return [(0, 999999)]
    ranges = []
    for part in str(spec).split(","):
        part = part.strip()
        if ":" in part:
            a, b = part.split(":", 1)
            ranges.append((int(a), int(b)))
        else:
            v = int(part)
            ranges.append((v, v))
    return ranges


def position_in_range(pos, ranges):
    return any(a <= pos <= b for a, b in ranges)


# ── Region construction ──────────────────────────────────────────────────────

def make_regions(top, traj, target_configs, out_dir,
                 cutoff_a=3.5, water_o_selection="name OW OH2",
                 calc_protein_shell=False,
                 protein_shell_settings=None,
                 calc_per_residue_shell=None, per_residue_settings=None):
    os.makedirs(out_dir, exist_ok=True)
    u = mda.Universe(top, traj, refresh_offsets=True, to_guess=())
    chain_kw = detect_chain_kw(u)

    segids = list(set(
        r.segid for r in u.residues
        if r.resname != "SOL" and len(r.segid.strip()) > 0
    ))
    print(f"Detected segids: {segids}")
    print_protein_info(u, chain_kw)

    available_chains = None
    if chain_kw == "chainid":
        available_chains = {getattr(r, "chainID", "").strip() for r in u.residues} - {""}
    elif chain_kw == "segid":
        available_chains = {r.segid.strip() for r in u.residues} - {""}

    regions = []
    for config in target_configs:
        name = config["name"]
        probe_type = config.get("probe_type", "water")
        core = build_core_selection(config, chain_kw, segids,
                                    available_chains=available_chains)
        desc = config.get("description", "")

        if probe_type == "water":
            sel = f"byres ({water_o_selection} and around {cutoff_a:.1f} {core})"
            static_sel = core
            mobile_sel = water_o_selection
        elif probe_type == "solute":
            resnames = config.get("resnames", [])
            rn = " ".join(resnames) if isinstance(resnames, list) else str(resnames)
            sel = f"byres (resname {rn} and around {cutoff_a:.1f} ({core}))"
            static_sel = core
            mobile_sel = f"resname {rn}"
        else:
            raise ValueError(f"Unknown probe_type '{probe_type}' in config {name}")

        sel_atoms = u.select_atoms(sel)
        n_atoms = sel_atoms.n_atoms
        n_res = sel_atoms.residues.n_residues
        if desc:
            print(f"Target {name}: {desc}")
        print(f"Selection for {name}: {sel} | atoms={n_atoms}, residues={n_res}")

        if n_atoms == 0:
            if probe_type == "water":
                if u.select_atoms(water_o_selection).n_atoms == 0:
                    raise ValueError(
                        f"Selection for {name} is empty and no atoms match "
                        f"'{water_o_selection}'. Provide topology with water or "
                        f"adjust WATER_O_SELECTION.")
            elif probe_type == "solute":
                rn = " ".join(config.get("resnames", [])) if isinstance(
                    config.get("resnames", []), list) else str(config.get("resnames", ""))
                if u.select_atoms(f"resname {rn}").n_atoms == 0:
                    raise ValueError(
                        f"Selection for {name} is empty and no atoms match "
                        f"'resname {rn}'. Check spelling/case.")
            # Empty at reference frame is OK for dynamic selections —
            # probes may visit during the trajectory. Warn and include.
            print(f"  WARNING: Selection for {name} is empty at reference frame "
                  f"(probes may visit during trajectory).")

        regions.append(RegionSpec(
            name, sel,
            os.path.join(out_dir, f"sp_{name}.csv"),
            os.path.join(out_dir, f"summary_{name}.txt"),
            description=desc,
            static_sel=static_sel,
            mobile_sel=mobile_sel,
            cutoff_a=float(cutoff_a),
            time_resolution_ns=float(config.get("time_resolution_ns", 0.01)),
            tau_max_ns=float(config.get("tau_max_ns", 25.0)),
            t0_spacing_ns=float(config.get("t0_spacing_ns", 0.5)),
        ))

    if calc_protein_shell:
        ps = protein_shell_settings or {}
        # mobile_sel defaults to water oxygens but can be overridden
        # (e.g. "resname ADP" for an ADP-only trajectory)
        ps_mobile = ps.get("mobile_sel", water_o_selection)
        # Exclude mobile from the static "protein" set so a cosolute whose
        # resname is in MDAnalysis's protein-keyword list (e.g. CHARMM "ACE"
        # = acetyl N-terminus cap, AMBER's NMA, etc.) doesn't count its own
        # neighbors as protein contacts. For mobile_sels that don't match
        # the protein keyword this is a no-op.
        ps_static = ps.get("static_sel", f"(protein and not ({ps_mobile}))")
        sel_ps = f"byres ({ps_mobile} and around {cutoff_a:.1f} ({ps_static}))"
        sel_atoms = u.select_atoms(sel_ps)
        print(f"Target protein_shell: {ps_mobile} within cutoff of protein.")
        print(f"Selection for protein_shell: {sel_ps} | "
              f"atoms={sel_atoms.n_atoms}, residues={sel_atoms.residues.n_residues}")
        if sel_atoms.n_atoms == 0:
            # Topology sanity check — does the probe species exist at all?
            # If 0 atoms match `ps_mobile` anywhere, it's a typo / wrong topology,
            # not a transient empty-at-frame-0 condition.
            if u.select_atoms(ps_mobile).n_atoms == 0:
                raise ValueError(
                    f"Selection for protein_shell is empty and no atoms match "
                    f"mobile_sel='{ps_mobile}'. Check spelling/case, or set "
                    f"CALC_PROTEIN_SHELL = False.")
            # Probe atoms exist but none happen to be within cutoff of protein
            # at the reference frame. This is OK for dynamic selections — they
            # may visit during the trajectory (matches cavity-block behavior).
            print(f"  WARNING: protein_shell empty at reference frame "
                  f"({ps_mobile} present in topology but none within "
                  f"{cutoff_a:.1f} Å of protein at frame 0; "
                  f"probes may visit during trajectory).")
        regions.append(RegionSpec(
            "protein_shell", sel_ps,
            os.path.join(out_dir, "sp_protein_shell.csv"),
            os.path.join(out_dir, "summary_protein_shell.txt"),
            description=f"{ps_mobile} within cutoff of protein (global shell).",
            static_sel=ps_static,
            mobile_sel=ps_mobile,
            cutoff_a=float(cutoff_a),
            time_resolution_ns=float(ps.get("time_resolution_ns", 0.01)),
            tau_max_ns=float(ps.get("tau_max_ns", 5.0)),
            t0_spacing_ns=float(ps.get("t0_spacing_ns", 0.5)),
        ))

    # Per-residue solvation shell
    res_ranges = parse_residue_range(calc_per_residue_shell)
    if res_ranges:
        prs = per_residue_settings or {}
        chains = get_protein_chains(u, chain_kw)
        if not chains:
            print("WARNING: No protein chains detected for per-residue analysis.")
        else:
            # Build position -> [(chain_label, resid), ...] mapping
            pos_map = {}  # {position: [(chain, resid), ...]}
            for label, resids in chains.items():
                min_resid = resids[0]
                for rid in resids:
                    pos = rid - min_resid  # 0-based position within chain
                    pos_map.setdefault(pos, []).append((label, rid))

            per_res_dir = os.path.join(out_dir, "per_residue")
            os.makedirs(per_res_dir, exist_ok=True)
            n_added = 0
            for pos in sorted(pos_map):
                if not position_in_range(pos, res_ranges):
                    continue
                equiv = pos_map[pos]  # [(chain, resid), ...]
                # Build combined selection across all equivalent residues
                if chain_kw == "segid":
                    parts = [f"(resid {rid} and segid {cl})"
                             for cl, rid in equiv]
                else:
                    parts = [f"(resid {rid} and chainid {cl})"
                             for cl, rid in equiv]
                core = " or ".join(parts)
                sel = (f"byres ({water_o_selection} and "
                       f"around {cutoff_a:.1f} ({core}))")

                # Get resname from first chain for labeling
                ref_resid = equiv[0][1]
                ref_res = u.select_atoms(f"resid {ref_resid} and protein")
                resname = ref_res.residues[0].resname if ref_res.n_atoms > 0 else "UNK"
                name = f"res_{pos}_{resname}"

                regions.append(RegionSpec(
                    name, sel,
                    os.path.join(per_res_dir, f"sp_{name}.csv"),
                    os.path.join(per_res_dir, f"summary_{name}.txt"),
                    description=(f"Water shell of {resname} pos {pos} "
                                 f"({len(equiv)} chains combined)."),
                    static_sel=f"({core})",
                    mobile_sel=water_o_selection,
                    cutoff_a=float(cutoff_a),
                    time_resolution_ns=float(prs.get("time_resolution_ns", 0.01)),
                    tau_max_ns=float(prs.get("tau_max_ns", 5.0)),
                    t0_spacing_ns=float(prs.get("t0_spacing_ns", 0.5)),
                ))
                n_added += 1
            print(f"Per-residue solvation: {n_added} positions, "
                  f"{len(chains)} chains each.")

    # Return universe to caller to avoid re-loading (MDA XTC reader bug)
    return regions, u


# ── Settings validation ──────────────────────────────────────────────────────

def validate_and_compute_settings(regions, dt_traj_ns, n_frames_total):
    traj_length_ns = n_frames_total * dt_traj_ns
    warnings_list = []
    errors_list = []

    print("\n" + "=" * 70)
    print("SETTINGS VALIDATION")
    print("=" * 70)
    print(f"Trajectory: {n_frames_total} frames, dt={dt_traj_ns:.4f} ns/frame, "
          f"total={traj_length_ns:.1f} ns")
    print("-" * 70)

    for region in regions:
        print(f"\n[{region.name}]")

        stride = max(1, round(region.time_resolution_ns / dt_traj_ns))
        actual_res = stride * dt_traj_ns
        tau_max_frames = max(1, round(region.tau_max_ns / actual_res))
        t0_step = max(1, round(region.t0_spacing_ns / actual_res))

        n_strided = len(range(0, n_frames_total, stride))
        valid_origin_frames = max(0, n_strided - tau_max_frames)
        valid_origin_range_ns = valid_origin_frames * actual_res
        n_origins = valid_origin_frames // t0_step if t0_step > 0 else 0

        region.stride = stride
        region.tau_max_frames = tau_max_frames
        region.t0_step = t0_step
        region.actual_resolution_ns = actual_res
        region.n_origins_estimated = n_origins
        region.valid_origin_range_ns = valid_origin_range_ns

        print(f"  User settings (ns):")
        print(f"    time_resolution_ns = {region.time_resolution_ns} "
              f"-> stride = {stride} (actual: {actual_res:.4f} ns)")
        print(f"    tau_max_ns = {region.tau_max_ns} -> tau_max_frames = {tau_max_frames}")
        print(f"    t0_spacing_ns = {region.t0_spacing_ns} -> t0_step = {t0_step}")
        print(f"  Derived:")
        print(f"    Valid origin range: 0 to {valid_origin_range_ns:.1f} ns "
              f"({valid_origin_frames} strided frames)")
        print(f"    Estimated time origins: {n_origins}")

        if actual_res != region.time_resolution_ns:
            msg = (f"WARNING: Requested resolution {region.time_resolution_ns} ns "
                   f"not achievable; using {actual_res:.4f} ns")
            print(f"  {msg}")
            warnings_list.append(f"[{region.name}] {msg}")

        if region.tau_max_ns > traj_length_ns / 2:
            msg = (f"WARNING: tau_max_ns ({region.tau_max_ns}) > trajectory/2 "
                   f"({traj_length_ns / 2:.1f} ns) - very few time origins!")
            print(f"  {msg}")
            warnings_list.append(f"[{region.name}] {msg}")
        elif region.tau_max_ns > traj_length_ns / 3:
            msg = (f"WARNING: tau_max_ns ({region.tau_max_ns}) > trajectory/3 "
                   f"({traj_length_ns / 3:.1f} ns) - may have poor statistics")
            print(f"  {msg}")
            warnings_list.append(f"[{region.name}] {msg}")

        if n_origins < 20:
            msg = f"ERROR: Only {n_origins} time origins - need at least 20!"
            print(f"  {msg}")
            errors_list.append(f"[{region.name}] {msg}")
        elif n_origins < 50:
            msg = (f"WARNING: Only {n_origins} time origins - "
                   f"consider smaller t0_spacing_ns")
            print(f"  {msg}")
            warnings_list.append(f"[{region.name}] {msg}")

        if valid_origin_frames <= 0:
            msg = (f"ERROR: No valid time origins! tau_max_ns ({region.tau_max_ns}) "
                   f"too large for trajectory ({traj_length_ns:.1f} ns)")
            print(f"  {msg}")
            errors_list.append(f"[{region.name}] {msg}")

    print("\n" + "-" * 70)
    if errors_list:
        print("\nERRORS FOUND:")
        for err in errors_list:
            print(f"  {err}")
        raise ValueError(
            f"Settings validation failed with {len(errors_list)} error(s).")
    if warnings_list:
        print(f"\n{len(warnings_list)} warning(s) - calculations will proceed.")
    else:
        print("\nAll settings validated successfully.")
    print("=" * 70 + "\n")


# ── IO helpers ───────────────────────────────────────────────────────────────

def save_sp_csv(tau_ns, S, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Atomic write: write to temp file then rename, so partial writes
    # from a killed job never leave a corrupt CSV at the final path.
    tmp_path = path + ".tmp"
    np.savetxt(tmp_path, np.column_stack([tau_ns, S]),
               delimiter=",", header="tau_ns,SP", comments="", fmt="%.8f")
    os.replace(tmp_path, path)


def write_run_log(regions, out_dir, top_path, traj_path,
                  start_frame, stop_frame, intermittency,
                  first_shell_a, water_o_selection, do_exp_fit, n_procs,
                  dt_ns=None, n_frames=None):
    log_path = os.path.join(out_dir, "run_settings.log")
    traj_length_ns = n_frames * dt_ns if (n_frames and dt_ns) else None

    lines = [
        "=" * 60, "RUN SETTINGS LOG", "=" * 60, "",
        "Trajectory:",
        f"  TOP_PATH = {top_path}",
        f"  TRAJ_PATH = {traj_path}",
        f"  n_frames = {n_frames}",
        f"  dt(traj) = {dt_ns:.6f} ns/frame" if dt_ns else "  dt(traj) = unknown",
        f"  total_length = {traj_length_ns:.2f} ns" if traj_length_ns else "  total_length = unknown",
        "",
        "Global settings:",
        f"  OUT_DIR = {out_dir}",
        f"  START_FRAME = {start_frame}",
        f"  STOP_FRAME = {stop_frame}",
        f"  INTERMITTENCY = {intermittency}",
        f"  FIRST_SHELL_A = {first_shell_a}",
        f"  WATER_O_SELECTION = {water_o_selection}",
        f"  DO_EXP_FIT = {do_exp_fit}",
        f"  N_PROCS = {n_procs}",
        "", "=" * 60, "PER-TARGET SETTINGS", "=" * 60,
    ]

    for r in regions:
        lines += [
            "", f"[{r.name}]",
            f"  Description: {r.description}", "",
            "  User settings (ns):",
            f"    time_resolution_ns = {r.time_resolution_ns}",
            f"    tau_max_ns = {r.tau_max_ns}",
            f"    t0_spacing_ns = {r.t0_spacing_ns}", "",
            "  Computed (frames):",
            f"    stride = {r.stride}",
            f"    tau_max_frames = {r.tau_max_frames}",
            f"    t0_step = {r.t0_step}",
            f"    actual_resolution_ns = {r.actual_resolution_ns:.6f}", "",
            "  Validation:",
            f"    valid_origin_range_ns = {r.valid_origin_range_ns:.2f}",
            f"    n_origins_estimated = {r.n_origins_estimated}",
        ]

    with open(log_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_summary(region, sp_result, fit1, fit2, out_path, do_exp_fit=True):
    """Write per-region text summary with SP results and fit parameters."""
    start = sp_result["start_frame"]
    stop = sp_result["stop_frame"]
    S = sp_result["S"]
    counts = sp_result["counts"]
    dt_ns = sp_result["dt_ns"]
    stride = region.stride

    sp_end = float(S[-1]) if S.size else float("nan")
    n_tau_end = int(counts[-1]) if counts.size else 0
    c_est = float(S[-1]) if S.size else 0.0
    tau_res_ns = float(np.trapezoid(S - c_est, sp_result["tau_ns"]))

    lines = [
        f"[{region.name}] frames {start}:{stop} (exclusive)",
    ]
    if region.description:
        lines.append(f"Description: {region.description}")

    lines += [
        "",
        "User settings (ns):",
        f"  time_resolution_ns = {region.time_resolution_ns}",
        f"  tau_max_ns = {region.tau_max_ns}",
        f"  t0_spacing_ns = {region.t0_spacing_ns}",
        "",
        "Computed (frames):",
        f"  stride = {stride}",
        f"  tau_max_frames = {region.tau_max_frames}",
        f"  t0_step = {region.t0_step}",
        f"  dt(traj) = {dt_ns:.6f} ns/frame",
        f"  dt(analysis) = {region.actual_resolution_ns:.6f} ns/frame",
        "",
        "Results:",
        f"  SP(tau_max) = {sp_end:.6f} (N0-weighted count@tau_max={n_tau_end})",
        f"  Residence time (integral): {tau_res_ns:.6f} ns",
    ]

    avg = sp_result["avg_residues"]
    if np.isfinite(avg):
        lines.append(
            f"  Average # selected residues: {avg:.1f} +/- {sp_result['std_residues']:.1f} "
            f"(n={sp_result['n_count_frames']}, step={sp_result['count_step']})")
    else:
        lines.append("  Average # selected residues: (skipped)")

    lines.append(f"  Computation time: {sp_result['time_taken']:.2f} s")

    if fit1 is not None:
        lines += [
            "",
            "Single-exp (alpha + c = 1):",
            "  P(t) = alpha * exp(-t/tau) + c",
            f"  alpha = {fit1['alpha']:.6f} (= 1 - c)",
            f"  tau   = {fit1['tau']:.6f} ns +/- {fit1['perr_tau']:.6f}",
            f"  c     = {fit1['c']:.6f} +/- {fit1['perr_c']:.6f}",
            f"  R^2   = {fit1['r_squared']:.6f}",
        ]
        if fit1["apparent_res_time"] is not None:
            lines.append(f"  Apparent residence time = {fit1['apparent_res_time']:.6f} ns")
        lines.append(
            f"  AIC = {fit1['aic']:.2f}, AICc = {fit1['aicc']:.2f}, BIC = {fit1['bic']:.2f}")

    if fit2 is not None:
        lines += [
            "",
            "Bi-exp (alpha1 + alpha2 + c = 1):",
            "  P(t) = alpha1 * exp(-t/tau1) + alpha2 * exp(-t/tau2) + c",
            f"  alpha1 = {fit2['alpha1']:.6f} (= u*(1-c), u={fit2['u']:.4f})",
            f"  tau1   = {fit2['tau1']:.6f} ns +/- {fit2['perr_tau1']:.6f}",
            f"  alpha2 = {fit2['alpha2']:.6f} (= (1-u)*(1-c))",
            f"  tau2   = {fit2['tau2']:.6f} ns +/- {fit2['perr_tau2']:.6f}",
            f"  c      = {fit2['c']:.6f} +/- {fit2['perr_c']:.6f}",
            f"  R^2    = {fit2['r_squared']:.6f}",
        ]
        if fit2["apparent_res_time"] is not None:
            lines.append(f"  Apparent residence time = {fit2['apparent_res_time']:.6f} ns")
        lines.append(f"  Fitted residence time  = {fit2['fitted_res_time']:.6f} ns")
        lines.append(f"  Half-life (fast)    = {fit2['t_half_fast']:.6f} ns")
        lines.append(f"  Half-life (slow)    = {fit2['t_half_slow']:.6f} ns")
        if fit2["t_half_overall"] is not None:
            lines.append(f"  Half-life (overall) = {fit2['t_half_overall']:.6f} ns")
        lines.append(
            f"  AIC = {fit2['aic']:.2f}, AICc = {fit2['aicc']:.2f}, BIC = {fit2['bic']:.2f}")

    if do_exp_fit and fit1 is None and fit2 is None:
        lines.append("\nAll fits failed.")
    elif not do_exp_fit:
        lines.append("\nFitting skipped (DO_EXP_FIT=False).")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_aggregate_csvs(all_results, output_dir, count_data=None,
                         model_free_data=None):
    """Write single_exp and bi_exp aggregate CSVs from list of (name, fit1, fit2).

    count_data: optional dict {name: (avg_residues, std_residues)} for occupancy columns.
    model_free_data: optional dict {name: dict} from fitting.model_free_metrics().
    """
    os.makedirs(output_dir, exist_ok=True)
    if count_data is None:
        count_data = {}
    if model_free_data is None:
        model_free_data = {}

    # Collect model-free field names from first entry
    mf_fields = []
    for v in model_free_data.values():
        mf_fields = sorted(v.keys())
        break

    single_fields = [
        "Region", "avg_count", "std_count",
        "alpha", "tau", "c", "perr_tau", "perr_c",
        "R2", "apparent_res_time",
    ] + mf_fields + ["AIC", "AICc", "BIC"]

    bi_fields = [
        "Region", "avg_count", "std_count",
        "alpha1", "tau1", "alpha2", "tau2", "c", "u",
        "perr_tau1", "perr_tau2", "perr_c", "perr_u",
        "R2", "apparent_res_time", "fitted_res_time",
        "t_half_fast", "t_half_slow", "t_half_overall",
    ] + mf_fields + ["AIC", "AICc", "BIC"]

    single_rows = []
    bi_rows = []

    def _fmt(v, prec=6):
        if v is None:
            return ""
        return f"{v:.{prec}f}"

    for name, fit1, fit2 in all_results:
        avg_c, std_c = count_data.get(name, (None, None))
        mf = model_free_data.get(name, {})
        mf_row = {k: _fmt(v) for k, v in mf.items()}

        if fit1 is not None:
            row = {
                "Region": name,
                "avg_count": _fmt(avg_c, 1),
                "std_count": _fmt(std_c, 1),
                "alpha": _fmt(fit1["alpha"]),
                "tau": _fmt(fit1["tau"]),
                "c": _fmt(fit1["c"]),
                "perr_tau": _fmt(fit1["perr_tau"]),
                "perr_c": _fmt(fit1["perr_c"]),
                "R2": _fmt(fit1["r_squared"]),
                "apparent_res_time": _fmt(fit1["apparent_res_time"]),
                "AIC": _fmt(fit1["aic"], 4),
                "AICc": _fmt(fit1["aicc"], 4),
                "BIC": _fmt(fit1["bic"], 4),
            }
            row.update(mf_row)
            single_rows.append(row)
        if fit2 is not None:
            row = {
                "Region": name,
                "avg_count": _fmt(avg_c, 1),
                "std_count": _fmt(std_c, 1),
                "alpha1": _fmt(fit2["alpha1"]),
                "tau1": _fmt(fit2["tau1"]),
                "alpha2": _fmt(fit2["alpha2"]),
                "tau2": _fmt(fit2["tau2"]),
                "c": _fmt(fit2["c"]),
                "u": _fmt(fit2["u"]),
                "perr_tau1": _fmt(fit2["perr_tau1"]),
                "perr_tau2": _fmt(fit2["perr_tau2"]),
                "perr_c": _fmt(fit2["perr_c"]),
                "perr_u": _fmt(fit2["perr_u"]),
                "R2": _fmt(fit2["r_squared"]),
                "apparent_res_time": _fmt(fit2["apparent_res_time"]),
                "fitted_res_time": _fmt(fit2["fitted_res_time"]),
                "t_half_fast": _fmt(fit2["t_half_fast"]),
                "t_half_slow": _fmt(fit2["t_half_slow"]),
                "t_half_overall": _fmt(fit2["t_half_overall"]),
                "AIC": _fmt(fit2["aic"], 4),
                "AICc": _fmt(fit2["aicc"], 4),
                "BIC": _fmt(fit2["bic"], 4),
            }
            row.update(mf_row)
            bi_rows.append(row)

    if single_rows:
        path = os.path.join(output_dir, "single_exp_fitting_results.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=single_fields)
            w.writeheader()
            w.writerows(single_rows)
        print(f"Single-exp results: {path} ({len(single_rows)} regions)")

    if bi_rows:
        path = os.path.join(output_dir, "bi_exp_fitting_results.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=bi_fields)
            w.writeheader()
            w.writerows(bi_rows)
        print(f"Bi-exp results: {path} ({len(bi_rows)} regions)")
