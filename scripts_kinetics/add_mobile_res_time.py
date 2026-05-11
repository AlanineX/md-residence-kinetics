"""Add `mobile_res_time` column to every fit CSV under results_v4/.

No re-fit needed — the new column is a pure arithmetic combination of
existing alpha1/alpha2/tau1/tau2 (bi-exp) or tau (single-exp). Walks
results_v4/v4_*/{single,bi}_exp_fitting_results.csv and the canonical
method2_* aliases. Skips files that already have the column.

Usage:
  python3 add_mobile_res_time.py                 # all datasets
  python3 add_mobile_res_time.py --dry           # report only, no writes
"""
import argparse
import csv
import io
import shutil
import sys
from pathlib import Path

RESULTS = Path("/home/alan/working/groel_new/results_v4")


def _f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _mobile_bi(row):
    a1, t1 = _f(row.get("alpha1")), _f(row.get("tau1"))
    a2, t2 = _f(row.get("alpha2")), _f(row.get("tau2"))
    if None in (a1, t1, a2, t2):
        return ""
    denom = a1 + a2
    if denom <= 0:
        return ""
    return f"{(a1 * t1 + a2 * t2) / denom:.6f}"


def _mobile_single(row):
    t = _f(row.get("tau"))
    if t is None or t <= 0:
        return ""
    return f"{t:.6f}"


def _patch_csv(path, kind, dry=False):
    """kind in {'single','bi'}. Adds `mobile_res_time` after `fitted_res_time`
    (bi) or after `apparent_res_time` (single). Idempotent."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open() as f:
        rdr = csv.DictReader(f)
        header = list(rdr.fieldnames or [])
        rows = list(rdr)
    if "mobile_res_time" in header:
        return ("skip", 0, "already has mobile_res_time")

    if kind == "bi":
        anchor = "fitted_res_time" if "fitted_res_time" in header \
                 else "apparent_res_time"
        compute = _mobile_bi
    else:
        anchor = "apparent_res_time" if "apparent_res_time" in header \
                 else "tau"
        compute = _mobile_single

    if anchor not in header:
        return ("skip", 0, f"no anchor column '{anchor}' in {path.name}")

    insert_at = header.index(anchor) + 1
    new_header = header[:insert_at] + ["mobile_res_time"] + header[insert_at:]
    for r in rows:
        r["mobile_res_time"] = compute(r)

    if dry:
        return ("dry", len(rows), f"would add at col {insert_at}")

    # Write atomically (write to tmp, then replace).
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_header)
        w.writeheader()
        for r in rows:
            # ensure unknown leftover keys (rare) are dropped to match schema
            w.writerow({k: r.get(k, "") for k in new_header})
    tmp.replace(path)
    return ("patched", len(rows), f"added at col {insert_at}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true",
                    help="report only, don't write")
    ap.add_argument("--results-dir", default=str(RESULTS))
    args = ap.parse_args()
    root = Path(args.results_dir).resolve()

    targets = []
    for d in sorted(root.glob("v4_*")):
        if not d.is_dir():
            continue
        # M1 dataset dir (not _method2): has bi_exp / single_exp at top level
        if not d.name.endswith("_method2"):
            targets += [
                (d / "single_exp_fitting_results.csv", "single"),
                (d / "bi_exp_fitting_results.csv",     "bi"),
            ]
        else:
            # M2 dir: same two filenames + 4 method-specific canonical files
            targets += [
                (d / "single_exp_fitting_results.csv",       "single"),
                (d / "bi_exp_fitting_results.csv",           "bi"),
                (d / "method2_mle_single_exp.csv",           "single"),
                (d / "method2_mle_bi_exp.csv",               "bi"),
                (d / "method2_km_ls_single_exp.csv",         "single"),
                (d / "method2_km_ls_bi_exp.csv",             "bi"),
            ]

    n_total = n_patched = n_skip = 0
    for path, kind in targets:
        n_total += 1
        out = _patch_csv(path, kind, dry=args.dry)
        if out is None:
            print(f"  [miss] {path}")
            continue
        action, nrows, msg = out
        if action == "patched":
            n_patched += 1
            print(f"  [ok  ] {path.relative_to(root)}  ({nrows} rows, {msg})")
        elif action == "dry":
            n_patched += 1
            print(f"  [dry ] {path.relative_to(root)}  ({nrows} rows, {msg})")
        else:
            n_skip += 1
            print(f"  [skip] {path.relative_to(root)}  ({msg})")

    print(f"\n  total scanned: {n_total}, patched: {n_patched}, "
          f"skipped: {n_skip}")


if __name__ == "__main__":
    main()
