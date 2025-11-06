import argparse
import os
import re
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Config fallbacks (optional)
# ---------------------------
# If you have FAIR_STRATEGY in config.py, we’ll try to import it; otherwise default to 1 (mini-max)
try:
    from config import FAIR_STRATEGY
except Exception:
    FAIR_STRATEGY = 1  # 1=minimax, 2=avgFair, 3=propFair


# ---------------------------
# Helpers
# ---------------------------
def latest_run_dir(base="logs"):
    """Find latest timestamped run directory under logs/."""
    p = Path(base)
    if not p.exists():
        return None
    # run_YYYYMMDD_HHMMSS
    runs = sorted([d for d in p.iterdir() if d.is_dir() and d.name.startswith("run_")])
    return runs[-1] if runs else None

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def list_ap_counts(scheme_dir, scheme):
    """
    Look for acc*.txt files and extract AP count (e.g., accfixedMLOdataSetAPs8.txt -> 8).
    Returns a sorted list of ints.
    """
    if not Path(scheme_dir).exists():
        return []
    if scheme == "fixed":
        pattern = "accfixedMLOdataSetAPs*.txt"
    elif scheme == "random":
        pattern = "accRandMLOdataSetAPs*.txt"
    elif scheme == "rl":
        pattern = "accRlMLOdataSetAPs*.txt"
    elif scheme == "frl":
        # any of the three fairness prefixes; we’ll unify later
        pattern = "acc*FrlMLOdataSetAPs*.txt"
    else:
        return []

    ap_counts = set()
    for f in Path(scheme_dir).glob(pattern):
        m = re.search(r"APs(\d+)\.txt$", f.name)
        if m:
            ap_counts.add(int(m.group(1)))
    return sorted(ap_counts)

def frl_prefixes(fair_strategy):
    if fair_strategy == 1:
        return "accMiniMaxFrlMLOdataSetAPs", "cdfMiniMaxFrlMLOdataSetAPs"
    elif fair_strategy == 2:
        return "accAvgFairFrlMLOdataSetAPs", "cdfAvgFairFrlMLOdataSetAPs"
    else:
        return "accPropFairFrlMLOdataSetAPs", "cdfPropFairFrlMLOdataSetAPs"

def load_acc(scheme_dir, scheme, ap_count, fair_strategy):
    """
    Load acc matrices (MaxSim x N_APs). Returns ndarray or None if missing.
    """
    if scheme == "fixed":
        fname = f"accfixedMLOdataSetAPs{ap_count}.txt"
    elif scheme == "random":
        fname = f"accRandMLOdataSetAPs{ap_count}.txt"
    elif scheme == "rl":
        fname = f"accRlMLOdataSetAPs{ap_count}.txt"
    else:  # frl
        prefix, _ = frl_prefixes(fair_strategy)
        fname = f"{prefix}{ap_count}.txt"

    fpath = Path(scheme_dir) / fname
    if not fpath.exists():
        return None
    return np.loadtxt(fpath)

def load_cdf(scheme_dir, scheme, ap_count, fair_strategy):
    """
    Load cdf matrices (MaxIter x (MaxSim*N_APs)). Returns ndarray or None.
    """
    if scheme == "fixed":
        fname = f"cdfFixedMLOdataSetAPs{ap_count}.txt"
    elif scheme == "random":
        fname = f"cdfRandMLOdataSetAPs{ap_count}.txt"
    elif scheme == "rl":
        fname = f"cdfRlMLOdataSetAPs{ap_count}.txt"
    else:  # frl
        _, prefix_cdf = frl_prefixes(fair_strategy)
        fname = f"{prefix_cdf}{ap_count}.txt"

    fpath = Path(scheme_dir) / fname
    if not fpath.exists():
        return None
    return np.loadtxt(fpath)

def cdf_min_rate_vector(cdf_mat, n_aps):
    """
    MATLAB logic:
      - cdf_mat: (MaxIter x (MaxSim * N_APs))
      - Take mean over iterations -> shape (MaxSim*N_APs,)
      - Reshape to (N_APs, MaxSim)
      - Take min across APs -> vector (MaxSim,)
    """
    if cdf_mat.ndim != 2:
        return None
    max_iter, cols = cdf_mat.shape
    if cols % n_aps != 0:
        return None
    max_sim = cols // n_aps

    mean_over_time = cdf_mat.mean(axis=0)              # (MaxSim*N_APs,)
    mat = mean_over_time.reshape(n_aps, max_sim)       # (N_APs, MaxSim)
    min_per_realization = mat.min(axis=0)              # (MaxSim,)
    return min_per_realization

def temporal_moving_average_first_realization(cdf_mat, n_aps):
    """
    Take the first realization columns [0 .. N_APs-1].
    For each iteration, avg across APs -> series (MaxIter,)
    Then cumulative moving average over iterations.
    """
    max_iter, cols = cdf_mat.shape
    if cols < n_aps:
        return None
    first_real_cols = cdf_mat[:, 0:n_aps]                 # (MaxIter x N_APs)
    per_iter_avg = first_real_cols.mean(axis=1)           # (MaxIter,)
    cumsum = np.cumsum(per_iter_avg)
    t = np.arange(1, len(per_iter_avg) + 1, dtype=float)
    moving_avg = cumsum / t
    return moving_avg

def per_ap_means(acc_mat):  # acc_mat: (MaxSim x N_APs)
    """Mean per AP across realizations + overall average across APs."""
    if acc_mat.ndim != 2:
        return None, None
    means_per_ap = acc_mat.mean(axis=0)          # (N_APs,)
    overall = means_per_ap.mean()
    return means_per_ap, overall


# ---------------------------
# Plotters
# ---------------------------
def plot_cdf_min_rate(fig_dir, ap_count, data_by_scheme):
    """
    data_by_scheme: dict {scheme_name: vector_of_min_rates}
    """
    if not data_by_scheme:
        return
    plt.figure()
    # Use a simple ECDF approximation
    for label, vec in data_by_scheme.items():
        if vec is None or len(vec) == 0:
            continue
        x = np.sort(vec)
        y = np.arange(1, len(x) + 1) / float(len(x))
        plt.plot(x, y, label=label)
    plt.xlabel("Ach. data rate (Mbps)")
    plt.ylabel("Empirical CDF")
    plt.legend()
    out = Path(fig_dir) / f"cdf_min_rate_APs{ap_count}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def plot_temporal(fig_dir, ap_count, ma_by_scheme):
    """
    ma_by_scheme: dict {scheme_name: moving_avg_series}
    """
    if not ma_by_scheme:
        return
    plt.figure()
    for label, series in ma_by_scheme.items():
        if series is None or len(series) == 0:
            continue
        plt.plot(series, label=label)
    plt.ylabel("Ach. data rate (Mbps)")
    plt.xlabel("Simulation time (iterations)")
    plt.legend()
    out = Path(fig_dir) / f"temporal_moving_avg_APs{ap_count}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def plot_bars(fig_dir, ap_count, per_ap_by_scheme, single_real_by_scheme=None):
    """
    per_ap_by_scheme: dict {scheme_name: vector per-AP means}
    single_real_by_scheme: dict {scheme_name: vector per-AP means for first realization} (optional)
    """
    if not per_ap_by_scheme:
        return
    labels = sorted(per_ap_by_scheme.keys())
    # per-AP bar chart (stacked by scheme as grouped bars)
    # Align on AP index
    apset = None
    for v in per_ap_by_scheme.values():
        apset = len(v)
        break
    if apset is None:
        return

    x = np.arange(apset)
    width = 0.8 / max(1, len(labels))

    plt.figure()
    for i, lab in enumerate(labels):
        y = per_ap_by_scheme[lab]
        if y is None:
            continue
        plt.bar(x + i*width, y, width=width, label=lab)
    plt.xticks(x + (len(labels)-1)*width/2.0, [f"AP{k+1}" for k in range(apset)], rotation=0)
    plt.xlabel("AP")
    plt.ylabel("Avg. achieved rate (Mbps)")
    plt.legend()
    out = Path(fig_dir) / f"per_ap_bar_APs{ap_count}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    # optional: single-realization bar
    if single_real_by_scheme:
        plt.figure()
        for i, lab in enumerate(labels):
            y = single_real_by_scheme.get(lab)
            if y is None:
                continue
            plt.bar(x + i*width, y, width=width, label=lab)
        plt.xticks(x + (len(labels)-1)*width/2.0, [f"AP{k+1}" for k in range(apset)], rotation=0)
        plt.xlabel("AP (single realization)")
        plt.ylabel("Avg. achieved rate (Mbps)")
        plt.legend()
        out = Path(fig_dir) / f"per_ap_bar_single_real_APs{ap_count}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

def plot_density(fig_dir, ap_counts, overall_by_scheme):
    """
    overall_by_scheme: dict {scheme_name: {ap_count: overall_avg_rate}}
    """
    if not overall_by_scheme:
        return
    plt.figure()
    for label, mapping in overall_by_scheme.items():
        xs, ys = [], []
        for ap in sorted(mapping.keys()):
            xs.append(ap)
            ys.append(mapping[ap])
        if xs:
            plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("Number of APs (network density)")
    plt.ylabel("Ach. data rate (Mbps)")
    plt.legend()
    out = Path(fig_dir) / "density_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


# ---------------------------
# Main logic
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate figures from logged simulation data.")
    parser.add_argument("--logdir", type=str, default=None, help="Path to logs/run_YYYYMMDD_HHMMSS/")
    parser.add_argument("--fair", type=int, default=None, choices=[1,2,3], help="FRL fairness strategy (1:minimax, 2:avg, 3:prop). Default = config.FAIR_STRATEGY.")
    args = parser.parse_args()

    logdir = args.logdir
    if logdir is None:
        lr = latest_run_dir("logs")
        if lr is None:
            print("No logs found under logs/. Nothing to plot.")
            return
        logdir = str(lr)

    fair = args.fair if args.fair is not None else FAIR_STRATEGY

    fixed_dir = Path(logdir) / "fixed"
    random_dir = Path(logdir) / "random"
    rl_dir = Path(logdir) / "rl"
    frl_dir = Path(logdir) / "frl"

    fig_dir = Path(logdir) / "figures"
    ensure_dir(fig_dir)

    available = {
        "Fixed": (fixed_dir.exists(), fixed_dir),
        "Random": (random_dir.exists(), random_dir),
        "RL": (rl_dir.exists(), rl_dir),
        "FRL": (frl_dir.exists(), frl_dir),
    }

    # Discover AP counts per scheme
    ap_counts_global = set()
    ap_counts_by_scheme = {}
    for label, (exists, d) in available.items():
        if not exists:
            ap_counts_by_scheme[label] = []
            continue
        scheme_key = label.lower()
        cps = list_ap_counts(d, scheme_key)
        ap_counts_by_scheme[label] = cps
        ap_counts_global.update(cps)

    if not ap_counts_global:
        print(f"No acc*.txt files found under {logdir}. Nothing to plot.")
        return

    ap_counts_global = sorted(ap_counts_global)

    # Build density plot data
    density_overall = { "Fixed": {}, "Random": {}, "RL": {}, "FRL": {} }

    # For each AP count, attempt all figures that we can with available data
    for ap in ap_counts_global:
        # CDF min-rate vectors
        cdf_min_by_scheme = {}
        temporal_ma_by_scheme = {}
        per_ap_means_by_scheme = {}
        single_real_by_scheme = {}

        # Iterate schemes
        for label, (exists, d) in available.items():
            if not exists:
                continue
            if ap not in ap_counts_by_scheme[label]:
                # This scheme wasn't run for this AP count
                continue

            scheme_key = label.lower()
            acc = load_acc(d, scheme_key, ap, fair)
            cdf = load_cdf(d, scheme_key, ap, fair)

            # Per-AP means + overall (for density curve)
            if acc is not None and acc.size > 0:
                means_per_ap, overall = per_ap_means(acc)
                per_ap_means_by_scheme[label] = means_per_ap
                density_overall[label][ap] = overall

                # "Single realization" bar = mean over iterations from first realization in CDF
                # If CDF available, approximate single-real per-AP mean as:
                #   mean over iterations of columns [0..N_APs-1], per AP.
                if cdf is not None and cdf.ndim == 2 and cdf.shape[1] >= ap:
                    single_first_real = cdf[:, 0:ap].mean(axis=0)  # (N_APs,)
                    single_real_by_scheme[label] = single_first_real

            # CDF min and temporal moving average (first realization)
            if cdf is not None and cdf.size > 0:
                cdf_min_by_scheme[label] = cdf_min_rate_vector(cdf, ap)
                temporal_ma_by_scheme[label] = temporal_moving_average_first_realization(cdf, ap)

        # Plot per-AP bars (if we have anything)
        if per_ap_means_by_scheme:
            plot_bars(fig_dir, ap, per_ap_means_by_scheme, single_real_by_scheme)

        # Plot CDF of min AP rate (only includes present schemes)
        if cdf_min_by_scheme:
            plot_cdf_min_rate(fig_dir, ap, cdf_min_by_scheme)

        # Temporal moving average
        if temporal_ma_by_scheme:
            plot_temporal(fig_dir, ap, temporal_ma_by_scheme)

    # Density curve across AP counts
    # Remove empty scheme entries
    density_overall = {k: v for k, v in density_overall.items() if v}
    if density_overall:
        plot_density(fig_dir, ap_counts_global, density_overall)

    print(f"Figures saved to: {fig_dir}")

if __name__ == "__main__":
    main()
