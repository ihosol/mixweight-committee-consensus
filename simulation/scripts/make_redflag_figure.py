#!/usr/bin/env python3
"""Red-flag demonstration figure for the committee-capture-risk evaluation method.

Emulated scenario: stake-concentration drift within a monitored top-k coalition in
a committee-based PoS localnet. The figure shows the three phases the method
detects as concentration grows --- safe, acting, violated/infeasible --- and marks
the red-flag (saturation) epoch, with the realised committee-threshold capture
overlaid to confirm the flag is not a false alarm.

The certificate is recomputed at the chosen threshold from the logged coalition
masses (alpha, beta) and realised mixing (lambda), so the script is independent of
the threshold the keeper happened to run with; it works on the theta=1/3 red-flag
suite and as an offline preview on any drift suite.

Usage:
  python3 poc/cosmos/scripts/make_redflag_figure.py \
      --suite article_rbhc_redflag_bft --theta 0.3333 --eps 0.5
"""
import argparse, csv, math, os, statistics as st
from math import comb
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]


def binom_tail(p, q, m):
    p = min(max(p, 0.0), 1.0)
    return sum(comb(m, i) * p**i * (1 - p)**(m - i) for i in range(q, m + 1))


def read_cell(suite, cell):
    """Return per-epoch {epoch: (alpha, beta, lambda)} averaged over seeds, and
    per-epoch realised capture seat counts {epoch: [seat_count, ...]}."""
    root = REPO / "poc" / "cosmos" / "artifacts" / suite / cell
    if not root.exists():
        return {}, {}
    masses = defaultdict(lambda: ([], [], [], []))
    seats = defaultdict(list)
    for seed_dir in sorted(root.glob("seed_*")):
        res = seed_dir / "results"
        sf = res / "epoch_summary_latest.csv"
        if sf.exists():
            for r in csv.DictReader(open(sf)):
                e = int(r["epoch_idx"])
                masses[e][0].append(float(r["mean_risk_alpha_ppm"]) / 1e6)
                masses[e][1].append(float(r["mean_risk_beta_ppm"]) / 1e6)
                masses[e][2].append(float(r["mean_lambda_auto_ppm"]) / 1e6)
                masses[e][3].append(float(r.get("mean_gini_ppm", 0) or 0) / 1e6)
        df = res / "epoch_draws_latest.csv"
        if df.exists():
            rows = list(csv.DictReader(open(df)))
            if not rows:
                continue
            col = "tracked_seats_share" if "tracked_seats_share" in rows[0] else "attacker_seats_share"
            for r in rows:
                try:
                    seats[int(r["epoch_idx"])].append(float(r[col]))
                except (KeyError, ValueError):
                    pass
    m = {e: (st.mean(a), st.mean(b), st.mean(l), st.mean(g))
         for e, (a, b, l, g) in masses.items() if a}
    return m, seats


def main():
    ap = argparse.ArgumentParser(description="Red-flag committee-capture demonstration figure")
    ap.add_argument("--suite", default="article_rbhc_redflag_bft")
    ap.add_argument("--theta", type=float, default=1.0 / 3.0)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--m", type=int, default=9, help="committee size")
    ap.add_argument("--lammax", type=float, default=0.55)
    ap.add_argument("--baseline-cell", default="cc_drift_baseline")
    ap.add_argument("--hybrid-cell", default="cc_drift_hybrid_concave")
    ap.add_argument("--out", default=str(REPO / "papers" / "risk_budget_controller_2026_ua"
                                        / "figures" / "fig_redflag_bft.png"))
    args = ap.parse_args()

    m, theta, eps, lammax = args.m, args.theta, args.eps, args.lammax
    q = max(1, math.ceil(theta * m))

    base_masses, base_seats = read_cell(args.suite, args.baseline_cell)
    hyb_masses, hyb_seats = read_cell(args.suite, args.hybrid_cell)
    if not hyb_masses and not base_masses:
        raise SystemExit(f"No artifacts found under suite '{args.suite}'. Run the suite first.")

    # alpha trajectory is shared; take it from whichever cell is present.
    ref = hyb_masses or base_masses
    epochs = sorted(ref)

    def realised(seats):
        return {e: 100.0 * sum(1 for s in seats[e] if round(s * m) >= q) / len(seats[e])
                for e in seats if seats[e]}

    base_real = realised(base_seats)

    B0, Blam, Bmax, phase = {}, {}, {}, {}
    for e in epochs:
        a, b, _, _g = ref[e]
        B0[e] = binom_tail(a, q, m)
        Bmax[e] = binom_tail((1 - lammax) * a + lammax * b, q, m)
        if e in hyb_masses:
            ha, hb, hl, _hg = hyb_masses[e]
            Blam[e] = binom_tail((1 - hl) * ha + hl * hb, q, m)
        phase[e] = "SAFE" if B0[e] <= eps else ("ACTING" if Bmax[e] <= eps else "VIOLATED")

    alert = next((e for e in epochs if phase[e] == "VIOLATED"), None)

    # Saturation threshold (empirical: coalition mass at the alert epoch); the
    # concentration headroom counts down to zero as the red flag is reached.
    astar = ref[alert][0] if alert is not None else max(ref[e][0] for e in epochs)
    gini = {e: ref[e][3] for e in epochs}
    headroom = {e: max(0.0, astar - ref[e][0]) for e in epochs}

    # ---- console + CSV sidecar ----------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    side = out.with_suffix(".csv")
    print(f"theta={theta:.4f} (q={q}/{m}), eps={eps}, lambda_max={lammax}")
    print(f"{'ep':>3} {'alpha':>6} {'B0':>5} {'Blam':>5} {'Bmax':>5} {'realP':>7} {'phase':>9}")
    with open(side, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "alpha", "gini", "headroom", "B0", "B_lambda",
                    "B_lambdamax", "realised_capture_pct", "phase"])
        for e in epochs:
            a = ref[e][0]
            rp = base_real.get(e, float("nan"))
            w.writerow([e, f"{a:.4f}", f"{gini[e]:.4f}", f"{headroom[e]:.4f}",
                        f"{B0[e]:.4f}", f"{Blam.get(e, float('nan')):.4f}",
                        f"{Bmax[e]:.4f}", f"{rp:.2f}", phase[e]])
            if e in (epochs[0], epochs[len(epochs) // 2], epochs[-1]) or e == alert:
                bl = Blam.get(e, float("nan"))
                print(f"{e:>3} {a:6.3f} {B0[e]:5.2f} {bl:5.2f} {Bmax[e]:5.2f} {rp:6.1f}% {phase[e]:>9}"
                      + ("  <-- RED FLAG" if e == alert else ""))
    if alert is not None:
        print(f"red-flag (saturation) epoch: {alert}")

    # ---- figure --------------------------------------------------------------
    fig, axL = plt.subplots(figsize=(8.2, 4.6))
    pcolor = {"SAFE": "#2ca02c", "ACTING": "#ff7f0e", "VIOLATED": "#d62728"}
    seen = set()
    for e in epochs:
        axL.axvspan(e - 0.5, e + 0.5, color=pcolor[phase[e]], alpha=0.07,
                    label=phase[e] if phase[e] not in seen else None)
        seen.add(phase[e])

    ax_e = epochs
    axL.plot(ax_e, [B0[e] for e in ax_e], color="black", lw=2.0, ls="--",
             label=r"$B_t(0)$ uncontrolled certificate")
    if any(e in Blam for e in ax_e):
        axL.plot([e for e in ax_e if e in Blam], [Blam[e] for e in ax_e if e in Blam],
                 color="#d62728", lw=2.0, label=r"$B_t(\lambda_t)$ controlled (concave)")
    axL.axhline(eps, color="green", lw=1.3, ls=":", label=fr"budget $\epsilon={eps}$")
    if alert is not None:
        axL.axvline(alert, color="#7f0000", lw=1.2)
        axL.annotate("red flag\n(saturation)", xy=(alert, eps), xytext=(alert + 1.0, eps - 0.22),
                     fontsize=8, color="#7f0000",
                     arrowprops=dict(arrowstyle="->", color="#7f0000", lw=1.0))
    axL.set_xlabel("epoch")
    axL.set_ylabel(fr"committee-capture certificate ($\vartheta=1/3$)")
    axL.set_ylim(0, 1.0)

    axR = axL.twinx()
    if base_real:
        re = sorted(base_real)
        axR.plot(re, [base_real[e] for e in re], color="#555555", lw=1.2, marker="o",
                 ms=3, alpha=0.8, label=r"realised $P(\geq\vartheta)$ uncontrolled")
    axR.set_ylabel(r"realised capture $P(\geq\vartheta)$, %")
    axR.set_ylim(0, 100)

    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7.5, framealpha=0.9)
    axL.set_title("Red-flag demonstration: certificate transition under concentration drift")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    print(f"wrote {side}")

    # ---- companion figure: concentration headroom vs Gini --------------------
    # The headroom alpha*-alpha_t counts down to zero at the red flag, while the
    # stake-Gini still looks moderate there: the certificate flags danger a
    # global decentralisation index misses.
    fig2, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(epochs, [headroom[e] for e in epochs], color="#1f77b4", lw=2.2,
            marker="o", ms=3, label=r"concentration headroom $\alpha_t^{\ast}-\alpha_t$")
    ax.plot(epochs, [gini[e] for e in epochs], color="#555555", lw=2.0, ls="--",
            label="stake Gini")
    ax.axhline(0.0, color="#1f77b4", lw=0.8, ls=":")
    if alert is not None:
        ax.axvline(alert, color="#7f0000", lw=1.2)
        ax.annotate(f"red flag: headroom $=0$,\nGini $={gini[alert]:.2f}$ (looks moderate)",
                    xy=(alert, gini[alert]), xytext=(alert + 1.0, max(gini.values()) * 0.55),
                    fontsize=8, color="#7f0000",
                    arrowprops=dict(arrowstyle="->", color="#7f0000", lw=1.0))
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Concentration headroom vanishes at the red flag while the Gini stays moderate")
    fig2.tight_layout()
    out2 = out.parent / "fig_headroom_gini.png"
    fig2.savefig(out2, dpi=200)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
