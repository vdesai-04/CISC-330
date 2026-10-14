"""Runs the Part 10 study: tables + plots (MLE vs MRE, REM vs MRE)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from error_analysis import simulate, find_max_mle

MAX_MRE = 5.0
mles = np.arange(0, 11)
res = {(m, x): simulate(m, float(x)) for m in range(3) for x in mles}

print("Worst-case MRE (mm) over 129,600 combinations")
print("MLE | " + " | ".join(f"M{m+1}" for m in range(3)) + " | all")
for x in mles:
    w = [res[(m, x)][0].max() for m in range(3)]
    print(f"{x:3d} | " + " | ".join(f"{v:5.2f}" for v in w) + f" | {max(w):5.2f}")
mx = find_max_mle()
print(f"\nmaxMLE (all markers, all directions, MRE < {MAX_MRE} mm) = {mx:.3f} mm  -> integer-grid answer: {int(np.floor(mx))} mm")
print("Slope worst MRE / MLE per marker:", [round(res[(m,1)][0].max(),3) for m in range(3)])

# ---- Plot 1: MLE vs MRE (worst case, plus mean) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for m, c in zip(range(3), ["#1f77b4", "#d55e00", "#2a9d5c"]):
    ax.plot(mles, [res[(m, x)][0].max() for x in mles], "o-", color=c, label=f"M{m+1} worst case")
ax.plot(mles, [np.mean([res[(m, x)][0].mean() for m in range(3)]) for x in mles], "k--", label="mean (all markers)")
ax.axhline(MAX_MRE, color="crimson", ls=":", label="maxMRE = 5 mm")
ax.axvline(mx, color="grey", ls=":"); ax.annotate(f"maxMLE ≈ {mx:.2f} mm", (mx, 0.3), xytext=(mx+0.2, 0.3))
ax.set_xlabel("Marker localization error, MLE (mm on detector)"); ax.set_ylabel("Marker reconstruction error, MRE (mm)")
ax.set_title("MLE vs MRE (all 360×360 direction combinations)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/part10_mle_vs_mre.png", dpi=200)

# ---- REM analysis, pooled over every marker and every MLE in 1..10 ----
mre_all = np.concatenate([res[(m, x)][0].ravel() for m in range(3) for x in mles[1:]])
rem_all = np.concatenate([res[(m, x)][1].ravel() for m in range(3) for x in mles[1:]])
bad = mre_all >= MAX_MRE
print(f"\nPooled samples: {mre_all.size:,}; with MRE >= 5 mm: {bad.sum():,}")
print(f"Pearson r(REM, MRE) = {np.corrcoef(rem_all, mre_all)[0,1]:.3f}")
print(f"Among MRE >= 5 mm: min REM = {rem_all[bad].min():.4f} mm, median REM = {np.median(rem_all[bad]):.3f} mm")
print(f"Max MRE among samples with REM < 0.01 mm: {mre_all[rem_all<0.01].max():.2f} mm   (count {np.sum(rem_all<0.01):,})")
print("\nREM threshold T | caught (REM>=T) among MRE>=5 | false alarms among MRE<5")
for T in [0.01, 0.1, 0.5, 1, 2, 3, 5]:
    flag = rem_all >= T
    print(f"{T:6.2f} mm      | {100*flag[bad].mean():6.1f}%                    | {100*flag[~bad].mean():5.1f}%")
# per-MLE view at the MLE that straddles the limit
x = 7
mre7 = np.concatenate([res[(m, x)][0].ravel() for m in range(3)]); rem7 = np.concatenate([res[(m, x)][1].ravel() for m in range(3)])
print(f"\nAt MLE = 7 mm: max REM = {rem7.max():.2f} mm; REM of the worst-MRE sample = {rem7[mre7.argmax()]:.3f} mm")

# ---- Plot 2: REM vs MRE ----
rng = np.random.default_rng(1)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
sel = rng.choice(mre_all.size, 150_000, replace=False)
axs[0].scatter(mre_all[sel], rem_all[sel], s=1, alpha=.15, color="#1f77b4", rasterized=True)
axs[0].axvline(MAX_MRE, color="crimson", ls=":", label="maxMRE = 5 mm")
axs[0].set_xlabel("MRE (mm)"); axs[0].set_ylabel("REM (mm)"); axs[0].set_title("REM vs MRE, MLE 1–10 mm (150k random samples)")
axs[0].legend(); axs[0].grid(alpha=.3)
for m, c in zip(range(3), ["#1f77b4", "#d55e00", "#2a9d5c"]):
    mre, rem = res[(m, 7)]
    axs[1].scatter(mre.ravel()[::7], rem.ravel()[::7], s=2, alpha=.3, color=c, label=f"M{m+1}", rasterized=True)
axs[1].axvline(MAX_MRE, color="crimson", ls=":")
axs[1].set_xlabel("MRE (mm)"); axs[1].set_ylabel("REM (mm)"); axs[1].set_title("MLE = 7 mm only"); axs[1].legend(); axs[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/part10_rem_vs_mre.png", dpi=200)
