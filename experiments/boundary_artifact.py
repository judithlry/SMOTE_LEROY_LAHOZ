"""
Experiment 2 — Boundary artifact of SMOTE.

Illustrates Theorem 3.6 from Sakho et al. (2024):
    "Do we need rebalancing strategies? A theoretical and empirical study
     around SMOTE and its variants"

Theorem 3.6 shows that SMOTE density vanishes near the boundary of the
support of the minority class distribution. Specifically, for z near the
boundary (||z|| >= R - ε), the density is bounded by O(ε^{1/4}).

This experiment provides a VISUAL illustration that the paper does not
include: scatter plots of original vs SMOTE-generated points on bounded
supports, showing the rarefaction near the boundary.

We test on two geometries:
    1. Disk B(0, 3) — matches Theorem 3.6 assumption (X = B(0,R))
    2. Annulus (ring) — a non-convex support, showing that SMOTE also
       fills the hole (Lemma 3.1: SMOTE ⊂ Conv(X)), which is a failure
       mode beyond just boundary artifacts.

For each, we also plot the radial density to quantify the boundary effect.

Usage:
    python experiments/boundary_artifact.py

Output:
    figures/boundary_artifact.pdf
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────

def sample_disk(n: int, R: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform on disk B(0, R)."""
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = R * np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def sample_annulus(n: int, R_outer: float, R_inner: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Uniform on annulus {x : R_inner <= ||x|| <= R_outer}.

    This is interesting because:
    - The support is non-convex (has a hole)
    - Lemma 3.1 says SMOTE ⊂ Conv(X), so SMOTE will generate points
      INSIDE the hole, which is outside the true support
    - There are TWO boundaries (inner + outer) where Theorem 3.6 applies
    """
    angles = rng.uniform(0, 2 * np.pi, n)
    # Uniform on annulus: sample r² uniformly in [R_inner², R_outer²]
    r_sq = rng.uniform(R_inner**2, R_outer**2, n)
    radii = np.sqrt(r_sq)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


# ──────────────────────────────────────────────
# SMOTE generation
# ──────────────────────────────────────────────

def generate_smote_points(X_minority: np.ndarray, K: int, n_synthetic: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Generate SMOTE synthetic points from minority samples.

    We create a dummy majority class to use imblearn's SMOTE.
    """
    n = len(X_minority)
    K_eff = min(K, n - 1)

    # Majority class: far away so it doesn't interfere with minority NN
    n_majority = n + n_synthetic
    X_majority = rng.standard_normal((n_majority, 2)) * 0.01 + np.array([100, 100])

    X_full = np.vstack([X_minority, X_majority])
    y_full = np.array([1] * n + [0] * n_majority)

    smote = SMOTE(
        k_neighbors=K_eff,
        random_state=int(rng.integers(1e6)),
    )
    X_res, y_res = smote.fit_resample(X_full, y_full)

    # Extract only synthetic minority points
    Z = X_res[len(X_full):]
    Z = Z[y_res[len(X_full):] == 1]
    return Z


# ──────────────────────────────────────────────
# Radial density estimation
# ──────────────────────────────────────────────

def radial_density(points: np.ndarray, bins: int = 50,
                   r_max: float = None) -> tuple[np.ndarray, np.ndarray]:
    """Estimate radial density f(r) from 2D points.

    For uniform distribution on disk B(0,R), the theoretical density of
    the radius is f(r) = 2r/R² for r in [0, R].
    """
    radii = np.linalg.norm(points, axis=1)
    if r_max is None:
        r_max = radii.max() * 1.05
    bin_edges = np.linspace(0, r_max, bins + 1)
    counts, _ = np.histogram(radii, bins=bin_edges)
    # Normalize by annular area to get density
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    annular_areas = 2 * np.pi * bin_centers * bin_widths
    density = counts / (len(points) * annular_areas + 1e-10)
    return bin_centers, density


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────

def run_experiment():
    rng = np.random.default_rng(42)
    n = 2000        # number of original minority samples
    K = 5           # default SMOTE parameter
    R = 3.0         # radius

    print("=" * 60)
    print("Experiment 2: Boundary artifact visualization")
    print("Illustrating Theorem 3.6")
    print("=" * 60)

    # ── Generate data ──
    print("Generating original samples...")
    X_disk = sample_disk(n, R, rng)
    X_annulus = sample_annulus(n, R_outer=R, R_inner=1.2, rng=rng)

    print("Generating SMOTE samples (K=5)...")
    n_synth = 2000
    Z_disk = generate_smote_points(X_disk, K=K, n_synthetic=n_synth, rng=rng)
    Z_annulus = generate_smote_points(X_annulus, K=K, n_synthetic=n_synth, rng=rng)

    print("Generating SMOTE samples (K=0.1n)...")
    Z_disk_bigK = generate_smote_points(
        X_disk, K=int(0.1 * n), n_synthetic=n_synth, rng=rng
    )
    Z_annulus_bigK = generate_smote_points(
        X_annulus, K=int(0.1 * n), n_synthetic=n_synth, rng=rng
    )

    # ── Plot ──
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    scatter_kw = dict(s=3, alpha=0.5, edgecolors="none")

    # Row 1: Disk
    # (a) Original
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(X_disk[:, 0], X_disk[:, 1], c="#377eb8", label="Original", **scatter_kw)
    circle = plt.Circle((0, 0), R, fill=False, color="black", linestyle="--", linewidth=1)
    ax.add_patch(circle)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title(f"Disk $B(0,{R:.0f})$ — Original ($n={n}$)", fontsize=11)
    ax.legend(fontsize=8)

    # (b) SMOTE K=5
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(X_disk[:, 0], X_disk[:, 1], c="#377eb8", label="Original", **scatter_kw)
    ax.scatter(Z_disk[:, 0], Z_disk[:, 1], c="#e41a1c", label=f"SMOTE ($K=5$)", **scatter_kw)
    circle = plt.Circle((0, 0), R, fill=False, color="black", linestyle="--", linewidth=1)
    ax.add_patch(circle)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title("SMOTE $K=5$: boundary rarefaction", fontsize=11)
    ax.legend(fontsize=8)

    # (c) Radial density comparison
    ax = fig.add_subplot(gs[0, 2])
    r_orig, d_orig = radial_density(X_disk, bins=40, r_max=3.5)
    r_smote, d_smote = radial_density(Z_disk, bins=40, r_max=3.5)
    r_smoteK, d_smoteK = radial_density(Z_disk_bigK, bins=40, r_max=3.5)
    ax.plot(r_orig, d_orig, color="#377eb8", linewidth=1.5, label="Original")
    ax.plot(r_smote, d_smote, color="#e41a1c", linewidth=1.5, label="SMOTE $K=5$")
    ax.plot(r_smoteK, d_smoteK, color="#4daf4a", linewidth=1.5, label="SMOTE $K=0.1n$")
    ax.axvline(x=R, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Radius $r$", fontsize=10)
    ax.set_ylabel("Radial density", fontsize=10)
    ax.set_title("Radial density — Thm 3.6: drop near $r=R$", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    # Annotation
    ax.annotate(
        "Boundary\nartifact",
        xy=(R - 0.15, 0.02), xytext=(R - 1.0, 0.08),
        fontsize=9, color="#e41a1c",
        arrowprops=dict(arrowstyle="->", color="#e41a1c"),
    )

    # Row 2: Annulus (non-convex support)
    R_inner = 1.2

    # (a) Original
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(X_annulus[:, 0], X_annulus[:, 1], c="#377eb8", label="Original", **scatter_kw)
    for r, ls in [(R, "--"), (R_inner, "--")]:
        circle = plt.Circle((0, 0), r, fill=False, color="black", linestyle=ls, linewidth=1)
        ax.add_patch(circle)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title(f"Annulus ($R_{{in}}={R_inner}$, $R_{{out}}={R:.0f}$) — Original", fontsize=11)
    ax.legend(fontsize=8)

    # (b) SMOTE K=5
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(X_annulus[:, 0], X_annulus[:, 1], c="#377eb8", label="Original", **scatter_kw)
    ax.scatter(Z_annulus[:, 0], Z_annulus[:, 1], c="#e41a1c", label="SMOTE ($K=5$)", **scatter_kw)
    for r, ls in [(R, "--"), (R_inner, "--")]:
        circle = plt.Circle((0, 0), r, fill=False, color="black", linestyle=ls, linewidth=1)
        ax.add_patch(circle)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title("Annulus + SMOTE: fills the hole (Lemma 3.1)", fontsize=11)
    ax.legend(fontsize=8)

    # (c) Radial density comparison
    ax = fig.add_subplot(gs[1, 2])
    r_orig, d_orig = radial_density(X_annulus, bins=40, r_max=3.5)
    r_smote, d_smote = radial_density(Z_annulus, bins=40, r_max=3.5)
    r_smoteK, d_smoteK = radial_density(Z_annulus_bigK, bins=40, r_max=3.5)
    ax.plot(r_orig, d_orig, color="#377eb8", linewidth=1.5, label="Original")
    ax.plot(r_smote, d_smote, color="#e41a1c", linewidth=1.5, label="SMOTE $K=5$")
    ax.plot(r_smoteK, d_smoteK, color="#4daf4a", linewidth=1.5, label="SMOTE $K=0.1n$")
    ax.axvline(x=R, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=R_inner, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Radius $r$", fontsize=10)
    ax.set_ylabel("Radial density", fontsize=10)
    ax.set_title("Radial density — artifact at both boundaries", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax.annotate(
        "Leak into\nthe hole",
        xy=(0.8, 0.01), xytext=(0.3, 0.06),
        fontsize=9, color="#e41a1c",
        arrowprops=dict(arrowstyle="->", color="#e41a1c"),
    )

    fig.suptitle(
        "SMOTE boundary artifact (Theorem 3.6)\n"
        "Density vanishes as $\\varepsilon^{1/4}$ near support boundary; "
        "SMOTE fills non-convex holes (Lemma 3.1)",
        fontsize=13,
        y=1.02,
    )
    plt.savefig("figures/boundary_artifact.pdf", bbox_inches="tight", dpi=150)
    plt.savefig("figures/boundary_artifact.png", bbox_inches="tight", dpi=150)
    print("\n✓ Saved figures/boundary_artifact.{pdf,png}")


if __name__ == "__main__":
    run_experiment()
