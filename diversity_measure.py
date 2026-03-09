"""
Experiment 1 — Diversity measure of SMOTE as a function of n and K.

Illustrates Theorem 3.4 and Corollary 3.5 from Sakho et al. (2024):
    "Do we need rebalancing strategies? A theoretical and empirical study
     around SMOTE and its variants"

The paper shows (Theorem 3.4) that when K/n → 0, SMOTE observations
concentrate around their central points — i.e., SMOTE asymptotically
*copies* the original minority samples. We reproduce their diversity
measure protocol (Appendix B) on THREE different distributions to show
the result is not distribution-specific:

    1. Uniform on [-3,3]^2   (same as paper, for reference)
    2. Uniform on a disk B(0, 3)
    3. 2D Gaussian mixture (3 components)

The diversity metric d(Z, X) = C̄(Z, X) / C̄(X̃, X) compares SMOTE
diversity against the ideal case of fresh i.i.d. samples.

Usage:
    python experiments/diversity_measure.py

Output:
    figures/diversity_measure.pdf
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Distribution samplers
# ──────────────────────────────────────────────

def sample_uniform_square(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform on [-3, 3]^2 — same as paper (reference)."""
    return rng.uniform(-3, 3, size=(n, 2))


def sample_uniform_disk(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform on disk B(0, 3) — tests a non-rectangular bounded support.

    Relevant because Theorem 3.6 assumes X = B(0, R).
    """
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = 3.0 * np.sqrt(rng.uniform(0, 1, n))  # sqrt for uniform density
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def sample_gaussian_mixture(n: int, rng: np.random.Generator) -> np.ndarray:
    """Mixture of 3 Gaussians — tests a distribution that violates
    Assumption 3.3 (density not bounded below on convex hull of support).

    This is interesting because Theorem 3.4 relies on Assumption 3.3.
    """
    centers = np.array([[-2, -2], [2, 2], [-2, 2]])
    sigma = 0.8
    assignments = rng.integers(0, 3, size=n)
    samples = np.zeros((n, 2))
    for k in range(3):
        mask = assignments == k
        count = mask.sum()
        samples[mask] = rng.normal(loc=centers[k], scale=sigma, size=(count, 2))
    return samples


DISTRIBUTIONS = {
    r"Uniform on $[-3,3]^2$": sample_uniform_square,
    r"Uniform on disk $B(0,3)$": sample_uniform_disk,
    r"Gaussian mixture (3 comp.)": sample_gaussian_mixture,
}

# ──────────────────────────────────────────────
# Diversity metric (Appendix B of the paper)
# ──────────────────────────────────────────────

def average_nn_distance(Z: np.ndarray, X: np.ndarray) -> float:
    """C(Z, X) = (1/m) Σ ||Z_i - X_{(1)}(Z_i)||_2

    Measures how far generated points Z are from the original points X.
    """
    nn = NearestNeighbors(n_neighbors=1).fit(X)
    distances, _ = nn.kneighbors(Z)
    return distances.mean()


def compute_diversity(
    sampler,
    n: int,
    K_values: dict[str, int],
    m: int = 1000,
    n_repeats: int = 75,
    seed: int = 42,
) -> dict[str, float]:
    """Compute d(Z, X) = C̄(Z, X) / C̄(X̃, X) for each K setting.

    Protocol from Appendix B:
    1. Generate X (n samples from distribution)
    2. Generate Z (m SMOTE samples from X) → compute C(Z, X)
    3. Generate X̃ (m fresh samples from same distribution) → C(X̃, X)
    4. Repeat n_repeats times, average, take ratio
    """
    rng = np.random.default_rng(seed)

    # Storage
    c_smote = {name: [] for name in K_values}
    c_reference = []

    for rep in range(n_repeats):
        X = sampler(n, rng)

        # Reference: fresh samples from same distribution
        X_tilde = sampler(m, rng)
        c_reference.append(average_nn_distance(X_tilde, X))

        # SMOTE for each K
        for name, K in K_values.items():
            K_eff = min(K, n - 1)  # K cannot exceed n-1
            if K_eff < 1:
                c_smote[name].append(np.nan)
                continue

            try:
                # Build a dummy binary classification problem
                # n minority + enough majority to trigger SMOTE
                n_majority = max(n, 2 * n)
                X_full = np.vstack([X, rng.standard_normal((n_majority, 2))])
                y_full = np.array([1] * n + [0] * n_majority)

                smote = SMOTE(k_neighbors=K_eff, random_state=int(rng.integers(1e6)))
                X_res, y_res = smote.fit_resample(X_full, y_full)

                # Extract synthetic minority samples
                Z_all = X_res[len(X_full):]
                Z_all = Z_all[y_res[len(X_full):] == 1]

                if len(Z_all) < m:
                    Z = Z_all
                else:
                    idx = rng.choice(len(Z_all), size=m, replace=False)
                    Z = Z_all[idx]

                c_smote[name].append(average_nn_distance(Z, X))
            except Exception:
                c_smote[name].append(np.nan)

    c_ref_mean = np.nanmean(c_reference)
    results = {}
    for name in K_values:
        c_mean = np.nanmean(c_smote[name])
        results[name] = c_mean / c_ref_mean if c_ref_mean > 0 else np.nan

    return results


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────

def get_K_values(n: int) -> dict[str, int]:
    """K settings matching the paper + one extra."""
    values = {
        r"$K = 5$": 5,
        r"$K = \sqrt{n}$": max(1, int(np.sqrt(n))),
        r"$K = 0.01n$": max(1, int(0.01 * n)),
        r"$K = 0.1n$": max(1, int(0.1 * n)),
        r"$K = 0.3n$": max(1, int(0.3 * n)),
    }
    return values


def run_experiment():
    # Reduced n range for tractability (paper goes up to 50k, we do up to 2000)
    n_values = [50, 100, 200, 400, 700, 1000, 1500, 2000]
    n_repeats = 30  # paper uses 75, reduced for speed

    print("=" * 60)
    print("Experiment 1: Diversity measure d(Z, X)")
    print("Illustrating Theorem 3.4 / Corollary 3.5")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    colors = {
        r"$K = 5$": "#e41a1c",
        r"$K = \sqrt{n}$": "#984ea3",
        r"$K = 0.01n$": "#377eb8",
        r"$K = 0.1n$": "#4daf4a",
        r"$K = 0.3n$": "#ff7f00",
    }
    markers = {
        r"$K = 5$": "o",
        r"$K = \sqrt{n}$": "D",
        r"$K = 0.01n$": "s",
        r"$K = 0.1n$": "^",
        r"$K = 0.3n$": "v",
    }

    for idx, (dist_name, sampler) in enumerate(DISTRIBUTIONS.items()):
        print(f"\n--- {dist_name} ---")

        ax = fig.add_subplot(gs[0, idx])
        all_results = {name: [] for name in colors}

        for n in n_values:
            K_values = get_K_values(n)
            print(f"  n = {n:>5d} ...", end=" ", flush=True)
            results = compute_diversity(
                sampler, n, K_values, m=500, n_repeats=n_repeats, seed=42 + n
            )
            print("done")
            for name in K_values:
                all_results[name].append(results[name])

        for name in colors:
            ax.plot(
                n_values,
                all_results[name],
                color=colors[name],
                marker=markers[name],
                markersize=5,
                linewidth=1.5,
                label=name,
            )

        ax.set_xlabel("Number of minority samples ($n$)", fontsize=11)
        if idx == 0:
            ax.set_ylabel(r"Diversity $d(\mathbf{Z}, \mathbf{X})$", fontsize=11)
        ax.set_title(dist_name, fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)

    fig.suptitle(
        "SMOTE diversity measure for different distributions and $K$ settings\n"
        "(Theorem 3.4: $K=5$ leads to copying; $K/n \\to 0$ slowly gives more diversity)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig("figures/diversity_measure.pdf", bbox_inches="tight", dpi=150)
    plt.savefig("figures/diversity_measure.png", bbox_inches="tight", dpi=150)
    print("\n✓ Saved figures/diversity_measure.{pdf,png}")


if __name__ == "__main__":
    run_experiment()
