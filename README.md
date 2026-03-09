# Numerical experiments on SMOTE

Numerical illustrations accompanying the presentation of:

> Sakho, Malherbe & Scornet (2024). *Do we need rebalancing strategies? A theoretical and empirical study around SMOTE and its variants.* [arXiv:2402.03819v5](https://arxiv.org/abs/2402.03819v5)

Prepared for the *Guidelines in ML* course (M2 Mathématiques & IA, Université Paris-Saclay).

## Experiments

### Experiment 1 — Diversity measure (`experiments/diversity_measure.py`)

Illustrates **Theorem 3.4** and **Corollary 3.5**: when K is fixed (default K=5), SMOTE asymptotically copies the original minority samples. We reproduce the diversity metric from Appendix B on **three different distributions** (uniform square, uniform disk, Gaussian mixture) to show the phenomenon is distribution-agnostic.

### Experiment 2 — Boundary artifact (`experiments/boundary_artifact.py`)

Illustrates **Theorem 3.6**: SMOTE density vanishes near the boundary of the minority class support. We provide **scatter plots** (absent from the paper) on a disk and an annulus, plus radial density profiles. The annulus case additionally demonstrates **Lemma 3.1** (SMOTE fills non-convex holes).

## Usage

```bash
pip install -r requirements.txt
mkdir -p figures
python experiments/diversity_measure.py
python experiments/boundary_artifact.py
```

Figures are saved in `figures/`.

## Note on methodology

We used an LLM (Claude, Anthropic) as an aid for code structuring and discussion of theoretical results. The experimental design, distribution choices, and analysis are our own.

## Reference

Original paper code: [github.com/artefactory/smote_strategies_study](https://github.com/artefactory/smote_strategies_study)
