# system/convergence_analysis.py
# Convergence diagnostics for Bayesian Optimization
#
# Answers the question: "Have we found the global minimum?"
# Run: python convergence_analysis.py [path_to_result.csv]
#
# Produces 5 diagnostic plots + textual summary:
#   1. Best-so-far convergence curve
#   2. GP predicted variance heatmap (unexplored regions)
#   3. Expected Improvement trajectory
#   4. Parameter convergence (are recent suggestions clustering?)
#   5. Cross-validation error (does the GP model fit well?)

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from itertools import combinations

# Add parent path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from BO import _normalize_params


# ============================================================================
# Data loading
# ============================================================================

def load_results(csv_path=None):
    """Load and prepare result.csv data."""
    if csv_path is None:
        csv_path = config.RESULTS_CSV_FILE
    df = pd.read_csv(csv_path)
    # Sort by sim_id to preserve chronological order
    df = df.sort_values('sim_id').reset_index(drop=True)
    return df


def get_normalized_X_and_y(df):
    """Extract normalized parameters and log-transformed costs."""
    param_names = list(config.SWEEP_PARAMETERS.keys())
    valid = df.dropna(subset=['cost'])
    valid = valid[valid['cost'] > 0]

    X = np.array([
        [row[name] for name in param_names]
        for _, row in valid.iterrows()
    ])
    # Normalize to [0,1]
    X_norm = np.zeros_like(X)
    for i, name in enumerate(param_names):
        lo = config.SWEEP_PARAMETERS[name]['min']
        hi = config.SWEEP_PARAMETERS[name]['max']
        X_norm[:, i] = (X[:, i] - lo) / (hi - lo)

    # Log-transform cost (same as BO.py)
    y = -np.log(valid['cost'].values)

    return X_norm, y, valid


# ============================================================================
# Test 1: Convergence Curve
# ============================================================================

def plot_convergence_curve(df, ax):
    """Best-so-far cost vs simulation number."""
    costs = df.sort_values('sim_id')['cost'].values
    best_so_far = np.minimum.accumulate(costs)

    ax.plot(range(1, len(costs) + 1), costs, '.', alpha=0.3, color='gray', label='All sims')
    ax.plot(range(1, len(best_so_far) + 1), best_so_far, '-', color='red', linewidth=2,
            label='Best so far')
    ax.set_xlabel('Simulation #')
    ax.set_ylabel('Cost')
    ax.set_yscale('log')
    ax.set_title('1. Convergence Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Compute improvement rate in last 20% of iterations
    n = len(best_so_far)
    last_20pct = max(1, n // 5)
    recent_improvement = best_so_far[-last_20pct] - best_so_far[-1]
    total_improvement = best_so_far[0] - best_so_far[-1]

    return {
        'best_cost': best_so_far[-1],
        'recent_improvement': recent_improvement,
        'total_improvement': total_improvement,
        'improvement_ratio': recent_improvement / total_improvement if total_improvement > 0 else 0,
        'n_sims': n,
    }


# ============================================================================
# Test 2: GP Posterior Variance (unexplored regions)
# ============================================================================

def plot_gp_variance(X_norm, y, ax):
    """
    Train a GP and show predicted variance across the parameter space.
    High variance = unexplored regions = potential for better solutions.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gp.fit(X_norm, y)

    # Sample random points across the space
    n_test = 5000
    rng = np.random.RandomState(42)
    X_test = rng.rand(n_test, X_norm.shape[1])

    _, y_std = gp.predict(X_test, return_std=True)

    # Show distribution of uncertainties
    ax.hist(y_std, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(y_std), color='red', linestyle='--', label=f'Mean = {np.mean(y_std):.3f}')
    ax.axvline(np.percentile(y_std, 95), color='orange', linestyle='--',
               label=f'P95 = {np.percentile(y_std, 95):.3f}')
    ax.set_xlabel('GP Predicted Std Dev (log-cost space)')
    ax.set_ylabel('Count')
    ax.set_title('2. GP Uncertainty Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Could a hidden minimum exist?
    # If max uncertainty > distance from best to second-best, there might be unexplored optima
    y_best = np.max(y)  # remember: y = -log(cost), so max y = best
    max_std = np.max(y_std)

    return {
        'mean_std': np.mean(y_std),
        'max_std': max_std,
        'p95_std': np.percentile(y_std, 95),
        'best_y': y_best,
        'gp': gp,
        'X_test': X_test,
    }


# ============================================================================
# Test 3: Expected Improvement
# ============================================================================

def plot_expected_improvement(gp_result, ax):
    """
    Expected Improvement: how much better does the GP *expect* to find?
    Low EI across the space = high confidence we've found the optimum.
    """
    gp = gp_result['gp']
    X_test = gp_result['X_test']
    y_best = gp_result['best_y']

    y_pred, y_std = gp.predict(X_test, return_std=True)

    # EI formula (maximization, since y = -log(cost))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (y_pred - y_best) / y_std
        ei = (y_pred - y_best) * norm.cdf(z) + y_std * norm.pdf(z)
        ei[y_std < 1e-10] = 0.0

    # Convert EI from log-cost-space to cost-space for interpretability
    # Best cost = exp(-y_best)
    best_cost = np.exp(-y_best)

    ax.hist(ei, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(np.max(ei), color='red', linestyle='--',
               label=f'Max EI = {np.max(ei):.4f}')
    ax.axvline(np.mean(ei), color='blue', linestyle='--',
               label=f'Mean EI = {np.mean(ei):.4f}')
    ax.set_xlabel('Expected Improvement (log-cost space)')
    ax.set_ylabel('Count')
    ax.set_title('3. Expected Improvement Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # What does max EI mean in cost terms?
    # If current best is cost=C, and EI predicts improvement delta in -log space:
    # new_cost ~ C * exp(-max_EI)
    potential_best = best_cost * np.exp(-np.max(ei))

    return {
        'max_ei': np.max(ei),
        'mean_ei': np.mean(ei),
        'best_cost': best_cost,
        'potential_best_cost': potential_best,
        'potential_improvement_pct': (best_cost - potential_best) / best_cost * 100,
    }


# ============================================================================
# Test 4: Parameter Convergence
# ============================================================================

def plot_parameter_convergence(df, ax):
    """
    Are recent BO suggestions clustering in parameter space?
    If yes -> the optimizer has "decided" where the optimum is.
    """
    param_names = list(config.SWEEP_PARAMETERS.keys())
    df_sorted = df.sort_values('sim_id').reset_index(drop=True)
    n = len(df_sorted)

    # Normalize parameters
    X_norm = np.zeros((n, len(param_names)))
    for i, name in enumerate(param_names):
        lo = config.SWEEP_PARAMETERS[name]['min']
        hi = config.SWEEP_PARAMETERS[name]['max']
        X_norm[:, i] = (df_sorted[name].values - lo) / (hi - lo)

    # Calculate rolling std of each parameter (window = 10)
    window = min(10, n // 3)
    if window < 3:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
        return {}

    rolling_stds = {}
    for i, name in enumerate(param_names):
        stds = pd.Series(X_norm[:, i]).rolling(window=window).std().values
        rolling_stds[name] = stds
        ax.plot(range(1, n + 1), stds, label=name, alpha=0.8)

    ax.set_xlabel('Simulation #')
    ax.set_ylabel(f'Rolling Std (window={window})')
    ax.set_title('4. Parameter Convergence (rolling std)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Final rolling std for each parameter
    final_stds = {name: rolling_stds[name][-1] for name in param_names
                  if not np.isnan(rolling_stds[name][-1])}

    return {
        'final_rolling_stds': final_stds,
        'converged_params': [name for name, std in final_stds.items() if std < 0.1],
        'exploring_params': [name for name, std in final_stds.items() if std >= 0.1],
    }


# ============================================================================
# Test 5: GP Cross-Validation
# ============================================================================

def plot_cv_error(X_norm, y, ax):
    """
    5-fold cross-validation: does the GP model fit the data well?
    Poor fit = model is unreliable = can't trust convergence conclusions.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.model_selection import KFold

    kernel = Matern(nu=2.5)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y_pred_all = np.zeros_like(y)
    y_std_all = np.zeros_like(y)

    for train_idx, test_idx in kf.split(X_norm):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42)
        gp.fit(X_norm[train_idx], y[train_idx])
        pred, std = gp.predict(X_norm[test_idx], return_std=True)
        y_pred_all[test_idx] = pred
        y_std_all[test_idx] = std

    # Plot predicted vs actual
    ax.errorbar(y, y_pred_all, yerr=2 * y_std_all, fmt='.', alpha=0.3, color='steelblue',
                ecolor='lightblue', capsize=0, label='Predictions +/- 2 std')
    lims = [min(y.min(), y_pred_all.min()), max(y.max(), y_pred_all.max())]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')
    ax.set_xlabel('Actual -log(cost)')
    ax.set_ylabel('Predicted -log(cost)')
    ax.set_title('5. GP Cross-Validation (5-fold)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Metrics
    residuals = y - y_pred_all
    rmse = np.sqrt(np.mean(residuals ** 2))
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)

    # Coverage: fraction of actuals within 2-sigma of prediction
    within_2sigma = np.mean(np.abs(residuals) <= 2 * y_std_all)

    return {
        'rmse': rmse,
        'r2': r2,
        'coverage_2sigma': within_2sigma,
    }


# ============================================================================
# Bonus: Parameter Space Coverage
# ============================================================================

def analyze_space_coverage(X_norm):
    """How well has the parameter space been sampled?"""
    n, d = X_norm.shape

    # 1. Divide each dimension into 5 bins, count empty bins
    n_bins = 5
    total_cells = n_bins ** d
    occupied = set()
    for row in X_norm:
        cell = tuple(min(int(row[j] * n_bins), n_bins - 1) for j in range(d))
        occupied.add(cell)
    coverage = len(occupied) / total_cells

    # 2. Pairwise 2D coverage (more practical for high-d)
    n_bins_2d = 10
    pair_coverages = {}
    for i, j in combinations(range(d), 2):
        cells = set()
        for row in X_norm:
            ci = min(int(row[i] * n_bins_2d), n_bins_2d - 1)
            cj = min(int(row[j] * n_bins_2d), n_bins_2d - 1)
            cells.add((ci, cj))
        pair_coverages[(i, j)] = len(cells) / (n_bins_2d ** 2)

    # 3. Minimum pairwise distance (detect clustering)
    from scipy.spatial.distance import pdist
    dists = pdist(X_norm)

    return {
        'full_coverage_6d': coverage,
        'mean_2d_coverage': np.mean(list(pair_coverages.values())),
        'min_2d_coverage': min(pair_coverages.values()),
        'max_2d_coverage': max(pair_coverages.values()),
        'min_pairwise_dist': np.min(dists),
        'mean_pairwise_dist': np.mean(dists),
        'median_pairwise_dist': np.median(dists),
        'n_samples': n,
        'n_dims': d,
    }


# ============================================================================
# Main Analysis
# ============================================================================

def run_analysis(csv_path=None):
    """Run all convergence diagnostics and generate report."""
    print("=" * 70)
    print("CONVERGENCE ANALYSIS - PS_Opt_V2")
    print("=" * 70)

    # Load data
    df = load_results(csv_path)
    X_norm, y, valid_df = get_normalized_X_and_y(df)
    print(f"\nLoaded {len(df)} total simulations ({len(valid_df)} with valid cost)")

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Bayesian Optimization Convergence Diagnostics', fontsize=14, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])  # for summary text

    # Run tests
    print("\n[1/5] Convergence curve...")
    conv = plot_convergence_curve(valid_df, ax1)

    print("[2/5] GP posterior variance...")
    gp_result = plot_gp_variance(X_norm, y, ax2)

    print("[3/5] Expected Improvement...")
    ei_result = plot_expected_improvement(gp_result, ax3)

    print("[4/5] Parameter convergence...")
    param_conv = plot_parameter_convergence(df, ax4)

    print("[5/5] GP cross-validation...")
    cv_result = plot_cv_error(X_norm, y, ax5)

    print("[+] Space coverage analysis...")
    coverage = analyze_space_coverage(X_norm)

    # Summary panel
    ax6.axis('off')
    summary_lines = [
        "CONVERGENCE SUMMARY",
        "=" * 35,
        "",
        f"Total sims: {conv['n_sims']}",
        f"Best cost: {conv['best_cost']:.4f}",
        "",
        "--- Convergence ---",
        f"Last 20% improvement: {conv['recent_improvement']:.4f}",
        f"  ({conv['improvement_ratio']*100:.1f}% of total)",
        "",
        "--- GP Model ---",
        f"CV R^2: {cv_result['r2']:.3f}",
        f"CV RMSE: {cv_result['rmse']:.3f}",
        f"2-sigma coverage: {cv_result['coverage_2sigma']*100:.0f}%",
        "",
        "--- Expected Improvement ---",
        f"Max EI: {ei_result['max_ei']:.4f}",
        f"Potential best: {ei_result['potential_best_cost']:.4f}",
        f"Potential gain: {ei_result['potential_improvement_pct']:.1f}%",
        "",
        "--- Space Coverage ---",
        f"Mean 2D coverage: {coverage['mean_2d_coverage']*100:.0f}%",
    ]
    ax6.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax6.transAxes,
             fontfamily='monospace', fontsize=9, verticalalignment='top')

    # Save figure
    output_dir = os.path.dirname(csv_path) if csv_path else config.SIMULATION_CSV_DIR
    fig_path = os.path.join(output_dir, 'convergence_analysis.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    plt.show()

    # Print detailed textual report
    print("\n" + "=" * 70)
    print("DETAILED REPORT")
    print("=" * 70)

    # Verdict
    print("\n>>> CONVERGENCE VERDICT <<<\n")

    confidence_score = 0
    max_score = 5
    reasons = []

    # 1. Convergence curve plateau
    if conv['improvement_ratio'] < 0.02:
        confidence_score += 1
        reasons.append("[V] Convergence curve plateaued (last 20% < 2% of total improvement)")
    else:
        reasons.append(f"[X] Still improving (last 20% = {conv['improvement_ratio']*100:.1f}% of total)")

    # 2. Low GP variance
    if gp_result['p95_std'] < 0.3:
        confidence_score += 1
        reasons.append(f"[V] GP uncertainty low (P95 std = {gp_result['p95_std']:.3f})")
    else:
        reasons.append(f"[X] GP uncertainty still high (P95 std = {gp_result['p95_std']:.3f})")

    # 3. Low Expected Improvement
    if ei_result['potential_improvement_pct'] < 2.0:
        confidence_score += 1
        reasons.append(f"[V] Expected Improvement negligible ({ei_result['potential_improvement_pct']:.1f}%)")
    else:
        reasons.append(f"[X] GP predicts possible {ei_result['potential_improvement_pct']:.1f}% improvement")

    # 4. Good GP fit
    if cv_result['r2'] > 0.8:
        confidence_score += 1
        reasons.append(f"[V] GP model fits well (R^2 = {cv_result['r2']:.3f})")
    else:
        reasons.append(f"[X] GP model fit poor (R^2 = {cv_result['r2']:.3f}) - conclusions unreliable")

    # 5. Parameters converged
    if param_conv.get('converged_params') and len(param_conv.get('converged_params', [])) >= 4:
        confidence_score += 1
        reasons.append(f"[V] Parameters converged: {param_conv['converged_params']}")
    else:
        reasons.append(f"[X] Parameters still exploring: {param_conv.get('exploring_params', [])}")

    for r in reasons:
        print(f"  {r}")

    print(f"\n  CONFIDENCE SCORE: {confidence_score}/{max_score}")

    if confidence_score >= 4:
        print("  CONCLUSION: HIGH confidence - likely near global minimum")
    elif confidence_score >= 3:
        print("  CONCLUSION: MODERATE confidence - probably close, but some uncertainty remains")
    elif confidence_score >= 2:
        print("  CONCLUSION: LOW confidence - more iterations recommended")
    else:
        print("  CONCLUSION: VERY LOW confidence - significantly more exploration needed")

    # Warnings about bounds
    print("\n>>> BOUND SATURATION WARNING <<<\n")
    param_names = list(config.SWEEP_PARAMETERS.keys())
    for name in param_names:
        vals = valid_df[name].values
        lo = config.SWEEP_PARAMETERS[name]['min']
        hi = config.SWEEP_PARAMETERS[name]['max']
        # Check what fraction of top-20 results hit bounds
        top20 = valid_df.nsmallest(20, 'cost')
        at_min = (np.abs(top20[name] - lo) / (hi - lo) < 0.02).sum()
        at_max = (np.abs(top20[name] - hi) / (hi - lo) < 0.02).sum()
        if at_min >= 5:
            print(f"  [!] {name}: {at_min}/20 best results at LOWER bound ({lo:.2e})")
            print(f"      -> Consider extending lower bound")
        if at_max >= 5:
            print(f"  [!] {name}: {at_max}/20 best results at UPPER bound ({hi:.2e})")
            print(f"      -> Consider extending upper bound")

    return {
        'convergence': conv,
        'gp_variance': gp_result,
        'expected_improvement': ei_result,
        'parameter_convergence': param_conv,
        'cross_validation': cv_result,
        'space_coverage': coverage,
        'confidence_score': confidence_score,
    }


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_analysis(csv_path)
