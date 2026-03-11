"""
Visualization Module
Creates clear, publication-quality plots for air quality data and model results
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import logging
from typing import List, Dict, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import FIGURES_DIR, AQI_CATEGORIES, AQI_COLORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'figure.constrained_layout.use': False,   # we control layout manually
})

# AQI band colours (India CPCB standard)
AQI_BAND = {
    'Good':         (0,   50,  '#55A84F'),
    'Satisfactory': (51,  100, '#A3C853'),
    'Moderate':     (101, 200, '#FFF833'),
    'Poor':         (201, 300, '#F29C33'),
    'Very Poor':    (301, 400, '#E93F33'),
    'Severe':       (401, 500, '#AF2D24'),
}

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
           '#3B1F2B', '#44BBA4', '#E94F37', '#393E41']


def _save(fig: plt.Figure, save_path, label: str):
    if save_path:
        out = FIGURES_DIR / save_path
        fig.savefig(out, bbox_inches='tight')
        logger.info(f"Saved {label} to {out}")
    plt.show()
    plt.close(fig)


def _add_aqi_bands(ax: plt.Axes):
    for name, (lo, hi, color) in AQI_BAND.items():
        ax.axhspan(lo, hi, alpha=0.08, color=color, zorder=0)


def _aqi_category(val: float) -> str:
    for name, (lo, hi, _) in AQI_BAND.items():
        if lo <= val <= hi:
            return name
    return 'Severe'


def plot_time_series(df: pd.DataFrame,
                     columns: List[str],
                     date_col: str = 'Date',
                     title: str = "Air Quality Time Series",
                     save_path: str = None):
    """Time-series with 30-day rolling mean and AQI category bands."""
    cols = [c for c in columns if c in df.columns]
    if not cols:
        logger.warning("None of the requested columns exist in the DataFrame.")
        return

    n   = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    import matplotlib.dates as mdates
    for ax, col in zip(axes, cols):
        x = df[date_col] if date_col in df.columns else pd.RangeIndex(len(df))
        y = df[col]

        if col == 'AQI':
            _add_aqi_bands(ax)

        ax.plot(x, y, linewidth=0.9, color=PALETTE[0], alpha=0.50, label='Daily')
        roll = y.rolling(30, min_periods=1).mean()
        ax.plot(x, roll, linewidth=2.2, color=PALETTE[1], label='30-day avg', zorder=5)

        mean_val = y.mean()
        ax.axhline(mean_val, color='grey', linestyle=':', linewidth=1.2, alpha=0.7)
        # Anchor the mean label inside the axes to avoid right-edge clipping
        ax.text(0.99, mean_val, f'mean={mean_val:.0f} ',
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=9, color='grey',
                clip_on=True)

        ax.set_ylabel(col, fontsize=12)
        ax.set_title(f'{col}  —  Daily values & 30-day rolling mean', fontsize=13)

        if col == 'AQI':
            patches = [mpatches.Patch(color=c, alpha=0.5,
                                      label=f'{nm}  ({lo}–{hi})')
                       for nm, (lo, hi, c) in AQI_BAND.items()]
            ax.legend(handles=patches, loc='upper right', fontsize=8,
                      title='AQI Category', title_fontsize=9, framealpha=0.85)
        else:
            ax.legend(loc='upper left', framealpha=0.7)

    try:
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    except Exception:
        pass
    axes[-1].set_xlabel('Date', fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(title, fontsize=16, fontweight='bold')
    _save(fig, save_path, 'time series')


def plot_correlation_matrix(df: pd.DataFrame,
                            columns: List[str] = None,
                            title: str = "Feature Correlation Matrix",
                            save_path: str = None):
    """Annotated correlation heatmap (lower triangle only)."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns = [c for c in columns if c in df.columns]
    if len(columns) < 2:
        logger.warning("Not enough numeric columns for correlation matrix.")
        return

    corr = df[columns].corr()
    n    = len(columns)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), max(8, n * 0.8)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn_r', center=0, square=True,
                linewidths=0.5, linecolor='white',
                cbar_kws={"shrink": 0.75, "label": "Pearson r"},
                ax=ax, annot_kws={"size": 9})
    ax.set_title(title, fontsize=15, pad=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    _save(fig, save_path, 'correlation matrix')


def plot_feature_importance(importance_df: pd.DataFrame,
                            top_n: int = 15,
                            title: str = "Feature Importance",
                            save_path: str = None):
    """Horizontal bar chart with value labels, coloured by importance score."""
    top = importance_df.head(top_n).copy().sort_values('Importance', ascending=True)

    row_h = 0.62                          # height budget per feature row (inches)
    fig, ax = plt.subplots(figsize=(12, max(5, len(top) * row_h)))
    norm   = plt.Normalize(top['Importance'].min(), top['Importance'].max())
    colors = plt.cm.RdYlGn(norm(top['Importance'].values))
    bars   = ax.barh(top['Feature'], top['Importance'],
                     color=colors, edgecolor='white', height=0.60)

    for bar, val in zip(bars, top['Importance']):
        ax.text(bar.get_width() + top['Importance'].max() * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=9)

    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, top['Importance'].max() * 1.22)
    ax.tick_params(axis='y', labelsize=10, pad=6)
    fig.tight_layout()
    _save(fig, save_path, 'feature importance')


def plot_predictions(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     title: str = "Actual vs Predicted AQI",
                     save_path: str = None):
    """3-panel: scatter coloured by AQI category + time-series + residual histogram."""
    from sklearn.metrics import r2_score, mean_absolute_error
    from scipy import stats
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)

    # --- Scatter ---
    ax = axes[0]
    cat_colors = [AQI_BAND[_aqi_category(v)][2] for v in y_true]
    ax.scatter(y_true, y_pred, c=cat_colors, alpha=0.45, s=18, zorder=5)
    lims = [min(y_true.min(), y_pred.min()) - 10,
            max(y_true.max(), y_pred.max()) + 10]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect fit')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual AQI'); ax.set_ylabel('Predicted AQI')
    ax.set_title('Scatter: Actual vs Predicted')
    ax.text(0.03, 0.97, f'R² = {r2:.3f}\nRMSE = {rmse:.1f}\nMAE  = {mae:.1f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    patches = [mpatches.Patch(color=c, alpha=0.7, label=nm)
               for nm, (lo, hi, c) in AQI_BAND.items()]
    ax.legend(handles=patches, fontsize=8, loc='lower right', title='AQI Cat.')

    # --- Time series ---
    ax = axes[1]
    idx = np.arange(len(y_true))
    ax.fill_between(idx, y_true, y_pred, alpha=0.18, color='orange', label='Error band')
    ax.plot(idx, y_true, linewidth=1.8, color=PALETTE[0], label='Actual',    alpha=0.85)
    ax.plot(idx, y_pred, linewidth=1.8, color=PALETTE[1], label='Predicted', alpha=0.85)
    _add_aqi_bands(ax)
    ax.set_xlabel('Test Sample Index'); ax.set_ylabel('AQI')
    ax.set_title('Predicted vs Actual (Test Set)')
    ax.legend(framealpha=0.8)

    # --- Residual histogram + normal fit ---
    ax = axes[2]
    residuals = y_true - y_pred
    ax.hist(residuals, bins=40, color=PALETTE[2], edgecolor='white',
            alpha=0.80, density=True)
    mu, std = stats.norm.fit(residuals)
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, mu, std), linewidth=2,
            color='black', label=f'N(μ={mu:.1f}, σ={std:.1f})')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Residual (Actual − Predicted)'); ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend(fontsize=9)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    _save(fig, save_path, 'predictions')


def plot_model_comparison(comparison_df: pd.DataFrame,
                          metric: str = 'RMSE',
                          title: str = None,
                          save_path: str = None):
    """Side-by-side bar charts for RMSE, MAE and R² with data labels and best-model highlight."""
    metrics_available = [m for m in ['RMSE', 'MAE', 'R2'] if m in comparison_df.columns]
    if not metrics_available:
        logger.warning("No recognised metrics (RMSE, MAE, R2) in comparison_df.")
        return

    n_met = len(metrics_available)
    fig, axes = plt.subplots(1, n_met, figsize=(7 * n_met, 6))
    if n_met == 1:
        axes = [axes]

    for ax, met in zip(axes, metrics_available):
        asc = met in ('RMSE', 'MAE', 'MSE')
        sdf = comparison_df.sort_values(met, ascending=asc).reset_index(drop=True)

        bar_colors = [PALETTE[0] if i == 0 else '#AECFD8' for i in range(len(sdf))]
        bars = ax.bar(sdf['Model'], sdf[met],
                      color=bar_colors, edgecolor='white', width=0.6)

        # Highlight best with gold border
        bars[0].set_edgecolor('gold'); bars[0].set_linewidth(2.5)

        # Value labels
        max_v = sdf[met].max()
        for bar, val in zip(bars, sdf[met]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_v * 0.01,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        direction = '↓ Lower is better' if asc else '↑ Higher is better'
        ax.set_title(f'{met}  ({direction})', fontsize=13)
        ax.set_ylabel(met)
        ax.set_xticks(range(len(sdf)))
        ax.set_xticklabels(sdf['Model'], rotation=40, ha='right', fontsize=10)
        ax.set_ylim(0, max_v * 1.26)
        # Centre the star annotation over the best (first) bar
        best_x = bars[0].get_x() + bars[0].get_width() / 2
        ax.text(best_x, max_v * 1.18, '★ Best', fontsize=9,
                color='goldenrod', ha='center', va='bottom', fontweight='bold')

    if title is None:
        title = "Model Performance Comparison"
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(title, fontsize=16, fontweight='bold')
    _save(fig, save_path, 'model comparison')


def plot_aqi_distribution(aqi_values: np.ndarray,
                          title: str = "AQI Distribution",
                          save_path: str = None):
    """Colour-coded histogram by AQI category + pie chart of category breakdown."""
    aqi = np.asarray(aqi_values, dtype=float)
    cat_counts = {name: int(np.sum((aqi >= lo) & (aqi <= hi)))
                  for name, (lo, hi, _) in AQI_BAND.items()}

    fig = plt.figure(figsize=(18, 7))
    gs  = fig.add_gridspec(1, 3, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])

    # --- Histogram coloured by category ---
    for name, (lo, hi, color) in AQI_BAND.items():
        mask = (aqi >= lo) & (aqi <= hi)
        if mask.sum():
            ax1.hist(aqi[mask], bins=40, color=color, alpha=0.85,
                     edgecolor='white', label=f'{name}  (n={mask.sum()})',
                     range=(lo, min(hi, aqi.max() + 1)))

    ax1.axvline(np.mean(aqi),   color='black',  linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(aqi):.0f}')
    ax1.axvline(np.median(aqi), color='dimgrey', linestyle=':', linewidth=2,
                label=f'Median = {np.median(aqi):.0f}')
    ax1.set_xlabel('AQI', fontsize=13); ax1.set_ylabel('Number of Days', fontsize=13)
    ax1.set_title('AQI Frequency Distribution by Category', fontsize=14)
    ax1.legend(fontsize=9, loc='upper right')
    stats_txt = (f"Min   : {aqi.min():.0f}\n"
                 f"Max   : {aqi.max():.0f}\n"
                 f"Mean  : {np.mean(aqi):.0f}\n"
                 f"Median: {np.median(aqi):.0f}\n"
                 f"Std   : {np.std(aqi):.0f}")
    ax1.text(0.02, 0.97, stats_txt, transform=ax1.transAxes, fontsize=9,
             va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))

    # --- Pie chart (label has % already – skip autopct to avoid duplication) ---
    non_zero = [(v, f'{k}\n{v} days\n({v/len(aqi)*100:.1f}%)', AQI_BAND[k][2])
                for k, v in cat_counts.items() if v > 0]
    if non_zero:
        sz, lb, cl = zip(*non_zero)
        ax2.pie(sz, labels=lb, colors=cl, explode=[0.04]*len(sz),
                startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=1.5),
                textprops={'fontsize': 9})
    ax2.set_title('AQI Category Breakdown', fontsize=14, pad=12)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(title, fontsize=16, fontweight='bold')
    _save(fig, save_path, 'AQI distribution')


def plot_residuals(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   title: str = "Residual Analysis",
                   save_path: str = None):
    """4-panel: residuals vs predicted, distribution+KDE, Q-Q, residuals over time."""
    from scipy import stats
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.35, s=15, color=PALETTE[0])
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.axhline( 2*residuals.std(), color='orange', linestyle=':', linewidth=1.5, label='±2σ')
    ax.axhline(-2*residuals.std(), color='orange', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Predicted AQI'); ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Predicted Values')
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.hist(residuals, bins=35, color=PALETTE[2], alpha=0.75, edgecolor='white',
            density=True, label='Residuals')
    mu, std = stats.norm.fit(residuals)
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, mu, std), 'k-', linewidth=2,
            label=f'Normal fit  μ={mu:.1f}, σ={std:.1f}')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Residual'); ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist='norm')
    ax.scatter(osm, osr, s=12, color=PALETTE[0], alpha=0.6)
    ax.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2, label='Normal line')
    ax.set_xlabel('Theoretical Quantiles'); ax.set_ylabel('Sample Quantiles')
    ax.set_title('Q-Q Plot  (Normality Check)')
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(residuals, linewidth=0.8, alpha=0.65, color=PALETTE[0])
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    roll = pd.Series(residuals).rolling(30, min_periods=1).mean()
    ax.plot(roll, color=PALETTE[1], linewidth=2, label='30-day rolling mean')
    ax.set_xlabel('Sample Index'); ax.set_ylabel('Residual')
    ax.set_title('Residuals Over Test Set')
    ax.legend(fontsize=9)
    rmse = np.sqrt(np.mean(residuals**2))
    ax.text(0.02, 0.95, f'RMSE = {rmse:.2f}', transform=ax.transAxes, fontsize=10,
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(title, fontsize=16, fontweight='bold')
    _save(fig, save_path, 'residual analysis')


def plot_learning_curve(history: Dict,
                        title: str = "LSTM Training History",
                        save_path: str = None):
    """Loss & MAE training curves with best-epoch marker."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (tr_key, val_key, ylabel) in zip(
            axes,
            [('loss', 'val_loss', 'Loss (MSE)'),
             ('mae',  'val_mae',  'MAE')]):
        if tr_key in history:
            ax.plot(history[tr_key], linewidth=2, color=PALETTE[0],
                    label='Train', marker='o', markersize=3)
        if val_key in history:
            ax.plot(history[val_key], linewidth=2, color=PALETTE[1],
                    label='Validation', marker='s', markersize=3)
            best_ep = int(np.argmin(history[val_key]))
            best_v  = history[val_key][best_ep]
            ax.axvline(best_ep, color='green', linestyle='--', linewidth=1.5,
                       alpha=0.8, label=f'Best epoch {best_ep}  ({best_v:.4f})')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(f'Model {ylabel}')
        ax.legend(framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(title, fontsize=16, fontweight='bold')
    _save(fig, save_path, 'learning curve')


# ── Bonus: Seasonal Analysis ──────────────────────────────────────────────────
def plot_seasonal_analysis(df: pd.DataFrame,
                           target_col: str = 'AQI',
                           date_col: str = 'Date',
                           title: str = "Seasonal AQI Analysis",
                           save_path: str = None):
    """Month-wise box plots + year-wise grouped bar showing seasonal patterns."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d['Month']     = d[date_col].dt.month
    d['MonthName'] = d[date_col].dt.strftime('%b')
    d['Year']      = d[date_col].dt.year

    month_order = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Month-wise box plot ---
    ax = axes[0]
    month_data = [d[d['MonthName'] == m][target_col].dropna().values
                  for m in month_order]
    bp = ax.boxplot(month_data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.3))
    for patch, data in zip(bp['boxes'], month_data):
        if len(data):
            med = np.median(data)
            for nm, (lo, hi, color) in AQI_BAND.items():
                if lo <= med <= hi:
                    patch.set_facecolor(color); patch.set_alpha(0.75); break
    _add_aqi_bands(ax)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(month_order)
    ax.set_xlabel('Month'); ax.set_ylabel(target_col)
    ax.set_title('Month-wise AQI Distribution')
    for i, data in enumerate(month_data):
        if len(data):
            ax.text(i + 1, np.median(data) + 5, f'{np.median(data):.0f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                              edgecolor='none', alpha=0.75))

    # --- Year × Month grouped bar ---
    ax = axes[1]
    years = sorted(d['Year'].unique())
    x     = np.arange(12)
    width = 0.8 / max(len(years), 1)
    for i, yr in enumerate(years):
        vals = []
        for m in month_order:
            row = d[(d['Year'] == yr) & (d['MonthName'] == m)]
            vals.append(row[target_col].mean() if len(row) else np.nan)
        offset = (i - len(years) / 2) * width + width / 2
        ax.bar(x + offset, vals, width * 0.9,
               label=str(yr), color=PALETTE[i % len(PALETTE)], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(month_order)
    ax.set_xlabel('Month'); ax.set_ylabel(f'Mean {target_col}')
    ax.set_title('Year-wise Monthly Average AQI')
    ax.legend(title='Year', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    _add_aqi_bands(ax)

    fig.tight_layout(rect=[0, 0, 0.87, 0.95])   # leave room for year-legend + suptitle
    fig.suptitle(title, fontsize=16, fontweight='bold')
    _save(fig, save_path, 'seasonal analysis')


def plot_yearly_trend(df: pd.DataFrame,
                      target_col: str = 'AQI',
                      date_col: str = 'Date',
                      title: str = "Year-wise AQI Trend",
                      save_path: str = None):
    """Bar chart of annual mean AQI with ±1σ error bars, coloured by AQI category."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d['Year']   = d[date_col].dt.year
    yearly = d.groupby('Year')[target_col].agg(['mean','median','std']).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    _add_aqi_bands(ax)
    ax.bar(yearly['Year'], yearly['mean'],
           color=[AQI_BAND[_aqi_category(v)][2] for v in yearly['mean']],
           alpha=0.75, edgecolor='white', width=0.6, label='Annual mean')
    ax.errorbar(yearly['Year'], yearly['mean'], yerr=yearly['std'],
                fmt='none', color='black', capsize=5, linewidth=1.5, label='±1 std dev')
    ax.plot(yearly['Year'], yearly['median'],
            'o--', color='black', linewidth=2, markersize=7, label='Median')
    for _, row in yearly.iterrows():
        ax.text(row['Year'], row['mean'] + row['std'] + 6,
                f"{row['mean']:.0f}", ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.75))
    ax.set_xlabel('Year', fontsize=12); ax.set_ylabel('AQI', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(yearly['Year'])
    ax.set_xticklabels([str(int(y)) for y in yearly['Year']], rotation=0)
    patches = [mpatches.Patch(color=c, alpha=0.5, label=f'{nm} ({lo}–{hi})')
               for nm, (lo, hi, c) in AQI_BAND.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=8,
              title='AQI Category', title_fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save(fig, save_path, 'yearly trend')


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.preprocessing import preprocess_data
    df = load_data("Delhi")
    df_clean = preprocess_data(df, remove_outliers=False)
    plot_time_series(df_clean, ['AQI'], save_path='test_ts.png')
    plot_seasonal_analysis(df_clean, save_path='test_seasonal.png')
    plot_aqi_distribution(df_clean['AQI'].values, save_path='test_dist.png')
    plot_yearly_trend(df_clean, save_path='test_yearly.png')
    print("Done.")
