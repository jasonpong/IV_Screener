import pandas as pd
import numpy as np
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor, as_completed


class IVScanner:
    """Rank variables by Information Value for fraud rule candidate selection.

    Uses LightGBM tree-based binning (consistent with ThresholdAnalyzer) to
    bin each variable, then computes Weight of Evidence and Information Value.

    IV interpretation:
        < 0.02  Useless
        0.02–0.1  Weak
        0.1–0.3   Medium
        0.3–0.5   Strong
        > 0.5     Very strong (check for overfitting / data leakage)

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str - binary 0/1 target column
    min_bin_size : int - minimum obs per bin (default 100)
    max_categories : int - cap for categorical values before grouping (default 15)
    snap_to : int, float, or None - round numeric splits to nearest multiple (default 1)

    Example
    -------
    >>> scanner = IVScanner(df, 'is_fraud')
    >>> results = scanner.scan(['device_age', 'login_count', 'ip_risk_score'])
    >>> print(results)
    >>>
    >>> # Or scan all non-target columns
    >>> results = scanner.scan()
    """

    def __init__(self, df, target_col, min_bin_size=100, max_categories=15, snap_to=1):
        self.df = df
        self.target_col = target_col
        self.min_bin_size = min_bin_size
        self.max_categories = max_categories
        self.snap_to = snap_to
        self.results = None

    def scan(self, variables=None, n_jobs=1):
        """Compute IV for each variable and return a ranked summary.

        Parameters
        ----------
        variables : list of str, optional
            Columns to scan. Defaults to all columns except target.
        n_jobs : int
            Number of parallel workers. Use 1 for debugging, -1 for all cores.

        Returns
        -------
        pd.DataFrame with columns:
            variable, iv, strength, n_bins, min_fraud_rate, max_fraud_rate,
            max_lift, n_records, missing_pct
        """
        if variables is None:
            variables = [c for c in self.df.columns if c != self.target_col]

        if n_jobs == 1:
            rows = [self._score_variable(var) for var in variables]
        else:
            rows = self._parallel_scan(variables, n_jobs)

        self.results = (
            pd.DataFrame(rows)
            .sort_values('iv', ascending=False)
            .reset_index(drop=True)
        )
        return self.results

    # -- Core IV computation --------------------------------------------- #

    def _score_variable(self, variable):
        """Compute IV and summary stats for a single variable."""
        col = self.df[[variable, self.target_col]].copy()
        n_total = len(self.df)
        n_valid = col[variable].notna().sum()
        missing_pct = (1 - n_valid / n_total) * 100 if n_total else 0

        # Drop nulls for binning
        col = col.dropna(subset=[variable])

        if len(col) < self.min_bin_size or col[self.target_col].nunique() < 2:
            return self._empty_row(variable, n_total, missing_pct, reason='insufficient data')

        is_numeric = pd.api.types.is_numeric_dtype(col[variable])

        try:
            if is_numeric:
                bins = self._get_numeric_bins(col, variable)
            else:
                bins = self._get_categorical_bins(col, variable)

            iv, n_bins, min_fr, max_fr = self._compute_iv(col, bins, variable)
        except Exception:
            return self._empty_row(variable, n_total, missing_pct, reason='binning failed')

        pop_rate = col[self.target_col].mean() * 100
        max_lift = (max_fr / pop_rate) if pop_rate > 0 else 0

        return {
            'variable': variable,
            'iv': round(iv, 4),
            'strength': self._iv_strength(iv),
            'n_bins': n_bins,
            'min_fraud_rate': round(min_fr, 2),
            'max_fraud_rate': round(max_fr, 2),
            'max_lift': round(max_lift, 1),
            'n_records': int(n_valid),
            'missing_pct': round(missing_pct, 1),
        }

    def _compute_iv(self, col, bins, variable):
        """Calculate WoE and IV from binned data."""
        tmp = col.copy()
        tmp['bin'] = bins

        grouped = tmp.groupby('bin', observed=True)[self.target_col].agg(['sum', 'count'])
        grouped.columns = ['fraud', 'total']
        grouped['non_fraud'] = grouped['total'] - grouped['fraud']

        total_fraud = grouped['fraud'].sum()
        total_non_fraud = grouped['non_fraud'].sum()

        if total_fraud == 0 or total_non_fraud == 0:
            fraud_rates = grouped['fraud'] / grouped['total'] * 100
            return 0.0, len(grouped), fraud_rates.min(), fraud_rates.max()

        # Distribution of fraud and non-fraud across bins
        # Add small constant to avoid log(0)
        eps = 0.5
        dist_fraud = (grouped['fraud'] + eps) / (total_fraud + eps * len(grouped))
        dist_non_fraud = (grouped['non_fraud'] + eps) / (total_non_fraud + eps * len(grouped))

        woe = np.log(dist_non_fraud / dist_fraud)
        iv = ((dist_non_fraud - dist_fraud) * woe).sum()

        fraud_rates = grouped['fraud'] / grouped['total'] * 100

        return iv, len(grouped), fraud_rates.min(), fraud_rates.max()

    # -- Binning (mirrors ThresholdAnalyzer) ----------------------------- #

    def _get_numeric_bins(self, col, variable):
        """LightGBM tree-based binning, consistent with ThresholdAnalyzer."""
        model = lgb.LGBMClassifier(
            max_depth=4,
            min_child_samples=self.min_bin_size,
            n_estimators=1,
            random_state=42,
            verbose=-1,
        )
        model.fit(col[[variable]], col[self.target_col])

        tree = model.booster_.dump_model()['tree_info'][0]['tree_structure']
        splits = sorted(set(self._extract_splits(tree)))

        if splits and self.snap_to is not None:
            splits = sorted(set(
                round(s / self.snap_to) * self.snap_to for s in splits
            ))

        if not splits:
            return pd.qcut(col[variable], q=5, duplicates='drop')

        return pd.cut(
            col[variable],
            bins=[-np.inf] + splits + [np.inf],
            include_lowest=True,
        )

    def _get_categorical_bins(self, col, variable):
        """Keep top categories by volume, group the rest into 'Other'."""
        counts = col[variable].value_counts()

        if len(counts) <= self.max_categories:
            return col[variable]

        top_cats = set(counts.nlargest(self.max_categories).index)
        return col[variable].where(col[variable].isin(top_cats), other='Other')

    def _extract_splits(self, node):
        """Recursively pull split thresholds from the tree."""
        if 'threshold' not in node:
            return []
        splits = [node['threshold']]
        if 'left_child' in node:
            splits.extend(self._extract_splits(node['left_child']))
        if 'right_child' in node:
            splits.extend(self._extract_splits(node['right_child']))
        return splits

    # -- Helpers --------------------------------------------------------- #

    def _parallel_scan(self, variables, n_jobs):
        """Run scoring in parallel for large variable lists."""
        if n_jobs == -1:
            import os
            n_jobs = os.cpu_count()

        rows = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(self._score_variable, var): var
                for var in variables
            }
            for future in as_completed(futures):
                rows.append(future.result())
        return rows

    @staticmethod
    def _iv_strength(iv):
        if iv < 0.02:
            return 'Useless'
        elif iv < 0.1:
            return 'Weak'
        elif iv < 0.3:
            return 'Medium'
        elif iv < 0.5:
            return 'Strong'
        else:
            return 'Very Strong'

    @staticmethod
    def _empty_row(variable, n_total, missing_pct, reason=''):
        return {
            'variable': variable,
            'iv': 0.0,
            'strength': reason,
            'n_bins': 0,
            'min_fraud_rate': 0.0,
            'max_fraud_rate': 0.0,
            'max_lift': 0.0,
            'n_records': int(n_total),
            'missing_pct': round(missing_pct, 1),
        }
