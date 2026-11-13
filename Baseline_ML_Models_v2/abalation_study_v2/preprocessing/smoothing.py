"""
preprocessing/smoothing.py
==========================
Window-based outlier detection + local smoothing (peak removal), vendored
verbatim from Code/Denoising/CRRT_stream_denoising_abalation.ipynb so build_streams
can apply the SAME peak-removal step as the denoising notebook.
"""

import numpy as np
import pandas as pd


def detect_and_smooth_outliers_skip_nan(
    df, time_col, pid_col, feat_cols,
    visit_col='Visit_no',
    window='40min', threshold=2, min_periods=1, neighbor_count=4,
    fill_nan=True, verbose=False):

    df2 = df.copy()
    df2[time_col] = pd.to_datetime(df2[time_col], errors='coerce')
    df2 = df2.sort_values([pid_col, visit_col, time_col]).reset_index(drop=False)
    df2 = df2.rename(columns={'index': 'orig_index'})
    df2['timestamp'] = df2[time_col]
    df2['win_id'] = df2['timestamp'].dt.floor(window)

    smoothed = df2.copy()
    half = neighbor_count // 2
    next_count = neighbor_count - half

    for (pid, visit), grp in df2.groupby([pid_col, visit_col], sort=False):
        if grp.empty:
            continue
        pos_idx = grp.index.to_numpy()
        idx_arr_local = np.arange(len(grp))
        win_groups = grp.groupby('win_id', sort=False)
        outlier_masks, nan_masks = {}, {}

        for f in feat_cols:
            s = grp[f]
            nan_mask = s.isna()
            nan_masks[f] = nan_mask.to_numpy()
            st = win_groups[f].agg(['count', 'mean', 'std']).rename(columns={'count': 'cnt'})
            if min_periods > 1:
                st.loc[st['cnt'] < min_periods, ['mean', 'std']] = np.nan
            mean_win = grp['win_id'].map(st['mean'])
            std_win = grp['win_id'].map(st['std'])
            z = (s - mean_win) / std_win
            mask = (std_win > 0) & z.abs().gt(threshold) & (~nan_mask)
            outlier_masks[f] = mask.to_numpy()

        group_means = grp[feat_cols].mean(numeric_only=True)

        for f in feat_cols:
            s_vals = grp[f].to_numpy(dtype=float)
            outliers = outlier_masks[f]
            nan_mask = nan_masks[f]
            valid_mask = (~outliers) & (~nan_mask)
            valid_idx_local = idx_arr_local[valid_mask]
            valid_vals = s_vals[valid_mask]

            targets = list(idx_arr_local[outliers])
            if fill_nan:
                targets += list(idx_arr_local[nan_mask])
            for i_local in targets:
                insert_pos = np.searchsorted(valid_idx_local, i_local)
                prev_start = max(0, insert_pos - half)
                prev_neighbors = valid_vals[prev_start:insert_pos]
                next_neighbors = valid_vals[insert_pos:insert_pos + next_count]
                neighbors = np.concatenate([prev_neighbors, next_neighbors])
                search_radius = neighbor_count
                while neighbors.size < neighbor_count and search_radius < len(valid_idx_local):
                    search_radius += neighbor_count
                    ps = max(0, insert_pos - search_radius // 2)
                    ne = min(len(valid_idx_local), insert_pos + search_radius // 2)
                    neighbors = np.concatenate([valid_vals[ps:insert_pos], valid_vals[insert_pos:ne]])
                if neighbors.size > 0:
                    new_val = float(np.mean(neighbors))
                else:
                    gm = group_means.get(f, np.nan)
                    new_val = float(gm) if not np.isnan(gm) else s_vals[i_local]
                if not np.isnan(new_val):
                    smoothed.loc[pos_idx[i_local], f] = new_val

    return smoothed.drop(columns=['win_id', 'timestamp', 'orig_index']).reset_index(drop=True)
