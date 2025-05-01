import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def load_lob_data(filepath: str) -> pd.DataFrame:
    """
    Load LOB data from CSV. Adjust column names as needed.
    Assumes columns: 'timestamp', 'bid1_p', 'bid1_s', ..., 'bidN_p', 'bidN_s', 'ask1_p', 'ask1_s', ..., 'symbol'
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


def _compute_ofi_side(px: np.ndarray, sz: np.ndarray) -> np.ndarray:
    """
    Vectorized OFI for one side (bid or ask) as per Cont et al. (2014).
    """
    # compute previous px and sz
    px_prev = np.empty_like(px)
    sz_prev = np.empty_like(sz)
    px_prev[0], sz_prev[0] = px[0], sz[0]
    px_prev[1:], sz_prev[1:] = px[:-1], sz[:-1]

    dp = px - px_prev
    dq = sz - sz_prev

    # apply rules
    ofi = np.where(dp > 0,  sz_prev,
           np.where(dp < 0, -sz_prev,
                    dq))
    return ofi


def compute_best_level_ofi(df: pd.DataFrame) -> np.ndarray:
    """
    Compute best-level OFI per event (level 1).
    """
    bid_p = df['bid1_p'].to_numpy()
    bid_s = df['bid1_s'].to_numpy()
    ask_p = df['ask1_p'].to_numpy()
    ask_s = df['ask1_s'].to_numpy()

    ofi_bid = _compute_ofi_side(bid_p, bid_s)
    # for ask, flip sign
    ofi_ask = -_compute_ofi_side(ask_p, ask_s)

    # best-ofi is sum
    return ofi_bid + ofi_ask


def compute_multilevel_ofi(df: pd.DataFrame, max_levels: int = 10) -> np.ndarray:
    """
    Compute multi-level OFI array (n_events, max_levels).
    """
    ofi_levels = []
    for lvl in range(1, max_levels+1):
        bid_p = df[f'bid{lvl}_p'].to_numpy()
        bid_s = df[f'bid{lvl}_s'].to_numpy()
        ask_p = df[f'ask{lvl}_p'].to_numpy()
        ask_s = df[f'ask{lvl}_s'].to_numpy()
        lvl_ofi = _compute_ofi_side(bid_p, bid_s) - _compute_ofi_side(ask_p, ask_s)
        ofi_levels.append(lvl_ofi)
    return np.stack(ofi_levels, axis=1)


def aggregate_ofi_array(timestamps: pd.Series, ofi_array: np.ndarray, freq: str = '1T') -> pd.DataFrame:
    """
    Aggregate each OFI column array over fixed time bars.
    Returns DataFrame with same number of columns as ofi_array.
    """
    ts_index = pd.DatetimeIndex(timestamps)
    ofi_df = pd.DataFrame(ofi_array, index=ts_index,
                          columns=[f'ofi_{i+1}' for i in range(ofi_array.shape[1])])
    return ofi_df.resample(freq).sum()


def compute_integrated_ofi(multi_ofi_df: pd.DataFrame) -> (pd.Series, np.ndarray):
    """
    Compute integrated OFI via first PCA component (L1 normalized).
    Returns (integrated_ofi, weights).
    """
    pca = PCA(n_components=1)
    pca.fit(multi_ofi_df)
    comp = pca.components_[0]
    weights = comp / np.sum(np.abs(comp))
    integrated = multi_ofi_df.values @ weights
    return pd.Series(integrated, index=multi_ofi_df.index), weights


def compute_cross_asset_partial(int_df: pd.DataFrame) -> pd.DataFrame:
    """
    Partial cross-asset OFI: for each asset, sum of other assets' integrated OFI.
    """
    total = int_df.sum(axis=1)
    return total.values.reshape(-1,1) - int_df


def compute_cross_asset_regression(int_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Regression-based cross-asset OFI: regress each asset's returns on others' OFIs.
    price_df: indexed same as int_df, contains mid-price series for returns.
    """
    returns = price_df.pct_change().fillna(0)
    assets = int_df.columns
    cross_reg = pd.DataFrame(index=int_df.index, columns=assets)
    for a in assets:
        X = int_df.drop(columns=a)
        y = returns[a]
        lr = LinearRegression().fit(X, y)
        cross_reg[a] = X.values @ lr.coef_
    return cross_reg


if __name__ == '__main__':
    # rectify file paths and symbols
    file_map = {
        'AAPL': 'first_25000_rows.csv',
        # add more symbols if available
    }
    freq = '1T'

    # storage
    best_df = {}
    int_df = {}
    mid_df = {}

    for sym, path in file_map.items():
        df = load_lob_data(path)
        best = compute_best_level_ofi(df)
        best_df[sym] = aggregate_ofi_array(df['timestamp'], best.reshape(-1,1), freq)['ofi_1']

        multi = compute_multilevel_ofi(df, max_levels=10)
        multi_bar = aggregate_ofi_array(df['timestamp'], multi, freq)
        integ, w = compute_integrated_ofi(multi_bar)
        int_df[sym] = integ

        mid = (df['bid1_p'] + df['ask1_p'])/2
        mid_df[sym] = mid.resample(freq, on=df['timestamp']).last().ffill()

    best_feat = pd.DataFrame(best_df)
    int_feat = pd.DataFrame(int_df)
    mid_feat = pd.DataFrame(mid_df)

    cross_partial = compute_cross_asset_partial(int_feat)
    cross_reg = compute_cross_asset_regression(int_feat, mid_feat)

    # save or return features
    best_feat.to_csv('best_ofi.csv')
    int_feat.to_csv('integrated_ofi.csv')
    cross_partial.to_csv('cross_partial.csv')
    cross_reg.to_csv('cross_regression.csv')