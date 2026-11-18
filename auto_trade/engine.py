"""ポートフォリオのバックテストと成績指標。"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_returns(weights: pd.DataFrame, prices_jpy: pd.DataFrame, costs: pd.Series) -> pd.Series:
    """日次リターン(円ベース)。

    - weights は「その日の終値で決めた目標の持ち高」。shift(1)で翌日の値動きに効かせる
    - 持ち高が変わった分(売買額)にコストを掛けて引く
    - 現金の利息は0として扱う(保守的)
    """
    w = weights.shift(1).fillna(0.0)
    ret = prices_jpy.pct_change(fill_method=None).fillna(0.0)
    turnover = w.diff().abs()
    turnover.iloc[0] = w.iloc[0].abs()
    c = costs.reindex(w.columns).fillna(0.0).to_numpy()
    return (w * ret).sum(axis=1) - (turnover * c).sum(axis=1)


def sharpe(r: pd.Series) -> float:
    s = r.std()
    return float("nan") if s == 0 or np.isnan(s) else r.mean() / s * np.sqrt(TRADING_DAYS)


def summarize(r: pd.Series, weights: pd.DataFrame = None) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    mdd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / years) - 1
    out = {
        "Total": eq.iloc[-1] - 1, "CAGR": cagr,
        "Vol": r.std() * np.sqrt(TRADING_DAYS), "Sharpe": sharpe(r),
        "MaxDD": mdd, "Calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
    }
    if weights is not None:  # 平均の投資比率と、年間の売買回転率
        w = weights.reindex(r.index).fillna(0.0)
        out["Invested"] = w.sum(axis=1).mean()
        out["Turnover/yr"] = w.diff().abs().sum(axis=1).sum() / years
    return out


def yearly_returns(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1
