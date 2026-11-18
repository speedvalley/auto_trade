"""ペーパートレード台帳:実際のお金は動かさず、判断どおりに売買したと仮定して記録する。

同じ日に2回実行しても二重に記録しない。終値で約定した想定(実際は値段がずれる)。
"""
import os

import pandas as pd

import config


def _paths():
    os.makedirs(config.LEDGER_DIR, exist_ok=True)
    return os.path.join(config.LEDGER_DIR, "equity.csv"), os.path.join(config.LEDGER_DIR, "weights.csv")


def previous_weights() -> pd.Series:
    """直近に記録した持ち高(なければ空)。"""
    _, w_path = _paths()
    if not os.path.exists(w_path):
        return pd.Series(dtype=float)
    w = pd.read_csv(w_path, parse_dates=["date"])
    last = w[w["date"] == w["date"].max()]
    return last.set_index("ticker")["weight"]


def update(asof: pd.Timestamp, target: pd.Series, prices_jpy: pd.DataFrame, costs: pd.Series) -> pd.DataFrame:
    """目標の持ち高で台帳を更新し、資産推移を返す。"""
    eq_path, w_path = _paths()
    if os.path.exists(eq_path):
        eq = pd.read_csv(eq_path, index_col=0, parse_dates=True)
    else:
        eq = pd.DataFrame(columns=["equity", "invested"])
    if len(eq) and asof <= eq.index[-1]:
        return eq  # 記録済みの日

    prev_w = previous_weights()
    if len(eq):
        last = eq.index[-1]
        move = prices_jpy.loc[asof] / prices_jpy.loc[last] - 1
        gross = (prev_w * move.reindex(prev_w.index).fillna(0.0)).sum()
        equity = eq["equity"].iloc[-1] * (1 + gross)
    else:
        equity = float(config.INITIAL_CAPITAL)
    diff = target.sub(prev_w.reindex(target.index).fillna(0.0)).abs()
    equity *= 1 - (diff * costs.reindex(target.index).fillna(0.0)).sum()

    eq.loc[asof] = [equity, target.sum()]
    eq.to_csv(eq_path)
    rows = pd.DataFrame({"date": asof, "ticker": target.index, "weight": target.values})
    rows = rows[rows["weight"] > 0]
    rows.to_csv(w_path, mode="a", header=not os.path.exists(w_path), index=False)
    return eq
