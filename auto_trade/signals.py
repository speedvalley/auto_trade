"""戦略の組み立て:長期・短期のシグナルを作り、市場配分と合わせて目標の持ち高にする。"""
from itertools import product

import numpy as np
import pandas as pd

import config
from data import MarketData
from engine import TRADING_DAYS
from strategies import long_term_weights, short_term_positions


def market_assets(data: MarketData, market: str) -> list:
    return [a for a, m in data.asset_market.items() if m == market]


def build_long(data: MarketData) -> pd.DataFrame:
    """長期戦略の持ち高(市場ごとに計算し、1つの表にまとめる。値は市場枠に対する割合)。"""
    out = pd.DataFrame(0.0, index=data.local_ff.index, columns=data.local_ff.columns)
    for m, cfg in config.MARKETS.items():
        cols = market_assets(data, m)
        if cols:
            out[cols] = long_term_weights(data.local_ff[cols], top_k=cfg["top_k"], **config.LONG)
    return out


def short_to_weights(pos: pd.DataFrame, data: MarketData) -> pd.DataFrame:
    """短期シグナル(0/1)を市場枠に対する割合へ。1銘柄 1/SHORT_SLOTS、超えたら等分に縮める。"""
    out = pd.DataFrame(0.0, index=pos.index, columns=pos.columns)
    for m in config.MARKETS:
        cols = market_assets(data, m)
        if cols:
            denom = pos[cols].sum(axis=1).clip(lower=config.SHORT_SLOTS)
            out[cols] = pos[cols].div(denom, axis=0)
    return out


def assemble(long_w: pd.DataFrame, short_w: pd.DataFrame, data: MarketData,
             long_share: float = None) -> pd.DataFrame:
    """市場配分 × (長期 と 短期の資金配分) を掛け合わせ、全資金に対する持ち高にする。"""
    ls = config.LONG_SHARE if long_share is None else long_share
    total = pd.DataFrame(0.0, index=long_w.index, columns=long_w.columns)
    for m in config.MARKETS:
        cols = market_assets(data, m)
        if cols:
            total[cols] = config.MARKET_WEIGHTS[m] * (ls * long_w[cols] + (1 - ls) * short_w[cols])
    return total.clip(upper=config.MAX_ASSET_WEIGHT)


class ShortEngine:
    """短期戦略のウォークフォワード。

    候補の数値の組み合わせ全部について持ち高を先に計算しておく(指標は過去しか見ないので
    全期間で計算しても未来は混ざらない)。毎年「その年より前のデータだけ」で最良の組を選び、
    翌年はその組を使う。
    """

    def __init__(self, data: MarketData):
        self.data = data
        keys = list(config.SHORT_GRID)
        self.combos = [dict(zip(keys, v)) for v in product(*config.SHORT_GRID.values())]
        idx = data.local_ff.index
        local_ret = data.local_ff.pct_change(fill_method=None).fillna(0.0)
        costs = data.costs
        self.pos, self.ret = [], []
        for c in self.combos:
            p = pd.DataFrame(0.0, index=idx, columns=data.local_ff.columns)
            for a in p.columns:
                s = data.local_raw[a].dropna()
                if len(s) >= 300:
                    p[a] = short_term_positions(s, **c, **config.SHORT_FIXED).reindex(idx).ffill().fillna(0.0)
            held = p.shift(1).fillna(0.0)
            turnover = held.diff().abs()
            turnover.iloc[0] = held.iloc[0].abs()
            self.pos.append(p)
            self.ret.append(held * local_ret - turnover * costs.reindex(p.columns).to_numpy())

    def best_combo(self, market: str, end_exclusive: pd.Timestamp):
        """end_exclusive より前のデータだけで、市場内の平均シャープが最高の組を選ぶ。"""
        cols = market_assets(self.data, market)
        idx = self.data.local_raw.index
        mask = idx < end_exclusive
        valid = self.data.local_raw.loc[mask, cols].notna().sum() >= config.MIN_TRAIN_DAYS
        cols = list(valid[valid].index)
        if not cols:
            return None
        scores = []
        for r in self.ret:
            train = r.loc[mask, cols]
            sh = (train.mean() / train.std() * np.sqrt(TRADING_DAYS)).replace([np.inf, -np.inf], np.nan)
            scores.append(sh.fillna(0.0).mean())
        return int(np.argmax(scores))

    def oos_positions(self, first_year: int):
        """テスト年ごとに、その年より前だけで選んだ組の持ち高をつなぎ合わせる。"""
        idx = self.data.local_ff.index
        out = pd.DataFrame(0.0, index=idx, columns=self.data.local_ff.columns)
        log = []
        for year in range(first_year, idx[-1].year + 1):
            rows = idx.year == year
            for m in config.MARKETS:
                best = self.best_combo(m, pd.Timestamp(year, 1, 1))
                if best is None:
                    continue
                cols = market_assets(self.data, m)
                out.loc[rows, cols] = self.pos[best].loc[rows, cols]
                log.append({"year": year, "market": m, **self.combos[best]})
        return out, pd.DataFrame(log)

    def live_positions(self, asof: pd.Timestamp):
        """今日時点の短期シグナル。asof までのデータで数値を選ぶ。"""
        out = pd.DataFrame(0.0, index=[asof], columns=self.data.local_ff.columns)
        choices = {}
        for m in config.MARKETS:
            best = self.best_combo(m, asof + pd.Timedelta(days=1))
            if best is None:
                continue
            cols = market_assets(self.data, m)
            out.loc[asof, cols] = self.pos[best].loc[asof, cols]
            choices[m] = self.combos[best]
        return out, choices
