"""自動テスト:  python3 tests/test_core.py

「未来を見ていないか」「コストを引いているか」など、成績の信頼性に関わる部分を確かめる。
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
import news
from data import clean_close
from engine import portfolio_returns
from strategies import apply_min_hold, long_term_weights, short_term_positions


def _prices(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)


def test_no_lookahead_in_engine():
    """k日目に初めて持ち高を決めても、k日目の値動きでは儲からない(翌日から効く)。"""
    p = _prices()
    w = pd.DataFrame({"A": 0.0}, index=p.index)
    k = 100
    w.iloc[k:, 0] = 1.0
    r = portfolio_returns(w, p.to_frame("A"), pd.Series({"A": 0.0}))
    assert (r.iloc[: k + 1] == 0).all(), "決めた当日までのリターンは0のはず"
    assert r.iloc[k + 1] != 0


def test_costs_reduce_returns():
    p = _prices()
    w = pd.DataFrame({"A": (np.arange(len(p)) % 20 < 10).astype(float)}, index=p.index)
    free = portfolio_returns(w, p.to_frame("A"), pd.Series({"A": 0.0})).sum()
    paid = portfolio_returns(w, p.to_frame("A"), pd.Series({"A": 0.005})).sum()
    assert paid < free


def test_indicators_use_only_past():
    """後ろのデータを書き換えても、過去の持ち高は変わらない。"""
    p = _prices()
    kw = dict(fast=10, slow=50, rsi_window=5, buy_below=30, er_threshold=0.3)
    a = short_term_positions(p, **kw)
    q = p.copy()
    q.iloc[400:] *= 3
    b = short_term_positions(q, **kw)
    assert (a.iloc[:400] == b.iloc[:400]).all()


def test_min_hold():
    pos = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    out = apply_min_hold(pos, 3)
    changes = np.flatnonzero(np.diff(out) != 0) + 1
    assert all(np.diff(changes) >= 3), f"売買の間隔が短すぎる: {out}"


def test_long_term_weights_valid():
    idx = pd.bdate_range("2019-01-01", periods=800)
    up = pd.Series(np.linspace(100, 300, 800), index=idx)
    down = pd.Series(np.linspace(300, 100, 800), index=idx)
    w = long_term_weights(pd.DataFrame({"UP": up, "DOWN": down}), top_k=2, mom_lookback=252,
                          mom_skip=21, trend_window=200)
    last = w.iloc[-1]
    assert last["UP"] == 0.5 and last["DOWN"] == 0.0, "上昇中だけを選び、残りは現金のはず"
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_clean_close_fixes_split_and_glitch():
    s = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=60))
    s.iloc[:30] = 1000.0          # 分割の未反映(前半だけ10倍)
    s.iloc[45:47] = 10.0          # 2日だけの誤データ
    notes = []
    out = clean_close(s, "X", notes)
    assert out.max() < 150 and out.min() > 50 and len(out) == 58 and len(notes) == 2


def test_news_validation():
    universe = {"AAPL"}
    today = datetime.date(2026, 11, 18)
    good = {"ticker": "AAPL", "date": "2026-11-17", "headline": "h", "event_type": "earnings",
            "sentiment": 1, "confidence": 0.8, "source_url": "https://example.com/a"}
    assert news.validate_item(good, universe, today) == []
    assert news.validate_item({**good, "sentiment": 5}, universe, today)
    assert news.validate_item({**good, "source_url": ""}, universe, today)
    assert news.validate_item({**good, "date": "2026-12-01"}, universe, today)
    assert news.validate_item({**good, "ticker": "ZZZZ"}, universe, today)


def test_news_overlay():
    today = datetime.date(2026, 11, 18)
    base = pd.Series({"AAPL": 0.05, "MSFT": 0.05, "NVDA": 0.05})
    mk = lambda t, ev, s: {"ticker": t, "date": "2026-11-18", "headline": "h", "event_type": ev,
                           "sentiment": s, "confidence": 1.0, "source_url": "https://x"}
    items = [mk("AAPL", "fraud", -2), mk("MSFT", "earnings", -2), mk("NVDA", "earnings", 2)]
    final, _ = news.overlay(base, items, today)
    assert final["AAPL"] == 0.0                      # 重大な悪材料は持たない
    assert abs(final["MSFT"] - 0.025) < 1e-12        # 悪材料は半分
    assert abs(final["NVDA"] - 0.0625) < 1e-12       # 好材料は1.25倍
    assert final.max() <= config.MAX_ASSET_WEIGHT


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("OK  ", t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", t.__name__, e)
    print(f"\n{len(tests) - failed}/{len(tests)} 合格")
    sys.exit(1 if failed else 0)
