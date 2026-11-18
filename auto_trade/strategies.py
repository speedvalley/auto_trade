"""戦略の本体。どれも「その日の終値までの情報」だけで判断する(未来は見ない)。"""
import numpy as np
import pandas as pd

from indicators import efficiency_ratio, rsi, sma


# ============================================================
# 長期戦略:モメンタム + 200日線フィルター(月末に見直し)
# ============================================================
def long_term_weights(prices: pd.DataFrame, top_k: int, mom_lookback: int,
                      mom_skip: int, trend_window: int) -> pd.DataFrame:
    """月末ごとに、勢いの強い上位 top_k 銘柄を等分で持つ(市場枠に対する割合)。

    - 勢い = 約1年前から約1か月前までの値上がり率(直近1か月は反動が出やすいので除く)
    - 条件: 勢いがプラス かつ 価格が200日平均より上。満たす銘柄が足りなければ残りは現金
    """
    mom = prices.shift(mom_skip) / prices.shift(mom_lookback) - 1
    above = prices > prices.rolling(trend_window).mean()
    eligible = mom.where(above & (mom > 0))

    idx = prices.index
    is_month_end = ~idx.to_period("M").duplicated(keep="last")
    # データの最終日は、本当に月の最終営業日のときだけ「月末」とみなす
    is_month_end[-1] = (idx[-1] + pd.offsets.BDay(1)).month != idx[-1].month

    w = pd.DataFrame(np.nan, index=idx, columns=prices.columns)
    for d in idx[is_month_end]:
        picks = eligible.loc[d].dropna().nlargest(top_k).index
        w.loc[d] = 0.0
        w.loc[d, picks] = 1.0 / top_k
    return w.ffill().fillna(0.0)  # 次の月末まで同じ持ち高を続ける


# ============================================================
# 短期戦略:レジーム適応(効率比でトレンド/レンジを判定し、順張り/逆張りを切り替え)
# ============================================================
def apply_min_hold(pos: np.ndarray, n: int) -> np.ndarray:
    """持ち高を変えたら、最低 n 日は変えない(コストのかさむ頻繁な売買を防ぐ)。"""
    out = pos.copy()
    hold = n  # 最初は制限なし
    for i in range(1, len(out)):
        want = pos[i]
        if want != out[i - 1] and hold < n:
            want = out[i - 1]
        hold = hold + 1 if want == out[i - 1] else 1
        out[i] = want
    return out


def short_term_positions(close: pd.Series, fast: int, slow: int, rsi_window: int,
                         buy_below: float, er_threshold: float, er_window: int = 20,
                         sell_above: float = 60, er_hysteresis: float = 0.1,
                         min_hold: int = 5) -> pd.Series:
    """持つ(1)/持たない(0)を返す。

    ヒステリシス: トレンド相場に入る基準(er_threshold)より、出る基準
    (er_threshold - er_hysteresis)を低くして、境目でのふらつきを防ぐ。
    """
    er = efficiency_ratio(close, er_window)
    trending = pd.Series(np.nan, index=close.index)
    trending[er >= er_threshold] = 1
    trending[er < er_threshold - er_hysteresis] = 0
    trending = trending.ffill()

    trend_pos = (sma(close, fast) > sma(close, slow)).astype(float)  # 順張り

    r = rsi(close, rsi_window)                                       # 逆張り
    mr = pd.Series(np.nan, index=close.index)
    mr[r < buy_below] = 1
    mr[r > sell_above] = 0
    mr = mr.ffill().fillna(0.0)

    raw = trend_pos.where(trending == 1, mr)
    raw = raw.where(trending.notna(), 0.0).fillna(0.0)  # 指標が未完成の期間は何もしない
    return pd.Series(apply_min_hold(raw.to_numpy(), min_hold), index=close.index)
