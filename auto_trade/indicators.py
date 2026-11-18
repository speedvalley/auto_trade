"""テクニカル指標(価格から計算する「ものさし」)。どれも過去のデータだけを使う。"""
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """単純移動平均:直近 window 日の平均価格。"""
    return close.rolling(window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI:0〜100。低いほど売られすぎ、高いほど買われすぎ。"""
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """効率比:1に近い=一直線(トレンド相場)、0に近い=行ったり来たり(レンジ相場)。"""
    net_move = (close - close.shift(window)).abs()
    total_move = close.diff().abs().rolling(window).sum()
    return net_move / total_move
