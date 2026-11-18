"""株価データの取得・円換算・品質チェック。"""
import datetime
import os
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

import config


@dataclass
class MarketData:
    local_raw: pd.DataFrame   # 現地通貨の終値(休場日はNaNのまま)
    local_ff: pd.DataFrame    # 同・休場日を前日の値で埋めたもの
    jpy: pd.DataFrame         # 円換算した終値(損益の計算に使う)
    asset_market: dict        # 銘柄 -> 市場名
    costs: pd.Series          # 銘柄ごとの売買コスト
    warnings: list = field(default_factory=list)


def _read_close(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["Close"].astype(float)


def load_close(ticker: str) -> pd.Series:
    """終値を返す。data/manual/ にCSVがあればそれを優先し、なければyfinanceから取る。"""
    manual = os.path.join(config.MANUAL_DIR, f"{ticker}.csv")
    if os.path.exists(manual):
        return _read_close(manual).dropna()

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    path = os.path.join(config.CACHE_DIR, f"{ticker}.csv")
    if os.path.exists(path):  # 今日保存したものがあれば再ダウンロードしない
        if datetime.date.fromtimestamp(os.path.getmtime(path)) == datetime.date.today():
            return _read_close(path).dropna()

    df = yf.download(ticker, start=config.START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError("データが取得できません")
    df[["Close"]].to_csv(path)
    return df["Close"].astype(float)


def clean_close(s: pd.Series, ticker: str, notes: list) -> pd.Series:
    """データ提供元の不具合を直す。1日で40%超動いた日を調べ、次のどれかで処理する。

    - 数日以内に元の水準へ戻る → 一時的な誤データなので、その日を削除
    - 戻らず、変動が 1/n 倍(n=2,3,4,5,10,20)に近い → 株式分割の未反映なので、過去を補正
    - どちらでもない → 本当の急変動かもしれないので、そのまま残して警告
    """
    s = s.copy()
    seen = set()
    while True:
        r = s.pct_change()
        todo = [d for d in r[r.abs() > 0.4].index if d not in seen]
        if not todo:
            return s
        d = todo[0]
        seen.add(d)
        i = s.index.get_loc(d)
        base, ratio = s.iloc[i - 1], s.iloc[i] / s.iloc[i - 1]
        window = s.iloc[i + 1:i + 6] / base
        back = window[(window > 0.75) & (window < 1.33)]
        if len(back):
            j = s.index.get_loc(back.index[0])
            notes.append(f"{ticker}: {d.date()} 〜 {s.index[j - 1].date()} の誤データ{j - i}日を削除")
            s = s.drop(s.index[i:j])
            continue
        inv = 1 / ratio if ratio < 1 else ratio
        n = round(inv)
        if n in (2, 3, 4, 5, 10, 20) and abs(inv - n) / n < 0.06:
            s.iloc[:i] = s.iloc[:i] * ratio
            notes.append(f"{ticker}: {d.date()} の株式分割(1:{n})が未反映だったため過去価格を補正")
        else:
            notes.append(f"{ticker}: {d.date()} に1日で{ratio - 1:+.0%}の変動(本物かデータ異常か要確認。補正せず)")


def load_all() -> MarketData:
    warnings, local, asset_market, cost = [], {}, {}, {}
    for market, cfg in config.MARKETS.items():
        for t in cfg["tickers"]:
            try:
                s = load_close(t)
            except Exception as e:  # 1銘柄の失敗で全体を止めない
                warnings.append(f"{t}: 取得失敗のため除外 ({e})")
                continue
            s = clean_close(s[s > 0], t, warnings)
            local[t] = s
            asset_market[t] = market
            cost[t] = cfg["cost"]

    local_raw = pd.DataFrame(local).sort_index()
    local_raw = local_raw[~local_raw.index.duplicated()]
    local_ff = local_raw.ffill(limit=5)

    # 円換算:ドル建ての銘柄には、その日のドル円を掛ける
    fx = {cur: load_close(sym).reindex(local_raw.index).ffill() for cur, sym in config.FX.items()}
    jpy = local_ff.copy()
    for t, market in asset_market.items():
        cur = config.MARKETS[market]["currency"]
        if cur != "JPY":
            jpy[t] = local_ff[t] * fx[cur]

    last = local_raw.apply(lambda s: s.last_valid_index())
    for t, d in last.items():
        if (local_raw.index[-1] - d).days > 7:
            warnings.append(f"{t}: 最新データが古い({d.date()})")
    return MarketData(local_raw, local_ff, jpy, asset_market, pd.Series(cost), warnings)
