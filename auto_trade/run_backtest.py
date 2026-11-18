"""バックテスト:  python3 run_backtest.py   (価格のみ。ニュースは過去検証できないので含めない)

評価期間は WF_FIRST_TEST_YEAR 年以降。短期戦略の数値は毎年「その年より前のデータだけ」で選ぶ。
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from data import load_all
from engine import portfolio_returns, summarize, yearly_returns
from signals import ShortEngine, assemble, build_long, market_assets, short_to_weights

warnings.filterwarnings("ignore")


def fmt(table: pd.DataFrame) -> str:
    pct = ["Total", "CAGR", "Vol", "MaxDD", "Invested"]
    f = {c: "{:.1%}".format for c in pct if c in table}
    f.update({c: "{:.2f}".format for c in ["Sharpe", "Calmar", "Turnover/yr"] if c in table})
    return table.to_string(formatters=f)


def main():
    data = load_all()
    for w in data.warnings:
        print("警告:", w)
    start = pd.Timestamp(config.WF_FIRST_TEST_YEAR, 1, 1)
    print(f"銘柄数 {len(data.asset_market)} / 評価期間 {start.date()} 〜 {data.jpy.index[-1].date()}")

    long_w = build_long(data)
    eng = ShortEngine(data)
    pos, choices = eng.oos_positions(config.WF_FIRST_TEST_YEAR)
    short_w = short_to_weights(pos, data)

    variants = {
        "Combined": assemble(long_w, short_w, data),
        "LongOnly": assemble(long_w, short_w, data, long_share=1.0),
        "ShortOnly": assemble(long_w, short_w, data, long_share=0.0),
    }
    bench_w = pd.DataFrame(0.0, index=data.jpy.index, columns=data.jpy.columns)
    for m, cfg in config.MARKETS.items():
        bench_w[cfg["benchmark"]] = config.MARKET_WEIGHTS[m]  # 3市場の指数ETFを等分で持ち続ける
    variants["Benchmark"] = bench_w

    rets, rows = {}, {}
    for name, w in variants.items():
        r = portfolio_returns(w, data.jpy, data.costs).loc[start:]
        rets[name] = r
        rows[name] = summarize(r, w)
    print("\n===== 全体(円ベース) =====")
    print(fmt(pd.DataFrame(rows).T))

    # 市場別:その市場だけの Combined と、その市場の指数ETFを比べる
    mrows = {}
    for m, cfg in config.MARKETS.items():
        cols = market_assets(data, m)
        w = variants["Combined"][cols] / config.MARKET_WEIGHTS[m]
        r = portfolio_returns(w, data.jpy[cols], data.costs[cols]).loc[start:]
        b = data.jpy[cfg["benchmark"]].pct_change(fill_method=None).fillna(0.0).loc[start:]
        mrows[f"{m} Combined"] = summarize(r, w)
        mrows[f"{m} {cfg['benchmark']}"] = summarize(b)
    print("\n===== 市場別(各市場の指数ETFと比較) =====")
    print(fmt(pd.DataFrame(mrows).T[["Total", "CAGR", "Sharpe", "MaxDD"]]))

    yearly = pd.DataFrame({k: yearly_returns(rets[k]) for k in ["Combined", "LongOnly", "ShortOnly", "Benchmark"]})
    print("\n===== 年別リターン =====")
    print(yearly.to_string(float_format="{:.1%}".format))

    print("\n===== 短期戦略:毎年選ばれた数値(安定しているか確認) =====")
    if len(choices):
        print(choices.pivot_table(index="year", columns="market", values="er_threshold").to_string())

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(rows).T.to_csv(os.path.join(config.OUTPUT_DIR, "backtest_summary.csv"))
    yearly.to_csv(os.path.join(config.OUTPUT_DIR, "backtest_yearly.csv"))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for k, r in rets.items():
        eq = (1 + r).cumprod()
        a1.plot(eq, label=k, lw=2.5 if k == "Combined" else 1.2)
        if k in ("Combined", "Benchmark"):
            a2.plot(eq / eq.cummax() - 1, label=k)
    a1.set_title("Growth of 1.0 (JPY, out-of-sample)")
    a1.legend()
    a2.set_title("Drawdown")
    a2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.OUTPUT_DIR, "backtest.png"), dpi=110)
    print(f"\n保存: {config.OUTPUT_DIR}/ (backtest.png, backtest_summary.csv, backtest_yearly.csv)")
    print("注意: 銘柄は現在の有名銘柄を選んでいるため、生存者バイアス(後から見て勝った銘柄に偏る)があります。")


if __name__ == "__main__":
    main()
