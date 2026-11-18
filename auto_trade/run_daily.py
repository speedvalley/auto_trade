"""毎日の実行:  python3 run_daily.py

データ更新 → 長期・短期シグナル → ニュース補正 → ペーパートレード台帳 → 日次レポート
実際の注文は出さない。reports/ のレポートを見て、人間が判断する。
"""
import datetime
import os
import warnings

import pandas as pd

import config
import news
import paper
from data import load_all
from signals import ShortEngine, assemble, build_long, market_assets, short_to_weights

warnings.filterwarnings("ignore")


def pct(x: float) -> str:
    return f"{x:.1%}"


def main():
    data = load_all()
    asof = data.jpy.dropna(how="all").index[-1]
    today = asof.date()

    # 1) 価格から決まる目標の持ち高(長期・短期)
    long_w = build_long(data).loc[[asof]]
    eng = ShortEngine(data)
    pos, choices = eng.live_positions(asof)
    short_w = short_to_weights(pos, data)
    long_part = assemble(long_w, short_w, data, long_share=1.0).iloc[0] * config.LONG_SHARE
    short_part = assemble(long_w, short_w, data, long_share=0.0).iloc[0] * (1 - config.LONG_SHARE)
    base = assemble(long_w, short_w, data).iloc[0]

    # 2) ニュースで補正
    items, news_errors, news_date = news.load_items(today)
    final, notes = news.overlay(base, items, today)

    # 3) ペーパートレード台帳を更新
    prev = paper.previous_weights()
    eq = paper.update(asof, final, data.jpy, data.costs)

    # 4) レポート
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    lines = [f"# 日次レポート {today}", "",
             f"- 基準日(最新の終値): {today} / 資産(ペーパー): {eq['equity'].iloc[-1]:,.0f} 円 "
             f"/ 投資比率 {pct(final.sum())} / 現金 {pct(1 - final.sum())}", ""]

    lines += ["## 目標の持ち高", "",
              "| 銘柄 | 市場 | 長期 | 短期 | ニュース補正 | 最終 | 前回 |", "|---|---|---|---|---|---|---|"]
    held = final[(final > 0) | (base > 0)].sort_values(ascending=False)
    for t in held.index:
        n = notes.get(t)
        mult = f"×{n['mult']:g}" if n else "-"
        lines.append(f"| {t} | {data.asset_market[t]} | {pct(long_part[t])} | {pct(short_part[t])} | "
                     f"{mult} | **{pct(final[t])}** | {pct(prev.get(t, 0.0))} |")
    if held.empty:
        lines.append("| (全額現金) | | | | | | |")

    lines += ["", "## 前回からの変更(売買の指示案)", ""]
    changes = (final - prev.reindex(final.index).fillna(0.0))
    changes = changes[changes.abs() >= 0.005].sort_values()
    if changes.empty:
        lines.append("- 変更なし")
    for t, d in changes.items():
        lines.append(f"- {'買い増し/新規' if d > 0 else '売却/縮小'}: {t} ({d:+.1%})")

    lines += ["", "## ニュースの影響", ""]
    if not notes:
        lines.append("- 目標の持ち高に関係するニュースはありません。")
    for t, n in notes.items():
        lines.append(f"- **{t}**: {n['why']}")
        for it in sorted(n["items"], key=lambda x: x["date"], reverse=True)[:3]:
            lines.append(f"  - {it['date']} [{it['event_type']}, 点数{it['sentiment']:+d}, 確度{it['confidence']:.1f}] "
                         f"{it['headline']} ({it['source_url']})")

    lines += ["", "## 短期戦略で今選ばれている数値", ""]
    for m, c in choices.items():
        lines.append(f"- {m}: {c}")

    lines += ["", "## 警告・確認事項", ""]
    warn = list(data.warnings) + [f"ニュース: {e}" for e in news_errors]
    if news_date is None:
        warn.append("ニュースファイルがありません(価格だけで判断しています)")
    elif (today - news_date).days > config.NEWS["stale_warn_days"]:
        warn.append(f"ニュースが古い(最新 {news_date})。Claude Codeにニュース収集を依頼してください")
    lines += [f"- {w}" for w in warn] or ["- なし"]
    lines += ["", "---", "これは検証用のペーパートレードの記録で、投資助言ではありません。実際の売買は自己責任で。"]

    path = os.path.join(config.REPORT_DIR, f"{today}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n保存: {path}")


if __name__ == "__main__":
    main()
