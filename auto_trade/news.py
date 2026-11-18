"""ニュース層:Claude Codeが集めたJSONを検証し、持ち高を補正する。

ニュースは「主役」ではなく「補正役」。価格から決めた持ち高に対して、
  - 重大な悪材料(不正・倒産・上場廃止)→ 持たない
  - 悪い点数が続く → 半分にする / 良い点数が続く → 1.25倍にする(上限あり)
という調整だけを行う。
"""
import datetime
import glob
import json
import os

import pandas as pd

import config

EVENT_TYPES = {"earnings", "guidance", "regulation", "legal", "fraud", "bankruptcy",
               "delisting", "m&a", "product", "macro", "management", "supply", "other"}
BLOCK_EVENTS = {"fraud", "bankruptcy", "delisting"}
REQUIRED = ["ticker", "date", "headline", "event_type", "sentiment", "confidence", "source_url"]


def validate_item(item: dict, universe: set, today: datetime.date) -> list:
    """1件のニュースを検証し、問題点のリストを返す(空なら合格)。"""
    errs = []
    for k in REQUIRED:
        if k not in item:
            errs.append(f"項目 {k} がない")
    if errs:
        return errs
    if item["ticker"] not in universe:
        errs.append(f"未知の銘柄 {item['ticker']}")
    try:
        d = datetime.date.fromisoformat(item["date"])
        if d > today:
            errs.append("日付が未来")
    except ValueError:
        errs.append("日付の形式が不正(YYYY-MM-DD)")
    if item["event_type"] not in EVENT_TYPES:
        errs.append(f"event_type が不正: {item['event_type']}")
    if not isinstance(item["sentiment"], int) or not -2 <= item["sentiment"] <= 2:
        errs.append("sentiment は -2〜2 の整数")
    if not isinstance(item["confidence"], (int, float)) or not 0 <= item["confidence"] <= 1:
        errs.append("confidence は 0〜1")
    if not str(item["source_url"]).startswith("http"):
        errs.append("source_url は必須(http で始まる根拠のURL)")
    return errs


def load_items(today: datetime.date):
    """news/*.json を読み、合格したニュースだけを返す。 (items, errors, 最新ファイルの日付)"""
    universe = {t for cfg in config.MARKETS.values() for t in cfg["tickers"]}
    items, errors, latest = [], [], None
    for path in sorted(glob.glob(os.path.join(config.NEWS_DIR, "*.json"))):
        name = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            latest = max(latest or datetime.date.min, datetime.date.fromisoformat(doc["as_of"]))
            raw = doc["items"]
        except Exception as e:
            errors.append(f"{name}: 読み込み失敗 ({e})")
            continue
        for i, it in enumerate(raw):
            errs = validate_item(it, universe, today)
            if errs:
                errors.append(f"{name}[{i}]: {', '.join(errs)}")
            else:
                items.append(it)
    return items, errors, latest


def _age(item: dict, today: datetime.date) -> int:
    return (today - datetime.date.fromisoformat(item["date"])).days


def score(items: list, ticker: str, today: datetime.date) -> float:
    """銘柄のニュース点数 = Σ (sentiment × confidence × 時間で減る重み)。"""
    total = 0.0
    for it in items:
        age = _age(it, today)
        if it["ticker"] == ticker and 0 <= age <= config.NEWS["max_age_days"]:
            total += it["sentiment"] * it["confidence"] * 0.5 ** (age / config.NEWS["half_life_days"])
    return total


def overlay(weights: pd.Series, items: list, today: datetime.date):
    """目標の持ち高をニュースで補正する。 (補正後の持ち高, 銘柄ごとの説明)"""
    final, notes = weights.copy(), {}
    for t in weights.index:
        mine = [it for it in items if it["ticker"] == t and 0 <= _age(it, today) <= config.NEWS["max_age_days"]]
        if not mine:
            continue
        s = score(items, t, today)
        blocked = [it for it in mine if it["event_type"] in BLOCK_EVENTS and it["sentiment"] <= -1]
        if blocked:
            mult, why = 0.0, f"重大な悪材料({blocked[0]['event_type']})のため持たない"
        elif s <= config.NEWS["neg_cut"]:
            mult, why = 0.5, f"悪材料が続く(点数 {s:+.2f})ので半分に"
        elif s >= config.NEWS["pos_boost"]:
            mult, why = 1.25, f"好材料が続く(点数 {s:+.2f})ので1.25倍(上限あり)"
        else:
            mult, why = 1.0, f"補正なし(点数 {s:+.2f})"
        final[t] = min(weights[t] * mult, config.MAX_ASSET_WEIGHT)
        notes[t] = {"mult": mult, "why": why, "score": s, "items": mine}
    return final, notes
