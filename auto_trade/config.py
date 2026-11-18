"""設定ファイル。調整したい数値はここだけ変えればよい。"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")    # ダウンロードした株価
MANUAL_DIR = os.path.join(BASE_DIR, "data", "manual")  # 自分で置く株価CSV(例: PSEの個別銘柄)
NEWS_DIR = os.path.join(BASE_DIR, "news")              # ニュースのJSON
LEDGER_DIR = os.path.join(BASE_DIR, "ledger")          # ペーパートレードの記録
REPORT_DIR = os.path.join(BASE_DIR, "reports")         # 日次レポート
OUTPUT_DIR = os.path.join(BASE_DIR, "output")          # バックテストの結果

# ---- 市場ごとの設定 ----
# currency: 株価の通貨 / benchmark: 比較の基準(その市場の指数ETF)
# cost: 売買1回あたりのコスト(手数料+スプレッド等の概算。実際の証券会社に合わせて要調整)
# top_k: 長期戦略で保有する最大銘柄数
# ※ フィリピンは yfinance で個別株が取れないため、米国上場の連動ETF(EPHE)とPLDT(PHI)で代用
MARKETS = {
    "US": {"currency": "USD", "benchmark": "SPY", "cost": 0.0010, "top_k": 3,
           "tickers": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "JNJ"]},
    "JP": {"currency": "JPY", "benchmark": "1306.T", "cost": 0.0015, "top_k": 3,
           "tickers": ["1306.T", "7203.T", "6758.T", "9984.T", "8306.T", "6861.T",
                       "9432.T", "8035.T", "8001.T", "4063.T"]},
    "PH": {"currency": "USD", "benchmark": "EPHE", "cost": 0.0030, "top_k": 2,
           "tickers": ["EPHE", "PHI"]},
}
FX = {"USD": "JPY=X"}  # 円換算に使う為替(1ドル=何円)。円建て資産は換算不要
START = "2013-01-01"

# ---- 資金配分 ----
MARKET_WEIGHTS = {"US": 1 / 3, "JP": 1 / 3, "PH": 1 / 3}  # 市場ごとの配分
LONG_SHARE = 0.7          # 資金のうち長期戦略に使う割合(残りが短期戦略)
MAX_ASSET_WEIGHT = 0.15   # 1銘柄の上限(全資金に対する割合)
INITIAL_CAPITAL = 1_000_000  # ペーパートレードの元手(円)

# ---- 長期戦略(月1回見直し):モメンタム + 200日線フィルター ----
LONG = {"mom_lookback": 252, "mom_skip": 21, "trend_window": 200}

# ---- 短期戦略:レジーム適応(ヒステリシス+最低保有日数つき) ----
SHORT_GRID = {  # ウォークフォワードで毎年この中から選ぶ
    "fast": [10, 20], "slow": [50, 100], "rsi_window": [2, 5],
    "buy_below": [20, 30], "er_threshold": [0.3, 0.45],
}
SHORT_FIXED = {"er_window": 20, "sell_above": 60, "er_hysteresis": 0.1, "min_hold": 5}
SHORT_SLOTS = 5  # 短期シグナル1銘柄に市場枠の 1/5 を割り当てる

# ---- ウォークフォワード ----
WF_FIRST_TEST_YEAR = 2019  # この年から毎年、過去だけで調整して採点する
MIN_TRAIN_DAYS = 500       # 調整に使うデータの最低日数

# ---- ニュース補正 ----
NEWS = {
    "max_age_days": 30,      # これより古いニュースは使わない
    "half_life_days": 3.0,   # ニュースの影響が半分になる日数
    "neg_cut": -1.5,         # 点数がこれ以下なら保有を半分に
    "pos_boost": 1.5,        # 点数がこれ以上なら保有を1.25倍に(上限あり)
    "stale_warn_days": 3,    # ニュースファイルがこれより古いと警告
}
