import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df_aapl = yf.download(
  "AAPL",
  start ="2023-01-05",
  auto_adjust = False,
  multi_level_index=False,
  progress=False,
  timeout=20
)
if df_aapl.empty:
    raise ValueError("株価データがない")

if isinstance(df_aapl.columns, pd.MultiIndex):
  level = 0 if "Close" in df_aapl.columnsget_level_values(0) else -1
  df_aapl.columns = df_aapl.columns.get_level_values(level)

  if "Adj Close" not in df_aapl.columns:
    df_aapl["Adj Close"] = df_aapl["Close"]
if "Volume" not in df_aapl.columns:
    df_aapl["Volume"] = 0

    price_cols = [
  "Open", "High", "Low",
  "Close", "Adj Close", "Volume"
]

    df_aapl[price_cols] = df_aapl[price_cols].apply(
  pd.to_numeric, errors="coerce"
)

    df_aapl[price_cols] = df_aapl[price_cols].replace(
  [float("inf"), float("-inf")],
        float("nan")
        )

    if "Adj Close" not in df_aapl.columns:
    df_aapl["Adj Close"] = df_aapl["Close"]   # 列が無いなら Close をそのままコピー
else:
    df_aapl["Adj Close"] = df_aapl["Adj Close"].fillna(df_aapl["Close"])

if "Volume" not in df_aapl.columns:
    df_aapl["Volume"] = 0                      # 列が無いなら 0 で作る
else:
    df_aapl["Volume"] = df_aapl["Volume"].fillna(0)

    df_aapl = df_aapl.dropna(
    subset = ["Open", "High", "Low", "Close"]
)

    if df_aapl.empty:
    raise ValueError("使用できる株価データがありません")

df_aapl["SMA14"] = df_aapl["Close"].rolling(14).mean()
df_aapl["SMA50"] = df_aapl["Close"].rolling(50).mean()
# df_aapl.head(50)

# 始値から終値までの変化率 (%) を計算する
# Open が 0 の行は NaN にして、ゼロ除算を避ける
df_aapl["change_pct"] = (
    (df_aapl["Close"] - df_aapl["Open"])
    / df_aapl["Open"].replace(0, float("nan"))
    * 100
)
# df_aapl.tail(20)

display(df_aapl.head())
display(df_aapl.tail())

print("機械学習用データの欠損値:")
print(df_aapl[price_cols].isna().sum())

print("機械学習用データの欠損値:")
print(df_aapl[price_cols].isna().sum())

import matplotlib.pyplot as plt

# 日本語が入ってるフォントを指定する (Windows)
plt.rcParams["font.family"] = "Meiryo"      # Yu Gothic でもOK
# マイナス記号が □ になるのを防ぐ
plt.rcParams["axes.unicode_minus"] = False

df_aapl[["Close", "SMA14", "SMA50"]].plot(
    figsize=(14,5),
    title="AAPL / Close とSMA"
)
plt.ylabel("Price / USD")
plt.grid(alpha=0.2)
plt.show()

df_aapl["change_pct"].tail(100).plot(
    figsize=(14,4),
    title="AAPL / 始値から終値への変動率"
)

plt.ylabel("%")
plt.axhline(0, color="gray", linestyle="--", linewidth=1)

plt.grid(alpha=0.2)
plt.show()

df_meta = yf.download(
  "META",
  start ="2023-01-05",
  auto_adjust = False,
  multi_level_index=False,
  progress=False,
  timeout=20
)
df_gold = yf.download(
  "GC=F",
  start ="2023-01-05",
  auto_adjust = False,
  multi_level_index=False,
  progress=False,
  timeout=20
)

for df in (df_meta, df_gold):
    # 列が MultiIndex (yfinance の複数階層) ならフラットにする
    if isinstance(df.columns, pd.MultiIndex):  # colums から修正
        # "Close" が0階層目にあれば0 なければ最後の階層を使う
        level = (0 if "Close" in df.columns.get_level_values(0) else -1)  # in を追加
        df.columns = df.columns.get_level_values(level)

    # 始値から終値までの変化率 (%) を計算する
    df["change_pct"] = (
        (df["Close"] - df["Open"]) / df["Open"].replace(0, float("nan")) * 100
    )

    ax = df_aapl["Close"].plot(
    figsize=(14,  4),
    label="Apple",
    color="red"
)

ax = df_meta["Close"].plot(
    ax=ax,
    label="Meta",
    color="blue"
)

plt.title("AAPL vs META: Close")
plt.legend()
plt.show()

ax =df_aapl["change_pct"].tail(100).plot(
    figsize=(14, 4),
    label="Apple")
df_meta["change_pct"].tail(100).plot(
    ax=ax,label="Meta")
df_gold["change_pct"].tail(100).plot(
    ax=ax,label="Gold Futures")
plt.title("Daily change (%) - last 100 observations")
plt.legend()
plt.show()