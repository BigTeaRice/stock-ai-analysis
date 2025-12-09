#!/usr/bin/env python3
import yfinance as yf, json, os, datetime, argparse

STOCKS = [
    "0700.HK", "9988.HK", "3690.HK", "1810.HK",
    "AAPL", "MSFT", "TSLA", "GOOGL", "NVDA"
]
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_one(symbol: str):
    file_name = symbol.replace(".", "") + ".json"
    file_path = os.path.join(DATA_DIR, file_name)
    try:
        df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)

        # 终极兼容：强制把列名变成字符串，避免 tuple 键
        df.columns = [str(c) for c in df.columns]
        records = []
        for idx, row in df.iterrows():
            d = {str(k): v for k, v in row.to_dict().items()}
            d["Date"] = idx.strftime("%Y-%m-%d")
            records.append(d)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=0)
        print(f"[{datetime.datetime.now():%F %T}] {symbol} → {len(records)} 条记录")
    except Exception as e:
        print(f"[ERROR] {symbol} : {e}")

def job():
    for s in STOCKS:
        fetch_one(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", help="单独更新某只股票")
    args = parser.parse_args()
    if args.stock:
        fetch_one(args.stock)
    else:
        job()
