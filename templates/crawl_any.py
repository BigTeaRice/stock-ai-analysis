#!/usr/bin/env python3
# crawl_any.py
import yfinance as yf, json, os, datetime, argparse

STOCKS = [
    "0700.HK", "9988.HK", "3690.HK", "1810.HK",   # 港股
    "AAPL", "MSFT", "TSLA", "GOOGL", "NVDA"        # 美股
]
DATA_DIR = "data"               # 与 index.html 同级
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_one(symbol: str):
    """下载最近 1 年日线并保存为 data/{symbol}.json"""
    file_name = symbol.replace(".", "") + ".json"   # 0700.HK → 0700HK.json
    file_path = os.path.join(DATA_DIR, file_name)
    try:
        df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        records = df.reset_index().to_dict('records')
        # 统一日期格式
        for r in records:
            r['Date'] = r['Date'].strftime('%Y-%m-%d')
        with open(file_path, 'w', encoding='utf-8') as f:
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
