import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time

class StockDataFetcher:
    def __init__(self, tickers=None):
        """
        初始化数据抓取器
        :param tickers: 股票代码列表，默认为港股科技龙头
        """
        if tickers is None:
            self.tickers = ['0700.HK', '9988.HK', '3690.HK']  # 腾讯、阿里、美团
        else:
            self.tickers = tickers

        # 设置数据保存路径（对应你项目中的 data 文件夹）
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_single_stock(self, ticker, period="1d", interval="1m"):
        """
        抓取单只股票数据
        """
        try:
            stock = yf.Ticker(ticker)
            # 获取最近1天的1分钟K线数据（适合日内高频分析）
            df = stock.history(period=period, interval=interval)

            if df.empty:
                print(f"⚠️ 未获取到 {ticker} 的数据")
                return None

            # --- 数据预处理与特征工程 ---
            # 1. 重置索引，将 Datetime 变为普通列
            df.reset_index(inplace=True)

            # 2. 计算基础技术指标 (对应你之前图2的需求)
            df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20周期简单移动平均
            df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50周期简单移动平均
            df['RSI'] = self.calculate_rsi(df['Close'])           # 计算RSI

            # 3. 标记数据时间
            df['Fetch_Time'] = datetime.now()

            print(f"✅ 成功抓取 {ticker} 数据，共 {len(df)} 条记录")
            return df

        except Exception as e:
            print(f"❌ 抓取 {ticker} 时出错: {e}")
            return None

    def calculate_rsi(self, prices, period=14):
        """
        计算相对强弱指数 (RSI)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def save_to_csv(self, df, ticker):
        """
        将数据保存为 CSV 文件
        """
        # 文件名格式：data/0700.HK_20240520.csv
        date_str = datetime.now().strftime("%Y%m%d")
        file_path = os.path.join(self.data_dir, f"{ticker}_{date_str}.csv")

        # 如果文件已存在，则追加数据（避免重复）
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['Datetime'])
            combined_df.to_csv(file_path, index=False)
            print(f"💾 数据已追加至: {file_path}")
        else:
            df.to_csv(file_path, index=False)
            print(f"💾 新建文件并保存: {file_path}")

    def run_once(self, save_to_file=True):
        """
        运行一次抓取任务
        :return: 返回包含所有股票数据的字典 (内存模式)
        """
        all_data = {}
        for ticker in self.tickers:
            df = self.fetch_single_stock(ticker)

            if df is not None and save_to_file:
                self.save_to_csv(df, ticker)

            if df is not None:
                all_data[ticker] = df

        return all_data

    def run_continuous(self, interval_minutes=1):
        """
        持续定时抓取（演示用，实际生产建议用 cron 或 APScheduler）
        """
        print(f"🚀 开始定时抓取任务，每 {interval_minutes} 分钟更新一次...")
        while True:
            self.run_once(save_to_file=True)
            time.sleep(interval_minutes * 60) # 转换为秒


# --- 使用示例 ---
if __name__ == "__main__":
    # 实例化抓取器
    fetcher = StockDataFetcher()

    # 方式 A: 运行一次并保存到 data/ 文件夹 (适合定时任务)
    fetcher.run_once()

    # 方式 B: 获取数据到内存中 (适合 AI 模型直接调用)
    # stock_data_dict = fetcher.run_once(save_to_file=False)
    # print("内存中的数据示例:")
    # print(stock_data_dict['0700.HK'].head())