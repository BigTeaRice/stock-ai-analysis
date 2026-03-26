import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys

# --- 1. 數據與計算類 ---
class AlphaVisualizer:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def fetch_and_prepare(self):
        print(f"正在下載 {self.ticker} 數據...")
        # 使用 yfinance 下載數據
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        
        if df.empty: 
            return None
        
        # --- 修復重點：正確處理多層索引 ---
        # 原始數據結構是 ('Close', '0700.HK')，我們需要取第一個元素 [0]
        new_columns = []
        for col in df.columns:
            if isinstance(col, tuple):
                # 如果是元組，取第一個元素 (例如 'Close')
                col_name = col[0]
            else:
                # 如果是字串，直接使用
                col_name = str(col)
            
            # 統一轉為首字母大寫 (例如 'close' -> 'Close')
            clean_name = col_name.strip().capitalize()
            new_columns.append(clean_name)
        
        df.columns = new_columns
        
        # 處理 Adj Close 的命名差異
        if 'Adj close' in df.columns: 
            df.rename(columns={'Adj close': 'Adj Close'}, inplace=True)
        
        # 安全檢查 Volume
        if 'Volume' in df.columns:
            df = df[df['Volume'] > 0]
        else:
            print("警告：數據中找不到 'Volume' 欄位，跳過過濾。")
            
        # 預處理
        df = df.ffill().bfill().replace([np.inf, -np.inf], 0).fillna(0)
        self.data = df
        return df

    def calculate_alpha_001(self, df):
        returns = df['Close'].pct_change().fillna(0)
        volatility = returns.rolling(20, min_periods=1).std().fillna(0)
        return volatility

    def calculate_alpha_003(self, df):
        rank_open = df['Open'].rank(pct=True)
        rank_vol = df['Volume'].rank(pct=True)
        corr = rank_open.rolling(10, min_periods=1).corr(rank_vol).fillna(0)
        return corr

    # --- 2. 繪圖函數 ---
    def plot_alpha_001(self, df):
        volatility = self.calculate_alpha_001(df)
        
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        ax1.plot(df.index, df['Close'], label='Close Price', color='#1f77b4', alpha=0.7)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price ($)', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        ax2 = ax1.twinx()
        ax2.plot(df.index, volatility, label='Alpha_001 (Volatility)', color='#d62728', alpha=0.8)
        ax2.set_ylabel('Volatility (Risk)', color='#d62728')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        
        plt.title(f'{self.ticker}: Price vs. Alpha_001 (Volatility Risk)', fontsize=16)
        fig.tight_layout()
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        
        plt.show()

    def plot_alpha_003(self, df):
        alpha_003 = self.calculate_alpha_003(df)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
        
        ax1.plot(df.index, df['Close'], label='Close Price', color='#1f77b4')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{self.ticker}: Alpha_003 (Sentiment) vs Price')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper left')
        
        ax2.plot(df.index, alpha_003, label='Alpha_003 (Open-Volume Corr)', color='#2ca02c', alpha=0.6)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.axhline(0.8, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(-0.8, color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2.fill_between(df.index, alpha_003, 0, where=(alpha_003 > 0), color='red', alpha=0.1)
        ax2.fill_between(df.index, alpha_003, 0, where=(alpha_003 < 0), color='green', alpha=0.1)
        
        ax2.set_ylabel('Factor Value')
        ax2.set_xlabel('Date')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()

# --- 3. 執行主程式 ---
if __name__ == "__main__":
    # 預設值
    default_ticker = "0700.HK"
    start_date = "2022-01-01"
    end_date = "2023-01-01"
    
    # 檢查是否有輸入參數
    if len(sys.argv) > 1:
        target_ticker = sys.argv[1].upper()
        print(f">>> 指定目標股票: {target_ticker}")
    else:
        target_ticker = default_ticker
        print(f">>> 未指定股票，使用預設值: {target_ticker}")

    visualizer = AlphaVisualizer(target_ticker, start_date, end_date)
    data = visualizer.fetch_and_prepare()
    
    if data is not None:
        print("數據準備完成，正在生成圖表...")
        try:
            visualizer.plot_alpha_001(data)
            visualizer.plot_alpha_003(data)
            print("圖表繪製完畢。")
        except Exception as e:
            print(f"繪圖時發生錯誤: {e}")
    else:
        print(f"錯誤：無法獲取 {target_ticker} 的數據，請檢查代碼是否正確。")