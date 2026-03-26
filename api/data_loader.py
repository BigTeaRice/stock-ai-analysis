import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional

class DataLoader:
    def __init__(self):
        pass

    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        從網絡下載股票數據
        :param ticker: 股票代碼 (例如: 0700.HK, AAPL)
        :param start_date: 開始日期 (格式: 'YYYY-MM-DD')
        :param end_date: 結束日期 (格式: 'YYYY-MM-DD')
        :return: 包含 OHLCV 數據的 DataFrame
        """
        try:
            print(f"从网络下载数据: {ticker}")
            # 下載數據，progress=False 隱藏下載進度條
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # 檢查數據是否為空
            if df is None or df.empty:
                print(f"警告: 無法獲取 {ticker} 的數據，或數據為空。")
                return pd.DataFrame()

            # --- 核心修復：處理時區問題 ---
            # yfinance 返回的數據索引通常帶有 UTC 時區 (tz-aware)
            # 而我們傳入的 end_date 是字符串或無時區時間 (tz-naive)
            # 直接比較會報錯，因此需要移除索引的時區信息
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # -----------------------------

            # 確保數據按日期排序
            df.sort_index(inplace=True)
            
            return df

        except Exception as e:
            print(f"下载数据出错: {e}")
            return pd.DataFrame()

    def load_local_data(self, file_path: str) -> pd.DataFrame:
        """
        從本地 CSV 文件加載數據
        :param file_path: CSV 文件路徑
        :return: DataFrame
        """
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            return df
        except Exception as e:
            print(f"加载本地数据出错: {e}")
            return pd.DataFrame()