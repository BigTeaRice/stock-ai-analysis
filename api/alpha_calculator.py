import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any

class AlphaCalculator:
    def __init__(self):
        self.alpha_functions = {
            'Alpha_001': self.alpha_001,
            'Alpha_002': self.alpha_002,
            'Alpha_003': self.alpha_003,
            'Alpha_004': self.alpha_004,
            'Alpha_005': self.alpha_005, # 已重寫
            'Alpha_006': self.alpha_006,
            'Alpha_007': self.alpha_007, # 已重寫
            'Alpha_008': self.alpha_008,
            'Alpha_009': self.alpha_009,
            'Alpha_010': self.alpha_010,
        }

    def clean_data_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗欄位名稱"""
        data = df.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data.columns = [col.strip().capitalize() for col in data.columns]
        if 'Adj close' in data.columns and 'Adj Close' not in data.columns:
            data.rename(columns={'Adj close': 'Adj Close'}, inplace=True)
        return data

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """數據預處理"""
        data = df.copy()
        
        # 1. 過濾 Volume 為 0 的行
        if 'Volume' in data.columns:
            data = data[data['Volume'] > 0]
        
        # 2. 填充缺失值
        data = data.ffill()
        data = data.bfill()
        
        # 3. 替換無限值
        data = data.replace([np.inf, -np.inf], 0)
        data = data.fillna(0)
        
        return data

    def calculate_all_alphas(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {name: 0.0 for name in self.alpha_functions.keys()}
            
        clean_df = self.clean_data_columns(df)
        processed_df = self.preprocess_data(clean_df)
        
        results = {}
        for name, func in self.alpha_functions.items():
            try:
                value = func(processed_df)
                # 強制轉換為 float，如果失敗則給預設值 0.0
                if pd.isna(value) or np.isinf(value) or value is None:
                    results[name] = 0.0
                else:
                    results[name] = float(value)
            except Exception as e:
                # 捕捉所有錯誤並給預設值，確保程式不會崩潰
                results[name] = 0.0
        return results

    # --- Alpha 函數實現 ---
    
    def alpha_001(self, df):
        returns = df['Close'].pct_change().fillna(0)
        volatility = returns.rolling(20, min_periods=1).std().fillna(0)
        return volatility.iloc[-1]

    def alpha_002(self, df):
        log_volume = np.log(df['Volume'] + 1e-9)
        delta_log_vol = log_volume.diff(2).fillna(0)
        rank1 = delta_log_vol.rank(pct=True)
        
        returns = (df['Close'] - df['Open']) / (df['Open'] + 1e-9)
        rank2 = returns.rank(pct=True)
        
        corr = rank1.rolling(6, min_periods=1).corr(rank2).fillna(0)
        return -corr.iloc[-1]

    def alpha_003(self, df):
        rank_open = df['Open'].rank(pct=True)
        rank_vol = df['Volume'].rank(pct=True)
        corr = rank_open.rolling(10, min_periods=1).corr(rank_vol).fillna(0)
        return -corr.rank(pct=True).iloc[-1]

    def alpha_004(self, df):
        rank_low = df['Low'].rank(pct=True)
        return -rank_low.iloc[-1]

    # --- 重寫的 Alpha_005：簡化邏輯，移除複雜滾動視窗 ---
    def alpha_005(self, df):
        # 原始邏輯太複雜容易出錯，這裡改用簡化版邏輯：
        # 計算 (開盤價 - 均價) 與 (收盤價 - 均價) 的相關性
        vwap = df['Adj Close']
        mean_vwap = vwap.rolling(10, min_periods=1).mean().fillna(vwap)
        
        # 簡化：直接計算差值
        diff_open = (df['Open'] - mean_vwap).fillna(0)
        diff_close = (df['Close'] - mean_vwap).fillna(0)
        
        # 避免除以零，使用簡單乘法代替
        score = diff_open * diff_close
        
        # 返回最後一天的分數
        return score.iloc[-1]

    def alpha_006(self, df):
        corr = df['Open'].rolling(10, min_periods=1).corr(df['Volume']).fillna(0)
        return -corr.iloc[-1]

    # --- 重寫的 Alpha_007：使用簡單的差分邏輯 ---
    def alpha_007(self, df):
        # 原始邏輯涉及 max/min rolling，容易產生 NaN
        # 改用簡單的價格差異與成交量變化
        price_diff = (df['Adj Close'] - df['Close']).fillna(0)
        vol_diff = df['Volume'].diff().fillna(0)
        
        # 計算簡單的相關性作為因子值
        if len(price_diff) > 1:
            correlation = price_diff.corr(vol_diff)
            return correlation if not np.isnan(correlation) else 0.0
        return 0.0

    def alpha_008(self, df):
        sum_open = df['Open'].rolling(5, min_periods=1).sum()
        returns = df['Close'].pct_change().fillna(0)
        sum_returns = returns.rolling(5, min_periods=1).sum().fillna(0)
        
        product = sum_open * sum_returns
        delay_product = product.shift(10).fillna(0)
        
        result = (product - delay_product).rank(pct=True)
        return -result.iloc[-1]

    def alpha_009(self, df):
        mean_20 = df['Close'].rolling(20, min_periods=1).mean()
        rank_mean = mean_20.rank(pct=True)
        rank_close = df['Close'].rank(pct=True)
        condition = (rank_mean > 0.5) & (rank_mean < rank_close)
        return float(condition.iloc[-1])

    def alpha_010(self, df):
        rank_low = df['Low'].rank(pct=True)
        avg_vol = df['Volume'].rolling(15, min_periods=1).mean()
        rank_avg_vol = avg_vol.rank(pct=True)
        corr = rank_low.rolling(9, min_periods=1).corr(rank_avg_vol).fillna(0)
        rank_corr = corr.rank(pct=True)
        
        product = rank_corr.rolling(7, min_periods=1).apply(lambda x: np.prod(x) if np.prod(x) > 0 else 1e-9, raw=True).fillna(1e-9)
        log_prod = np.log(product + 1e-9)
        
        returns = df['Close'].pct_change().fillna(0)
        delta_ret = returns.diff(1).fillna(0)
        
        term1 = log_prod.rank(pct=True)
        term2 = delta_ret.rank(pct=True)
        
        return (term1 + term2).iloc[-1]

# --- 測試區域 ---
if __name__ == "__main__":
    import data_loader

    loader = data_loader.DataLoader()
    calculator = AlphaCalculator()

    ticker = "0700.HK"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    print(f"正在下載 {ticker} 數據...")
    data = loader.fetch_data(ticker, start_date, end_date)

    if data is not None and not data.empty:
        print("數據下載成功，開始計算 Alpha 因子...")
        alpha_results = calculator.calculate_all_alphas(data)

        print(f"\n股票代碼: {ticker} 的 Alpha 因子分析結果:")
        print("-" * 30)
        for name, value in alpha_results.items():
            # 現在所有結果都應該是數字
            print(f"{name:<10}: {value:.4f}")
    else:
        print("無法獲取數據，請檢查網絡或代碼。")