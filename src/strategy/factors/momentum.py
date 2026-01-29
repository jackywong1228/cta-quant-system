"""
动量因子计算
包含: ROC, RSI, MACD, 综合动量得分
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class MomentumFactors:
    """动量因子计算器"""
    
    def __init__(
        self,
        roc_period: int = 20,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ):
        """
        初始化因子参数
        
        Args:
            roc_period: ROC 计算周期 (默认20个5分钟=100分钟)
            rsi_period: RSI 计算周期
            macd_fast: MACD 快线周期
            macd_slow: MACD 慢线周期
            macd_signal: MACD 信号线周期
        """
        self.roc_period = roc_period
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
        logger.info(f"动量因子初始化: ROC={roc_period}, RSI={rsi_period}, MACD=({macd_fast},{macd_slow},{macd_signal})")
    
    def calc_roc(self, prices: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        计算 ROC (Rate of Change) 变动率
        
        公式: (close - close[n]) / close[n] * 100
        
        Args:
            prices: 价格序列
            period: 计算周期，默认使用初始化参数
        
        Returns:
            ROC 值序列
        """
        period = period or self.roc_period
        roc = (prices - prices.shift(period)) / prices.shift(period) * 100
        return roc
    
    def calc_rsi(self, prices: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        计算 RSI (Relative Strength Index) 相对强弱指标
        
        公式: 100 - 100 / (1 + avg_gain / avg_loss)
        
        Args:
            prices: 价格序列
            period: 计算周期
        
        Returns:
            RSI 值序列 (0-100)
        """
        period = period or self.rsi_period
        
        # 计算价格变化
        delta = prices.diff()
        
        # 分离涨跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均涨跌 (使用 EMA)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        # 计算 RS 和 RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calc_macd(
        self, 
        prices: pd.Series,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算 MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: 价格序列
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
        
        Returns:
            (MACD线, 信号线, 柱状图)
        """
        fast = fast or self.macd_fast
        slow = slow or self.macd_slow
        signal = signal or self.macd_signal
        
        # 计算 EMA
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # MACD 线 = 快线 - 慢线
        macd_line = ema_fast - ema_slow
        
        # 信号线 = MACD 线的 EMA
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # 柱状图 = MACD 线 - 信号线
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calc_zscore(self, series: pd.Series, window: int = 60) -> pd.Series:
        """
        计算滚动 Z-score 标准化
        
        公式: (x - mean) / std
        
        Args:
            series: 原始序列
            window: 滚动窗口
        
        Returns:
            Z-score 标准化后的序列
        """
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)
        
        return zscore
    
    def calc_momentum_score(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        weights: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        计算综合动量得分
        
        Args:
            df: 包含价格数据的 DataFrame
            price_col: 价格列名
            weights: 因子权重，默认 {'roc': 0.4, 'rsi': 0.3, 'macd': 0.3}
        
        Returns:
            添加了因子列的 DataFrame
        """
        weights = weights or {'roc': 0.4, 'rsi': 0.3, 'macd': 0.3}
        
        df = df.copy()
        prices = df[price_col]
        
        # 计算各因子
        logger.info("计算 ROC 因子...")
        df['roc'] = self.calc_roc(prices)
        
        logger.info("计算 RSI 因子...")
        df['rsi'] = self.calc_rsi(prices)
        # RSI 转换为 -50 到 50 的范围
        df['rsi_centered'] = df['rsi'] - 50
        
        logger.info("计算 MACD 因子...")
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calc_macd(prices)
        
        # Z-score 标准化
        logger.info("Z-score 标准化...")
        df['roc_z'] = self.calc_zscore(df['roc'])
        df['rsi_z'] = self.calc_zscore(df['rsi_centered'])
        df['macd_z'] = self.calc_zscore(df['macd_hist'])
        
        # 计算综合得分
        logger.info("计算综合动量得分...")
        df['momentum_score'] = (
            weights['roc'] * df['roc_z'] +
            weights['rsi'] * df['rsi_z'] +
            weights['macd'] * df['macd_z']
        )
        
        # 统计
        valid_scores = df['momentum_score'].dropna()
        logger.info(f"动量得分统计: mean={valid_scores.mean():.4f}, std={valid_scores.std():.4f}")
        logger.info(f"有效数据: {len(valid_scores)}/{len(df)} ({len(valid_scores)/len(df)*100:.1f}%)")
        
        return df
    
    def calc_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """
        计算因子 IC (Information Coefficient)
        
        IC = corr(factor, future_return)
        
        Args:
            factor: 因子值
            returns: 未来收益率
        
        Returns:
            IC 值 (-1 到 1)
        """
        # 对齐并删除缺失值
        aligned = pd.concat([factor, returns], axis=1).dropna()
        
        if len(aligned) < 10:
            return np.nan
        
        ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        return ic
    
    def calc_ic_series(
        self,
        df: pd.DataFrame,
        factor_col: str,
        price_col: str = 'close',
        forward_periods: int = 1
    ) -> pd.Series:
        """
        计算滚动 IC 序列
        
        Args:
            df: 数据
            factor_col: 因子列名
            price_col: 价格列名
            forward_periods: 未来收益计算周期
        
        Returns:
            每日 IC 值序列
        """
        df = df.copy()
        
        # 计算未来收益
        df['future_return'] = df[price_col].pct_change(forward_periods).shift(-forward_periods)
        
        # 按日期分组计算 IC
        if 'datetime' in df.columns:
            df['date'] = df['datetime'].dt.date
            ic_series = df.groupby('date').apply(
                lambda x: self.calc_ic(x[factor_col], x['future_return'])
            )
        else:
            # 整体计算
            ic = self.calc_ic(df[factor_col], df['future_return'])
            ic_series = pd.Series([ic])
        
        return ic_series


def test_momentum_factors():
    """测试动量因子计算"""
    # 创建测试数据
    np.random.seed(42)
    n = 500
    
    # 模拟价格走势 (带趋势 + 噪声)
    trend = np.linspace(100, 120, n)
    noise = np.random.randn(n) * 2
    prices = trend + noise
    
    df = pd.DataFrame({
        'datetime': pd.date_range('2025-01-01 09:00', periods=n, freq='5min'),
        'close': prices,
    })
    
    # 计算因子
    mf = MomentumFactors()
    result = mf.calc_momentum_score(df)
    
    print("\n因子统计:")
    print(result[['roc', 'rsi', 'macd_hist', 'momentum_score']].describe())
    
    return result


if __name__ == '__main__':
    test_momentum_factors()
