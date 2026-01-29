#!/usr/bin/env python3
"""
运行完整回测示例
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from loguru import logger

from src.strategy import MomentumFactors, SignalGenerator
from src.backtest import run_full_backtest


def create_sample_data(n: int = 2000) -> pd.DataFrame:
    """创建模拟数据"""
    np.random.seed(42)
    
    # 模拟带趋势的价格序列
    # 价格在 3300-3500 之间波动，带有一些趋势
    base_price = 3400
    trend = np.cumsum(np.random.randn(n) * 0.5)  # 随机游走趋势
    noise = np.random.randn(n) * 5  # 噪声
    prices = base_price + trend + noise
    
    # 确保价格为正
    prices = np.maximum(prices, 3000)
    
    # 生成 OHLC
    df = pd.DataFrame({
        'datetime': pd.date_range('2025-01-02 09:00', periods=n, freq='5min'),
        'open': prices + np.random.randn(n) * 2,
        'high': prices + np.abs(np.random.randn(n) * 5),
        'low': prices - np.abs(np.random.randn(n) * 5),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n).astype(float),
    })
    
    # 确保 OHLC 逻辑正确
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def main():
    logger.info("=" * 50)
    logger.info("CTA 动量策略回测")
    logger.info("=" * 50)
    
    # 1. 准备数据
    logger.info("生成模拟数据...")
    df = create_sample_data(2000)  # 约17个交易日的数据
    logger.info(f"数据量: {len(df)} 条")
    logger.info(f"时间范围: {df['datetime'].min()} - {df['datetime'].max()}")
    
    # 2. 计算因子
    logger.info("\n计算动量因子...")
    mf = MomentumFactors(
        roc_period=20,   # 100分钟动量
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
    )
    df = mf.calc_momentum_score(df)
    
    # 3. 生成信号
    logger.info("\n生成交易信号...")
    sg = SignalGenerator(
        long_threshold=1.5,
        short_threshold=-1.5,
        close_threshold=0.5,
        stop_loss=0.025,  # -2.5% 止损
    )
    df = sg.generate_signals(df)
    
    # 4. 运行回测
    logger.info("\n运行回测...")
    result = run_full_backtest(
        df=df,
        initial_capital=1_000_000,
        strategy_name="MomentumStrategy_RB",
    )
    
    # 5. 输出结果
    logger.info("\n回测完成!")
    logger.info(f"报告路径: {result['report'].get('report_path', 'N/A')}")
    logger.info(f"图表路径: {result['report'].get('chart_path', 'N/A')}")
    
    return result


if __name__ == '__main__':
    main()
