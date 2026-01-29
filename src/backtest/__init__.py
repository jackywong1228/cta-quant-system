"""
回测模块 - 回测引擎、绩效指标、报告生成
"""

from .engine import BacktestEngine, Trade, Position, Account
from .metrics import PerformanceMetrics
from .report import BacktestReport, run_full_backtest

__all__ = [
    "BacktestEngine",
    "Trade",
    "Position",
    "Account",
    "PerformanceMetrics",
    "BacktestReport",
    "run_full_backtest",
]
