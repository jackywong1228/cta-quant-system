"""
绩效指标计算模块
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class PerformanceMetrics:
    """绩效指标计算器"""
    
    def __init__(
        self,
        risk_free_rate: float = 0.03,  # 无风险利率 3%
        trading_days: int = 252,        # 年交易日
        periods_per_day: int = 48,      # 每天交易周期数 (5分钟线: 4小时*12=48)
    ):
        """
        初始化
        
        Args:
            risk_free_rate: 年化无风险利率
            trading_days: 年交易日数
            periods_per_day: 每天的数据周期数
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.periods_per_day = periods_per_day
        self.periods_per_year = trading_days * periods_per_day
    
    def calc_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        """计算收益率序列"""
        if 'equity' not in equity_curve.columns:
            return pd.Series()
        return equity_curve['equity'].pct_change().dropna()
    
    def calc_total_return(self, equity_curve: pd.DataFrame) -> float:
        """计算总收益率"""
        if len(equity_curve) < 2:
            return 0.0
        start = equity_curve['equity'].iloc[0]
        end = equity_curve['equity'].iloc[-1]
        return (end - start) / start
    
    def calc_annualized_return(self, equity_curve: pd.DataFrame) -> float:
        """计算年化收益率"""
        total_return = self.calc_total_return(equity_curve)
        n_periods = len(equity_curve)
        
        if n_periods <= 1:
            return 0.0
        
        # 年化：(1 + total_return) ^ (periods_per_year / n_periods) - 1
        years = n_periods / self.periods_per_year
        if years <= 0:
            return 0.0
        
        annualized = (1 + total_return) ** (1 / years) - 1
        return annualized
    
    def calc_volatility(self, equity_curve: pd.DataFrame) -> float:
        """计算年化波动率"""
        returns = self.calc_returns(equity_curve)
        if len(returns) < 2:
            return 0.0
        
        # 年化波动率 = 标准差 * sqrt(periods_per_year)
        vol = returns.std() * np.sqrt(self.periods_per_year)
        return vol
    
    def calc_sharpe_ratio(self, equity_curve: pd.DataFrame) -> float:
        """
        计算夏普比率
        Sharpe = (年化收益 - 无风险利率) / 年化波动率
        """
        ann_return = self.calc_annualized_return(equity_curve)
        ann_vol = self.calc_volatility(equity_curve)
        
        if ann_vol <= 0:
            return 0.0
        
        sharpe = (ann_return - self.risk_free_rate) / ann_vol
        return sharpe
    
    def calc_sortino_ratio(self, equity_curve: pd.DataFrame) -> float:
        """
        计算索提诺比率
        Sortino = (年化收益 - 无风险利率) / 下行波动率
        """
        returns = self.calc_returns(equity_curve)
        ann_return = self.calc_annualized_return(equity_curve)
        
        # 下行波动率（只计算负收益）
        downside_returns = returns[returns < 0]
        if len(downside_returns) < 2:
            return 0.0
        
        downside_vol = downside_returns.std() * np.sqrt(self.periods_per_year)
        
        if downside_vol <= 0:
            return 0.0
        
        sortino = (ann_return - self.risk_free_rate) / downside_vol
        return sortino
    
    def calc_max_drawdown(self, equity_curve: pd.DataFrame) -> Dict:
        """
        计算最大回撤
        
        Returns:
            {
                'max_drawdown': 最大回撤比例,
                'max_drawdown_duration': 最大回撤持续时间,
                'peak_date': 峰值日期,
                'trough_date': 谷底日期,
            }
        """
        if len(equity_curve) < 2:
            return {'max_drawdown': 0.0, 'max_drawdown_duration': 0}
        
        equity = equity_curve['equity']
        
        # 计算滚动最大值
        rolling_max = equity.cummax()
        
        # 计算回撤
        drawdown = (equity - rolling_max) / rolling_max
        
        # 最大回撤
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # 找到峰值位置
        peak_idx = equity[:max_dd_idx].idxmax() if max_dd_idx > 0 else 0
        
        result = {
            'max_drawdown': abs(max_dd),
            'peak_idx': peak_idx,
            'trough_idx': max_dd_idx,
        }
        
        if 'datetime' in equity_curve.columns:
            result['peak_date'] = equity_curve.loc[peak_idx, 'datetime']
            result['trough_date'] = equity_curve.loc[max_dd_idx, 'datetime']
        
        return result
    
    def calc_calmar_ratio(self, equity_curve: pd.DataFrame) -> float:
        """
        计算卡玛比率
        Calmar = 年化收益 / 最大回撤
        """
        ann_return = self.calc_annualized_return(equity_curve)
        max_dd = self.calc_max_drawdown(equity_curve)['max_drawdown']
        
        if max_dd <= 0:
            return 0.0
        
        calmar = ann_return / max_dd
        return calmar
    
    def calc_win_rate(self, trades: pd.DataFrame) -> float:
        """计算胜率"""
        if trades.empty or 'pnl' not in trades.columns:
            return 0.0
        
        # 只看平仓交易
        close_trades = trades[trades['action'] == 'close']
        if close_trades.empty:
            return 0.0
        
        win_trades = (close_trades['pnl'] > 0).sum()
        total_trades = len(close_trades)
        
        return win_trades / total_trades if total_trades > 0 else 0.0
    
    def calc_profit_factor(self, trades: pd.DataFrame) -> float:
        """
        计算盈亏比
        Profit Factor = 总盈利 / 总亏损
        """
        if trades.empty or 'pnl' not in trades.columns:
            return 0.0
        
        close_trades = trades[trades['action'] == 'close']
        
        gross_profit = close_trades[close_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(close_trades[close_trades['pnl'] < 0]['pnl'].sum())
        
        if gross_loss <= 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calc_avg_trade(self, trades: pd.DataFrame) -> Dict:
        """计算平均交易统计"""
        if trades.empty or 'pnl' not in trades.columns:
            return {'avg_pnl': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0}
        
        close_trades = trades[trades['action'] == 'close']
        
        avg_pnl = close_trades['pnl'].mean() if not close_trades.empty else 0.0
        
        win_trades = close_trades[close_trades['pnl'] > 0]
        avg_win = win_trades['pnl'].mean() if not win_trades.empty else 0.0
        
        loss_trades = close_trades[close_trades['pnl'] < 0]
        avg_loss = loss_trades['pnl'].mean() if not loss_trades.empty else 0.0
        
        return {
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0.0,
        }
    
    def calc_all_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame
    ) -> Dict:
        """
        计算所有绩效指标
        
        Returns:
            完整的绩效指标字典
        """
        metrics = {}
        
        # 收益指标
        metrics['total_return'] = self.calc_total_return(equity_curve)
        metrics['annualized_return'] = self.calc_annualized_return(equity_curve)
        metrics['volatility'] = self.calc_volatility(equity_curve)
        
        # 风险调整收益
        metrics['sharpe_ratio'] = self.calc_sharpe_ratio(equity_curve)
        metrics['sortino_ratio'] = self.calc_sortino_ratio(equity_curve)
        metrics['calmar_ratio'] = self.calc_calmar_ratio(equity_curve)
        
        # 回撤
        dd_info = self.calc_max_drawdown(equity_curve)
        metrics['max_drawdown'] = dd_info['max_drawdown']
        
        # 交易统计
        metrics['total_trades'] = len(trades[trades['action'] == 'close']) if not trades.empty else 0
        metrics['win_rate'] = self.calc_win_rate(trades)
        metrics['profit_factor'] = self.calc_profit_factor(trades)
        
        avg_trade = self.calc_avg_trade(trades)
        metrics.update(avg_trade)
        
        # 手续费
        if not trades.empty and 'commission' in trades.columns:
            metrics['total_commission'] = trades['commission'].sum()
        
        return metrics
    
    def format_metrics(self, metrics: Dict) -> str:
        """格式化输出绩效指标"""
        lines = [
            "=" * 50,
            "              回测绩效报告",
            "=" * 50,
            "",
            "【收益指标】",
            f"  总收益率:      {metrics.get('total_return', 0)*100:>10.2f}%",
            f"  年化收益率:    {metrics.get('annualized_return', 0)*100:>10.2f}%",
            f"  年化波动率:    {metrics.get('volatility', 0)*100:>10.2f}%",
            "",
            "【风险指标】",
            f"  最大回撤:      {metrics.get('max_drawdown', 0)*100:>10.2f}%",
            f"  夏普比率:      {metrics.get('sharpe_ratio', 0):>10.2f}",
            f"  索提诺比率:    {metrics.get('sortino_ratio', 0):>10.2f}",
            f"  卡玛比率:      {metrics.get('calmar_ratio', 0):>10.2f}",
            "",
            "【交易统计】",
            f"  总交易次数:    {metrics.get('total_trades', 0):>10d}",
            f"  胜率:          {metrics.get('win_rate', 0)*100:>10.2f}%",
            f"  盈亏比:        {metrics.get('profit_factor', 0):>10.2f}",
            f"  平均盈利:      {metrics.get('avg_win', 0):>10.2f}",
            f"  平均亏损:      {metrics.get('avg_loss', 0):>10.2f}",
            f"  盈亏比(金额):  {metrics.get('win_loss_ratio', 0):>10.2f}",
            "",
            f"  总手续费:      {metrics.get('total_commission', 0):>10.2f}",
            "=" * 50,
        ]
        
        return "\n".join(lines)
