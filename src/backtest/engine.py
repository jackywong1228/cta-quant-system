"""
回测引擎
事件驱动的期货回测系统
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Callable
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..strategy.signal_gen import Signal
from ..strategy.position import PositionManager, RiskManager


@dataclass
class Trade:
    """交易记录"""
    id: int
    datetime: datetime
    direction: int          # 1=多, -1=空
    action: str             # 'open', 'close'
    price: float
    lots: int
    value: float            # 合约价值
    commission: float       # 手续费
    slippage_cost: float    # 滑点成本
    pnl: float = 0.0        # 平仓盈亏
    reason: str = ""


@dataclass
class Position:
    """持仓"""
    direction: int = 0      # 1=多, -1=空, 0=无
    lots: int = 0
    entry_price: float = 0.0
    entry_time: datetime = None
    unrealized_pnl: float = 0.0


@dataclass 
class Account:
    """账户"""
    initial_capital: float
    equity: float
    cash: float
    margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_total: float = 0.0


class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_rate: float = 0.0001,    # 手续费率 0.01%
        slippage_ticks: int = 1,            # 滑点跳数
        tick_size: float = 1.0,             # 最小变动价位
        contract_multiplier: int = 10,      # 合约乘数
        margin_rate: float = 0.1,           # 保证金率
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_ticks: 滑点跳数
            tick_size: 最小变动价位
            contract_multiplier: 合约乘数
            margin_rate: 保证金率
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.contract_multiplier = contract_multiplier
        self.margin_rate = margin_rate
        
        # 组件
        self.position_manager = PositionManager(
            contract_multiplier=contract_multiplier,
            margin_rate=margin_rate
        )
        self.risk_manager = RiskManager()
        
        # 状态
        self.account: Account = None
        self.position: Position = None
        self.trades: List[Trade] = []
        self.equity_curve: List[dict] = []
        
        self._trade_id = 0
        
        logger.info(f"回测引擎初始化: capital={initial_capital:,}, commission={commission_rate}, slippage={slippage_ticks}跳")
    
    def reset(self):
        """重置回测状态"""
        self.account = Account(
            initial_capital=self.initial_capital,
            equity=self.initial_capital,
            cash=self.initial_capital,
        )
        self.position = Position()
        self.trades = []
        self.equity_curve = []
        self._trade_id = 0
        self.risk_manager.peak_equity = self.initial_capital
    
    def _calc_commission(self, value: float) -> float:
        """计算手续费"""
        return value * self.commission_rate
    
    def _calc_slippage(self, price: float, direction: int) -> float:
        """
        计算滑点后的成交价
        
        Args:
            price: 信号价格
            direction: 交易方向 (1=买入, -1=卖出)
        
        Returns:
            实际成交价
        """
        slippage = self.slippage_ticks * self.tick_size * direction
        return price + slippage
    
    def _open_position(self, dt: datetime, price: float, direction: int, reason: str = "") -> Optional[Trade]:
        """开仓"""
        if self.position.direction != 0:
            logger.warning("已有持仓，无法开仓")
            return None
        
        # 计算成交价（含滑点）
        fill_price = self._calc_slippage(price, direction)
        
        # 计算仓位
        lots = self.position_manager.calc_position_size(
            capital=self.account.cash,
            price=fill_price,
        )
        
        if lots <= 0:
            logger.warning("资金不足，无法开仓")
            return None
        
        # 计算成本
        value = lots * fill_price * self.contract_multiplier
        commission = self._calc_commission(value)
        slippage_cost = abs(fill_price - price) * lots * self.contract_multiplier
        margin = value * self.margin_rate
        
        # 更新账户
        self.account.cash -= (margin + commission)
        self.account.margin = margin
        self.account.commission_total += commission
        
        # 更新持仓
        self.position.direction = direction
        self.position.lots = lots
        self.position.entry_price = fill_price
        self.position.entry_time = dt
        
        # 记录交易
        self._trade_id += 1
        trade = Trade(
            id=self._trade_id,
            datetime=dt,
            direction=direction,
            action='open',
            price=fill_price,
            lots=lots,
            value=value,
            commission=commission,
            slippage_cost=slippage_cost,
            reason=reason,
        )
        self.trades.append(trade)
        
        logger.debug(f"开仓: {dt} {'多' if direction == 1 else '空'} {lots}手 @ {fill_price:.2f}")
        
        return trade
    
    def _close_position(self, dt: datetime, price: float, reason: str = "") -> Optional[Trade]:
        """平仓"""
        if self.position.direction == 0:
            return None
        
        direction = self.position.direction
        lots = self.position.lots
        entry_price = self.position.entry_price
        
        # 计算成交价（含滑点，平仓方向相反）
        fill_price = self._calc_slippage(price, -direction)
        
        # 计算盈亏
        value = lots * fill_price * self.contract_multiplier
        pnl = (fill_price - entry_price) * direction * lots * self.contract_multiplier
        commission = self._calc_commission(value)
        slippage_cost = abs(fill_price - price) * lots * self.contract_multiplier
        
        # 更新账户
        self.account.cash += self.account.margin + pnl - commission
        self.account.margin = 0
        self.account.realized_pnl += pnl
        self.account.commission_total += commission
        
        # 更新持仓
        old_direction = self.position.direction
        self.position.direction = 0
        self.position.lots = 0
        self.position.entry_price = 0
        self.position.entry_time = None
        self.position.unrealized_pnl = 0
        
        # 更新风控
        self.risk_manager.daily_pnl += pnl
        
        # 记录交易
        self._trade_id += 1
        trade = Trade(
            id=self._trade_id,
            datetime=dt,
            direction=old_direction,
            action='close',
            price=fill_price,
            lots=lots,
            value=value,
            commission=commission,
            slippage_cost=slippage_cost,
            pnl=pnl,
            reason=reason,
        )
        self.trades.append(trade)
        
        logger.debug(f"平仓: {dt} {lots}手 @ {fill_price:.2f}, PnL={pnl:.2f}")
        
        return trade
    
    def _update_equity(self, dt: datetime, price: float):
        """更新权益"""
        # 计算未实现盈亏
        if self.position.direction != 0:
            self.position.unrealized_pnl = (
                (price - self.position.entry_price) * 
                self.position.direction * 
                self.position.lots * 
                self.contract_multiplier
            )
            self.account.unrealized_pnl = self.position.unrealized_pnl
        else:
            self.account.unrealized_pnl = 0
        
        # 更新总权益
        self.account.equity = self.account.cash + self.account.margin + self.account.unrealized_pnl
        
        # 更新风控
        self.risk_manager.update_equity(self.account.equity)
        
        # 记录权益曲线
        self.equity_curve.append({
            'datetime': dt,
            'equity': self.account.equity,
            'cash': self.account.cash,
            'position': self.position.direction * self.position.lots,
            'unrealized_pnl': self.account.unrealized_pnl,
            'realized_pnl': self.account.realized_pnl,
        })
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        运行回测
        
        Args:
            df: 包含 datetime, close, signal, signal_reason 的 DataFrame
        
        Returns:
            回测结果 DataFrame
        """
        logger.info(f"开始回测: {len(df)} 条数据")
        
        self.reset()
        
        last_date = None
        
        for idx, row in df.iterrows():
            dt = row['datetime']
            price = row['close']
            signal = row.get('signal', Signal.HOLD.value)
            reason = row.get('signal_reason', '')
            
            # 日切重置
            current_date = dt.date() if hasattr(dt, 'date') else dt
            if last_date is not None and current_date != last_date:
                self.risk_manager.reset_daily()
            last_date = current_date
            
            # 检查风控
            if not self.risk_manager.can_trade():
                signal = Signal.HOLD.value
            
            # 检查止损
            if self.position.direction != 0:
                if self.risk_manager.check_stop_loss(
                    self.position.entry_price, price, self.position.direction
                ):
                    self._close_position(dt, price, "止损")
                    signal = Signal.HOLD.value
            
            # 执行信号
            if signal == Signal.LONG.value:
                if self.position.direction == 0:
                    self._open_position(dt, price, 1, reason)
            
            elif signal == Signal.SHORT.value:
                if self.position.direction == 0:
                    self._open_position(dt, price, -1, reason)
            
            elif signal == Signal.CLOSE.value:
                if self.position.direction != 0:
                    self._close_position(dt, price, reason)
            
            # 更新权益
            self._update_equity(dt, price)
            
            # 检查日内亏损限制
            self.risk_manager.check_daily_limit(self.initial_capital)
            
            # 检查最大回撤
            self.risk_manager.check_max_drawdown(self.account.equity)
        
        # 强制平仓（如果还有持仓）
        if self.position.direction != 0:
            last_row = df.iloc[-1]
            self._close_position(last_row['datetime'], last_row['close'], "回测结束平仓")
            self._update_equity(last_row['datetime'], last_row['close'])
        
        logger.info(f"回测完成: {len(self.trades)} 笔交易")
        
        return pd.DataFrame(self.equity_curve)
    
    def get_trades(self) -> pd.DataFrame:
        """获取交易记录"""
        if not self.trades:
            return pd.DataFrame()
        
        records = []
        for t in self.trades:
            records.append({
                'id': t.id,
                'datetime': t.datetime,
                'direction': '多' if t.direction == 1 else '空',
                'action': t.action,
                'price': t.price,
                'lots': t.lots,
                'value': t.value,
                'commission': t.commission,
                'slippage': t.slippage_cost,
                'pnl': t.pnl,
                'reason': t.reason,
            })
        
        return pd.DataFrame(records)
