"""
仓位管理模块
"""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


class PositionManager:
    """仓位管理器"""
    
    def __init__(
        self,
        method: str = "fixed",
        fixed_pct: float = 0.1,
        atr_risk_pct: float = 0.02,
        atr_period: int = 20,
        max_position_pct: float = 0.3,
        contract_multiplier: int = 10,  # 螺纹钢每手10吨
        margin_rate: float = 0.1,       # 保证金率10%
    ):
        """
        初始化仓位管理器
        
        Args:
            method: 仓位计算方法 ('fixed', 'atr', 'kelly')
            fixed_pct: 固定仓位比例
            atr_risk_pct: ATR风险比例（每笔交易风险占资金比例）
            atr_period: ATR计算周期
            max_position_pct: 最大仓位比例
            contract_multiplier: 合约乘数
            margin_rate: 保证金率
        """
        self.method = method
        self.fixed_pct = fixed_pct
        self.atr_risk_pct = atr_risk_pct
        self.atr_period = atr_period
        self.max_position_pct = max_position_pct
        self.contract_multiplier = contract_multiplier
        self.margin_rate = margin_rate
        
        logger.info(f"仓位管理器初始化: method={method}, fixed_pct={fixed_pct}, max={max_position_pct}")
    
    def calc_atr(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        计算 ATR (Average True Range)
        
        Args:
            df: 包含 high, low, close 的 DataFrame
            period: 计算周期
        
        Returns:
            ATR 序列
        """
        period = period or self.atr_period
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calc_position_size(
        self,
        capital: float,
        price: float,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> int:
        """
        计算开仓手数
        
        Args:
            capital: 可用资金
            price: 当前价格
            atr: ATR值（用于ATR方法）
            volatility: 波动率（用于波动率调整）
        
        Returns:
            开仓手数
        """
        if self.method == "fixed":
            # 固定仓位：使用固定比例的资金
            position_value = capital * self.fixed_pct
        
        elif self.method == "atr":
            # ATR仓位：根据ATR调整仓位，风险固定
            if atr is None or atr <= 0:
                position_value = capital * self.fixed_pct
            else:
                # 每手风险 = ATR * 合约乘数
                risk_per_contract = atr * self.contract_multiplier
                # 总风险金额 = 资金 * 风险比例
                total_risk = capital * self.atr_risk_pct
                # 手数 = 总风险 / 每手风险
                lots = total_risk / risk_per_contract
                position_value = lots * price * self.contract_multiplier * self.margin_rate
        
        elif self.method == "volatility":
            # 波动率调整：波动率高时减少仓位
            if volatility is None or volatility <= 0:
                vol_factor = 1.0
            else:
                # 假设目标波动率为 0.02 (2%)
                target_vol = 0.02
                vol_factor = min(target_vol / volatility, 2.0)  # 最多放大2倍
            position_value = capital * self.fixed_pct * vol_factor
        
        else:
            position_value = capital * self.fixed_pct
        
        # 应用最大仓位限制
        max_position_value = capital * self.max_position_pct
        position_value = min(position_value, max_position_value)
        
        # 计算手数
        contract_value = price * self.contract_multiplier
        margin_per_lot = contract_value * self.margin_rate
        
        if margin_per_lot <= 0:
            return 0
        
        lots = int(position_value / margin_per_lot)
        
        return max(lots, 0)
    
    def calc_position_value(self, lots: int, price: float) -> float:
        """
        计算持仓市值
        
        Args:
            lots: 手数
            price: 价格
        
        Returns:
            持仓市值
        """
        return lots * price * self.contract_multiplier
    
    def calc_margin(self, lots: int, price: float) -> float:
        """
        计算保证金
        
        Args:
            lots: 手数
            price: 价格
        
        Returns:
            所需保证金
        """
        return self.calc_position_value(lots, price) * self.margin_rate


class RiskManager:
    """风控管理器"""
    
    def __init__(
        self,
        stop_loss: float = 0.025,       # 单笔止损 2.5%
        daily_loss_limit: float = 0.05, # 日内最大亏损 5%
        max_drawdown: float = 0.15,     # 最大回撤 15%
        max_position_pct: float = 0.3,  # 最大仓位
    ):
        """
        初始化风控管理器
        
        Args:
            stop_loss: 单笔止损比例
            daily_loss_limit: 日内亏损限制
            max_drawdown: 最大回撤限制
            max_position_pct: 最大仓位比例
        """
        self.stop_loss = stop_loss
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown
        self.max_position_pct = max_position_pct
        
        # 状态
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.is_trading_allowed = True
        
        logger.info(f"风控管理器初始化: stop_loss={stop_loss}, daily_limit={daily_loss_limit}, max_dd={max_drawdown}")
    
    def reset_daily(self):
        """每日重置"""
        self.daily_pnl = 0.0
        self.is_trading_allowed = True
    
    def update_equity(self, equity: float):
        """更新权益峰值"""
        if equity > self.peak_equity:
            self.peak_equity = equity
    
    def check_stop_loss(self, entry_price: float, current_price: float, direction: int) -> bool:
        """
        检查是否触发止损
        
        Args:
            entry_price: 开仓价
            current_price: 当前价
            direction: 方向 (1=多, -1=空)
        
        Returns:
            是否应该止损
        """
        if entry_price <= 0:
            return False
        
        pnl_pct = (current_price - entry_price) / entry_price * direction
        return pnl_pct <= -self.stop_loss
    
    def check_daily_limit(self, initial_equity: float) -> bool:
        """
        检查是否达到日内亏损限制
        
        Args:
            initial_equity: 当日初始权益
        
        Returns:
            是否应该停止交易
        """
        if initial_equity <= 0:
            return False
        
        daily_return = self.daily_pnl / initial_equity
        if daily_return <= -self.daily_loss_limit:
            self.is_trading_allowed = False
            logger.warning(f"触发日内亏损限制: {daily_return*100:.2f}%")
            return True
        return False
    
    def check_max_drawdown(self, current_equity: float) -> bool:
        """
        检查是否达到最大回撤
        
        Args:
            current_equity: 当前权益
        
        Returns:
            是否应该停止交易
        """
        if self.peak_equity <= 0:
            return False
        
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown >= self.max_drawdown:
            self.is_trading_allowed = False
            logger.warning(f"触发最大回撤限制: {drawdown*100:.2f}%")
            return True
        return False
    
    def can_trade(self) -> bool:
        """是否可以交易"""
        return self.is_trading_allowed
