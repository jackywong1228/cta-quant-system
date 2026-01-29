"""
交易信号生成模块
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from loguru import logger
from enum import Enum
from datetime import time as dt_time


class Signal(Enum):
    """交易信号枚举"""
    LONG = 1      # 做多
    SHORT = -1    # 做空
    CLOSE = 0     # 平仓
    HOLD = 99     # 持有不动


class SignalGenerator:
    """交易信号生成器"""
    
    def __init__(
        self,
        long_threshold: float = 1.5,
        short_threshold: float = -1.5,
        close_threshold: float = 0.5,
        stop_loss: float = 0.025,  # -2.5% 止损
        trading_start: str = "09:00",
        trading_end: str = "14:30",
        force_close_time: str = "14:55",
    ):
        """
        初始化信号生成器
        
        Args:
            long_threshold: 做多阈值（动量得分 > 此值开多）
            short_threshold: 做空阈值（动量得分 < 此值开空）
            close_threshold: 平仓阈值（|动量得分| < 此值平仓）
            stop_loss: 止损比例 (负数)
            trading_start: 交易开始时间
            trading_end: 最后开仓时间
            force_close_time: 强制平仓时间
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.close_threshold = close_threshold
        self.stop_loss = stop_loss
        
        self.trading_start = dt_time.fromisoformat(trading_start)
        self.trading_end = dt_time.fromisoformat(trading_end)
        self.force_close_time = dt_time.fromisoformat(force_close_time)
        
        logger.info(f"信号生成器初始化: long>{long_threshold}, short<{short_threshold}, "
                   f"close<|{close_threshold}|, stop={stop_loss*100}%")
    
    def _is_trading_time(self, dt: pd.Timestamp) -> bool:
        """检查是否在可交易时间内"""
        t = dt.time()
        
        # 日盘: 09:00-10:15, 10:30-11:30, 13:30-15:00
        day_session = (
            (dt_time(9, 0) <= t <= dt_time(10, 15)) or
            (dt_time(10, 30) <= t <= dt_time(11, 30)) or
            (dt_time(13, 30) <= t <= dt_time(15, 0))
        )
        
        # 夜盘: 21:00-23:00 (部分品种到02:30)
        night_session = dt_time(21, 0) <= t <= dt_time(23, 0)
        
        return day_session or night_session
    
    def _can_open_position(self, dt: pd.Timestamp) -> bool:
        """检查是否可以开仓（在14:30前）"""
        t = dt.time()
        
        # 日盘开仓限制
        if dt_time(9, 0) <= t <= dt_time(15, 0):
            return t <= self.trading_end
        
        # 夜盘可以开仓
        return dt_time(21, 0) <= t <= dt_time(23, 0)
    
    def _should_force_close(self, dt: pd.Timestamp) -> bool:
        """检查是否需要强制平仓"""
        t = dt.time()
        
        # 日盘14:55强平
        if dt_time(13, 30) <= t <= dt_time(15, 0):
            return t >= self.force_close_time
        
        # 夜盘22:55强平 (简化处理)
        if dt_time(21, 0) <= t <= dt_time(23, 0):
            return t >= dt_time(22, 55)
        
        return False
    
    def generate_signal(
        self,
        momentum_score: float,
        current_position: int,
        entry_price: Optional[float],
        current_price: float,
        dt: pd.Timestamp,
    ) -> Tuple[Signal, str]:
        """
        生成交易信号
        
        Args:
            momentum_score: 当前动量得分
            current_position: 当前持仓 (1=多, -1=空, 0=无)
            entry_price: 开仓价格
            current_price: 当前价格
            dt: 当前时间
        
        Returns:
            (信号, 原因)
        """
        # 1. 强制平仓检查
        if self._should_force_close(dt) and current_position != 0:
            return Signal.CLOSE, "强制平仓(收盘前)"
        
        # 2. 止损检查
        if current_position != 0 and entry_price is not None:
            pnl_pct = (current_price - entry_price) / entry_price * current_position
            if pnl_pct <= -self.stop_loss:
                return Signal.CLOSE, f"止损({pnl_pct*100:.2f}%)"
        
        # 3. 非交易时间
        if not self._is_trading_time(dt):
            return Signal.HOLD, "非交易时间"
        
        # 4. 生成新信号
        if pd.isna(momentum_score):
            return Signal.HOLD, "无因子数据"
        
        # 有持仓时的逻辑
        if current_position != 0:
            # 动量减弱，平仓
            if abs(momentum_score) < self.close_threshold:
                return Signal.CLOSE, f"动量减弱({momentum_score:.2f})"
            
            # 反向信号，平仓
            if current_position == 1 and momentum_score < self.short_threshold:
                return Signal.CLOSE, f"反向信号(多→空, {momentum_score:.2f})"
            if current_position == -1 and momentum_score > self.long_threshold:
                return Signal.CLOSE, f"反向信号(空→多, {momentum_score:.2f})"
            
            # 继续持有
            return Signal.HOLD, "持有"
        
        # 无持仓时的逻辑
        if not self._can_open_position(dt):
            return Signal.HOLD, "超过开仓时间"
        
        if momentum_score > self.long_threshold:
            return Signal.LONG, f"做多({momentum_score:.2f}>{self.long_threshold})"
        
        if momentum_score < self.short_threshold:
            return Signal.SHORT, f"做空({momentum_score:.2f}<{self.short_threshold})"
        
        return Signal.HOLD, "无信号"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量生成信号
        
        Args:
            df: 包含 datetime, close, momentum_score 的 DataFrame
        
        Returns:
            添加了 signal, signal_reason, position 列的 DataFrame
        """
        df = df.copy()
        
        signals = []
        reasons = []
        positions = []
        
        current_position = 0
        entry_price = None
        
        for idx, row in df.iterrows():
            signal, reason = self.generate_signal(
                momentum_score=row.get('momentum_score', np.nan),
                current_position=current_position,
                entry_price=entry_price,
                current_price=row['close'],
                dt=row['datetime'],
            )
            
            signals.append(signal.value)
            reasons.append(reason)
            
            # 更新持仓状态
            if signal == Signal.LONG:
                current_position = 1
                entry_price = row['close']
            elif signal == Signal.SHORT:
                current_position = -1
                entry_price = row['close']
            elif signal == Signal.CLOSE:
                current_position = 0
                entry_price = None
            
            positions.append(current_position)
        
        df['signal'] = signals
        df['signal_reason'] = reasons
        df['position'] = positions
        
        # 统计
        long_signals = (df['signal'] == Signal.LONG.value).sum()
        short_signals = (df['signal'] == Signal.SHORT.value).sum()
        close_signals = (df['signal'] == Signal.CLOSE.value).sum()
        
        logger.info(f"信号统计: 做多={long_signals}, 做空={short_signals}, 平仓={close_signals}")
        
        return df


def test_signal_generator():
    """测试信号生成"""
    from src.strategy.factors.momentum import MomentumFactors
    import numpy as np
    
    # 创建测试数据
    np.random.seed(42)
    n = 200
    
    # 模拟价格走势
    trend = np.linspace(3300, 3400, n)
    noise = np.random.randn(n) * 10
    prices = trend + noise
    
    df = pd.DataFrame({
        'datetime': pd.date_range('2025-01-20 09:00', periods=n, freq='5min'),
        'close': prices,
    })
    
    # 计算因子
    mf = MomentumFactors()
    df = mf.calc_momentum_score(df)
    
    # 生成信号
    sg = SignalGenerator(stop_loss=0.025)  # -2.5% 止损
    df = sg.generate_signals(df)
    
    print("\n信号示例:")
    print(df[df['signal'] != 99][['datetime', 'close', 'momentum_score', 'signal', 'signal_reason', 'position']].head(20))
    
    return df


if __name__ == '__main__':
    test_signal_generator()
