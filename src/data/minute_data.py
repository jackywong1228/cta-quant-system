"""
分钟线数据获取与处理
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from loguru import logger
from datetime import datetime, timedelta
import time

from .tushare_client import TushareClient


class MinuteDataLoader:
    """分钟线数据加载器"""
    
    def __init__(self, client: Optional[TushareClient] = None, data_dir: str = "data"):
        """
        初始化
        
        Args:
            client: Tushare客户端，如果不提供则自动创建
            data_dir: 数据存储目录
        """
        self.client = client or TushareClient()
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "minute"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("分钟线数据加载器初始化完成")
    
    # 品种到交易所的映射
    SYMBOL_EXCHANGE_MAP = {
        # 上期所 SHFE
        'RB': 'SHF', 'HC': 'SHF', 'CU': 'SHF', 'AL': 'SHF', 'ZN': 'SHF',
        'NI': 'SHF', 'SN': 'SHF', 'PB': 'SHF', 'AU': 'SHF', 'AG': 'SHF',
        'FU': 'SHF', 'BU': 'SHF', 'RU': 'SHF', 'SP': 'SHF', 'SS': 'SHF',
        'WR': 'SHF', 'AO': 'SHF',
        # 上期能源 INE
        'SC': 'INE', 'NR': 'INE', 'LU': 'INE', 'BC': 'INE',
        # 大商所 DCE
        'I': 'DCE', 'J': 'DCE', 'JM': 'DCE', 'M': 'DCE', 'Y': 'DCE',
        'P': 'DCE', 'C': 'DCE', 'CS': 'DCE', 'A': 'DCE', 'B': 'DCE',
        'JD': 'DCE', 'L': 'DCE', 'V': 'DCE', 'PP': 'DCE', 'EB': 'DCE',
        'EG': 'DCE', 'PG': 'DCE', 'RR': 'DCE', 'LH': 'DCE',
        # 郑商所 ZCE
        'SR': 'ZCE', 'CF': 'ZCE', 'TA': 'ZCE', 'MA': 'ZCE', 'OI': 'ZCE',
        'RM': 'ZCE', 'FG': 'ZCE', 'ZC': 'ZCE', 'AP': 'ZCE', 'CJ': 'ZCE',
        'UR': 'ZCE', 'SA': 'ZCE', 'PF': 'ZCE', 'PK': 'ZCE', 'CY': 'ZCE',
        'SF': 'ZCE', 'SM': 'ZCE',
    }
    
    def get_main_contract(self, symbol: str, trade_date: str = "") -> Optional[str]:
        """
        获取指定日期的主力合约代码
        
        Args:
            symbol: 品种代码 (如 'RB')
            trade_date: 交易日期 (YYYYMMDD)，为空则获取最新
        
        Returns:
            主力合约代码 (如 'RB2505.SHF')
        """
        try:
            exchange = self.SYMBOL_EXCHANGE_MAP.get(symbol.upper(), 'SHF')
            ts_code = f'{symbol.upper()}.{exchange}'
            
            if trade_date:
                df = self.client.pro.fut_mapping(ts_code=ts_code, trade_date=trade_date)
            else:
                df = self.client.pro.fut_mapping(ts_code=ts_code)
            
            if not df.empty:
                if trade_date:
                    return df.iloc[0]['mapping_ts_code']
                else:
                    # 获取最新的
                    df = df.sort_values('trade_date', ascending=False)
                    return df.iloc[0]['mapping_ts_code']
        except Exception as e:
            logger.warning(f"获取主力合约失败: {e}")
        return None
    
    def get_minute_data(
        self,
        ts_code: str,
        freq: str = "5min",
        start_date: str = "",
        end_date: str = "",
    ) -> pd.DataFrame:
        """
        获取分钟线数据
        
        Args:
            ts_code: 合约代码
            freq: 频率 (1min/5min/15min/30min/60min)
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
        
        Returns:
            分钟线数据 DataFrame
        """
        logger.info(f"获取 {ts_code} {freq} 数据 ({start_date} - {end_date})")
        
        try:
            df = self.client.pro.ft_mins(
                ts_code=ts_code,
                freq=freq,
                start_date=start_date,
                end_date=end_date,
            )
            
            if df is not None and not df.empty:
                # 处理时间列
                df['trade_time'] = pd.to_datetime(df['trade_time'])
                df = df.sort_values('trade_time').reset_index(drop=True)
                
                # 重命名列以统一格式
                df = df.rename(columns={
                    'trade_time': 'datetime',
                    'vol': 'volume'
                })
                
                logger.info(f"获取到 {len(df)} 条记录")
                return df
            else:
                logger.warning(f"未获取到数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取分钟数据失败: {e}")
            return pd.DataFrame()
    
    def get_continuous_minute_data(
        self,
        symbol: str,
        freq: str = "5min",
        start_date: str = "",
        end_date: str = "",
        save: bool = True
    ) -> pd.DataFrame:
        """
        获取主力连续合约的分钟线数据
        
        Args:
            symbol: 品种代码 (如 'RB')
            freq: 频率
            start_date: 开始日期
            end_date: 结束日期
            save: 是否保存到本地
        
        Returns:
            连续合约分钟线数据
        """
        logger.info(f"获取 {symbol} 主力连续 {freq} 数据")
        
        # 获取日期范围内的交易日
        trade_cal = self.client.pro.trade_cal(
            exchange='DCE',  # 使用大商所日历
            start_date=start_date,
            end_date=end_date,
            is_open='1'
        )
        trade_dates = trade_cal['cal_date'].tolist()
        
        all_data = []
        prev_contract = None
        
        for i, date in enumerate(trade_dates):
            # 获取当日主力合约
            main_contract = self.get_main_contract(symbol, date)
            
            if main_contract is None:
                continue
            
            # 记录换月
            if prev_contract and main_contract != prev_contract:
                logger.info(f"合约换月: {prev_contract} -> {main_contract} ({date})")
            prev_contract = main_contract
            
            # 获取当日分钟数据
            df = self.get_minute_data(
                ts_code=main_contract,
                freq=freq,
                start_date=date,
                end_date=date
            )
            
            if not df.empty:
                df['main_contract'] = main_contract
                df['symbol'] = symbol
                all_data.append(df)
            
            # API限流 (Tushare ft_mins 每分钟最多2次)
            time.sleep(35)  # 每次请求后等待35秒，确保不超限
            
            if (i + 1) % 10 == 0:
                logger.info(f"进度: {i+1}/{len(trade_dates)}")
        
        if not all_data:
            logger.warning(f"未获取到 {symbol} 的分钟数据")
            return pd.DataFrame()
        
        # 合并所有数据
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values('datetime').reset_index(drop=True)
        
        logger.info(f"{symbol} 连续合约数据: {len(result)} 条")
        
        # 保存到本地
        if save:
            filepath = self.raw_dir / f"{symbol}_{freq}_continuous.parquet"
            result.to_parquet(filepath, index=False)
            logger.info(f"数据已保存: {filepath}")
        
        return result
    
    def load_local_data(self, symbol: str, freq: str = "5min") -> pd.DataFrame:
        """
        从本地加载数据
        
        Args:
            symbol: 品种代码
            freq: 频率
        
        Returns:
            本地存储的数据
        """
        filepath = self.raw_dir / f"{symbol}_{freq}_continuous.parquet"
        
        if filepath.exists():
            df = pd.read_parquet(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            logger.info(f"从本地加载 {symbol} 数据: {len(df)} 条")
            return df
        else:
            logger.warning(f"本地数据不存在: {filepath}")
            return pd.DataFrame()


def download_minute_data(
    symbols: List[str],
    freq: str = "5min",
    start_date: str = "",
    end_date: str = "",
) -> dict:
    """
    批量下载分钟线数据
    
    Args:
        symbols: 品种列表
        freq: 频率
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        {symbol: DataFrame} 字典
    """
    loader = MinuteDataLoader()
    result = {}
    
    for symbol in symbols:
        logger.info(f"=== 下载 {symbol} ===")
        df = loader.get_continuous_minute_data(
            symbol=symbol,
            freq=freq,
            start_date=start_date,
            end_date=end_date,
            save=True
        )
        result[symbol] = df
        time.sleep(2)  # API限流
    
    return result
