"""
Tushare 数据客户端
获取国内商品期货数据
"""

import pandas as pd
import tushare as ts
from pathlib import Path
from loguru import logger
from typing import Optional, List
import yaml


class TushareClient:
    """Tushare 数据接口封装"""
    
    def __init__(self, token: Optional[str] = None, config_path: str = "config/config.yaml"):
        """
        初始化 Tushare 客户端
        
        Args:
            token: Tushare token，如果不提供则从配置文件读取
            config_path: 配置文件路径
        """
        if token is None:
            token = self._load_token_from_config(config_path)
        
        ts.set_token(token)
        self.pro = ts.pro_api()
        logger.info("Tushare 客户端初始化成功")
    
    def _load_token_from_config(self, config_path: str) -> str:
        """从配置文件加载 token"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(
                f"配置文件 {config_path} 不存在，请复制 config.example.yaml 并填入 token"
            )
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        token = config.get('data', {}).get('tushare', {}).get('token')
        if not token or token == "YOUR_TUSHARE_TOKEN_HERE":
            raise ValueError("请在配置文件中填入有效的 Tushare token")
        
        return token
    
    def get_futures_basic(self, exchange: str = "") -> pd.DataFrame:
        """
        获取期货合约基本信息
        
        Args:
            exchange: 交易所代码 (SHFE/DCE/CZCE/INE)
        
        Returns:
            期货合约信息 DataFrame
        """
        df = self.pro.fut_basic(exchange=exchange, fut_type='1')  # 1: 普通期货
        logger.info(f"获取期货合约信息: {len(df)} 条")
        return df
    
    def get_main_contracts(self, exchange: str = "") -> pd.DataFrame:
        """
        获取主力连续合约信息
        
        Args:
            exchange: 交易所代码
        
        Returns:
            主力合约信息 DataFrame
        """
        df = self.pro.fut_mapping(ts_code='', trade_date='')
        if exchange:
            df = df[df['ts_code'].str.contains(exchange)]
        logger.info(f"获取主力合约映射: {len(df)} 条")
        return df
    
    def get_daily_quotes(
        self,
        ts_code: str = "",
        trade_date: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> pd.DataFrame:
        """
        获取期货日线行情
        
        Args:
            ts_code: 合约代码
            trade_date: 交易日期 (YYYYMMDD)
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            日线行情 DataFrame
        """
        df = self.pro.fut_daily(
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
        )
        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
        logger.info(f"获取日线行情 {ts_code}: {len(df)} 条")
        return df
    
    def get_minute_quotes(
        self,
        ts_code: str,
        freq: str = "1min",
        start_date: str = "",
        end_date: str = "",
    ) -> pd.DataFrame:
        """
        获取期货分钟线行情
        
        Args:
            ts_code: 合约代码
            freq: 频率 (1min/5min/15min/30min/60min)
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            分钟线行情 DataFrame
        """
        df = self.pro.ft_mins(
            ts_code=ts_code,
            freq=freq,
            start_date=start_date,
            end_date=end_date,
        )
        if not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df = df.sort_values('trade_time').reset_index(drop=True)
        logger.info(f"获取分钟线 {ts_code} ({freq}): {len(df)} 条")
        return df
    
    def get_holding_data(
        self,
        trade_date: str,
        symbol: str = "",
        exchange: str = "",
    ) -> pd.DataFrame:
        """
        获取期货持仓数据（龙虎榜）
        
        Args:
            trade_date: 交易日期
            symbol: 品种代码
            exchange: 交易所
        
        Returns:
            持仓数据 DataFrame
        """
        df = self.pro.fut_holding(
            trade_date=trade_date,
            symbol=symbol,
            exchange=exchange,
        )
        logger.info(f"获取持仓数据 {trade_date}: {len(df)} 条")
        return df
    
    def get_warehouse_data(
        self,
        trade_date: str = "",
        symbol: str = "",
        exchange: str = "",
    ) -> pd.DataFrame:
        """
        获取期货仓单数据
        
        Args:
            trade_date: 交易日期
            symbol: 品种代码
            exchange: 交易所
        
        Returns:
            仓单数据 DataFrame
        """
        df = self.pro.fut_wsr(
            trade_date=trade_date,
            symbol=symbol,
            exchange=exchange,
        )
        logger.info(f"获取仓单数据: {len(df)} 条")
        return df


# 便捷函数
def get_client(token: Optional[str] = None) -> TushareClient:
    """获取 Tushare 客户端实例"""
    return TushareClient(token=token)
