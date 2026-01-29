"""
数据预处理模块
- 主力合约连续处理
- 缺失值处理
- 数据质量检查
- 数据存储
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger
from datetime import datetime, timedelta


class DataProcessor:
    """期货数据预处理器"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # 确保目录存在
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"数据处理器初始化: {self.data_dir}")
    
    def build_continuous_contract(
        self,
        daily_data: pd.DataFrame,
        mapping_data: pd.DataFrame,
        symbol: str,
        method: str = "back_adjust"
    ) -> pd.DataFrame:
        """
        构建主力连续合约
        
        Args:
            daily_data: 日线行情数据 (包含多个合约)
            mapping_data: 主力合约映射数据
            symbol: 品种代码 (如 'RB')
            method: 调整方法
                - 'back_adjust': 向后复权（推荐，保持最新价格真实）
                - 'forward_adjust': 向前复权
                - 'ratio_adjust': 比例复权
        
        Returns:
            主力连续合约 DataFrame
        """
        logger.info(f"构建 {symbol} 主力连续合约 (方法: {method})")
        
        # 过滤该品种的映射数据
        symbol_mapping = mapping_data[
            mapping_data['ts_code'].str.startswith(symbol)
        ].copy()
        
        if symbol_mapping.empty:
            logger.warning(f"未找到 {symbol} 的主力合约映射")
            return pd.DataFrame()
        
        # 按日期排序
        symbol_mapping['trade_date'] = pd.to_datetime(symbol_mapping['trade_date'])
        symbol_mapping = symbol_mapping.sort_values('trade_date')
        
        # 构建连续合约
        continuous_data = []
        
        for _, row in symbol_mapping.iterrows():
            trade_date = row['trade_date']
            main_contract = row['mapping_ts_code']
            
            # 获取该日该合约的数据
            day_data = daily_data[
                (daily_data['ts_code'] == main_contract) &
                (daily_data['trade_date'] == trade_date)
            ]
            
            if not day_data.empty:
                record = day_data.iloc[0].to_dict()
                record['main_contract'] = main_contract
                continuous_data.append(record)
        
        if not continuous_data:
            logger.warning(f"{symbol} 无有效数据")
            return pd.DataFrame()
        
        df = pd.DataFrame(continuous_data)
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 价格调整（处理合约换月跳空）
        if method == "back_adjust" and len(df) > 1:
            df = self._back_adjust(df)
        elif method == "ratio_adjust" and len(df) > 1:
            df = self._ratio_adjust(df)
        
        # 添加品种标识
        df['symbol'] = symbol
        
        logger.info(f"{symbol} 连续合约构建完成: {len(df)} 条记录")
        return df
    
    def _back_adjust(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        向后复权调整
        保持最新价格真实，历史价格调整
        """
        df = df.copy()
        
        # 找到合约切换点
        df['contract_change'] = df['main_contract'] != df['main_contract'].shift(1)
        
        # 计算调整因子
        adjust_factor = 0.0
        adjust_factors = [0.0]
        
        for i in range(1, len(df)):
            if df.iloc[i]['contract_change']:
                # 合约切换时计算价差
                prev_close = df.iloc[i-1]['close']
                curr_open = df.iloc[i]['open']
                gap = curr_open - prev_close
                adjust_factor += gap
            adjust_factors.append(adjust_factor)
        
        # 应用调整（从后往前调整历史价格）
        df['adjust_factor'] = adjust_factors
        max_factor = df['adjust_factor'].max()
        
        for col in ['open', 'high', 'low', 'close']:
            df[f'{col}_adj'] = df[col] - (max_factor - df['adjust_factor'])
        
        # 保留原始价格，添加调整后价格
        df['close_raw'] = df['close']
        df['close'] = df['close_adj']
        df['open'] = df['open_adj']
        df['high'] = df['high_adj']
        df['low'] = df['low_adj']
        
        # 清理临时列
        df = df.drop(columns=['contract_change', 'adjust_factor', 
                               'open_adj', 'high_adj', 'low_adj', 'close_adj'])
        
        return df
    
    def _ratio_adjust(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        比例复权调整
        使用收益率而非价差
        """
        df = df.copy()
        
        # 找到合约切换点
        df['contract_change'] = df['main_contract'] != df['main_contract'].shift(1)
        
        # 计算调整比例
        ratio = 1.0
        ratios = [1.0]
        
        for i in range(1, len(df)):
            if df.iloc[i]['contract_change']:
                prev_close = df.iloc[i-1]['close']
                curr_open = df.iloc[i]['open']
                if prev_close != 0:
                    ratio *= curr_open / prev_close
            ratios.append(ratio)
        
        df['ratio'] = ratios
        max_ratio = df['ratio'].max()
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * (max_ratio / df['ratio'])
        
        df = df.drop(columns=['contract_change', 'ratio'])
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df: 原始数据
        
        Returns:
            清洗后的数据
        """
        if df.empty:
            return df
        
        df = df.copy()
        original_len = len(df)
        
        # 1. 删除全空行
        df = df.dropna(how='all')
        
        # 2. 删除价格异常（价格为0或负数）
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # 3. 检查 OHLC 逻辑关系
        if all(col in df.columns for col in price_cols):
            # high >= low
            df = df[df['high'] >= df['low']]
            # high >= open, close
            df = df[(df['high'] >= df['open']) & (df['high'] >= df['close'])]
            # low <= open, close
            df = df[(df['low'] <= df['open']) & (df['low'] <= df['close'])]
        
        # 4. 删除成交量为0的数据（非交易日）
        if 'vol' in df.columns:
            df = df[df['vol'] > 0]
        
        # 5. 删除重复日期
        if 'trade_date' in df.columns:
            df = df.drop_duplicates(subset=['trade_date'], keep='last')
        
        # 6. 按日期排序
        if 'trade_date' in df.columns:
            df = df.sort_values('trade_date').reset_index(drop=True)
        
        cleaned_len = len(df)
        if original_len != cleaned_len:
            logger.info(f"数据清洗: {original_len} -> {cleaned_len} (删除 {original_len - cleaned_len} 条)")
        
        return df
    
    def fill_missing_data(
        self,
        df: pd.DataFrame,
        method: str = "ffill",
        max_gap: int = 5
    ) -> pd.DataFrame:
        """
        填充缺失数据
        
        Args:
            df: 数据
            method: 填充方法 ('ffill', 'interpolate')
            max_gap: 最大允许的连续缺失天数
        
        Returns:
            填充后的数据
        """
        if df.empty or 'trade_date' not in df.columns:
            return df
        
        df = df.copy()
        
        # 创建完整日期范围
        date_range = pd.date_range(
            start=df['trade_date'].min(),
            end=df['trade_date'].max(),
            freq='B'  # 工作日
        )
        
        # 重建索引
        df = df.set_index('trade_date')
        df = df.reindex(date_range)
        
        # 填充缺失值
        if method == 'ffill':
            df = df.ffill(limit=max_gap)
        elif method == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=max_gap)
            df = df.ffill(limit=max_gap)  # 非数值列用前向填充
        
        # 删除仍有缺失的行
        df = df.dropna(subset=['close'])
        
        # 重置索引
        df = df.reset_index()
        df = df.rename(columns={'index': 'trade_date'})
        
        return df
    
    def check_data_quality(self, df: pd.DataFrame, symbol: str = "") -> Dict:
        """
        数据质量检查报告
        
        Args:
            df: 数据
            symbol: 品种代码
        
        Returns:
            质量检查报告
        """
        report = {
            'symbol': symbol,
            'total_rows': len(df),
            'issues': []
        }
        
        if df.empty:
            report['issues'].append("数据为空")
            return report
        
        # 1. 检查日期范围
        if 'trade_date' in df.columns:
            report['start_date'] = df['trade_date'].min()
            report['end_date'] = df['trade_date'].max()
            report['trading_days'] = len(df)
        
        # 2. 检查缺失值
        missing = df.isnull().sum()
        if missing.sum() > 0:
            report['missing_values'] = missing[missing > 0].to_dict()
            report['issues'].append(f"存在缺失值: {missing.sum()} 个")
        
        # 3. 检查价格跳空
        if 'close' in df.columns and len(df) > 1:
            returns = df['close'].pct_change()
            large_gaps = returns[abs(returns) > 0.1]  # 单日涨跌超过10%
            if len(large_gaps) > 0:
                report['large_gaps'] = len(large_gaps)
                report['issues'].append(f"存在 {len(large_gaps)} 个大幅跳空 (>10%)")
        
        # 4. 检查数据连续性
        if 'trade_date' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('trade_date')
            date_diff = df_sorted['trade_date'].diff().dt.days
            large_gaps = date_diff[date_diff > 5]  # 超过5天的间隔
            if len(large_gaps) > 0:
                report['date_gaps'] = len(large_gaps)
                report['issues'].append(f"存在 {len(large_gaps)} 个日期间隔 (>5天)")
        
        # 5. 总结
        report['is_clean'] = len(report['issues']) == 0
        
        return report
    
    def save_data(
        self,
        df: pd.DataFrame,
        filename: str,
        format: str = "parquet"
    ) -> Path:
        """
        保存数据到文件
        
        Args:
            df: 数据
            filename: 文件名（不含扩展名）
            format: 格式 ('parquet', 'csv', 'feather')
        
        Returns:
            保存的文件路径
        """
        if format == "parquet":
            filepath = self.processed_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
        elif format == "csv":
            filepath = self.processed_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        elif format == "feather":
            filepath = self.processed_dir / f"{filename}.feather"
            df.to_feather(filepath)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        logger.info(f"数据已保存: {filepath}")
        return filepath
    
    def load_data(self, filename: str, format: str = "parquet") -> pd.DataFrame:
        """
        从文件加载数据
        
        Args:
            filename: 文件名（不含扩展名）
            format: 格式
        
        Returns:
            DataFrame
        """
        if format == "parquet":
            filepath = self.processed_dir / f"{filename}.parquet"
            return pd.read_parquet(filepath)
        elif format == "csv":
            filepath = self.processed_dir / f"{filename}.csv"
            return pd.read_csv(filepath, parse_dates=['trade_date'])
        elif format == "feather":
            filepath = self.processed_dir / f"{filename}.feather"
            return pd.read_feather(filepath)
        else:
            raise ValueError(f"不支持的格式: {format}")


def process_symbol(
    client,
    processor: DataProcessor,
    symbol: str,
    start_date: str,
    end_date: str,
    save: bool = True
) -> pd.DataFrame:
    """
    处理单个品种的完整流程
    
    Args:
        client: TushareClient 实例
        processor: DataProcessor 实例
        symbol: 品种代码
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        save: 是否保存到文件
    
    Returns:
        处理后的连续合约数据
    """
    logger.info(f"开始处理 {symbol} ({start_date} - {end_date})")
    
    # 1. 获取主力合约映射
    mapping = client.pro.fut_mapping(ts_code=f'{symbol}.', trade_date='')
    
    # 2. 获取所有相关合约的日线数据
    # 这里简化处理，实际可能需要分批获取
    daily_data = client.get_daily_quotes(
        ts_code='',
        start_date=start_date,
        end_date=end_date
    )
    
    # 过滤该品种
    daily_data = daily_data[daily_data['ts_code'].str.startswith(symbol)]
    
    # 3. 构建连续合约
    continuous = processor.build_continuous_contract(
        daily_data=daily_data,
        mapping_data=mapping,
        symbol=symbol,
        method='back_adjust'
    )
    
    # 4. 数据清洗
    continuous = processor.clean_data(continuous)
    
    # 5. 质量检查
    report = processor.check_data_quality(continuous, symbol)
    if not report['is_clean']:
        logger.warning(f"{symbol} 数据质量问题: {report['issues']}")
    
    # 6. 保存
    if save and not continuous.empty:
        processor.save_data(continuous, f"{symbol}_continuous")
    
    return continuous
