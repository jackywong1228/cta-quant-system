#!/usr/bin/env python3
"""
后台下载螺纹钢5分钟数据
运行方式: nohup python scripts/download_rb_5min.py > logs/download.log 2>&1 &
"""

import sys
sys.path.insert(0, '.')

import time
from datetime import datetime
from loguru import logger
from src.data import TushareClient, MinuteDataLoader

# 配置日志
logger.add("logs/download_rb.log", rotation="10 MB")

def main():
    logger.info("=" * 50)
    logger.info("开始下载螺纹钢5分钟数据")
    logger.info("=" * 50)
    
    loader = MinuteDataLoader()
    
    # 下载最近1年的数据
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = '20240101'  # 从2024年开始
    
    logger.info(f"日期范围: {start_date} - {end_date}")
    
    try:
        df = loader.get_continuous_minute_data(
            symbol='RB',
            freq='5min',
            start_date=start_date,
            end_date=end_date,
            save=True
        )
        
        logger.info(f"下载完成! 总记录数: {len(df)}")
        logger.info(f"数据范围: {df['datetime'].min()} - {df['datetime'].max()}")
        
    except Exception as e:
        logger.error(f"下载失败: {e}")
        raise

if __name__ == '__main__':
    main()
