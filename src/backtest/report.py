"""
回测报告生成模块
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from loguru import logger
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib 未安装，图表功能不可用")


class BacktestReport:
    """回测报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        metrics: Dict,
        strategy_name: str = "Strategy",
        save: bool = True,
    ) -> Dict:
        """
        生成完整回测报告
        
        Args:
            equity_curve: 权益曲线
            trades: 交易记录
            metrics: 绩效指标
            strategy_name: 策略名称
            save: 是否保存文件
        
        Returns:
            报告内容字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{strategy_name}_{timestamp}"
        
        report = {
            'name': report_name,
            'strategy': strategy_name,
            'timestamp': timestamp,
            'metrics': metrics,
        }
        
        # 生成图表
        if MATPLOTLIB_AVAILABLE and save:
            fig_path = self.output_dir / f"{report_name}_chart.png"
            self.plot_equity_curve(equity_curve, trades, metrics, fig_path, strategy_name)
            report['chart_path'] = str(fig_path)
        
        # 保存交易记录
        if save and not trades.empty:
            trades_path = self.output_dir / f"{report_name}_trades.csv"
            trades.to_csv(trades_path, index=False)
            report['trades_path'] = str(trades_path)
        
        # 保存权益曲线
        if save and not equity_curve.empty:
            equity_path = self.output_dir / f"{report_name}_equity.csv"
            equity_curve.to_csv(equity_path, index=False)
            report['equity_path'] = str(equity_path)
        
        # 生成文本报告
        if save:
            text_path = self.output_dir / f"{report_name}_report.txt"
            self.save_text_report(metrics, trades, text_path, strategy_name)
            report['report_path'] = str(text_path)
        
        logger.info(f"报告已生成: {report_name}")
        
        return report
    
    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        metrics: Dict,
        save_path: Optional[Path] = None,
        title: str = "Backtest Result",
    ):
        """绘制权益曲线图"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib 不可用，跳过绘图")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f'{title} - Backtest Report', fontsize=14, fontweight='bold')
        
        # 1. 权益曲线
        ax1 = axes[0]
        if 'datetime' in equity_curve.columns:
            x = pd.to_datetime(equity_curve['datetime'])
        else:
            x = range(len(equity_curve))
        
        equity = equity_curve['equity']
        
        # 绘制权益曲线
        ax1.plot(x, equity, 'b-', linewidth=1.5, label='Equity')
        ax1.fill_between(x, equity.iloc[0], equity, alpha=0.3)
        
        # 标注最大回撤区间
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd_idx = drawdown.idxmin()
        peak_idx = equity[:max_dd_idx].idxmax() if max_dd_idx > 0 else 0
        
        if isinstance(x, pd.DatetimeIndex) or hasattr(x, 'iloc'):
            ax1.axvline(x=x.iloc[peak_idx] if hasattr(x, 'iloc') else x[peak_idx], 
                       color='g', linestyle='--', alpha=0.5, label='Peak')
            ax1.axvline(x=x.iloc[max_dd_idx] if hasattr(x, 'iloc') else x[max_dd_idx], 
                       color='r', linestyle='--', alpha=0.5, label='Trough')
        
        ax1.set_ylabel('Equity', fontsize=10)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Equity Curve | Return: {metrics.get('total_return', 0)*100:.2f}% | "
                     f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | "
                     f"MaxDD: {metrics.get('max_drawdown', 0)*100:.2f}%")
        
        # 2. 回撤曲线
        ax2 = axes[1]
        ax2.fill_between(x, 0, drawdown * 100, color='red', alpha=0.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.set_ylim([drawdown.min() * 100 * 1.1, 1])
        ax2.grid(True, alpha=0.3)
        
        # 3. 持仓
        ax3 = axes[2]
        if 'position' in equity_curve.columns:
            position = equity_curve['position']
            ax3.fill_between(x, 0, position, where=position > 0, color='green', alpha=0.5, label='Long')
            ax3.fill_between(x, 0, position, where=position < 0, color='red', alpha=0.5, label='Short')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_ylabel('Position', fontsize=10)
            ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 格式化x轴
        if isinstance(x, pd.DatetimeIndex):
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        
        plt.close()
    
    def save_text_report(
        self,
        metrics: Dict,
        trades: pd.DataFrame,
        save_path: Path,
        strategy_name: str = "Strategy",
    ):
        """保存文本报告"""
        lines = [
            "=" * 60,
            f"        回测报告: {strategy_name}",
            f"        生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "【收益指标】",
            f"  总收益率:        {metrics.get('total_return', 0)*100:>12.2f}%",
            f"  年化收益率:      {metrics.get('annualized_return', 0)*100:>12.2f}%",
            f"  年化波动率:      {metrics.get('volatility', 0)*100:>12.2f}%",
            "",
            "【风险指标】",
            f"  最大回撤:        {metrics.get('max_drawdown', 0)*100:>12.2f}%",
            f"  夏普比率:        {metrics.get('sharpe_ratio', 0):>12.2f}",
            f"  索提诺比率:      {metrics.get('sortino_ratio', 0):>12.2f}",
            f"  卡玛比率:        {metrics.get('calmar_ratio', 0):>12.2f}",
            "",
            "【交易统计】",
            f"  总交易次数:      {metrics.get('total_trades', 0):>12d}",
            f"  胜率:            {metrics.get('win_rate', 0)*100:>12.2f}%",
            f"  盈亏比:          {metrics.get('profit_factor', 0):>12.2f}",
            f"  平均每笔盈亏:    {metrics.get('avg_pnl', 0):>12.2f}",
            f"  平均盈利:        {metrics.get('avg_win', 0):>12.2f}",
            f"  平均亏损:        {metrics.get('avg_loss', 0):>12.2f}",
            "",
            f"  总手续费:        {metrics.get('total_commission', 0):>12.2f}",
            "",
            "=" * 60,
        ]
        
        # 添加交易明细
        if not trades.empty:
            lines.extend([
                "",
                "【最近10笔交易】",
                "-" * 60,
            ])
            
            close_trades = trades[trades['action'] == 'close'].tail(10)
            for _, t in close_trades.iterrows():
                lines.append(
                    f"  {t['datetime']} | {t['direction']} | "
                    f"{t['lots']}手 @ {t['price']:.2f} | "
                    f"PnL: {t['pnl']:+.2f} | {t['reason']}"
                )
        
        lines.append("=" * 60)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        logger.info(f"文本报告已保存: {save_path}")


def run_full_backtest(
    df: pd.DataFrame,
    initial_capital: float = 1_000_000,
    strategy_name: str = "MomentumStrategy",
) -> Dict:
    """
    运行完整回测流程
    
    Args:
        df: 包含信号的数据
        initial_capital: 初始资金
        strategy_name: 策略名称
    
    Returns:
        回测结果
    """
    from .engine import BacktestEngine
    from .metrics import PerformanceMetrics
    
    # 1. 运行回测
    engine = BacktestEngine(initial_capital=initial_capital)
    equity_curve = engine.run(df)
    trades = engine.get_trades()
    
    # 2. 计算绩效
    pm = PerformanceMetrics()
    metrics = pm.calc_all_metrics(equity_curve, trades)
    
    # 3. 打印摘要
    print(pm.format_metrics(metrics))
    
    # 4. 生成报告
    reporter = BacktestReport()
    report = reporter.generate_report(
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        strategy_name=strategy_name,
        save=True,
    )
    
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'metrics': metrics,
        'report': report,
    }
