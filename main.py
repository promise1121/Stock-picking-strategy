# -*- coding: utf-8 -*-
"""
Enhanced Risk-Adjusted Momentum Strategy - Main Execution
"""

import warnings
warnings.filterwarnings('ignore')

import config
import data_loader
import backtest_engine
import reporting

def main():
    """
    主程序入口，协调所有模块的执行
    """
    
    print("🚀 增强版风险调整动量策略 - 开始执行")
    print("="*80)
    
    try:
        # 步骤1: 数据加载和预处理
        print("\n步骤1: 数据下载和预处理...")
        data = data_loader.load_data()
        if data is None:
            print("❌ 数据加载失败，程序退出")
            return
        
        # 步骤2: 回测执行
        print("\n步骤2: 执行回测...")
        if config.generate_recommendations_only:
            print("🔍 仅生成交易建议模式")
        else:
            print("📊 完整回测模式")
            
        results = backtest_engine.run_backtest(data)
        if results is None:
            print("❌ 回测执行失败，程序退出")
            return
        
        # 步骤3: 性能分析和报告生成
        if not config.generate_recommendations_only:
            print("\n步骤3: 性能分析和报告生成...")
            reporting.generate_report(results, data)
        
        print("\n✅ 程序执行完成！")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        print("请检查配置和数据")
        raise

if __name__ == "__main__":
    main()
