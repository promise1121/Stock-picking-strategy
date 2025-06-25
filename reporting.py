# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
import openpyxl
import warnings

warnings.filterwarnings('ignore')

def generate_report(results, data):
    """
    1:1还原原始代码的完整报告功能
    包括持仓建议汇总、性能指标计算、复杂图表绘制和Excel导出
    """
    
    # 从results和data中提取所需参数
    portfolio_value_over_time = results['portfolio_value_over_time']
    weights_df = results['weights_df']
    winning_periods = results['winning_periods']
    total_periods = results['total_periods']
    portfolio_log = results['portfolio_log']
    
    # 从data中提取参数
    rf_rate_annual_decimal = data['rf_rate_annual_decimal']
    start_date = data['start_date']
    end_date = data['end_date']
    
    # 导入配置模块
    import config
    
    # --- 输出所有持仓建议汇总 ---
    print("\n" + "="*80)
    print("📊 风险调整动量策略建议汇总 📊")
    print("="*80)

    if portfolio_log:
        # 获取最新的建议
        latest_decision_date = max(portfolio_log.keys())
        latest_recommendation = portfolio_log[latest_decision_date]
        latest_holdings = latest_recommendation[latest_recommendation > 0].sort_values(ascending=False)
        
        print(f"\n🔥 最新风险调整动量建议 ({latest_decision_date.date()}):")
        if not latest_holdings.empty:
            for ticker, weight in latest_holdings.items():
                print(f"   {ticker}: {weight:.2%}")
            
            total_weight = latest_holdings.sum()
            if config.apply_ndx_short_hedge:
                print(f"\n   多头总权重: {total_weight:.2%}")
                print(f"   NDX做空权重: {config.ndx_short_ratio:.2%}")
                cash_weight = 1.0 - total_weight - config.ndx_short_ratio
                if cash_weight > 0.01:
                    print(f"   现金权重: {cash_weight:.2%}")
            else:
                cash_weight = 1.0 - total_weight
                if cash_weight > 0.01:
                    print(f"   现金权重: {cash_weight:.2%}")
        else:
            print("   推荐: 全部现金")
        
        # 显示历史建议数量
        print(f"\n📈 总共生成了 {len(portfolio_log)} 期调仓建议")
        
        # 保存建议到单独的文件
        try:
            recommendations_df = pd.DataFrame(portfolio_log).T
            recommendations_df = recommendations_df[recommendations_df.sum(axis=1) > 0]  # 只保存有持仓的期间
            
            # 只保存有权重的列
            non_zero_columns = (recommendations_df != 0).any(axis=0)
            recommendations_summary = recommendations_df.loc[:, non_zero_columns]
            
            factor_suffix = "_Enhanced" if config.use_enhanced_factor else "_Original"
            hedge_suffix = f"_NDXShort{int(config.ndx_short_ratio*100)}pct" if config.apply_ndx_short_hedge else ""
            recommendations_filename = f'Risk_Adjusted_Momentum_Recommendations_{config.weighting_method}{factor_suffix}{hedge_suffix}.xlsx'
            
            with pd.ExcelWriter(recommendations_filename, engine='openpyxl') as writer:
                recommendations_summary.to_excel(writer, sheet_name='Risk_Adjusted_Recommendations', index_label='Decision_Date')
                
                # 创建一个格式化的建议表
                formatted_recommendations = recommendations_summary.copy()
                for col in formatted_recommendations.columns:
                    formatted_recommendations[col] = formatted_recommendations[col].apply(lambda x: f"{x:.2%}" if x > 0 else "")
                formatted_recommendations.to_excel(writer, sheet_name='Formatted_Recommendations', index_label='Decision_Date')
            
            print(f"💾 风险调整动量持仓建议已保存至: {recommendations_filename}")
            
        except Exception as e:
            print(f"保存建议文件时出错: {e}")

    print("="*80)

    # 检查是否有有效的净值数据
    if portfolio_value_over_time.empty:
        print("\n错误：回测后投资组合净值序列为空，无法进行绩效分析。")
        return
    
    print("\nPortfolio Value (Tail):")
    print(portfolio_value_over_time.tail())
    print("\nTarget Weights Log (Tail):")
    print(weights_df.tail())

    # --- 步驟 7: 計算並展示績效指標 ---
    print("\n步驟 7: 計算並展示績效指標...")

    # --- 準備計算 ---
    # 計算每日收益率 (從回測結果的淨值序列計算)
    analysis_daily_returns = portfolio_value_over_time.pct_change().dropna()
    print("Daily returns calculated from portfolio value.")
    
    # --- 新增: 下載基準數據並計算Alpha, Beta, Information Ratio ---
    alpha, beta, information_ratio = np.nan, np.nan, np.nan
    try:
        print("正在下載基準指數數據 (^NDX) 以計算 Alpha/Beta...")
        benchmark_returns = yf.download('^NDX', start=start_date, end=end_date, progress=False)['Close'].pct_change()

        # 合併和對齊收益率數據
        merged_returns = pd.concat([analysis_daily_returns, benchmark_returns], axis=1, join='inner')
        merged_returns.columns = ['portfolio', 'benchmark']
        
        if merged_returns.empty:
            print("警告: 策略和基準指數沒有重疊的交易日，無法計算 Alpha/Beta。")
        else:
            # 對齊無風險利率數據
            aligned_rf_rate = rf_rate_annual_decimal.reindex(merged_returns.index).ffill()
            daily_rf_rate = (1 + aligned_rf_rate)**(1/config.trading_days_per_year) - 1
            
            # 計算超額收益 (相對於無風險利率)
            portfolio_excess_returns = merged_returns['portfolio'] - daily_rf_rate
            benchmark_excess_returns = merged_returns['benchmark'] - daily_rf_rate

            # 計算Beta
            covariance = portfolio_excess_returns.cov(benchmark_excess_returns)
            benchmark_variance = benchmark_excess_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
            
            # 計算Alpha (年化)
            alpha_daily = portfolio_excess_returns.mean() - beta * benchmark_excess_returns.mean()
            alpha = alpha_daily * config.trading_days_per_year

            # 計算Information Ratio
            active_returns = merged_returns['portfolio'] - merged_returns['benchmark']
            tracking_error = active_returns.std() * np.sqrt(config.trading_days_per_year)
            
            # 使用策略的年化收益率和基準的年化收益率
            # 確保使用相同的時間段計算年化收益
            aligned_years = (merged_returns.index[-1] - merged_returns.index[0]).days / 365.25
            if aligned_years > 0:
                annualized_return_for_ir = (1 + merged_returns['portfolio']).prod()**(1/aligned_years) - 1
                annualized_benchmark_return_for_ir = (1 + merged_returns['benchmark']).prod()**(1/aligned_years) - 1
                information_ratio = (annualized_return_for_ir - annualized_benchmark_return_for_ir) / tracking_error if tracking_error > 0 else np.nan
            else:
                information_ratio = np.nan
            
            print(f"指標計算完成: Alpha={alpha:.4f}, Beta={beta:.3f}, IR={information_ratio:.3f}")

    except Exception as e:
        print(f"計算 Alpha/Beta/IR 時出錯: {e}")
        alpha, beta, information_ratio = np.nan, np.nan, np.nan
    
    # --- 準備相關參數 ---
    start_val = portfolio_value_over_time.iloc[0]
    end_val = portfolio_value_over_time.iloc[-1]
    start_dt = portfolio_value_over_time.index[0]
    end_dt = portfolio_value_over_time.index[-1]
    years_elapsed = (end_dt - start_dt).days / 365.25

    # --- 計算指標 ---
    cumulative_return = (end_val / start_val) - 1

    if years_elapsed <= 0:
        annualized_return = 0
    else:
        annualized_return = (end_val / start_val) ** (1 / years_elapsed) - 1

    annualized_volatility = analysis_daily_returns.std() * np.sqrt(config.trading_days_per_year)

    # 對齊無風險利率數據到策略的每日收益率日期
    aligned_rf_rate = rf_rate_annual_decimal.reindex(analysis_daily_returns.index).ffill()
    average_annual_risk_free_rate = aligned_rf_rate.mean()

    if annualized_volatility == 0 or np.isnan(annualized_volatility):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (annualized_return - average_annual_risk_free_rate) / annualized_volatility

    rolling_max = portfolio_value_over_time.cummax()
    daily_drawdown = (portfolio_value_over_time / rolling_max) - 1
    max_drawdown = daily_drawdown.min()

    if max_drawdown == 0 or np.isnan(max_drawdown):
        calmar_ratio = np.nan
    else:
        calmar_ratio = annualized_return / abs(max_drawdown)

    # Calculate Win Rate
    if total_periods > 0:
        win_rate = winning_periods / total_periods
    else:
        win_rate = np.nan

    # --- 整理並展示結果 ---
    performance_summary = {
        "權重方法": config.weighting_method.replace('_', ' ').title(),
        "回測期間": f"{start_dt.strftime('%Y-%m-%d')} 到 {end_dt.strftime('%Y-%m-%d')}" if pd.notna(start_dt) else "N/A",
        "期初淨值": f"{start_val:.4f}" if pd.notna(start_val) else "N/A",
        "期末淨值": f"{end_val:.4f}" if pd.notna(end_val) else "N/A",
        "累積報酬率": f"{cumulative_return:.2%}" if pd.notna(cumulative_return) else "N/A",
        "年化報酬率 (CAGR)": f"{annualized_return:.2%}" if pd.notna(annualized_return) else "N/A",
        "年化波動率": f"{annualized_volatility:.2%}" if pd.notna(annualized_volatility) else "N/A",
        "平均年化無風險利率": f"{average_annual_risk_free_rate:.2%}" if pd.notna(average_annual_risk_free_rate) else "N/A",
        "年化夏普比率": f"{sharpe_ratio:.3f}" if pd.notna(sharpe_ratio) else "N/A",
        "最大回撤 (MDD)": f"{max_drawdown:.2%}" if pd.notna(max_drawdown) else "N/A",
        "Calmar 比率": f"{calmar_ratio:.3f}" if pd.notna(calmar_ratio) else "N/A",
        "勝率 (按調倉期)": f"{win_rate:.2%}" if pd.notna(win_rate) else "N/A",
        "Alpha (年化, vs. ^NDX)": f"{alpha:.2%}" if pd.notna(alpha) else "N/A",
        "Beta (vs. ^NDX)": f"{beta:.3f}" if pd.notna(beta) else "N/A",
        "信息比率 (vs. ^NDX)": f"{information_ratio:.3f}" if pd.notna(information_ratio) else "N/A"
    }

    if config.apply_downside_frequency_filter:
        performance_summary["下行波動頻率過濾"] = f"啟用 (過去{config.downside_frequency_lookback_days}天 < {config.downside_frequency_threshold:.0%})"
    else:
        performance_summary["下行波動頻率過濾"] = "未啟用"

    if config.apply_volume_factor:
        performance_summary["成交量因子調整"] = f"啟用 (長回看{config.volume_long_lookback_days}天, 幅度{config.volume_multiplier_floor}-{config.volume_multiplier_cap})"
    else:
        performance_summary["成交量因子調整"] = "未啟用"

    if config.apply_stop_loss:
        performance_summary["個股止損機制"] = f"啟用 ({config.stop_loss_pct:.0%})"
    else:
        performance_summary["個股止損機制"] = "未啟用"

    if config.apply_ndx_short_hedge:
        performance_summary["NDX做空對沖"] = f"啟用 ({config.ndx_short_ratio:.0%} 做空^NDX)"
    else:
        performance_summary["NDX做空對沖"] = "未啟用"

    print("\n--- 策略績效總結 ---")
    for key, value in performance_summary.items():
        print(f"{key:<25}: {value}")
    print("----------------------")

    # --- 繪製績效圖表 ---
    try:
        # 下載標普500和納斯達克100的數據
        print("\n下載基準指數數據...")
        benchmark_data_download = yf.download(['^SPX', '^NDX'], start=start_date, end=end_date)
        
        # Try to get 'Adj Close', fallback to 'Close'
        try:
            benchmark_data = benchmark_data_download['Adj Close']
        except KeyError:
            print("Warning: 'Adj Close' not found for benchmark data, trying 'Close'.")
            benchmark_data = benchmark_data_download['Close']
            
        benchmark_data = benchmark_data.reindex(portfolio_value_over_time.index).ffill()
        print("\nDebug: benchmark_data after reindex and ffill:")
        print(benchmark_data.head())
        print(benchmark_data.tail())
        print("Debug: NaN counts in benchmark_data:")
        print(benchmark_data.isna().sum())

        # 計算基準指數的收益率
        benchmark_returns = benchmark_data.pct_change()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        print("\nDebug: benchmark_cumulative before normalization (and after initial cumprod):")
        print(benchmark_cumulative.head())

        # The first row of pct_change() is NaN, so the first row of cumprod() will also be NaN.
        # We set the first valid row of benchmark_cumulative to 1, as it's our starting point.
        
        # Find the first valid index for each column (benchmark)
        for col in benchmark_cumulative.columns:
            first_valid_idx = benchmark_cumulative[col].first_valid_index()
            if first_valid_idx is not None:
                # Set the first value to 1 (our normalized base)
                benchmark_cumulative.loc[first_valid_idx, col] = 1.0
            else:
                # If a column is all NaN (should not happen if benchmark_data was fine), fill with 1 to prevent errors.
                benchmark_cumulative[col] = 1.0 
        
        # Forward fill any NaNs that might have been at the beginning (e.g. if pct_change had multiple leading NaNs due to market holidays)
        benchmark_cumulative = benchmark_cumulative.ffill()

        print("\nDebug: benchmark_cumulative after setting first valid to 1 and ffill:")
        print(benchmark_cumulative.head())
        print(benchmark_cumulative.tail())
        print("Debug: NaN counts in benchmark_cumulative:")
        print(benchmark_cumulative.isna().sum())
        
        # 設置中文字體和樣式
        try:
            plt.style.use('seaborn-v0_8-whitegrid') # Try a more specific seaborn style
        except:
            print("Warning: 'seaborn-v0_8-whitegrid' style not found. Using default style.")
            
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 11 # Base font size
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # 創建子圖，使用GridSpec來更好地控制佈局
        fig = plt.figure(figsize=(20, 14)) # Larger figure for better spacing
        fig.suptitle('風險調整動量策略績效分析', fontsize=18, y=0.98)
        
        # Create a more balanced GridSpec layout with custom spacing
        gs = plt.GridSpec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.2, 1], 
                         hspace=0.25, wspace=0.25)
        
        # 1. 投資組合價值圖 (Top-Left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(portfolio_value_over_time.index, portfolio_value_over_time, 
                 label='策略組合', color='#1f77b4', linewidth=2.5)
        ax1.plot(benchmark_cumulative.index, benchmark_cumulative['^SPX'], 
                 label='標普500', color='#2ca02c', linewidth=1.8, alpha=0.8)
        ax1.plot(benchmark_cumulative.index, benchmark_cumulative['^NDX'], 
                 label='納斯達克100', color='#ff7f0e', linewidth=1.8, alpha=0.8)
        
        ax1.set_title('策略績效對比', fontsize=15, pad=15, fontweight='bold')
        ax1.set_ylabel('累積收益 (起始值=1)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
        
        # 添加年化收益率標籤
        strategy_annual_return = (portfolio_value_over_time.iloc[-1] ** (252/len(portfolio_value_over_time)) - 1) * 100
        spx_annual_return = (benchmark_cumulative['^SPX'].iloc[-1] ** (252/len(benchmark_cumulative)) - 1) * 100
        ndx_annual_return = (benchmark_cumulative['^NDX'].iloc[-1] ** (252/len(benchmark_cumulative)) - 1) * 100
        
        # Add a box for the returns
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
        textstr = f'年化收益率:\n策略: {strategy_annual_return:.1f}%\n標普500: {spx_annual_return:.1f}%\n納斯達克100: {ndx_annual_return:.1f}%'
        ax1.text(0.02, 0.85, textstr, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        # Format x-axis to show years
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # 2. 回撤圖 (Top-Right)
        ax2 = fig.add_subplot(gs[0, 1])
        strategy_drawdown = (portfolio_value_over_time / portfolio_value_over_time.cummax() - 1) * 100
        spx_drawdown = (benchmark_cumulative['^SPX'] / benchmark_cumulative['^SPX'].cummax() - 1) * 100
        ndx_drawdown = (benchmark_cumulative['^NDX'] / benchmark_cumulative['^NDX'].cummax() - 1) * 100
        
        ax2.fill_between(strategy_drawdown.index, strategy_drawdown, 0, 
                         color='#1f77b4', alpha=0.4, label='策略回撤')
        ax2.fill_between(spx_drawdown.index, spx_drawdown, 0, 
                         color='#2ca02c', alpha=0.3, label='標普500回撤')
        ax2.fill_between(ndx_drawdown.index, ndx_drawdown, 0, 
                         color='#ff7f0e', alpha=0.3, label='納斯達克100回撤')
        
        ax2.set_title('最大回撤比較', fontsize=15, pad=15, fontweight='bold')
        ax2.set_ylabel('回撤 (%)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
        
        # Format x-axis to show years
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Add max drawdown info
        max_strategy_dd = strategy_drawdown.min()
        max_spx_dd = spx_drawdown.min()
        max_ndx_dd = ndx_drawdown.min()
        
        dd_textstr = f'最大回撤:\n策略: {max_strategy_dd:.1f}%\n標普500: {max_spx_dd:.1f}%\n納斯達克100: {max_ndx_dd:.1f}%'
        ax2.text(0.02, 0.15, dd_textstr, transform=ax2.transAxes, fontsize=11,
                verticalalignment='bottom', bbox=props)
        
        # 3. 月度收益熱力圖 (Bottom-Left)
        ax3 = fig.add_subplot(gs[1, 0])
        monthly_returns_series = portfolio_value_over_time.resample('M').last().pct_change().dropna()

        if not monthly_returns_series.empty:
            if not isinstance(monthly_returns_series.index, pd.DatetimeIndex):
                monthly_returns_series.index = pd.to_datetime(monthly_returns_series.index)

            if isinstance(monthly_returns_series.index, pd.DatetimeIndex):
                monthly_returns_df = monthly_returns_series.to_frame(name='Returns')
                monthly_returns_df['Year'] = monthly_returns_df.index.year
                monthly_returns_df['Month'] = monthly_returns_df.index.month
                monthly_returns_pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Returns')
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                # Ensure all 12 months are columns, fill missing with NaN, then assign names
                monthly_returns_pivot = monthly_returns_pivot.reindex(columns=range(1,13))
                monthly_returns_pivot.columns = month_names[:len(monthly_returns_pivot.columns)]
                
                # Use a more distinct colormap for better visibility
                cmap = sns.diverging_palette(10, 133, as_cmap=True)
                
                sns.heatmap(monthly_returns_pivot, 
                            annot=True,
                            fmt='.1%',
                            cmap=cmap,
                            center=0,
                            cbar_kws={'label': '月度收益率', 'shrink': 0.8},
                            linewidths=1,
                            linecolor='white',
                            ax=ax3)
                
                # Rotate x-axis labels for better readability
                plt.setp(ax3.get_xticklabels(), rotation=0, ha='center')
                plt.setp(ax3.get_yticklabels(), rotation=0)
                
            else:
                print("Warning: Could not process monthly_returns.index as DatetimeIndex for heatmap.")
                ax3.text(0.5, 0.5, '月度收益數據索引錯誤', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, '月度收益數據不足', 
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax3.transAxes)
        
        ax3.set_title('月度收益熱力圖', fontsize=15, pad=15, fontweight='bold')
        
        # 4. 最新一期持倉分布餅圖 (Bottom-Right)
        ax4 = fig.add_subplot(gs[1, 1])
        if not weights_df.empty:
            latest_weights = weights_df.iloc[-1]
            latest_holdings = latest_weights[latest_weights > 0.001] # Filter small holdings for clarity

            if not latest_holdings.empty:
                # Sort holdings by value for better visual hierarchy
                latest_holdings = latest_holdings.sort_values(ascending=False)
                
                pie_labels = latest_holdings.index
                pie_sizes = latest_holdings.values
                
                # Use a more visually appealing color palette
                colors = plt.cm.tab20c(np.linspace(0, 1, len(pie_labels)))
                
                # Explode the largest slice slightly
                explode = np.zeros(len(pie_labels))
                explode[0] = 0.1  # Explode largest slice
                
                wedges, texts, autotexts = ax4.pie(pie_sizes, 
                                                  labels=None, # Labels will be in legend
                                                  autopct='%1.1f%%', 
                                                  startangle=90, 
                                                  colors=colors,
                                                  pctdistance=0.85,
                                                  explode=explode,
                                                  shadow=False,
                                                  wedgeprops=dict(width=0.5, edgecolor='w', linewidth=2))

                ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                
                # Format the date for the title
                latest_date_str = weights_df.index[-1].strftime("%Y-%m-%d")
                ax4.set_title(f'最新持倉分布 ({latest_date_str})', fontsize=15, pad=15, fontweight='bold')

                # Create a cleaner legend with percentage values
                legend_labels = [f'{label} ({size:.1%})' for label, size in zip(pie_labels, pie_sizes)]
                
                # Place legend to the right of the pie chart
                ax4.legend(wedges, legend_labels, 
                           title="持倉資產",
                           loc="center left", 
                           bbox_to_anchor=(1.0, 0.5),
                           frameon=True,
                           fontsize=10,
                           title_fontsize=12)
                
                # Style the percentage text on the pie
                plt.setp(autotexts, size=10, weight="bold", color="white")

            else:
                ax4.text(0.5, 0.5, '最新一期無持倉', 
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=ax4.transAxes,
                         fontsize=14)
        else:
            ax4.text(0.5, 0.5, '無持倉數據', 
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax4.transAxes,
                     fontsize=14)
        
        # 調整整體佈局
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.88)
        
        # 保存圖片
        plt.savefig('strategy_performance_enhanced_v3.png', dpi=300, bbox_inches='tight')
        print("\n增強版績效圖表已保存為 strategy_performance_enhanced_v3.png")
        
        # 單獨保存月度收益熱力圖
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns_pivot,
                    annot=True,
                    fmt='.1%',
                    cmap='RdYlGn',
                    center=0,
                    cbar_kws={'label': '月度收益率'})
        plt.title('月度收益熱力圖', pad=20)
        plt.tight_layout()
        plt.savefig('monthly_returns_heatmap_v3.png', dpi=300, bbox_inches='tight')
        print("月度收益熱力圖已保存為 monthly_returns_heatmap_v3.png")
        
        plt.show()

    except ImportError:
        print("\n警告：未安裝必要的套件，請執行：")
        print("pip install matplotlib seaborn")
    except Exception as e:
        print(f"\n繪製圖表時發生錯誤: {e}")

    # --- 步驟 8: 將結果輸出至Excel文件 ---
    print("\n步驟 8: 將結果輸出至Excel文件...")
    try:
        print("\n檢查 portfolio_value_over_time:")
        print(portfolio_value_over_time)
        print("\n檢查 weights_df:")
        print(weights_df)
        print("\n檢查 performance_summary:")
        print(performance_summary)

        # Construct filename based on strategy components
        factor_suffix = "_Enhanced" if config.use_enhanced_factor else "_Original"
        hedge_suffix = f"_NDXShort{int(config.ndx_short_ratio*100)}pct" if config.apply_ndx_short_hedge else ""
        excel_filename = f'Momentum_Backtest_Results_{config.weighting_method}{factor_suffix}{hedge_suffix}.xlsx'
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # 將投資組合淨值保存到第一個sheet
            pd.DataFrame(portfolio_value_over_time).to_excel(writer, sheet_name='Portfolio_Value', index_label='Date')

            # 將權重記錄保存到第二個sheet
            weights_df.to_excel(writer, sheet_name='Rebalance_Weights', index_label='Rebalance_Date')

            # 將績效指標保存到第三個sheet
            pd.DataFrame(list(performance_summary.items()), columns=['Metric', 'Value']).to_excel(
                writer, sheet_name='Performance', index=False)

        print("結果已成功導出至", excel_filename)
        print("工作表內容:")
        print("1. Portfolio_Value - 投資組合每日淨值")
        print("2. Rebalance_Weights - 調倉日權重配置")
        print("3. Performance - 績效指標匯總")

    except ModuleNotFoundError:
        print("錯誤：缺少openpyxl套件，請執行以下命令安裝：")
        print("pip install openpyxl")
    except Exception as e:
        print(f"導出Excel時發生錯誤: {str(e)}")

    # 查看最近一期的持倉
    if not weights_df.empty:
        try:
            # Get the last rebalancing date from the weights_df index
            latest_rebal_date = weights_df.index[-1]
            
            selected_weights = weights_df.loc[latest_rebal_date]
            holdings = selected_weights[selected_weights > 0].sort_values(ascending=False)
            print(f"\n{latest_rebal_date.date()} (最近一期) 持倉 ({config.weighting_method.replace('_', ' ').title()} Weighting):")
            if not holdings.empty:
                print(holdings)
            else:
                print("無持倉。")

        except IndexError:
            print("\n無法獲取最近的調倉日，權重記錄可能為空或索引錯誤。")
        except Exception as e:
            print(f"\n查詢最近一期持倉時發生錯誤: {e}")
    else:
        print("\n權重記錄為空，無法查詢最近一期持倉。")

    return performance_summary