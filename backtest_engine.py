 # -*- coding: utf-8 -*-
"""
Backtest Engine for the Risk-Adjusted Momentum Strategy.
从原始代码1:1还原的回测引擎逻辑
"""
import pandas as pd
import numpy as np

import config
import factor_engine

def run_backtest(data):
    """
    从原始代码1:1还原的回测执行函数
    """
    # --- Unpack Data ---
    daily_returns = data['daily_returns']
    asset_price_data_close = data['asset_price_data_close']
    asset_price_data_open = data['asset_price_data_open']
    asset_volume_data = data['asset_volume_data']
    rf_rate_annual_decimal = data['rf_rate_annual_decimal']
    ndx_daily_returns = data['ndx_daily_returns']
    nasdaq_100_tickers = data['nasdaq_100_tickers']
    
    # 打印增强因子配置信息
    print("\n" + "="*80)
    print("🚀 增强版风险调整动量策略")
    print("="*80)
    if config.use_enhanced_factor:
        print("✅ 使用增强版多维度因子:")
        print(f"   📈 多期动量组合: {config.enhanced_factor_config['momentum_periods']}天")
        print(f"   ⚖️ 动量权重: {config.enhanced_factor_config['momentum_weights']}")
        print(f"   🛡️ 下行风险调整: {'启用' if config.enhanced_factor_config.get('downside_vol_only') else '禁用'}")
        print(f"   📉 最大回撤调整: {'启用' if config.enhanced_factor_config.get('max_drawdown_adjustment') else '禁用'}")
        print(f"   🎯 收益质量评估: {'启用' if config.enhanced_factor_config.get('enable_quality_adjustment') else '禁用'}")
        print(f"   📊 增强成交量因子: {'启用' if config.enhanced_factor_config.get('enable_enhanced_volume') else '禁用'}")
        print(f"   🌊 市场状态适应: {'启用' if config.enhanced_factor_config.get('enable_regime_adjustment') else '禁用'}")
    else:
        print("⚡ 使用原版简单因子 (单期动量 + 总波动率)")

    print(f"   🔽 下行频率过滤: {'启用' if config.apply_downside_frequency_filter else '禁用'}")
    if config.apply_downside_frequency_filter:
        print(f"       回看期: {config.downside_frequency_lookback_days}天, 阈值: {config.downside_frequency_threshold:.0%}")
    print(f"   📈 成交量因子: {'启用' if config.apply_volume_factor else '禁用'}")
    print(f"   🛑 个股止损: {'启用' if config.apply_stop_loss else '禁用'} ({config.stop_loss_pct:.0%})")
    print(f"   🔻 NDX做空对冲: {'启用' if config.apply_ndx_short_hedge else '禁用'} ({config.ndx_short_ratio:.0%})")
    print("="*80)

    # --- Backtest Simulation Loop ---
    print("\nStarting Backtest Simulation Outline with delayed execution (trade on next open after decision)...\n")
    portfolio_log = {}
    momentum_scores_history = {}
    portfolio_value_over_time = pd.Series(index=daily_returns.index, dtype=float)
    first_rebalance_done = False
    initial_capital = 1.0
    winning_periods = 0
    total_periods = 0

    # --- Rebalance Date Logic ---
    if config.rebalance_frequency == 'weekly':
        all_decision_day_candidates = pd.date_range(start=config.backtest_start_date, end=config.end_date, freq='W-FRI')
        decision_dates_indices = daily_returns.index.searchsorted(all_decision_day_candidates, side='right') - 1
        decision_dates_indices = decision_dates_indices[decision_dates_indices >= 0]
        decision_dates = daily_returns.index[decision_dates_indices].unique()
    elif config.rebalance_frequency == 'monthly':
        all_decision_day_candidates = pd.date_range(start=config.backtest_start_date, end=config.end_date, freq='M')
        decision_dates_indices = daily_returns.index.searchsorted(all_decision_day_candidates, side='right') - 1
        decision_dates_indices = decision_dates_indices[decision_dates_indices >= 0]
        decision_dates = daily_returns.index[decision_dates_indices].unique()
    else:
        raise ValueError(f"Unsupported rebalance_frequency: {config.rebalance_frequency}")

    decision_dates = decision_dates[decision_dates >= pd.to_datetime(config.backtest_start_date)]
    decision_dates = decision_dates[decision_dates <= pd.to_datetime(config.end_date)]

    if len(decision_dates) == 0:
        print("\nError: No valid decision dates found for the specified frequency and date range.")
        return None
    print(f"Found {len(decision_dates)} decision dates from {decision_dates.min().date()} to {decision_dates.max().date()}")

    # 🔄 每日计算因子值
    print(f"\n🔄 开始每日因子计算（调仓频率：{config.rebalance_frequency}）")
    print("💡 每日记录因子值，确保因子分析的一致性")

    all_trading_dates = daily_returns.index[daily_returns.index >= pd.to_datetime(config.backtest_start_date)]
    all_trading_dates = all_trading_dates[all_trading_dates <= pd.to_datetime(config.end_date)]
    total_dates = len(all_trading_dates)

    # 确定所需的最长回看期
    if config.use_enhanced_factor:
        max_lookback_days_needed = max(factor_engine.enhanced_factor_calculator.momentum_periods)
        max_volume_lookback_needed = factor_engine.enhanced_factor_calculator.volume_long_period
    else:
        max_lookback_days_needed = int(config.lookback_months * (config.trading_days_per_year / 12))
        max_volume_lookback_needed = config.volume_long_lookback_days

    # 每日因子计算循环
    for idx, current_date in enumerate(all_trading_dates):
        if idx % 10 == 0 or idx == total_dates - 1:
            progress = (idx + 1) / total_dates * 100
            print(f"  📅 计算因子进度: {idx+1}/{total_dates} ({progress:.1f}%) - {current_date.date()}")

        current_date_iloc = daily_returns.index.get_loc(current_date)
        momentum_scores = {}
        eligible_tickers_post_downside_filter = set()

        for ticker in daily_returns.columns:
            passes_filter = not config.apply_downside_frequency_filter
            try:
                # Downside Frequency Filter
                if config.apply_downside_frequency_filter:
                    downside_start = max(0, current_date_iloc - config.downside_frequency_lookback_days + 1)
                    returns_for_freq = daily_returns[ticker].iloc[downside_start : current_date_iloc + 1]
                    if len(returns_for_freq) >= config.downside_frequency_lookback_days * 0.90:
                        negative_days = (returns_for_freq < 0).sum()
                        frequency = negative_days / len(returns_for_freq)
                        passes_filter = frequency < config.downside_frequency_threshold

                if passes_filter:
                    eligible_tickers_post_downside_filter.add(ticker)
                    
                    # 准备因子计算数据
                    returns_start = max(0, current_date_iloc - max_lookback_days_needed + 1)
                    stock_returns = daily_returns[ticker].iloc[returns_start : current_date_iloc + 1]
                    
                    volume_start = max(0, current_date_iloc - max_volume_lookback_needed + 1)
                    stock_volumes = asset_volume_data[ticker].iloc[volume_start : current_date_iloc + 1]

                    if not stock_returns.empty:
                        score, vol = factor_engine.calculate_risk_adj_momentum_and_vol(
                            stock_returns,
                            stock_volumes,
                            stock_volumes,
                            config.lookback_months, 
                            config.trading_days_per_year, 
                            ticker=ticker, 
                            date=current_date.date()
                        )
                        if np.isfinite(score): 
                            momentum_scores[ticker] = score
            except Exception:
                continue

        screened_scores = {t: s for t, s in momentum_scores.items() if t in eligible_tickers_post_downside_filter}
        momentum_scores_history[current_date] = screened_scores.copy()

    # 每日因子计算完成总结
    print(f"\n✅ 每日因子计算完成！")
    print(f"  📈 总计算日数: {len(momentum_scores_history)}")
    if momentum_scores_history:
        print(f"  📅 时间范围: {min(momentum_scores_history.keys()).date()} 至 {max(momentum_scores_history.keys()).date()}")
        avg_factors = sum(len(scores) for scores in momentum_scores_history.values()) / len(momentum_scores_history)
        print(f"  📊 平均每日因子数: {avg_factors:.1f}")

    return _execute_backtest_decisions(
        decision_dates, daily_returns, asset_price_data_close, asset_price_data_open, 
        asset_volume_data, ndx_daily_returns, nasdaq_100_tickers, momentum_scores_history,
        portfolio_log, portfolio_value_over_time, first_rebalance_done, initial_capital,
        winning_periods, total_periods
    )

def _execute_backtest_decisions(decision_dates, daily_returns, asset_price_data_close, 
                               asset_price_data_open, asset_volume_data, ndx_daily_returns,
                               nasdaq_100_tickers, momentum_scores_history, portfolio_log,
                               portfolio_value_over_time, first_rebalance_done, initial_capital,
                               winning_periods, total_periods):
    """执行回测决策循环"""
    
    print(f"\n📈 开始调仓决策循环")
    
    for i, decision_date in enumerate(decision_dates):
        print(f"--- Decision Date: {decision_date.date()} ---")

        # 确定交易执行日期
        trade_execution_date = None
        try:
            decision_idx = daily_returns.index.get_loc(decision_date)
            if decision_idx + 1 < len(daily_returns.index):
                trade_execution_date = daily_returns.index[decision_idx + 1]
                print(f"  Trade Execution Date: {trade_execution_date.date()}")
            else:
                if config.output_next_period_recommendation:
                    print("  No execution data available, but generating recommendation for next period.")
                else:
                    print(f"  Skipping last decision date {decision_date.date()}")
                    continue
        except KeyError:
            print(f"  Decision date {decision_date.date()} not found in trading days. Skipping.")
            continue

        # 获取预计算的因子值
        screened_momentum_scores = momentum_scores_history.get(decision_date, {})

        # 计算波动率权重（如需要）
        stock_volatilities_for_period = {}
        if config.weighting_method == 'inverse_volatility' and screened_momentum_scores:
            decision_date_iloc = daily_returns.index.get_loc(decision_date)
            lookback_days = int(config.lookback_months * (config.trading_days_per_year / 12))
            
            for ticker in screened_momentum_scores.keys():
                try:
                    start_iloc = max(0, decision_date_iloc - lookback_days + 1)
                    stock_returns = daily_returns[ticker].iloc[start_iloc: decision_date_iloc + 1]
                    stock_volumes_short = asset_volume_data[ticker].iloc[start_iloc: decision_date_iloc + 1]
                    
                    vol_start = max(0, decision_date_iloc - config.volume_long_lookback_days + 1)
                    stock_volumes_long = asset_volume_data[ticker].iloc[vol_start: decision_date_iloc + 1]

                    if not stock_returns.empty:
                        score, vol = factor_engine.calculate_risk_adj_momentum_and_vol(
                            stock_returns, stock_volumes_short, stock_volumes_long,
                            config.lookback_months, config.trading_days_per_year, 
                            ticker=ticker, date=decision_date.date()
                        )
                        if vol is not None and np.isfinite(vol) and vol > 0:
                            stock_volatilities_for_period[ticker] = vol
                except Exception:
                    continue

        # 构建投资组合权重
        target_weights = _construct_portfolio_weights(
            screened_momentum_scores, stock_volatilities_for_period, nasdaq_100_tickers, i, decision_dates, portfolio_log
        )

        portfolio_log[decision_date] = target_weights

        # 输出持仓建议
        _print_recommendations(decision_date, target_weights)

        # 如果只生成建议，跳过回测
        if config.generate_recommendations_only:
            continue

        if trade_execution_date is None:
            continue

        # 执行回测模拟
        first_rebalance_done, winning_periods, total_periods = _simulate_portfolio_performance(
            i, decision_date, trade_execution_date, decision_dates, daily_returns,
            asset_price_data_close, asset_price_data_open, asset_volume_data,
            ndx_daily_returns, target_weights, portfolio_log, portfolio_value_over_time,
            first_rebalance_done, initial_capital, winning_periods, total_periods
        )

    # 最终处理
    return _finalize_backtest_results(
        portfolio_value_over_time, portfolio_log, momentum_scores_history,
        winning_periods, total_periods, daily_returns
    )

def _construct_portfolio_weights(screened_scores, stock_volatilities, nasdaq_100_tickers, i, decision_dates, portfolio_log):
    """构建投资组合权重"""
    
    if not screened_scores:
        print("  No valid momentum scores, using previous weights or skipping.")
        if not portfolio_log:
            return pd.Series(0.0, index=nasdaq_100_tickers)
        else:
            prev_key = decision_dates[i-1] if i > 0 else None
            return portfolio_log.get(prev_key, pd.Series(0.0, index=nasdaq_100_tickers))

    scores_series = pd.Series(screened_scores)
    ranked_stocks = scores_series.sort_values(ascending=False)
    selected_tickers = ranked_stocks.head(config.num_portfolio_stocks).index.tolist()
    print(f"  Selected {len(selected_tickers)} stocks: {selected_tickers}")

    target_weights = pd.Series(0.0, index=nasdaq_100_tickers)

    if selected_tickers:
        if config.weighting_method == 'equal':
            print("  Using Equal Weighting.")
            target_weights.loc[selected_tickers] = 1.0 / len(selected_tickers)
        elif config.weighting_method == 'factor_value':
            print("  Using Factor Value Weighting.")
            selected_scores = scores_series.loc[selected_tickers]
            positive_scores = selected_scores[selected_scores > 0]
            if not positive_scores.empty and positive_scores.sum() != 0:
                target_weights.loc[positive_scores.index] = positive_scores / positive_scores.sum()
            else:
                print("  Factor Value: No positive scores. Fallback to Equal Weight.")
                target_weights.loc[selected_tickers] = 1.0 / len(selected_tickers)
        elif config.weighting_method == 'inverse_volatility':
            print("  Using Inverse Volatility Weighting.")
            valid_vols = {t: stock_volatilities.get(t) for t in selected_tickers 
                         if t in stock_volatilities and stock_volatilities.get(t, 0) > 1e-8}
            if valid_vols:
                vols_series = pd.Series(valid_vols)
                inverse_vols = 1.0 / vols_series
                target_weights.loc[inverse_vols.index] = inverse_vols / inverse_vols.sum()
            else:
                print("  Inverse Vol: No valid volatilities. Fallback to Equal Weight.")
                target_weights.loc[selected_tickers] = 1.0 / len(selected_tickers)

        # 应用最大单股权重约束
        if target_weights.sum() > 1e-6:
            target_weights = target_weights.clip(upper=config.max_single_stock_weight)
            current_total = target_weights.sum()
            if current_total < 0.999:
                print(f"  Max weight cap applied. Total: {current_total:.2%}")

        # 应用NDX空头对冲
        if config.apply_ndx_short_hedge:
            long_ratio = 1.0 - config.ndx_short_ratio
            target_weights = target_weights * long_ratio
            print(f"  NDX Short Hedge: Scaled to {long_ratio:.1%}")

    return target_weights

def _print_recommendations(decision_date, target_weights):
    """输出持仓建议"""
    print(f"\n=== 风险调整动量策略持仓建议 for {decision_date.date()} ===")
    holdings = target_weights[target_weights > 0].sort_values(ascending=False)
    
    if not holdings.empty:
        print("推荐持仓:")
        for ticker, weight in holdings.items():
            print(f"  {ticker}: {weight:.2%}")
        
        total_weight = holdings.sum()
        if config.apply_ndx_short_hedge:
            print(f"\n多头总权重: {total_weight:.2%}")
            print(f"NDX做空权重: {config.ndx_short_ratio:.2%}")
            cash_weight = 1.0 - total_weight - config.ndx_short_ratio
            if cash_weight > 0.01:
                print(f"现金权重: {cash_weight:.2%}")
        else:
            cash_weight = 1.0 - total_weight
            if cash_weight > 0.01:
                print(f"现金权重: {cash_weight:.2%}")
    else:
        print("推荐: 全部现金")
    print("=" * 50)

def _simulate_portfolio_performance(i, decision_date, trade_execution_date, decision_dates,
                                  daily_returns, asset_price_data_close, asset_price_data_open,
                                  asset_volume_data, ndx_daily_returns, target_weights,
                                  portfolio_log, portfolio_value_over_time, first_rebalance_done,
                                  initial_capital, winning_periods, total_periods):
    """模拟投资组合表现"""
    
    try:
        current_exec_iloc = daily_returns.index.get_loc(trade_execution_date)
    except KeyError:
        print(f"  Trade execution date {trade_execution_date.date()} not found. Skipping.")
        return first_rebalance_done, winning_periods, total_periods

    # 确定持有期结束点
    next_exec_iloc = len(daily_returns.index)
    if i + 1 < len(decision_dates):
        try:
            next_decision = decision_dates[i+1]
            next_decision_idx = daily_returns.index.get_loc(next_decision)
            if next_decision_idx + 1 < len(daily_returns.index):
                next_trade_date = daily_returns.index[next_decision_idx + 1]
                next_exec_iloc = daily_returns.index.get_loc(next_trade_date)
        except KeyError:
            pass

    # 创建可变权重副本（用于止损）
    effective_weights = target_weights.reindex(daily_returns.columns, fill_value=0.0).copy()

    # 止损：存储成本基础
    holdings_cost_basis = pd.Series(dtype=float)
    if config.apply_stop_loss and not effective_weights.empty:
        if trade_execution_date in asset_price_data_close.index:
            for ticker in effective_weights[effective_weights > 0].index:
                holdings_cost_basis[ticker] = asset_price_data_close.loc[trade_execution_date, ticker]

    # 更新投资组合价值
    if not first_rebalance_done:
        # 首次调仓
        portfolio_value_over_time.loc[trade_execution_date] = initial_capital
        
        turnover = effective_weights.abs().sum()
        if config.apply_ndx_short_hedge:
            turnover += config.ndx_short_ratio
        transaction_costs = initial_capital * turnover * config.transaction_cost_pct
        
        value_after_costs = initial_capital - transaction_costs
        
        # 计算首日收益
        if (trade_execution_date in asset_price_data_open.index and 
            trade_execution_date in asset_price_data_close.index):
            
            open_prices = asset_price_data_open.loc[trade_execution_date].reindex(effective_weights.index, fill_value=np.nan)
            close_prices = asset_price_data_close.loc[trade_execution_date].reindex(effective_weights.index, fill_value=np.nan)
            
            exec_returns = pd.Series(0.0, index=effective_weights.index)
            valid_mask = open_prices.notna() & close_prices.notna() & (open_prices > 0)
            exec_returns[valid_mask] = (close_prices[valid_mask] / open_prices[valid_mask]) - 1
            
            portfolio_return = (exec_returns * effective_weights).sum()
            
            # NDX对冲收益
            if (config.apply_ndx_short_hedge and ndx_daily_returns is not None and 
                trade_execution_date in ndx_daily_returns.index):
                ndx_return = ndx_daily_returns.loc[trade_execution_date, config.ndx_ticker]
                if pd.notna(ndx_return):
                    portfolio_return -= ndx_return * config.ndx_short_ratio
            
            if np.isfinite(portfolio_return):
                portfolio_value_over_time.loc[trade_execution_date] = value_after_costs * (1 + portfolio_return)
            else:
                portfolio_value_over_time.loc[trade_execution_date] = value_after_costs
        else:
            portfolio_value_over_time.loc[trade_execution_date] = value_after_costs
        
        first_rebalance_done = True
    else:
        # 后续调仓
        try:
            exec_loc = daily_returns.index.get_loc(trade_execution_date)
            if exec_loc > 0:
                prev_day = daily_returns.index[exec_loc - 1]
                start_value = portfolio_value_over_time.get(prev_day)
                
                if pd.isna(start_value):
                    temp_series = portfolio_value_over_time.loc[:prev_day].ffill()
                    start_value = temp_series.iloc[-1] if not temp_series.empty else initial_capital
            else:
                start_value = initial_capital
        except KeyError:
            start_value = initial_capital

        if pd.isna(start_value):
            start_value = initial_capital

        # 计算换手费用
        prev_weights = portfolio_log[decision_dates[i-1]].reindex(daily_returns.columns, fill_value=0.0)
        current_weights = effective_weights.reindex(daily_returns.columns, fill_value=0.0)
        
        turnover = (current_weights - prev_weights).abs().sum() / 2
        if config.apply_ndx_short_hedge:
            turnover += config.ndx_short_ratio
        transaction_costs = start_value * turnover * config.transaction_cost_pct
        
        value_after_costs = start_value - transaction_costs
        
        # 计算执行日收益（同首次调仓逻辑）
        if (trade_execution_date in asset_price_data_open.index and 
            trade_execution_date in asset_price_data_close.index):
            
            open_prices = asset_price_data_open.loc[trade_execution_date].reindex(effective_weights.index, fill_value=np.nan)
            close_prices = asset_price_data_close.loc[trade_execution_date].reindex(effective_weights.index, fill_value=np.nan)
            
            exec_returns = pd.Series(0.0, index=effective_weights.index)
            valid_mask = open_prices.notna() & close_prices.notna() & (open_prices > 0)
            exec_returns[valid_mask] = (close_prices[valid_mask] / open_prices[valid_mask]) - 1
            
            portfolio_return = (exec_returns * effective_weights).sum()
            
            if (config.apply_ndx_short_hedge and ndx_daily_returns is not None and 
                trade_execution_date in ndx_daily_returns.index):
                ndx_return = ndx_daily_returns.loc[trade_execution_date, config.ndx_ticker]
                if pd.notna(ndx_return):
                    portfolio_return -= ndx_return * config.ndx_short_ratio
            
            if np.isfinite(portfolio_return):
                portfolio_value_over_time.loc[trade_execution_date] = value_after_costs * (1 + portfolio_return)
            else:
                portfolio_value_over_time.loc[trade_execution_date] = value_after_costs
        else:
            portfolio_value_over_time.loc[trade_execution_date] = value_after_costs

    # 记录期初价值用于胜率计算
    period_start_value = portfolio_value_over_time.get(trade_execution_date)
    if pd.isna(period_start_value):
        period_start_value = value_after_costs

    # 每日循环：止损检查和价值更新
    try:
        exec_iloc = daily_returns.index.get_loc(trade_execution_date)
    except KeyError:
        return first_rebalance_done, winning_periods, total_periods

    for k_offset in range(1, next_exec_iloc - current_exec_iloc):
        day_iloc = current_exec_iloc + k_offset
        if day_iloc >= len(daily_returns.index):
            break
        
        current_day = daily_returns.index[day_iloc]
        prev_day = daily_returns.index[day_iloc - 1]
        
        prev_value = portfolio_value_over_time.get(prev_day)
        if pd.isna(prev_value):
            if prev_day == trade_execution_date:
                prev_value = period_start_value
            else:
                temp_series = portfolio_value_over_time.loc[:prev_day].ffill()
                prev_value = temp_series.iloc[-1] if not temp_series.empty else initial_capital
        
        if pd.isna(prev_value):
            prev_value = initial_capital

        # 止损检查
        if config.apply_stop_loss and not holdings_cost_basis.empty:
            for ticker in list(holdings_cost_basis.index):
                if effective_weights.get(ticker, 0) > 0:
                    if (current_day in asset_price_data_close.index and 
                        ticker in asset_price_data_close.columns):
                        current_price = asset_price_data_close.loc[current_day, ticker]
                        cost_price = holdings_cost_basis[ticker]
                        if (pd.notna(current_price) and pd.notna(cost_price) and cost_price > 0):
                            stop_price = cost_price * (1 - config.stop_loss_pct)
                            if current_price < stop_price:
                                print(f"  STOP-LOSS HIT: {ticker} on {current_day.date()}. "
                                     f"Price: {current_price:.2f} < SL: {stop_price:.2f}")
                                effective_weights[ticker] = 0.0

        # 计算当日收益
        daily_asset_returns = daily_returns.iloc[day_iloc]
        aligned_returns = daily_asset_returns.reindex(effective_weights.index, fill_value=0.0)
        day_portfolio_return = (aligned_returns * effective_weights).sum()
        
        # NDX对冲收益
        if (config.apply_ndx_short_hedge and ndx_daily_returns is not None and 
            current_day in ndx_daily_returns.index):
            ndx_return = ndx_daily_returns.loc[current_day, config.ndx_ticker]
            if pd.notna(ndx_return):
                day_portfolio_return -= ndx_return * config.ndx_short_ratio
        
        if not np.isfinite(day_portfolio_return):
            day_portfolio_return = 0.0

        portfolio_value_over_time.loc[current_day] = prev_value * (1 + day_portfolio_return)

    # 胜率计算
    if first_rebalance_done and current_exec_iloc < next_exec_iloc:
        period_end_date = daily_returns.index[next_exec_iloc - 1]
        period_end_value = portfolio_value_over_time.get(period_end_date)
        
        if (pd.notna(period_start_value) and pd.notna(period_end_value) and period_start_value != 0):
            period_return = (period_end_value / period_start_value) - 1
            if period_return > 0:
                winning_periods += 1
            total_periods += 1

    return first_rebalance_done, winning_periods, total_periods

def _finalize_backtest_results(portfolio_value_over_time, portfolio_log, momentum_scores_history,
                              winning_periods, total_periods, daily_returns):
    """完成回测结果处理"""
    
    # 最终处理
    portfolio_value_over_time = portfolio_value_over_time.reindex(daily_returns.index).ffill().dropna()
    weights_df = pd.DataFrame(portfolio_log).T

    print("\n--- Backtest Simulation Complete ---")

    # 输出建议汇总
    print("\n" + "="*80)
    print("📊 风险调整动量策略建议汇总 📊")
    print("="*80)

    if portfolio_log:
        latest_date = max(portfolio_log.keys())
        latest_recommendation = portfolio_log[latest_date]
        latest_holdings = latest_recommendation[latest_recommendation > 0].sort_values(ascending=False)
        
        print(f"\n🔥 最新风险调整动量建议 ({latest_date.date()}):")
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
        
        print(f"\n📈 总共生成了 {len(portfolio_log)} 期调仓建议")
        
        # 保存建议
        try:
            recommendations_df = pd.DataFrame(portfolio_log).T
            recommendations_df = recommendations_df[recommendations_df.sum(axis=1) > 0]
            
            non_zero_columns = (recommendations_df != 0).any(axis=0)
            recommendations_summary = recommendations_df.loc[:, non_zero_columns]
            
            factor_suffix = "_Enhanced" if config.use_enhanced_factor else "_Original"
            hedge_suffix = f"_NDXShort{int(config.ndx_short_ratio*100)}pct" if config.apply_ndx_short_hedge else ""
            filename = f'Risk_Adjusted_Momentum_Recommendations_{config.weighting_method}{factor_suffix}{hedge_suffix}.xlsx'
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                recommendations_summary.to_excel(writer, sheet_name='Risk_Adjusted_Recommendations', index_label='Decision_Date')
                
                formatted_recs = recommendations_summary.copy()
                for col in formatted_recs.columns:
                    formatted_recs[col] = formatted_recs[col].apply(lambda x: f"{x:.2%}" if x > 0 else "")
                formatted_recs.to_excel(writer, sheet_name='Formatted_Recommendations', index_label='Decision_Date')
            
            print(f"💾 建议已保存至: {filename}")
            
        except Exception as e:
            print(f"保存建议文件时出错: {e}")

    print("="*80)

    return {
        "portfolio_value_over_time": portfolio_value_over_time,
        "weights_df": weights_df,
        "portfolio_log": portfolio_log,
        "momentum_scores_history": momentum_scores_history,
        "winning_periods": winning_periods,
        "total_periods": total_periods
    }