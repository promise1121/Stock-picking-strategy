# -*- coding: utf-8 -*-
"""
Factor Engine for the Risk-Adjusted Momentum Strategy.
"""

import numpy as np
import pandas as pd
import config

# --- Enhanced Factor Calculation Function ---
class EnhancedRiskAdjustedMomentum:
    """增强版风险调整动量因子类"""
    
    def __init__(self):
        # 多期动量参数
        self.momentum_periods = [21, 63, 126]  # 1月、3月、6月
        self.momentum_weights = np.array([0.8, 0.0, 0.2])  # 权重递减
        self.momentum_weights = self.momentum_weights / self.momentum_weights.sum()
        
        # 风险调整参数
        self.volatility_period = 63
        self.downside_threshold = 0.0
        self.max_dd_period = 252
        self.quality_period = 63
        
        # 成交量参数
        self.volume_short_period = 21
        self.volume_long_period = 126
    
    def calculate_multi_period_momentum(self, returns):
        """计算多期动量组合"""
        momentum_scores = []
        
        for period in self.momentum_periods:
            if len(returns) >= period:
                period_returns = returns.iloc[-period:]
                total_return = (1 + period_returns).prod() - 1
                momentum_scores.append(total_return if np.isfinite(total_return) else 0.0)
            else:
                momentum_scores.append(0.0)
        
        return np.sum(np.array(momentum_scores) * self.momentum_weights)
    
    def calculate_downside_deviation(self, returns, threshold=0.0):
        """计算下行偏差（只考虑负收益）"""
        downside_returns = returns[returns < threshold]
        if len(downside_returns) > 2:
            return downside_returns.std()
        else:
            return returns.std()
    
    def calculate_max_drawdown_ratio(self, returns):
        """计算最大回撤比率"""
        if len(returns) < 10:
            return 0.0
        
        period_data = returns.iloc[-min(len(returns), self.max_dd_period):]
        cumulative = (1 + period_data).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1
        max_dd = drawdown.min()
        
        return abs(max_dd) if np.isfinite(max_dd) else 0.0
    
    def calculate_return_quality(self, returns):
        """计算收益质量评分"""
        if len(returns) < self.quality_period * 0.5:
            return 1.0
        
        quality_data = returns.iloc[-min(len(returns), self.quality_period):]
        
        # 1. 正收益天数占比
        positive_ratio = (quality_data > 0).sum() / len(quality_data)
        
        # 2. 收益稳定性
        if quality_data.std() > 0:
            stability = abs(quality_data.mean()) / quality_data.std()
        else:
            stability = 1.0
        
        # 3. 偏度调整
        try:
            from scipy import stats
            skewness = stats.skew(quality_data.dropna())
            skew_adjustment = 1.0 + max(-0.5, min(0.5, skewness / 2))
        except:
            skew_adjustment = 1.0
        
        # 综合质量评分
        quality_score = positive_ratio * stability * skew_adjustment
        return max(0.1, min(3.0, quality_score))
    
    def calculate_enhanced_volume_factor(self, volumes, returns):
        """增强版成交量因子"""
        if len(volumes) < max(self.volume_short_period, self.volume_long_period):
            return 1.0
        
        short_vol = volumes.iloc[-self.volume_short_period:].mean()
        long_vol = volumes.iloc[-self.volume_long_period:].mean()
        
        if long_vol <= 0:
            return 1.0
        
        # 1. 基础成交量比率
        volume_ratio = short_vol / long_vol
        
        # 2. 价量配合度
        recent_returns = returns.iloc[-self.volume_short_period:]
        recent_volumes = volumes.iloc[-self.volume_short_period:]
        
        if len(recent_returns) == len(recent_volumes) and len(recent_returns) > 5:
            try:
                price_vol_corr = recent_returns.corr(recent_volumes)
                if np.isfinite(price_vol_corr):
                    pv_adjustment = 1.0 + price_vol_corr * 0.2
                else:
                    pv_adjustment = 1.0
            except:
                pv_adjustment = 1.0
        else:
            pv_adjustment = 1.0
        
        # 3. 成交量相对排名
        long_period_volumes = volumes.iloc[-self.volume_long_period:]
        if len(long_period_volumes) > 10:
            volume_percentile = (long_period_volumes < short_vol).sum() / len(long_period_volumes)
            percentile_adjustment = 0.5 + volume_percentile
        else:
            percentile_adjustment = 1.0
        
        # 综合成交量因子
        enhanced_volume_factor = (volume_ratio * 0.4 + 
                                pv_adjustment * 0.3 + 
                                percentile_adjustment * 0.3)
        
        return max(0.3, min(2.5, enhanced_volume_factor))
    
    def calculate_market_regime_adjustment(self, returns):
        """市场状态调整"""
        if len(returns) < 63:
            return 1.0
        
        recent_returns = returns.iloc[-63:]
        
        # 波动率状态调整
        current_vol = recent_returns.std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252) if len(returns) > 126 else current_vol
        
        if historical_vol > 0:
            vol_ratio = current_vol / historical_vol
            vol_adjustment = 1.0 / (1.0 + max(0, vol_ratio - 1.0) * 0.3)
        else:
            vol_adjustment = 1.0
        
        # 趋势强度调整
        if len(returns) >= 20:
            cumulative_recent = (1 + recent_returns).cumprod()
            ma_20 = cumulative_recent.rolling(20).mean()
            current_price = cumulative_recent.iloc[-1]
            
            if len(ma_20.dropna()) > 0 and ma_20.iloc[-1] > 0:
                trend_strength = (current_price / ma_20.iloc[-1] - 1)
                trend_adjustment = 1.0 + max(-0.2, min(0.3, trend_strength))
            else:
                trend_adjustment = 1.0
        else:
            trend_adjustment = 1.0
        
        return vol_adjustment * trend_adjustment

# 创建全局增强因子计算器实例
if hasattr(config, 'enhanced_factor_config') and config.use_enhanced_factor:
    # 使用用户配置初始化增强因子计算器
    enhanced_factor_calculator = EnhancedRiskAdjustedMomentum()
    if config.enhanced_factor_config.get('momentum_periods'):
        enhanced_factor_calculator.momentum_periods = config.enhanced_factor_config['momentum_periods']
    if config.enhanced_factor_config.get('momentum_weights'):
        weights = np.array(config.enhanced_factor_config['momentum_weights'])
        enhanced_factor_calculator.momentum_weights = weights / weights.sum()
else:
    enhanced_factor_calculator = EnhancedRiskAdjustedMomentum()

def calculate_risk_adj_momentum_and_vol(
    stock_returns_daily, 
    stock_short_term_volumes, 
    stock_long_term_volumes,  
    lookback_months, 
    trading_days_per_year, 
    ticker="N/A", 
    date="N/A"
):
    """
    增强版风险调整动量和波动率计算函数
    可根据配置选择使用增强版或原版计算逻辑
    """
    
    # 根据配置选择计算方式
    if config.use_enhanced_factor:
        return _calculate_enhanced_factor(
            stock_returns_daily, stock_short_term_volumes, stock_long_term_volumes,
            lookback_months, trading_days_per_year, ticker, date
        )
    else:
        return _calculate_original_factor(
            stock_returns_daily, stock_short_term_volumes, stock_long_term_volumes,
            lookback_months, trading_days_per_year, ticker, date
        )

def _calculate_enhanced_factor(stock_returns_daily, stock_short_term_volumes, stock_long_term_volumes,
                              lookback_months, trading_days_per_year, ticker, date):
    """增强版因子计算"""
    try:
        # 检查数据充分性
        min_data_points = max(enhanced_factor_calculator.momentum_periods) * 0.8
        if len(stock_returns_daily) < min_data_points:
            return np.nan, np.nan
        
        # 1. 多期动量组合
        momentum_score = enhanced_factor_calculator.calculate_multi_period_momentum(stock_returns_daily)
        
        # 2. 增强风险调整
        if config.enhanced_factor_config.get('downside_vol_only', True):
            downside_vol = enhanced_factor_calculator.calculate_downside_deviation(stock_returns_daily)
        else:
            downside_vol = stock_returns_daily.std()  # 使用总波动率
        
        if config.enhanced_factor_config.get('max_drawdown_adjustment', True):
            max_dd_ratio = enhanced_factor_calculator.calculate_max_drawdown_ratio(stock_returns_daily)
        else:
            max_dd_ratio = 0.0
        
        # 综合风险度量
        risk_measure = downside_vol * (1 + max_dd_ratio)
        epsilon = 1e-8
        
        if risk_measure <= epsilon:
            base_risk_adj_score = 0.0
        else:
            base_risk_adj_score = momentum_score / risk_measure
        
        # 3. 收益质量调整
        if config.enhanced_factor_config.get('enable_quality_adjustment', True):
            quality_multiplier = enhanced_factor_calculator.calculate_return_quality(stock_returns_daily)
        else:
            quality_multiplier = 1.0
        
        # 4. 增强成交量调整
        if (config.apply_volume_factor and config.enhanced_factor_config.get('enable_enhanced_volume', True) and 
            not stock_short_term_volumes.empty and not stock_long_term_volumes.empty):
            volume_multiplier = enhanced_factor_calculator.calculate_enhanced_volume_factor(
                stock_short_term_volumes, stock_returns_daily)
        elif config.apply_volume_factor and not stock_short_term_volumes.empty and not stock_long_term_volumes.empty:
            # 使用简单成交量因子作为备选
            volume_multiplier = _calculate_simple_volume_factor(stock_short_term_volumes, stock_long_term_volumes)
        else:
            volume_multiplier = 1.0
        
        # 5. 市场状态调整
        if config.enhanced_factor_config.get('enable_regime_adjustment', True):
            regime_adjustment = enhanced_factor_calculator.calculate_market_regime_adjustment(stock_returns_daily)
        else:
            regime_adjustment = 1.0
        
        # 综合因子评分
        enhanced_score = (base_risk_adj_score * 
                        quality_multiplier * 
                        volume_multiplier * 
                        regime_adjustment)
        
        return enhanced_score, downside_vol
        
    except Exception as e:
        print(f"  ⚠️ Enhanced factor calculation failed for {ticker} on {date}: {str(e)}")
        # 回退到原版计算
        return _calculate_original_factor(
            stock_returns_daily, stock_short_term_volumes, stock_long_term_volumes,
            lookback_months, trading_days_per_year, ticker, date
        )

def _calculate_original_factor(stock_returns_daily, stock_short_term_volumes, stock_long_term_volumes,
                              lookback_months, trading_days_per_year, ticker, date):
    """原版简单因子计算（作为备选方案）"""
    lookback_days = int(lookback_months * (trading_days_per_year / 12))

    if len(stock_returns_daily) < lookback_days * 0.9:
        return np.nan, np.nan

    relevant_returns = stock_returns_daily.iloc[-lookback_days:]
    total_return = (1 + relevant_returns).prod() - 1
    volatility = relevant_returns.std()
    epsilon = 1e-8 

    if not np.isfinite(total_return):
        return np.nan, volatility if volatility is not None and np.isfinite(volatility) else np.nan

    if volatility is None or not np.isfinite(volatility) or np.isclose(volatility, 0):
        if np.isclose(total_return, 0):
            risk_adj_score = 0.0
        else:
            risk_adj_score = np.nan 
        original_vol = volatility if volatility is not None and np.isfinite(volatility) else np.nan
        return risk_adj_score, original_vol

    # 基础风险调整分数
    risk_adj_score = total_return / (volatility + epsilon)

    # 原版成交量调整
    if config.apply_volume_factor:
        volume_multiplier = _calculate_simple_volume_factor(stock_short_term_volumes, stock_long_term_volumes)
        if pd.notna(risk_adj_score):
            risk_adj_score *= volume_multiplier
    
    return risk_adj_score, volatility

def _calculate_simple_volume_factor(stock_short_term_volumes, stock_long_term_volumes):
    """简单成交量因子计算（原版逻辑）"""
    min_short_vol_days = len(stock_short_term_volumes) * 0.9
    min_long_vol_days = config.volume_long_lookback_days * 0.9

    if (stock_short_term_volumes.empty or stock_long_term_volumes.empty or
        len(stock_short_term_volumes) < min_short_vol_days or
        len(stock_long_term_volumes) < min_long_vol_days):
        return 1.0
    
    avg_volume_short_term = stock_short_term_volumes.mean()
    avg_volume_long_term = stock_long_term_volumes.mean()

    if (pd.isna(avg_volume_short_term) or pd.isna(avg_volume_long_term) or
        avg_volume_long_term is None or np.isclose(avg_volume_long_term, 0)):
        return 1.0
    
    volume_multiplier = avg_volume_short_term / avg_volume_long_term
    return max(config.volume_multiplier_floor, min(volume_multiplier, config.volume_multiplier_cap))
