# -*- coding: utf-8 -*-
"""
Enhanced Risk-Adjusted Momentum Strategy Configuration
"""

import datetime

# --- Parameters ---
nasdaq_100_tickers = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AVGO', 'ASML',
    'COST', 'PEP', 'ADBE', 'CSCO', 'TMUS', 'AMD', 'NFLX', 'INTC', 'CMCSA', 'INTU',
    'AMGN', 'TXN', 'QCOM', 'HON', 'AMAT', 'ISRG', 'BKNG', 'SBUX', 'VRTX', 'MDLZ',
    'GILD', 'ADP', 'LRCX', 'ADI', 'REGN', 'PYPL', 'MU', 'CSX', 'PANW', 'SNPS',
    'CDNS', 'MAR', 'MELI', 'KLAC', 'CRWD', 'ORLY', 'CTAS', 'ABNB', 'NXPI', 'PCAR',
    'EXC', 'AEP', 'PAYX', 'MNST', 'ROP', 'CPRT', 'FTNT', 'AZN', 'BKR', 'DXCM',
    'FAST', 'IDXX', 'WDAY', 'CEG', 'MRVL', 'ROST', 'ODFL', 'DDOG', 'TEAM', 'MCHP',
    'MRNA', 'ON', 'XEL', 'KDP', 'CTSH', 'EA', 'WBD', 'FANG', 'GEHC', 'BIIB',
    'ILMN', 'GFS', 'CSGP', 'ZS', 'WBA', 'DLTR', 'TTD', 'VRSK', 'ENPH', 'ALGN',
    'ZM', 'JD', 'LCID', 'SIRI', 'APP', 'PLTR', 'MSTR', 'PDD', 'TSM', 'LULU', 
    'SPOT', 'RBLX', 'UBER', 'SE', 'CART', 'CVNA', 'DASH', 'SHOP', 'SAP', 'NOW',
    'NET', 'CYBR', 'OKTA', 'CLS', 'NBIS', 'GLW', 'ALAB', 'CRDO' # Stock pool
]
nasdaq_100_tickers = sorted(list(set(nasdaq_100_tickers)))

# --- 添加無風險利率 Ticker ---
risk_free_ticker = '^IRX'  # US 13 Week Treasury Bill (^IRX)

start_date = '2019-06-01'  # Longer history for lookback
backtest_start_date = '2019-12-01'  # Start backtest after initial lookback period
end_date = datetime.date.today().strftime('%Y-%m-%d')

lookback_months = 1  # 1-month lookback
num_portfolio_stocks = 10  # Number of stocks to hold (Top N)
trading_days_per_year = 252
transaction_cost_pct = 0.002 # 0.1% transaction cost
max_single_stock_weight = 0.30 # Max weight for any single stock in the portfolio

# --- Weighting Method Parameter ---
weighting_method = 'factor_value'  # Options: 'equal', 'factor_value', 'inverse_volatility'

# --- New Filter Parameters ---
apply_downside_frequency_filter = True  # Set to False to disable this filter
downside_frequency_lookback_days = 63   # Lookback period for downside frequency
downside_frequency_threshold = 0.50     # Max allowed frequency of negative days (e.g., 0.40 for 40%)

# --- Enhanced Factor Parameters ---
apply_volume_factor = True             # Set to False to disable volume factor
volume_long_lookback_days = 63        # Approx 3 months for longer-term volume average
volume_multiplier_cap = 2           # Max value for the volume multiplier (legacy, kept for compatibility)
volume_multiplier_floor = 0.5         # Min value for the volume multiplier (legacy, kept for compatibility)

# --- Enhanced Factor Configuration ---
use_enhanced_factor = False              # Set to False to use original simple factor
enhanced_factor_config = {
    'momentum_periods': [21, 63, 126],    # Multi-period momentum (1M, 3M, 6M)
    'momentum_weights': [0.8, 0.0, 0.2],  # Weights for each momentum period
    'enable_quality_adjustment': True,     # Enable return quality scoring
    'enable_regime_adjustment': True,      # Enable market regime adjustment
    'enable_enhanced_volume': True,        # Enable enhanced volume factor
    'downside_vol_only': True,            # Use downside deviation instead of total volatility
    'max_drawdown_adjustment': True        # Include max drawdown in risk measure
}

# --- Rebalance Frequency Parameter ---
rebalance_frequency = 'monthly'  # Options: 'monthly', 'weekly'
# rebalance_frequency = 'monthly'

# --- Stop-Loss Parameters ---
apply_stop_loss = True            # Set to False to disable this stop-loss mechanism
stop_loss_pct = 0.10              # e.g., 0.10 for a 10% stop-loss from purchase/rebalance price

# --- Strategy Mode Parameters ---
generate_recommendations_only = False  # Set to True to only generate trading recommendations without backtesting
output_next_period_recommendation = True  # Set to True to output recommendation for next period even without execution data

# --- NDX Short Hedge Parameters ---
apply_ndx_short_hedge = False       # Set to False to disable NDX short hedging
ndx_short_ratio = 0.10            # Percentage of portfolio to short NDX (e.g., 0.20 for 20%)
ndx_ticker = '^NDX'               # NASDAQ-100 Index ticker for shorting
