# -*- coding: utf-8 -*-
"""
Data Loader for the Risk-Adjusted Momentum Strategy.
从原始代码1:1还原的数据下载和预处理逻辑
"""
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

import config

warnings.filterwarnings('ignore')

def load_data():
    """
    从原始代码1:1还原的数据下载和预处理函数
    
    Returns:
        A dictionary containing all necessary dataframes and variables
    """
    
    # --- Data Download ---
    all_tickers_to_download = config.nasdaq_100_tickers + [config.risk_free_ticker]
    if config.apply_ndx_short_hedge:
        all_tickers_to_download.append(config.ndx_ticker)
    
    try:
        # 下載數據
        downloaded_data = yf.download(all_tickers_to_download, start=config.start_date, end=config.end_date)

        if downloaded_data.empty:
            raise ValueError("Downloaded data is empty. Check ticker list or date range.")

        print("Downloaded data columns:", downloaded_data.columns)

        # Attempt to access data based on typical yfinance structure (metrics as top-level columns)
        try:
            if isinstance(downloaded_data.columns, pd.MultiIndex):
                # If MultiIndex, check if metrics are at the first level
                if 'Close' in downloaded_data.columns.get_level_values(0) and \
                   'Open' in downloaded_data.columns.get_level_values(0) and \
                   'Volume' in downloaded_data.columns.get_level_values(0):
                    
                    price_data_close = downloaded_data['Close']
                    price_data_open = downloaded_data['Open']
                    volume_data = downloaded_data['Volume']
                    print("Data accessed with metrics as top-level columns (using 'Close' for closing price).")
                else: # Metrics might be at the second level if grouped by ticker
                    print("Metrics not found at top level. Attempting to access by ticker grouping (using 'Close' for closing price).")
                    # Reconstruct: We want DataFrames where index is Date, columns are Tickers, values are the metric
                    price_data_close = downloaded_data.xs('Close', level=1, axis=1)
                    price_data_open = downloaded_data.xs('Open', level=1, axis=1)
                    volume_data = downloaded_data.xs('Volume', level=1, axis=1)
                    print("Data accessed and reorganized from ticker-grouped structure (using 'Close' for closing price).")
            else: # Single index columns, likely for a single ticker download or flat structure
                if 'Close' in downloaded_data.columns and \
                   'Open' in downloaded_data.columns and \
                   'Volume' in downloaded_data.columns:
                    # This path is taken if yfinance returns a flat structure.
                    # We assume the columns 'Close', 'Open', 'Volume' directly contain the data for the tickers.
                    # This is simpler if only one ticker was requested and returned, or if yf.download flattens it.
                    # The subsequent filtering `price_data_close[valid_nasdaq_tickers]` will handle ticker selection.
                    price_data_close = downloaded_data # All columns initially
                    price_data_open = downloaded_data  # All columns initially
                    volume_data = downloaded_data    # All columns initially
                    
                    price_data_close = downloaded_data # Let subsequent filtering handle ticker selection
                    price_data_open = downloaded_data
                    volume_data = downloaded_data
                    # Ensure these specific columns are selected if they exist, otherwise subsequent logic for asset_price_data_close fails.
                    # This part is still a bit weak if the flat structure isn't just for one ticker. The MultiIndex logic is preferred.

                else:
                     raise ValueError("Downloaded data has single-level columns but metrics 'Close', 'Open', 'Volume' are not all present directly or structure is unexpected.")


        except KeyError as e:
            raise ValueError(f"Failed to access standard data columns (e.g., 'Close'). Error: {e}. Columns: {downloaded_data.columns}")
        except Exception as e: # Other errors during data restructuring
            raise ValueError(f"Error restructuring downloaded data. Error: {e}. Columns: {downloaded_data.columns}")


        # --- From here, we assume price_data_close, price_data_open, volume_data are DataFrames ---
        # --- where index is Date, columns are Tickers (for that specific metric) ---

        #分開處理資產價格和無風險利率
        # Ensure all nasdaq_100_tickers are actually columns in the downloaded price_data_close
        valid_nasdaq_tickers = [t for t in config.nasdaq_100_tickers if t in price_data_close.columns]
        if len(valid_nasdaq_tickers) < len(config.nasdaq_100_tickers):
            print(f"Warning: Some NASDAQ tickers were not found in downloaded 'Close' data. Found: {len(valid_nasdaq_tickers)}/{len(config.nasdaq_100_tickers)}")
        asset_price_data_close = price_data_close[valid_nasdaq_tickers]

        # Ensure all nasdaq_100_tickers are actually columns in the downloaded price_data_open
        # If price_data_open is the entire downloaded_data (from flat structure case):
        valid_nasdaq_tickers_open = [t for t in config.nasdaq_100_tickers if t in price_data_open.columns]
        asset_price_data_open = price_data_open[valid_nasdaq_tickers_open]

        # Ensure all nasdaq_100_tickers are actually columns in the downloaded volume_data
        valid_nasdaq_tickers_volume = [t for t in config.nasdaq_100_tickers if t in volume_data.columns]
        asset_volume_data = volume_data[valid_nasdaq_tickers_volume]

        common_tickers_for_assets = list(set(asset_price_data_close.columns) & set(asset_price_data_open.columns) & set(asset_volume_data.columns))
        
        if not common_tickers_for_assets:
            raise ValueError("No common tickers found across Close, Open, and Volume data for the specified NASDAQ list.")

        nasdaq_100_tickers = sorted(common_tickers_for_assets)
        asset_price_data_close = asset_price_data_close[nasdaq_100_tickers]
        asset_price_data_open = asset_price_data_open[nasdaq_100_tickers]
        asset_volume_data = asset_volume_data[nasdaq_100_tickers]


        if config.risk_free_ticker not in price_data_close.columns:
            raise ValueError(f"Risk-free ticker {config.risk_free_ticker} not found in downloaded 'Close' price data columns: {price_data_close.columns.tolist()}")
        rf_data = price_data_close[[config.risk_free_ticker]] # Keep it as DataFrame

        # 檢查資產價格數據
        if asset_price_data_close.empty: # Open data emptiness is implicitly checked by common_tickers_for_assets
            raise ValueError("Asset price data (Close) is empty after filtering.")
        if asset_price_data_open.empty:
            raise ValueError("Asset price data (Open) is empty after filtering.")

        print(f"Successfully processed data for {len(nasdaq_100_tickers)} common tickers.")

        # 填充資產價格數據
        asset_price_data_close = asset_price_data_close.ffill().bfill()
        asset_price_data_open = asset_price_data_open.ffill().bfill() # Fill open prices as well
        print("Asset close and open price data download and initial fill complete.")

        # --- Fill and Process Volume Data ---
        if asset_volume_data.empty:
            print("Warning: Asset volume data download failed or is empty. Volume factor will not be effective.")
            # Create an empty dataframe with same index/columns as price_data to avoid errors later, fill with 0 or NaN
            asset_volume_data = pd.DataFrame(index=asset_price_data_close.index, columns=asset_price_data_close.columns).fillna(0)
        else:
            asset_volume_data = asset_volume_data[nasdaq_100_tickers] # Ensure columns match price data after potential drops
            asset_volume_data = asset_volume_data.ffill().bfill().fillna(0) # Fill NaNs, then fill any remaining with 0
            print("Asset volume data download and initial fill complete.")

        # 檢查無風險利率數據
        if rf_data.empty or (rf_data.isnull().all()).all():
            raise ValueError(f"Risk-free rate ({config.risk_free_ticker}) data download failed or is all NaN.")
        # 處理無風險利率數據 (填充)
        rf_rate_annual_decimal = rf_data[config.risk_free_ticker] / 100  # 轉換為小數, access column directly
        rf_rate_annual_decimal = rf_rate_annual_decimal.ffill().bfill()  # 填充 NaN
        print("Risk-free rate data download and initial fill complete.")

        # --- Process NDX Data for Short Hedging ---
        ndx_data = None
        ndx_daily_returns = None
        if config.apply_ndx_short_hedge:
            if config.ndx_ticker not in price_data_close.columns:
                print(f"Warning: NDX ticker {config.ndx_ticker} not found in downloaded data. Short hedge will be disabled.")
                # 注意：这里不能修改config的值，只能设置local变量
                ndx_hedge_enabled = False
            else:
                ndx_data = price_data_close[[config.ndx_ticker]]
                if ndx_data.empty or (ndx_data.isnull().all()).all():
                    print(f"Warning: NDX data ({config.ndx_ticker}) is empty or all NaN. Short hedge will be disabled.")
                    ndx_hedge_enabled = False
                else:
                    ndx_data = ndx_data.ffill().bfill()
                    ndx_daily_returns = ndx_data.pct_change().dropna()
                    ndx_hedge_enabled = True
                    print(f"NDX data ({config.ndx_ticker}) for short hedging processed successfully.")
        else:
            ndx_hedge_enabled = False

    except Exception as e:
        print(f"Error during data download or initial processing: {e}")
        print("Please check your ticker list, date range, internet connection, and yfinance installation.")
        raise

    # 計算資產日收益率
    daily_returns = asset_price_data_close.pct_change().dropna()
    print("\nDaily returns calculated.")

    # --- 對齊無風險利率和資產收益率數據 (步驟 4 邏輯) ---
    common_index = daily_returns.index.intersection(rf_rate_annual_decimal.index)
    daily_returns = daily_returns.loc[common_index]
    rf_rate_annual_decimal = rf_rate_annual_decimal.loc[common_index]

    # --- Align NDX data if short hedge is enabled ---
    if config.apply_ndx_short_hedge and 'ndx_daily_returns' in locals() and ndx_daily_returns is not None:
        common_index = common_index.intersection(ndx_daily_returns.index)
        daily_returns = daily_returns.loc[common_index]
        rf_rate_annual_decimal = rf_rate_annual_decimal.loc[common_index]
        ndx_daily_returns = ndx_daily_returns.loc[common_index]
        print(f"NDX data aligned for short hedging.")
    elif config.apply_ndx_short_hedge:
        # 如果配置要求NDX对冲但数据不可用，设置为None
        ndx_daily_returns = None

    print(f"Aligned data from {common_index.min().date()} to {common_index.max().date()}")
    # --- 無風險利率處理結束 ---

    return {
        "daily_returns": daily_returns,
        "asset_price_data_close": asset_price_data_close,
        "asset_price_data_open": asset_price_data_open,
        "asset_volume_data": asset_volume_data,
        "rf_rate_annual_decimal": rf_rate_annual_decimal,
        "ndx_daily_returns": ndx_daily_returns if 'ndx_daily_returns' in locals() else None,
        "nasdaq_100_tickers": nasdaq_100_tickers,
        "common_index": common_index,
        "start_date": config.start_date,
        "end_date": config.end_date
    }
