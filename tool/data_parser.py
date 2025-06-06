""" 
Functions and details : 
1. time:2023/1/1-2024/12/31
2. data should be from binance public api. Also I want the data to be in 1 hr k line. 
3. trading pair: BTCUSDT,ETHUSDT, LINKUSDT, XRPUSDT, SOLUSDT, BNBUSDT, GALAUSDT, GMTUSDT, ADAUSDT
4. Data should be save to:   /Users/mouyasushi/Desktop/Factor_ML/data
5. Add a function that I can : Combine the data into one big data frame ( put all trading pairs into a column called 'symbol', and I want to transfer them into e.g. btc, eth, etc  )

"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from typing import List, Dict, Union, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = "/Users/mouyasushi/Desktop/Factor_ML/data"
TRADING_PAIRS = ["BTCUSDT", "ETHUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT", 
                 "BNBUSDT", "GALAUSDT", "GMTUSDT", "ADAUSDT"]
SYMBOL_MAP = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
    "LINKUSDT": "link",
    "XRPUSDT": "xrp",
    "SOLUSDT": "sol",
    "BNBUSDT": "bnb",
    "GALAUSDT": "gala",
    "GMTUSDT": "gmt",
    "ADAUSDT": "ada"
}
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
INTERVAL = "1h"  # 1 hour kline

def ensure_data_directory() -> None:
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory: {DATA_DIR}")

def unix_timestamp(date_str: str) -> int:
    """Convert date string to unix timestamp in milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)

def fetch_kline_data(symbol: str, start_time: int, end_time: int, interval: str = INTERVAL) -> List[List]:
    """
    Fetch kline data from Binance API.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        start_time: Start time in unix timestamp (milliseconds)
        end_time: End time in unix timestamp (milliseconds)
        interval: Kline interval (default: '1h')
        
    Returns:
        List of kline data
    """
    all_klines = []
    
    # Binance has a limit of 1000 klines per request, so we need to make multiple requests
    current_start = start_time
    
    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_time,
            'limit': 1000
        }
        
        try:
            response = requests.get(BINANCE_BASE_URL, params=params)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
                
            all_klines.extend(klines)
            
            # Update start time for next request
            current_start = klines[-1][0] + 1
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            # Wait longer if we hit a rate limit
            if "429" in str(e):
                logger.warning("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
            else:
                time.sleep(5)
    
    return all_klines

def process_kline_data(klines: List[List], symbol: str) -> pd.DataFrame:
    """
    Process raw kline data into a pandas DataFrame.
    
    Args:
        klines: List of kline data from Binance API
        symbol: Trading pair symbol
        
    Returns:
        Processed DataFrame
    """
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                      'quote_asset_volume', 'taker_buy_base_asset_volume', 
                      'taker_buy_quote_asset_volume']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])
    
    # Convert timestamps to datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # Add symbol information
    df['symbol'] = symbol
    df['symbol_simple'] = SYMBOL_MAP.get(symbol, symbol.lower().replace('usdt', ''))
    
    # Drop unnecessary columns
    df = df.drop('ignore', axis=1)
    
    return df

def fetch_and_save_data(start_date: str = START_DATE, 
                        end_date: str = END_DATE, 
                        trading_pairs: List[str] = TRADING_PAIRS) -> Dict[str, str]:
    """
    Fetch kline data for all trading pairs and save to CSV files.
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        trading_pairs: List of trading pairs to fetch data for
        
    Returns:
        Dictionary mapping trading pairs to file paths
    """
    ensure_data_directory()
    
    start_time = unix_timestamp(start_date)
    end_time = unix_timestamp(end_date)
    
    file_paths = {}
    
    for symbol in trading_pairs:
        logger.info(f"Fetching data for {symbol}...")
        
        klines = fetch_kline_data(symbol, start_time, end_time)
        
        if not klines:
            logger.warning(f"No data found for {symbol}")
            continue
            
        df = process_kline_data(klines, symbol)
        
        # Save to CSV
        file_name = f"{symbol.lower()}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
        file_path = os.path.join(DATA_DIR, file_name)
        df.to_csv(file_path, index=False)
        
        file_paths[symbol] = file_path
        logger.info(f"Saved {len(df)} records for {symbol} to {file_path}")
    
    return file_paths

def combine_data(file_paths: Dict[str, str] = None) -> pd.DataFrame:
    """
    Combine all trading pair data into one big DataFrame.
    
    Args:
        file_paths: Dictionary mapping trading pairs to file paths.
                   If None, will try to find CSV files in the data directory.
                   
    Returns:
        Combined DataFrame with all trading pairs
    """
    if file_paths is None:
        # Find all CSV files in the data directory
        ensure_data_directory()
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        file_paths = {f.split('_')[0].upper() + 'USDT': os.path.join(DATA_DIR, f) for f in csv_files}
    
    all_data = []
    
    for symbol, file_path in file_paths.items():
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # Ensure the DataFrame has the required columns
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            if 'symbol_simple' not in df.columns:
                df['symbol_simple'] = SYMBOL_MAP.get(symbol, symbol.lower().replace('usdt', ''))
                
            all_data.append(df)
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    if not all_data:
        logger.warning("No data found to combine")
        return pd.DataFrame()
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Convert timestamps to datetime if they're not already
    if combined_df['open_time'].dtype != 'datetime64[ns]':
        combined_df['open_time'] = pd.to_datetime(combined_df['open_time'])
    
    if combined_df['close_time'].dtype != 'datetime64[ns]':
        combined_df['close_time'] = pd.to_datetime(combined_df['close_time'])
    
    return combined_df

def get_binance_data(start_date: str = START_DATE, 
                    end_date: str = END_DATE, 
                    trading_pairs: List[str] = TRADING_PAIRS,
                    combine: bool = True) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Main function to fetch, save, and optionally combine Binance data.
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        trading_pairs: List of trading pairs to fetch data for
        combine: Whether to combine all data into one DataFrame
        
    Returns:
        If combine=True: Combined DataFrame with all trading pairs
        If combine=False: Dictionary mapping trading pairs to file paths
    """
    file_paths = fetch_and_save_data(start_date, end_date, trading_pairs)
    
    if combine:
        return combine_data(file_paths)
    else:
        return file_paths

