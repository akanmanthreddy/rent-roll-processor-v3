import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

# This list can be shared or moved to a config module later
PRIMARY_RECORD_COLS = [
    'unit', 'unit_type', 'sq_ft', 'resident_code', 'resident_name',
    'market_rent', 'resident_deposit', 'other_deposit', 'move_in',
    'lease_expiration', 'move_out', 'balance'
]

def clean_and_convert_to_numeric(series: pd.Series) -> pd.Series:
    """
    Cleans a series by removing currency symbols, commas, and handling parentheses
    for negative numbers, then converts it to a numeric type.
    """
    if series.empty:
        return series

    cleaned = (
        series.astype(str)
        .str.replace(r'[$,]', '', regex=True)
        .str.replace(r'\((.*?)\)', r'-\1', regex=True)
        .str.strip()
        .replace(['', 'nan', 'None', 'null'], np.nan)
    )
    return pd.to_numeric(cleaned, errors='coerce')

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes DataFrame memory usage by downcasting numeric types and converting
    low-cardinality object columns to 'category' type.
    """
    logger.info(f"Initial memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['object']).columns:
        if len(df[col].unique()) / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')

    logger.info(f"Optimized memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df

def process_rent_roll_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the rent roll data using a fully vectorized approach.
    """
    logger.info(f"Starting processing with {len(df)} rows")
    
    ffill_cols = [col for col in PRIMARY_RECORD_COLS if col in df.columns]
    if ffill_cols:
        df[ffill_cols] = df[ffill_cols].ffill()
        logger.info(f"Forward-filled columns: {ffill_cols}")

    if 'unit' not in df.columns or 'charge_code' not in df.columns:
        logger.error(f"Missing required columns. Available columns: {list(df.columns)}")
        raise ValueError("Required columns 'unit' or 'charge_code' not found in the data")
    
    # Log initial data state
    logger.info(f"Rows before cleaning: {len(df)}")
    
    df = df.dropna(subset=['unit', 'charge_code'])
    logger.info(f"Rows after dropping NaN in unit/charge_code: {len(df)}")
    
    # Convert unit column to string before using string operations
    df['unit'] = df['unit'].astype(str)
    df = df[~df['unit'].str.contains(r',,,,,,,,,,,,,', na=False, regex=False)]
    logger.info(f"Rows after removing separator rows: {len(df)}")
    
    # Convert charge_code to string before comparison
    df['charge_code'] = df['charge_code'].astype(str)
    df = df[df['charge_code'].str.lower() != 'total']
    logger.info(f"Rows after removing 'total' rows: {len(df)}")

    if 'amount' not in df.columns:
        logger.error(f"'amount' column not found. Available columns: {list(df.columns)}")
        raise ValueError("'amount' column not found in the data")
        
    df['amount'] = clean_and_convert_to_numeric(df['amount'])
    df = df.dropna(subset=['amount'])
    logger.info(f"Rows after cleaning amount column: {len(df)}")
    
    if df.empty:
        logger.warning("DataFrame is empty after cleaning")
        return pd.DataFrame()

    pivot_df = df.pivot_table(
        index='unit',
        columns='charge_code',
        values='amount',
        aggfunc='sum'
    ).reset_index()
    logger.info(f"Pivoted DataFrame has {len(pivot_df)} units")

    available_cols = [col for col in ffill_cols if col in df.columns and col != 'unit']
    if available_cols:
        unit_details = df.groupby('unit')[available_cols].last().reset_index()
        final_df = pd.merge(unit_details, pivot_df, on='unit', how='left')
    else:
        final_df = pivot_df

    final_df.columns = [str(col).lower().strip().replace(' ', '_') for col in final_df.columns]

    for col in final_df.select_dtypes(include=['object']).columns:
        if 'date' in col or 'move' in col or 'lease' in col:
            final_df[col] = pd.to_datetime(final_df[col], errors='coerce')

    final_df = optimize_memory_usage(final_df)
    
    logger.info(f"Final DataFrame has {len(final_df)} rows and {len(final_df.columns)} columns")
    return final_df
