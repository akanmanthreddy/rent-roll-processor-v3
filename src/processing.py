"""
Processing module for rent roll data.
Handles data cleaning, pivoting, and transformation.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Columns that define a primary unit record
PRIMARY_RECORD_COLS = [
    'unit', 'unit_type', 'sq_ft', 'resident_code', 'resident_name',
    'market_rent', 'resident_deposit', 'other_deposit', 'move_in',
    'lease_expiration', 'move_out', 'balance'
]


def clean_and_convert_to_numeric(series: pd.Series) -> pd.Series:
    """
    Cleans currency and numeric data.
    Handles $, commas, parentheses for negatives.
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
    Optimizes DataFrame memory by downcasting types.
    """
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Downcast numeric types
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer', errors='ignore')
        df[col] = pd.to_numeric(df[col], downcast='float', errors='ignore')

    # Convert low-cardinality strings to category
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')

    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory optimized: {initial_memory:.2f}MB → {final_memory:.2f}MB")
    
    return df


def process_rent_roll_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main processing function for rent roll data.
    Cleans, pivots, and transforms the data.
    """
    if df.empty:
        logger.warning("Received empty DataFrame")
        return pd.DataFrame()
    
    logger.info(f"Starting processing with {len(df)} rows")
    
    # Forward-fill primary record information (but NOT move_out date)
    # Move-out is tenant-specific and shouldn't be carried forward
    cols_to_ffill = [col for col in PRIMARY_RECORD_COLS if col in df.columns and col != 'move_out']
    if cols_to_ffill:
        df[cols_to_ffill] = df[cols_to_ffill].ffill()
        logger.info(f"Forward-filled {len(cols_to_ffill)} columns (excluding move_out)")
    
    # Keep track of all primary columns for later (including move_out)
    available_primary_cols = [col for col in PRIMARY_RECORD_COLS if col in df.columns]

    # Validate required columns
    if 'unit' not in df.columns or 'charge_code' not in df.columns:
        missing = []
        if 'unit' not in df.columns:
            missing.append('unit')
        if 'charge_code' not in df.columns:
            missing.append('charge_code')
        raise ValueError(f"Required columns missing: {', '.join(missing)}")
    
    # Clean data
    initial_rows = len(df)
    
    # Remove rows without unit or charge_code
    df = df.dropna(subset=['unit', 'charge_code'])
    logger.info(f"Removed {initial_rows - len(df)} rows with missing unit/charge_code")
    
    # Convert to string for string operations
    df['unit'] = df['unit'].astype(str)
    df['charge_code'] = df['charge_code'].astype(str)
    
    # Remove separator rows (visual formatting rows in some reports)
    df = df[~df['unit'].str.contains(',,,,,', na=False, regex=False)]
    
    # Remove total/summary rows
    df = df[df['charge_code'].str.lower() != 'total']
    df = df[~df['charge_code'].str.lower().str.contains('summary', na=False)]
    
    logger.info(f"Rows after cleaning: {len(df)}")

    # Process amount column
    if 'amount' not in df.columns:
        raise ValueError("'amount' column not found in the data")
    
    df['amount'] = clean_and_convert_to_numeric(df['amount'])
    df = df.dropna(subset=['amount'])
    
    if df.empty:
        logger.warning("No valid data after cleaning")
        return pd.DataFrame()

    # Pivot charge codes into columns
    pivot_df = df.pivot_table(
        index='unit',
        columns='charge_code',
        values='amount',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    logger.info(f"Created pivot table with {len(pivot_df)} units")

    # Get unit details (last known values for each unit)
    # Use available_primary_cols which includes move_out
    detail_cols = [col for col in available_primary_cols if col in df.columns and col != 'unit']
    if detail_cols:
        unit_details = df.groupby('unit')[detail_cols].last().reset_index()
        final_df = pd.merge(unit_details, pivot_df, on='unit', how='outer')
    else:
        final_df = pivot_df

    # Clean column names
    final_df.columns = [str(col).lower().strip().replace(' ', '_') for col in final_df.columns]

    # Convert date columns
    date_cols = [col for col in final_df.columns 
                 if any(keyword in col for keyword in ['date', 'move', 'lease', 'expir'])]
    for col in date_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_datetime(final_df[col], errors='coerce')

    # Optimize memory
    final_df = optimize_memory_usage(final_df)
    
    logger.info(f"Processing complete: {len(final_df)} units, {len(final_df.columns)} columns")
    return final_df
