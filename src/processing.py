"""
Processing module for rent roll data.
Handles data cleaning, pivoting, and transformation.
"""

import pandas as pd
import numpy as np
import logging
from src.config import PRIMARY_RECORD_COLS, NO_FORWARD_FILL_COLS, FILTER_PATTERNS

logger = logging.getLogger(__name__)


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
    
    # Forward-fill primary record information (but NOT columns in NO_FORWARD_FILL_COLS)
    cols_to_ffill = [col for col in PRIMARY_RECORD_COLS if col in df.columns and col not in NO_FORWARD_FILL_COLS]
    
    # Create a group identifier based on unit and resident to prevent forward-fill across different tenants
    if 'resident_name' in df.columns:
        # Create groups where forward-fill should stop at boundaries
        df['fill_group'] = (df['unit'] + '_' + df['resident_name'].fillna('VACANT')).ne(
            (df['unit'] + '_' + df['resident_name'].fillna('VACANT')).shift()).cumsum()
        
        # Forward-fill only within each group
        if cols_to_ffill:
            df[cols_to_ffill] = df.groupby('fill_group')[cols_to_ffill].ffill()
            logger.info(f"Forward-filled {len(cols_to_ffill)} columns within tenant groups")
        
        # Drop the temporary column
        df = df.drop('fill_group', axis=1)
    else:
        # If no resident_name column, use original logic but exclude date columns for safety
        date_cols = ['move_in', 'lease_expiration']
        cols_to_ffill = [col for col in cols_to_ffill if col not in date_cols]
        if cols_to_ffill:
            df[cols_to_ffill] = df[cols_to_ffill].ffill()
            logger.info(f"Forward-filled {len(cols_to_ffill)} columns (excluding dates)")
    
    # Keep track of all primary columns for later (including those in NO_FORWARD_FILL_COLS)
    available_primary_cols = [col for col in PRIMARY_RECORD_COLS if col in df.columns]

    # Validate required columns
    if 'unit' not in df.columns:
        raise ValueError("Required column 'unit' not found in the data")
    
    # Clean data
    initial_rows = len(df)
    
    # Convert unit to string for string operations
    df['unit'] = df['unit'].astype(str)
    
    # Remove separator rows using patterns from config
    for pattern in FILTER_PATTERNS:
        if ',' in pattern:  # Separator pattern
            df = df[~df['unit'].str.contains(pattern, na=False, regex=False)]
    
    # IMPORTANT: Capture ALL units first (including vacant ones)
    all_units = df.drop_duplicates(subset=['unit'], keep='first')
    unit_info = all_units[['unit'] + [col for col in available_primary_cols if col in all_units.columns and col != 'unit']]
    logger.info(f"Found {len(unit_info)} unique units (including vacant)")
    
    # Now filter for rows with charges (for the pivot table)
    charge_rows = df.copy()
    
    # Check if charge_code exists
    if 'charge_code' in charge_rows.columns:
        charge_rows['charge_code'] = charge_rows['charge_code'].astype(str)
        # Remove rows without charge codes
        charge_rows = charge_rows.dropna(subset=['charge_code'])
        # Remove total/summary rows using patterns from config
        for pattern in FILTER_PATTERNS:
            if pattern in ['total', 'summary']:
                charge_rows = charge_rows[~charge_rows['charge_code'].str.lower().str.contains(pattern, na=False)]
        
        logger.info(f"Found {len(charge_rows)} rows with charges")
        
        # Process amount column if it exists
        if 'amount' in charge_rows.columns:
            charge_rows['amount'] = clean_and_convert_to_numeric(charge_rows['amount'])
            charge_rows = charge_rows.dropna(subset=['amount'])
            
            # Pivot charge codes into columns
            if not charge_rows.empty:
                pivot_df = charge_rows.pivot_table(
                    index='unit',
                    columns='charge_code',
                    values='amount',
                    aggfunc='sum',
                    fill_value=0
                ).reset_index()
                logger.info(f"Created pivot table with {len(pivot_df)} units with charges")
            else:
                pivot_df = pd.DataFrame({'unit': unit_info['unit'].unique()})
                logger.warning("No valid charges found")
        else:
            pivot_df = pd.DataFrame({'unit': unit_info['unit'].unique()})
    else:
        pivot_df = pd.DataFrame({'unit': unit_info['unit'].unique()})
        logger.warning("No charge_code column found")
    
    # Merge ALL unit information with the pivot table
    final_df = pd.merge(unit_info, pivot_df, on='unit', how='left')
    
    # For units not in pivot (vacant units), fill charge columns with 0
    charge_columns = [col for col in final_df.columns if col not in available_primary_cols and col != 'unit']
    for col in charge_columns:
        final_df[col] = final_df[col].fillna(0)
    
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
    
    # Add occupancy status for clarity
    if 'resident_name' in final_df.columns:
        final_df['occupancy_status'] = final_df['resident_name'].apply(
            lambda x: 'Vacant' if pd.isna(x) or str(x).strip() == '' else 'Occupied'
        )
    
    logger.info(f"Processing complete: {len(final_df)} units, {len(final_df.columns)} columns")
    if 'occupancy_status' in final_df.columns:
        vacant_count = (final_df['occupancy_status'] == 'Vacant').sum()
        occupied_count = (final_df['occupancy_status'] == 'Occupied').sum()
        logger.info(f"Units: {occupied_count} occupied, {vacant_count} vacant")
    
    return final_df
