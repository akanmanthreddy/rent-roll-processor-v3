"""
Processing module for rent roll data - Fixed charge aggregation for Yardi.
Properly handles multiple rows per unit with different charge codes.
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


def process_rent_roll_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main processing function for Yardi rent roll data.
    Properly handles multi-row structure where each unit has multiple charge rows.
    """
    if df.empty:
        logger.warning("Received empty DataFrame")
        return pd.DataFrame()
    
    logger.info(f"Starting processing with {len(df)} rows")
    logger.info(f"Columns available: {list(df.columns)}")
    
    # Ensure we have a unit column
    if 'unit' not in df.columns:
        possible_unit_cols = [col for col in df.columns if 'unit' in col.lower()]
        if possible_unit_cols:
            df = df.rename(columns={possible_unit_cols[0]: 'unit'})
        else:
            raise ValueError("Required column 'unit' not found")
    
    # Clean unit column
    df['unit'] = df['unit'].astype(str).str.strip()
    
    # Remove invalid units
    invalid_units = ['nan', 'NaN', 'NAN', 'null', 'NULL', 'None', 'NONE', '', ' ']
    df = df[~df['unit'].isin(invalid_units)]
    df = df[df['unit'].notna()]
    
    # Remove separator/total rows
    for pattern in FILTER_PATTERNS:
        if ',' in pattern:
            df = df[~df['unit'].str.contains(pattern, na=False, regex=False)]
        else:
            df = df[~df['unit'].str.lower().str.contains(pattern, na=False)]
    
    logger.info(f"After cleaning, {len(df)} rows remain")
    
    # Forward fill unit information within each unit group
    # This is critical for Yardi - unit info is in first row, charges in subsequent rows
    unit_info_cols = ['unit', 'unit_type', 'sq_ft', 'resident_code', 'resident_name',
                      'market_rent', 'resident_deposit', 'other_deposit', 
                      'move_in', 'lease_expiration', 'move_out', 'balance']
    
    # Forward fill these columns within each unit
    for col in unit_info_cols:
        if col in df.columns:
            df[col] = df.groupby('unit')[col].ffill()
    
    logger.info("Forward-filled unit information within groups")
    
    # Convert date columns
    date_cols = ['move_in', 'move_out', 'lease_expiration']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Clean numeric columns
    numeric_cols = ['market_rent', 'sq_ft', 'resident_deposit', 'other_deposit', 'balance', 'amount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_and_convert_to_numeric(df[col])
    
    # Now process charges
    if 'charge_code' in df.columns and 'amount' in df.columns:
        logger.info("Processing charges...")
        
        # Clean charge codes
        df['charge_code'] = df['charge_code'].astype(str).str.strip().str.lower()
        
        # Remove invalid charge codes
        df.loc[df['charge_code'].isin(['nan', 'null', 'none', '']), 'charge_code'] = pd.NA
        
        # Log charge distribution before filtering
        charge_summary = df.groupby('charge_code')['amount'].agg(['count', 'sum'])
        logger.info(f"Charge summary before filtering:\n{charge_summary.head(20)}")
        
        # Split into unit info and charges
        # Unit info: rows where charge_code is null/empty (first row per unit)
        unit_info_df = df[df['charge_code'].isna()].copy()
        
        # Charges: rows with valid charge codes and amounts
        charge_df = df[
            df['charge_code'].notna() & 
            (df['amount'].notna()) & 
            (df['amount'] != 0)
        ].copy()
        
        logger.info(f"Found {len(unit_info_df)} unit info rows")
        logger.info(f"Found {len(charge_df)} charge rows")
        
        # Get unique unit information (one row per unit)
        # Use the first row for each unit (which has the resident info)
        unit_cols = ['unit', 'unit_type', 'sq_ft', 'resident_code', 'resident_name',
                     'market_rent', 'resident_deposit', 'other_deposit',
                     'move_in', 'lease_expiration', 'move_out', 'balance']
        
        available_unit_cols = [col for col in unit_cols if col in df.columns]
        
        # Get unique units with their info
        units_df = df[available_unit_cols].drop_duplicates(subset=['unit'], keep='first')
        logger.info(f"Found {len(units_df)} unique units")
        
        # Pivot charges if we have any
        if not charge_df.empty:
            # Remove any charge codes that look like totals
            charge_df = charge_df[~charge_df['charge_code'].str.contains('total|summary', na=False)]
            
            # Pivot charges
            pivot_df = charge_df.pivot_table(
                index='unit',
                columns='charge_code',
                values='amount',
                aggfunc='sum',
                fill_value=0
            ).reset_index()
            
            logger.info(f"Pivoted charges: {len(pivot_df)} units, {len(pivot_df.columns)-1} charge types")
            logger.info(f"Charge columns: {list(pivot_df.columns)[:20]}")
            
            # Check if 'rent' column exists in pivot
            if 'rent' in pivot_df.columns:
                rent_summary = pivot_df['rent'].describe()
                logger.info(f"Rent summary:\n{rent_summary}")
                logger.info(f"Units with rent > 0: {(pivot_df['rent'] > 0).sum()}")
            
            # Merge unit info with charges
            final_df = pd.merge(units_df, pivot_df, on='unit', how='left')
            
            # Fill NaN values in charge columns with 0
            charge_columns = [col for col in pivot_df.columns if col != 'unit']
            for col in charge_columns:
                final_df[col] = final_df[col].fillna(0)
        else:
            logger.warning("No charges found to pivot")
            final_df = units_df.copy()
    else:
        logger.warning("No charge_code or amount column found")
        # Just get unique units
        unit_cols = [col for col in df.columns if col != 'charge_code' and col != 'amount']
        final_df = df[unit_cols].drop_duplicates(subset=['unit'], keep='first')
    
    # Add occupancy status
    if 'resident_name' in final_df.columns:
        final_df['occupancy_status'] = final_df['resident_name'].apply(
            lambda x: 'Vacant' if (pd.isna(x) or 
                                 str(x).strip() == '' or 
                                 str(x).upper() in ['VACANT', 'NAN', 'NULL', 'NONE'])
            else 'Occupied'
        )
    elif 'resident_code' in final_df.columns:
        final_df['occupancy_status'] = final_df['resident_code'].apply(
            lambda x: 'Vacant' if (pd.isna(x) or 
                                 str(x).strip() == '' or 
                                 str(x).upper() in ['VACANT', 'NAN', 'NULL', 'NONE'])
            else 'Occupied'
        )
    
    # Create actual_rent column
    if 'rent' in final_df.columns:
        final_df['actual_rent'] = final_df['rent']
        occupied_with_rent = ((final_df['occupancy_status'] == 'Occupied') & (final_df['rent'] > 0)).sum()
        occupied_total = (final_df['occupancy_status'] == 'Occupied').sum()
        logger.info(f"Rent check: {occupied_with_rent}/{occupied_total} occupied units have rent > 0")
    
    # Log final statistics
    logger.info(f"Final result: {len(final_df)} units, {len(final_df.columns)} columns")
    
    if 'lease_expiration' in final_df.columns:
        lease_count = final_df['lease_expiration'].notna().sum()
        logger.info(f"Lease expiration dates: {lease_count}/{len(final_df)} units")
    
    return final_df


def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes DataFrame memory by downcasting types.
    """
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer', errors='ignore')
        df[col] = pd.to_numeric(df[col], downcast='float', errors='ignore')

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')

    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory optimized: {initial_memory:.2f}MB → {final_memory:.2f}MB")
    
    return df
