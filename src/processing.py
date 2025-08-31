"""
Processing module for rent roll data - Fixed charge aggregation for Yardi.
Based on working version approach.
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
    Fixed to properly handle multi-row structure where each unit has multiple charge rows.
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
    
    # Convert unit to string first
    df['unit'] = df['unit'].astype(str).str.strip()
    
    # CRITICAL: Forward-fill unit column FIRST before any filtering
    # In Yardi format, unit appears only on first row, subsequent charge rows are empty
    df['unit'] = df['unit'].replace(['nan', 'NaN', 'NAN', 'null', 'NULL', 'None', 'NONE', ''], np.nan)
    df['unit'] = df['unit'].ffill()
    
    # Now remove rows with invalid units (after forward-fill)
    df = df[df['unit'].notna()]
    df = df[~df['unit'].str.contains(',,,,,', na=False, regex=False)]  # Remove separator rows
    
    # Remove total/summary rows
    for pattern in FILTER_PATTERNS:
        if ',' in pattern:
            df = df[~df['unit'].str.contains(pattern, na=False, regex=False)]
        else:
            df = df[~df['unit'].str.lower().str.contains(pattern, na=False)]
    
    logger.info(f"After cleaning, {len(df)} rows remain")
    
    # Now forward-fill other unit information columns within each unit
    # BUT NOT charge_code or amount
    cols_to_ffill = ['unit_type', 'sq_ft', 'resident_code', 'resident_name',
                     'market_rent', 'resident_deposit', 'other_deposit', 
                     'move_in', 'lease_expiration', 'balance']
    
    # Only forward-fill columns that exist
    cols_to_ffill = [col for col in cols_to_ffill if col in df.columns]
    
    # Create fill groups based on unit + resident to avoid forward-filling across different tenants
    if 'resident_name' in df.columns:
        df['fill_group'] = (df['unit'] + '_' + df['resident_name'].fillna('VACANT')).ne(
            (df['unit'] + '_' + df['resident_name'].fillna('VACANT')).shift()).cumsum()
        
        # Forward-fill within each group
        if cols_to_ffill:
            df[cols_to_ffill] = df.groupby('fill_group')[cols_to_ffill].ffill()
        
        df = df.drop('fill_group', axis=1)
    else:
        # Simple forward-fill if no resident_name
        if cols_to_ffill:
            df[cols_to_ffill] = df.groupby('unit')[cols_to_ffill].ffill()
    
    logger.info("Forward-filled unit information within groups")
    
    # Don't forward-fill move_out (as per config)
    if 'move_out' in df.columns:
        # move_out should only apply to specific records, not forward-filled
        pass
    
    # Convert date columns
    date_cols = ['move_in', 'move_out', 'lease_expiration']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Clean numeric columns (but preserve amount values!)
    numeric_cols = ['market_rent', 'sq_ft', 'resident_deposit', 'other_deposit', 'balance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_and_convert_to_numeric(df[col])
    
    # Clean amount column separately and carefully
    if 'amount' in df.columns:
        df['amount'] = clean_and_convert_to_numeric(df['amount'])
        logger.info(f"Amount column - Non-null: {df['amount'].notna().sum()}, Non-zero: {(df['amount'] != 0).sum()}")
    
    # Get ALL unique units first (including vacant)
    available_primary_cols = [col for col in PRIMARY_RECORD_COLS if col in df.columns]
    all_units = df.drop_duplicates(subset=['unit'], keep='first')
    unit_info = all_units[available_primary_cols].copy()
    logger.info(f"Found {len(unit_info)} unique units (including vacant)")
    
    # Now process charges if they exist
    if 'charge_code' in df.columns and 'amount' in df.columns:
        logger.info("Processing charges...")
        
        # Clean charge codes
        df['charge_code'] = df['charge_code'].astype(str).str.strip().str.lower()
        
        # Filter for rows with valid charge codes
        charge_rows = df[df['charge_code'].notna()].copy()
        charge_rows = charge_rows[~charge_rows['charge_code'].isin(['nan', 'null', 'none', ''])]
        
        # Remove total/summary rows
        charge_rows = charge_rows[~charge_rows['charge_code'].str.contains('total|summary', na=False)]
        
        logger.info(f"Found {len(charge_rows)} rows with charges")
        
        # Only keep rows with non-null amounts for pivoting
        charge_rows = charge_rows[charge_rows['amount'].notna()]
        
        if not charge_rows.empty:
            # Log charge distribution
            charge_summary = charge_rows.groupby('charge_code')['amount'].agg(['count', 'sum', 'mean'])
            logger.info(f"Charge distribution:\n{charge_summary}")
            
            # Pivot charge codes into columns
            pivot_df = charge_rows.pivot_table(
                index='unit',
                columns='charge_code',
                values='amount',
                aggfunc='sum',
                fill_value=0
            ).reset_index()
            
            logger.info(f"Created pivot table: {len(pivot_df)} units, {len(pivot_df.columns)-1} charge types")
            logger.info(f"Charge columns: {list(pivot_df.columns)}")
            
            # Check specific charge columns
            if 'rent' in pivot_df.columns:
                rent_stats = pivot_df['rent'].describe()
                logger.info(f"Rent statistics:\n{rent_stats}")
                logger.info(f"Units with rent > 0: {(pivot_df['rent'] > 0).sum()}")
        else:
            logger.warning("No valid charges found to pivot")
            pivot_df = pd.DataFrame({'unit': unit_info['unit'].unique()})
    else:
        logger.warning("No charge_code or amount columns found")
        pivot_df = pd.DataFrame({'unit': unit_info['unit'].unique()})
    
    # Merge ALL unit information with the pivot table
    # This ensures vacant units are included even if they have no charges
    final_df = pd.merge(unit_info, pivot_df, on='unit', how='left')
    
    # Fill NaN values in charge columns with 0
    charge_columns = [col for col in final_df.columns if col not in available_primary_cols]
    for col in charge_columns:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)
    
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
        if 'occupancy_status' in final_df.columns:
            occupied_with_rent = ((final_df['occupancy_status'] == 'Occupied') & (final_df['rent'] > 0)).sum()
            occupied_total = (final_df['occupancy_status'] == 'Occupied').sum()
            logger.info(f"Rent check: {occupied_with_rent}/{occupied_total} occupied units have rent > 0")
    else:
        final_df['actual_rent'] = 0
    
    # Final logging
    logger.info(f"Processing complete: {len(final_df)} units, {len(final_df.columns)} columns")
    
    if 'occupancy_status' in final_df.columns:
        vacant_count = (final_df['occupancy_status'] == 'Vacant').sum()
        occupied_count = (final_df['occupancy_status'] == 'Occupied').sum()
        logger.info(f"Units: {occupied_count} occupied, {vacant_count} vacant")
    
    # Log charge column statistics
    charge_cols = [c for c in final_df.columns if c not in available_primary_cols + ['occupancy_status', 'actual_rent']]
    if charge_cols:
        logger.info(f"Charge columns in output: {charge_cols}")
        for col in charge_cols[:10]:  # Log first 10 charge types
            non_zero = (final_df[col] != 0).sum()
            if non_zero > 0:
                logger.info(f"  {col}: {non_zero} non-zero values, total=${final_df[col].sum():,.2f}")
    
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
