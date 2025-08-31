"""
Processing module for rent roll data - Final fix for Yardi format.
Properly filters out 'nan' units and handles charge pivoting correctly.
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
    Main processing function for rent roll data - Fixed for Yardi format.
    Properly handles the unit/charge structure without duplicating units.
    Filters out invalid units including 'nan'.
    """
    if df.empty:
        logger.warning("Received empty DataFrame")
        return pd.DataFrame()
    
    logger.info(f"Starting processing with {len(df)} rows")
    logger.info(f"Initial columns: {list(df.columns)[:15]}")
    
    # Validate required columns
    if 'unit' not in df.columns:
        # Try to find a unit column
        possible_unit_cols = [col for col in df.columns if 'unit' in col.lower()]
        if possible_unit_cols:
            logger.info(f"'unit' column not found, using '{possible_unit_cols[0]}'")
            df = df.rename(columns={possible_unit_cols[0]: 'unit'})
        else:
            raise ValueError("Required column 'unit' not found in the data")
    
    # Convert unit to string and clean
    df['unit'] = df['unit'].astype(str).str.strip()
    
    # CRITICAL FIX: Remove invalid unit values including 'nan', empty strings, and actual NaN
    invalid_units = ['nan', 'NaN', 'NAN', 'null', 'NULL', 'None', 'NONE', '', ' ']
    df = df[~df['unit'].isin(invalid_units)]
    df = df[df['unit'].notna()]
    
    logger.info(f"After removing invalid units, {len(df)} rows remain")
    
    # Remove separator/total rows
    for pattern in FILTER_PATTERNS:
        if ',' in pattern:  # Separator pattern
            df = df[~df['unit'].str.contains(pattern, na=False, regex=False)]
        else:  # Total/summary patterns
            df = df[~df['unit'].str.lower().str.contains(pattern, na=False)]
    
    logger.info(f"After filtering patterns, {len(df)} rows remain")
    
    # Identify which columns we have
    available_primary_cols = [col for col in PRIMARY_RECORD_COLS if col in df.columns]
    logger.info(f"Available primary columns: {available_primary_cols}")
    
    # Strategy for Yardi: Group by unit first, then handle charges
    # Step 1: Get unique unit information (first occurrence of each unit)
    unit_info_df = df.drop_duplicates(subset=['unit'], keep='first').copy()
    
    # Keep only the primary columns for unit info
    unit_cols_to_keep = ['unit'] + [col for col in available_primary_cols if col in unit_info_df.columns and col != 'unit']
    unit_info = unit_info_df[unit_cols_to_keep].copy()
    
    logger.info(f"Found {len(unit_info)} unique units")
    
    # Step 2: Process charge information if it exists
    if 'charge_code' in df.columns and 'amount' in df.columns:
        logger.info("Processing charge codes...")
        
        # Clean charge codes and amounts
        df['charge_code'] = df['charge_code'].astype(str).str.strip()
        df['amount'] = clean_and_convert_to_numeric(df['amount'])
        
        # Filter for valid charges only
        charge_df = df[df['charge_code'].notna() & 
                       (df['charge_code'] != '') & 
                       (df['charge_code'] != 'nan') &
                       df['amount'].notna()].copy()
        
        # Remove any charge codes that look like totals
        charge_df = charge_df[~charge_df['charge_code'].str.lower().str.contains('total|summary', na=False)]
        
        if not charge_df.empty:
            logger.info(f"Found {len(charge_df)} rows with valid charges")
            
            # Pivot charges to columns
            try:
                pivot_df = charge_df.pivot_table(
                    index='unit',
                    columns='charge_code',
                    values='amount',
                    aggfunc='sum',
                    fill_value=0
                ).reset_index()
                
                logger.info(f"Created pivot table with {len(pivot_df)} units and {len(pivot_df.columns)-1} charge types")
                
                # Merge with unit info
                final_df = pd.merge(unit_info, pivot_df, on='unit', how='left')
                
            except Exception as e:
                logger.error(f"Error creating pivot table: {e}")
                final_df = unit_info.copy()
        else:
            logger.warning("No valid charges found after filtering")
            final_df = unit_info.copy()
    else:
        logger.info("No charge_code column found, using unit information only")
        final_df = unit_info.copy()
    
    # Fill NaN values in charge columns with 0
    charge_columns = [col for col in final_df.columns 
                     if col not in available_primary_cols and col != 'unit']
    for col in charge_columns:
        final_df[col] = final_df[col].fillna(0)
    
    # Clean column names
    final_df.columns = [str(col).lower().strip().replace(' ', '_') for col in final_df.columns]
    
    # Process standard columns
    # Clean numeric columns
    numeric_cols = ['market_rent', 'sq_ft', 'resident_deposit', 'other_deposit', 'balance']
    for col in numeric_cols:
        if col in final_df.columns:
            final_df[col] = clean_and_convert_to_numeric(final_df[col])
    
    # Convert date columns
    date_cols = ['move_in', 'move_out', 'lease_expiration']
    for col in date_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_datetime(final_df[col], errors='coerce')
    
    # Add occupancy status
    if 'resident_name' in final_df.columns:
        # Check for VACANT as resident name or empty/null values
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
    else:
        # If no resident info, check if there's rent
        if 'rent' in final_df.columns:
            final_df['occupancy_status'] = final_df['rent'].apply(
                lambda x: 'Occupied' if pd.notna(x) and x > 0 else 'Vacant'
            )
        else:
            final_df['occupancy_status'] = 'Unknown'
    
    # Create actual_rent column if we have charge data
    if 'rent' in final_df.columns:
        final_df['actual_rent'] = final_df['rent']
    elif charge_columns:
        # Try to identify rent from charge columns
        rent_columns = [col for col in charge_columns if 'rent' in col.lower() and 'credit' not in col.lower()]
        if rent_columns:
            # Use the first rent-related column
            final_df['actual_rent'] = final_df[rent_columns[0]]
            logger.info(f"Using '{rent_columns[0]}' as actual_rent")
    
    # Optimize memory
    final_df = optimize_memory_usage(final_df)
    
    # Final validation - remove any remaining invalid units that might have slipped through
    initial_count = len(final_df)
    final_df = final_df[~final_df['unit'].isin(['nan', 'NaN', 'NAN', 'null', 'NULL', 'None', 'NONE', '', ' '])]
    if initial_count != len(final_df):
        logger.warning(f"Removed {initial_count - len(final_df)} additional invalid units in final validation")
    
    # Final logging
    logger.info(f"Processing complete: {len(final_df)} units, {len(final_df.columns)} columns")
    
    if 'occupancy_status' in final_df.columns:
        occupied = (final_df['occupancy_status'] == 'Occupied').sum()
        vacant = (final_df['occupancy_status'] == 'Vacant').sum()
        logger.info(f"Occupancy: {occupied} occupied, {vacant} vacant")
    
    # Log charge columns found
    if charge_columns:
        logger.info(f"Charge types found: {charge_columns[:10]}")  # First 10 charge types
    
    return final_df
