"""
Processing module for rent roll data - Properly handles Yardi charge structure.
Preserves lease_expiration and correctly aggregates charges.
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
    Properly preserves lease_expiration and aggregates all charges per unit.
    """
    if df.empty:
        logger.warning("Received empty DataFrame")
        return pd.DataFrame()
    
    logger.info(f"Starting processing with {len(df)} rows")
    logger.info(f"Initial columns: {list(df.columns)[:20]}")
    
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
    
    # Remove invalid unit values
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
    
    # CRITICAL: Convert date columns BEFORE aggregation to preserve them
    date_cols = ['move_in', 'move_out', 'lease_expiration']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logger.info(f"Converted {col} to datetime")
    
    # Clean numeric columns before aggregation
    numeric_cols = ['market_rent', 'sq_ft', 'resident_deposit', 'other_deposit', 'balance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_and_convert_to_numeric(df[col])
    
    # Identify primary columns (unit info) vs charge columns
    primary_cols = ['unit', 'unit_type', 'sq_ft', 'resident_code', 'resident_name', 
                   'market_rent', 'resident_deposit', 'other_deposit', 
                   'move_in', 'lease_expiration', 'move_out', 'balance']
    
    available_primary_cols = [col for col in primary_cols if col in df.columns]
    logger.info(f"Available primary columns: {available_primary_cols}")
    
    # Strategy: 
    # 1. Get unit info from the first non-null occurrence of each field
    # 2. Process charges separately and pivot
    # 3. Merge them together
    
    # Step 1: Aggregate unit information (taking first non-null value for each field)
    agg_dict = {}
    for col in available_primary_cols:
        if col != 'unit':
            if col in ['balance', 'resident_deposit', 'other_deposit']:
                # For these, we might want to sum or take max
                agg_dict[col] = 'first'  # or 'max' or 'sum' depending on business logic
            else:
                # For most fields, take the first non-null value
                agg_dict[col] = 'first'
    
    # Group by unit and aggregate
    unit_info = df.groupby('unit', as_index=False).agg(agg_dict)
    
    logger.info(f"After aggregation, have {len(unit_info)} unique units")
    logger.info(f"Unit info columns: {list(unit_info.columns)}")
    
    # Check if lease_expiration survived
    if 'lease_expiration' in unit_info.columns:
        non_null_lease = unit_info['lease_expiration'].notna().sum()
        logger.info(f"Lease expiration data: {non_null_lease} units have lease expiration dates")
    else:
        logger.warning("lease_expiration column not found after aggregation!")
    
    # Step 2: Process charges if they exist
    if 'charge_code' in df.columns and 'amount' in df.columns:
        logger.info("Processing charge codes...")
        
        # Clean charge codes and amounts
        df['charge_code'] = df['charge_code'].astype(str).str.strip().str.lower()
        df['amount'] = clean_and_convert_to_numeric(df['amount'])
        
        # Filter for valid charges
        charge_df = df[
            df['charge_code'].notna() & 
            (df['charge_code'] != '') & 
            (df['charge_code'] != 'nan') &
            (df['charge_code'] != 'null') &
            df['amount'].notna() &
            (df['amount'] != 0)  # Only non-zero amounts
        ].copy()
        
        # Remove total/summary rows
        charge_df = charge_df[~charge_df['charge_code'].str.contains('total|summary', na=False)]
        
        if not charge_df.empty:
            logger.info(f"Found {len(charge_df)} rows with valid charges")
            
            # Show distribution of charge codes
            charge_distribution = charge_df['charge_code'].value_counts().head(10)
            logger.info(f"Top charge codes:\n{charge_distribution}")
            
            # Pivot charges
            try:
                pivot_df = charge_df.pivot_table(
                    index='unit',
                    columns='charge_code',
                    values='amount',
                    aggfunc='sum',  # Sum all charges of the same type for each unit
                    fill_value=0
                ).reset_index()
                
                logger.info(f"Pivot table created: {len(pivot_df)} units, {len(pivot_df.columns)-1} charge types")
                
                # Clean column names (charge codes)
                pivot_df.columns = [str(col).strip().lower().replace(' ', '_') for col in pivot_df.columns]
                
                # Merge with unit info
                final_df = pd.merge(unit_info, pivot_df, on='unit', how='left')
                
                # Fill NaN in charge columns with 0
                charge_columns = [col for col in pivot_df.columns if col != 'unit']
                for col in charge_columns:
                    if col in final_df.columns:
                        final_df[col] = final_df[col].fillna(0)
                
                logger.info(f"After merge: {len(final_df)} units with charges")
                
            except Exception as e:
                logger.error(f"Error creating pivot table: {e}")
                final_df = unit_info.copy()
        else:
            logger.warning("No valid charges found")
            final_df = unit_info.copy()
    else:
        logger.info("No charge_code column found")
        final_df = unit_info.copy()
    
    # Clean column names
    final_df.columns = [str(col).lower().strip().replace(' ', '_') for col in final_df.columns]
    
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
    else:
        final_df['occupancy_status'] = 'Unknown'
    
    # Create actual_rent column
    if 'rent' in final_df.columns:
        final_df['actual_rent'] = final_df['rent']
        logger.info(f"Rent column found. Non-zero rents: {(final_df['rent'] > 0).sum()}")
    else:
        # Try to find rent in charge columns
        rent_cols = [col for col in final_df.columns if 'rent' in col.lower() and col != 'petrent']
        if rent_cols:
            final_df['actual_rent'] = final_df[rent_cols[0]]
            logger.info(f"Using '{rent_cols[0]}' as actual_rent")
    
    # Final check on critical columns
    logger.info(f"Final columns ({len(final_df.columns)} total): {list(final_df.columns)[:30]}")
    
    critical_cols = ['unit', 'unit_type', 'sq_ft', 'resident_name', 'lease_expiration', 'move_in', 'rent']
    for col in critical_cols:
        if col in final_df.columns:
            non_null = final_df[col].notna().sum()
            logger.info(f"{col}: {non_null}/{len(final_df)} have data")
        else:
            logger.warning(f"{col}: NOT FOUND in final dataframe")
    
    # Optimize memory
    final_df = optimize_memory_usage(final_df)
    
    # Final validation
    initial_count = len(final_df)
    final_df = final_df[~final_df['unit'].isin(['nan', 'NaN', 'NAN', 'null', 'NULL', 'None', 'NONE', '', ' '])]
    if initial_count != len(final_df):
        logger.warning(f"Removed {initial_count - len(final_df)} invalid units in final validation")
    
    logger.info(f"Processing complete: {len(final_df)} units, {len(final_df.columns)} columns")
    
    if 'occupancy_status' in final_df.columns:
        occupied = (final_df['occupancy_status'] == 'Occupied').sum()
        vacant = (final_df['occupancy_status'] == 'Vacant').sum()
        logger.info(f"Occupancy: {occupied} occupied, {vacant} vacant")
    
    return final_df
