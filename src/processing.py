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
    
    # CRITICAL FIX: Forward fill unit information INCLUDING 'unit' column
    # But NOT 'amount' or 'charge_code' to preserve the actual charge values
    unit_info_cols = ['unit', 'unit_type', 'sq_ft', 'resident_code', 'resident_name',
                      'market_rent', 'resident_deposit', 'other_deposit', 
                      'move_in', 'lease_expiration', 'move_out', 'balance']
    
    # First, we need to identify unit groups before forward-filling 'unit'
    # Create a unit group identifier based on where unit values appear
    df['unit_group'] = df['unit'].notna().cumsum()
    
    # Now forward fill within each unit group
    for col in unit_info_cols:
        if col in df.columns:
            df[col] = df.groupby('unit_group')[col].ffill()
    
    # Drop the temporary unit_group column
    df = df.drop('unit_group', axis=1)
    
    logger.info("Forward-filled unit information within groups (preserving charge amounts)")
    
    # Log sample of data to verify amounts are preserved
    if 'charge_code' in df.columns and 'amount' in df.columns:
        sample_charges = df[df['charge_code'].notna()][['unit', 'charge_code', 'amount']].head(20)
        logger.info(f"Sample charge data after forward-fill:\n{sample_charges}")
    
    # Convert date columns
    date_cols = ['move_in', 'move_out', 'lease_expiration']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Clean numeric columns EXCEPT charge_code and amount initially
    # We'll handle amount specially to preserve the values
    numeric_cols = ['market_rent', 'sq_ft', 'resident_deposit', 'other_deposit', 'balance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_and_convert_to_numeric(df[col])
    
    # Now handle amount column carefully
    if 'amount' in df.columns:
        df['amount'] = clean_and_convert_to_numeric(df['amount'])
        logger.info(f"Amount column stats - Non-null: {df['amount'].notna().sum()}, Non-zero: {(df['amount'] != 0).sum()}")
    
    # Now process charges
    if 'charge_code' in df.columns and 'amount' in df.columns:
        logger.info("Processing charges...")
        
        # Clean charge codes
        df['charge_code'] = df['charge_code'].astype(str).str.strip().str.lower()
        
        # Replace 'nan' strings with actual NaN
        df.loc[df['charge_code'].isin(['nan', 'null', 'none', '']), 'charge_code'] = pd.NA
        
        # Log charge distribution before filtering
        charge_df_analysis = df[df['charge_code'].notna()].copy()
        if not charge_df_analysis.empty:
            charge_summary = charge_df_analysis.groupby('charge_code')['amount'].agg(['count', 'sum', 'mean'])
            logger.info(f"Charge summary before pivoting:\n{charge_summary}")
        
        # Get unique unit information (one row per unit)
        unit_cols = ['unit', 'unit_type', 'sq_ft', 'resident_code', 'resident_name',
                     'market_rent', 'resident_deposit', 'other_deposit',
                     'move_in', 'lease_expiration', 'move_out', 'balance']
        
        available_unit_cols = [col for col in unit_cols if col in df.columns]
        
        # Get unique units with their info (from first row of each unit)
        units_df = df[available_unit_cols].drop_duplicates(subset=['unit'], keep='first')
        logger.info(f"Found {len(units_df)} unique units")
        
        # Process charges - only include rows with valid charge codes AND non-zero amounts
        # Based on your insight: if a charge appears, it should have a value
        charge_df = df[
            df['charge_code'].notna() & 
            df['amount'].notna() & 
            (df['amount'] != 0)
        ].copy()
        
        logger.info(f"Found {len(charge_df)} charge rows with non-zero amounts")
        
        # Also log any charge rows that have 0 amounts (these might be data issues)
        zero_amount_charges = df[
            df['charge_code'].notna() & 
            ((df['amount'] == 0) | df['amount'].isna())
        ]
        if not zero_amount_charges.empty:
            logger.warning(f"Found {len(zero_amount_charges)} charge rows with 0 or null amounts - these are being excluded")
            logger.warning(f"Sample of excluded charges: {zero_amount_charges[['unit', 'charge_code', 'amount']].head(10)}")
        
        # Pivot charges if we have any
        if not charge_df.empty:
            # Remove any charge codes that look like totals
            charge_df = charge_df[~charge_df['charge_code'].str.contains('total|summary', na=False)]
            
            # Pivot charges - sum amounts for duplicate charge codes per unit
            pivot_df = charge_df.pivot_table(
                index='unit',
                columns='charge_code',
                values='amount',
                aggfunc='sum',
                fill_value=0
            ).reset_index()
            
            logger.info(f"Pivoted charges: {len(pivot_df)} units, {len(pivot_df.columns)-1} charge types")
            
            # Log detailed statistics for each charge type
            charge_stats = []
            for col in pivot_df.columns:
                if col != 'unit':
                    non_zero = (pivot_df[col] != 0).sum()
                    if non_zero > 0:  # Only log charges that have non-zero values
                        total = pivot_df[col].sum()
                        avg = pivot_df[col][pivot_df[col] > 0].mean() if non_zero > 0 else 0
                        charge_stats.append(f"  {col}: {non_zero} units, total: ${total:,.2f}, avg: ${avg:,.2f}")
            
            logger.info("Charge statistics (non-zero only):")
            for stat in charge_stats:
                logger.info(stat)
            
            # Merge unit info with charges
            final_df = pd.merge(units_df, pivot_df, on='unit', how='left')
            
            # Fill NaN values in charge columns with 0
            charge_columns = [col for col in pivot_df.columns if col != 'unit']
            for col in charge_columns:
                final_df[col] = final_df[col].fillna(0)
        else:
            logger.warning("No valid charges found to pivot (all amounts were 0 or null)")
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
    
    # Create actual_rent column (if rent charge exists)
    if 'rent' in final_df.columns:
        final_df['actual_rent'] = final_df['rent']
        occupied_with_rent = ((final_df['occupancy_status'] == 'Occupied') & (final_df['rent'] > 0)).sum()
        occupied_total = (final_df['occupancy_status'] == 'Occupied').sum()
        logger.info(f"Rent check: {occupied_with_rent}/{occupied_total} occupied units have rent > 0")
    else:
        # If no rent column from charges, use 0 or market_rent as fallback
        final_df['actual_rent'] = 0
        logger.info("No 'rent' charge found, setting actual_rent to 0")
    
    # Log final statistics
    logger.info(f"Final result: {len(final_df)} units, {len(final_df.columns)} columns")
    
    # Log summary of which charge columns made it to the final output
    charge_cols_in_output = [col for col in final_df.columns if col not in available_unit_cols and col not in ['occupancy_status', 'actual_rent']]
    if charge_cols_in_output:
        logger.info(f"Charge columns in final output: {charge_cols_in_output}")
    
    return final_df
