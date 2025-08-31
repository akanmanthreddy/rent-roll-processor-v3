"""
Data loader module for Yardi rent roll processing.
Fixed to handle all 14 columns including lease dates.
"""

import pandas as pd
import io
import logging
from typing import List, Tuple, Optional
from src.format_detector import detect_format, get_date_parser, clean_text_for_comparison
from src.config import FORMAT_PROFILES, DATA_END_MARKERS, DEFAULT_FORMAT_PROFILE

logger = logging.getLogger(__name__)


def find_header_and_data_start_excel_yardi(df: pd.DataFrame, format_info: dict) -> pd.DataFrame:
    """
    Specialized handler for Yardi Excel files with two-row headers.
    Properly combines all 14 columns including dates and charges.
    """
    logger.info("=== YARDI EXCEL PROCESSING ===")
    logger.info(f"Input DataFrame shape: {df.shape} (rows: {len(df)}, columns: {len(df.columns)})")
    
    # Find the header row containing Unit, Unit Type, etc.
    header_row_idx = -1
    for idx in range(min(20, len(df))):
        # Check all values in the row, not just non-null
        row_values = df.iloc[idx].values
        row_str = ' '.join(str(val).lower() for val in row_values)
        
        # Yardi header pattern
        if 'unit' in row_str and 'type' in row_str and 'resident' in row_str:
            header_row_idx = idx
            logger.info(f"Found header row at index {idx}")
            break
    
    if header_row_idx == -1:
        raise ValueError("Could not find header row with Unit/Type/Resident pattern")
    
    # Get COMPLETE header rows - all columns
    header1_full = df.iloc[header_row_idx].values  # Get as array to preserve all columns
    header2_full = df.iloc[header_row_idx + 1].values if header_row_idx + 1 < len(df) else ['' for _ in range(len(df.columns))]
    
    logger.info(f"Header row 1 has {len(header1_full)} columns")
    logger.info(f"Header row 2 has {len(header2_full)} columns")
    logger.info(f"Header 1 values: {[str(v) if pd.notna(v) else 'empty' for v in header1_full]}")
    logger.info(f"Header 2 values: {[str(v) if pd.notna(v) else 'empty' for v in header2_full]}")
    
    # Based on your description, here's the exact Yardi structure:
    # Row 4: Unit | Unit Type | Unit | Resident | Name | Market | Charge | Amount | Resident | Other | Move In | Lease | Move Out | Balance
    # Row 5: empty | empty | Sq Ft | empty | empty | Rent | Code | empty | Deposit | Deposit | empty | Expiration | empty | empty
    
    # Build the combined headers manually based on known Yardi structure
    combined_header = []
    yardi_header_map = [
        ('Unit', ''),           # Column 0: Unit
        ('Unit Type', ''),      # Column 1: Unit Type
        ('Unit', 'Sq Ft'),      # Column 2: Unit Sq Ft -> sq_ft
        ('Resident', ''),       # Column 3: Resident (Code)
        ('Name', ''),           # Column 4: Name
        ('Market', 'Rent'),     # Column 5: Market Rent
        ('Charge', 'Code'),     # Column 6: Charge Code
        ('Amount', ''),         # Column 7: Amount
        ('Resident', 'Deposit'), # Column 8: Resident Deposit
        ('Other', 'Deposit'),   # Column 9: Other Deposit
        ('Move In', ''),        # Column 10: Move In
        ('Lease', 'Expiration'), # Column 11: Lease Expiration
        ('Move Out', ''),       # Column 12: Move Out
        ('Balance', '')         # Column 13: Balance
    ]
    
    # Build combined headers using the map or actual values
    for i in range(min(len(df.columns), 14)):  # Process all 14 columns
        if i < len(yardi_header_map):
            h1_expected, h2_expected = yardi_header_map[i]
            
            # Get actual values
            h1_actual = str(header1_full[i]).strip() if i < len(header1_full) and pd.notna(header1_full[i]) else ''
            h2_actual = str(header2_full[i]).strip() if i < len(header2_full) and pd.notna(header2_full[i]) else ''
            
            # Clean 'nan' strings
            h1_actual = '' if h1_actual.lower() in ['nan', 'none'] else h1_actual
            h2_actual = '' if h2_actual.lower() in ['nan', 'none'] else h2_actual
            
            # Combine based on expected structure
            if h2_expected and h2_actual:  # If we expect and have a second row value
                combined = f"{h1_actual} {h2_actual}".strip()
            elif h1_actual:
                combined = h1_actual
            else:
                combined = f'column_{i}'
        else:
            # Beyond expected columns
            combined = f'column_{i}'
        
        combined_header.append(combined)
    
    # Add any remaining columns beyond 14
    for i in range(14, len(df.columns)):
        combined_header.append(f'column_{i}')
    
    logger.info(f"Combined headers ({len(combined_header)}): {combined_header}")
    
    # Find data section start
    data_start_idx = header_row_idx + 2  # Default
    
    for idx in range(header_row_idx + 2, min(header_row_idx + 10, len(df))):
        row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
        
        if any(marker in row_str for marker in ['current/notice/vacant', 'current residents', 'occupied']):
            data_start_idx = idx + 1
            logger.info(f"Found data section marker at row {idx}: {row_str[:100]}")
            break
    
    # Find data end
    data_end_idx = len(df)
    for idx in range(data_start_idx, len(df)):
        row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
        
        if any(marker in row_str for marker in DATA_END_MARKERS):
            data_end_idx = idx
            logger.info(f"Found end marker at row {idx}")
            break
    
    # Extract data
    data_df = df.iloc[data_start_idx:data_end_idx].copy()
    data_df.columns = combined_header
    
    logger.info(f"Extracted {len(data_df)} data rows")
    
    # Apply Yardi-specific column name normalization
    data_df = normalize_column_names_yardi(data_df)
    
    # Remove empty rows
    if 'unit' in data_df.columns:
        initial_count = len(data_df)
        data_df = data_df[data_df['unit'].notna() & (data_df['unit'].astype(str).str.strip() != '')]
        logger.info(f"Removed {initial_count - len(data_df)} empty rows")
    
    # Verify we have all critical columns
    critical_cols = ['unit', 'unit_type', 'sq_ft', 'resident_code', 'resident_name', 
                    'market_rent', 'charge_code', 'amount', 'resident_deposit', 
                    'other_deposit', 'move_in', 'lease_expiration', 'move_out', 'balance']
    
    logger.info("Column verification:")
    for col in critical_cols:
        if col in data_df.columns:
            non_null = data_df[col].notna().sum()
            logger.info(f"  ✓ {col}: {non_null} non-null values")
        else:
            logger.warning(f"  ✗ {col}: NOT FOUND")
    
    return data_df


def normalize_column_names_yardi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Yardi column names to standard format.
    """
    # Map the combined headers to standard names
    column_mappings = {
        'unit': 'unit',
        'unit type': 'unit_type',
        'unit sq ft': 'sq_ft',
        'resident': 'resident_code',
        'name': 'resident_name',
        'market rent': 'market_rent',
        'charge code': 'charge_code',
        'amount': 'amount',
        'resident deposit': 'resident_deposit',
        'other deposit': 'other_deposit',
        'move in': 'move_in',
        'lease expiration': 'lease_expiration',
        'move out': 'move_out',
        'balance': 'balance'
    }
    
    # Clean column names first
    cleaned_columns = {}
    for col in df.columns:
        # Convert to lowercase and strip
        cleaned = str(col).lower().strip()
        # Map to standard name if exists
        if cleaned in column_mappings:
            cleaned_columns[col] = column_mappings[cleaned]
        else:
            # Replace spaces with underscores for unmapped columns
            cleaned_columns[col] = cleaned.replace(' ', '_')
    
    df = df.rename(columns=cleaned_columns)
    
    logger.info(f"Final column names: {list(df.columns)}")
    
    return df


def find_header_and_data_start_excel(df: pd.DataFrame, format_info: dict = None) -> pd.DataFrame:
    """
    Main Excel processing function.
    """
    if format_info and format_info.get('format') == 'yardi' and format_info.get('confidence', 0) > 50:
        logger.info("Using Yardi-specific Excel processor")
        return find_header_and_data_start_excel_yardi(df, format_info)
    
    logger.info("Using generic Excel processor")
    # Generic processing would go here
    return df


def load_and_prepare_dataframe(file_buffer: io.BytesIO, filename: str) -> pd.DataFrame:
    """
    Main entry point for loading rent roll files.
    """
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if file_extension in ['xlsx', 'xls']:
        logger.info(f"Processing Excel file: {filename}")
        
        # Read Excel without interpreting headers
        try:
            df = pd.read_excel(file_buffer, header=None, engine='openpyxl')
            logger.info(f"Read Excel: {df.shape}")
        except Exception as e:
            logger.error(f"Error reading Excel: {e}")
            raise
        
        # Detect format
        format_info = detect_format(df=df, filename=filename, debug=True)
        logger.info(f"Detected: {format_info['format']} ({format_info['confidence']}%)")
        
        # Process with appropriate handler
        return find_header_and_data_start_excel(df, format_info)
    
    else:
        raise NotImplementedError(f"File type {file_extension} not yet implemented")
