"""
Data loader module for Yardi rent roll processing.
STANDALONE VERSION - No format detection, no external dependencies except pandas.
"""

import pandas as pd
import io
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def find_header_and_data_start_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the header rows and the start of the data section in an Excel DataFrame.
    Returns a cleaned DataFrame with proper headers.
    SPECIFIC TO YARDI FORMAT ONLY.
    """
    logger.info(f"Starting Excel processing with shape: {df.shape}")
    
    # Find the header row - look for Unit, Unit Type, Resident pattern
    header_row_idx = -1
    for idx in range(min(20, len(df))):
        row_values = df.iloc[idx].values
        row_str = ' '.join(str(val).lower() for val in row_values if pd.notna(val))
        
        if 'unit' in row_str and 'type' in row_str and 'resident' in row_str:
            header_row_idx = idx
            logger.info(f"Found header row at index {idx}")
            break
    
    if header_row_idx == -1:
        raise ValueError("Could not find the header row with Unit/Type/Resident pattern")
    
    # Get the two header rows
    header1 = df.iloc[header_row_idx].values
    header2 = df.iloc[header_row_idx + 1].values if header_row_idx + 1 < len(df) else [''] * len(df.columns)
    
    # Combine headers - this is the exact Yardi structure
    combined_header = []
    for i in range(len(df.columns)):
        h1 = str(header1[i]).strip() if i < len(header1) and pd.notna(header1[i]) else ''
        h2 = str(header2[i]).strip() if i < len(header2) and pd.notna(header2[i]) else ''
        
        # Clean 'nan' strings
        h1 = '' if h1.lower() in ['nan', 'none'] else h1
        h2 = '' if h2.lower() in ['nan', 'none'] else h2
        
        if h1 and h2:
            combined = f"{h1} {h2}".strip()
        elif h1:
            combined = h1
        elif h2:
            combined = h2
        else:
            combined = f'column_{i}'
        
        combined_header.append(combined)
    
    logger.info(f"Combined headers: {combined_header}")
    
    # Find data start - look for "Current/Notice/Vacant Residents" or similar
    data_start_idx = header_row_idx + 2  # Default
    
    for idx in range(header_row_idx + 2, min(header_row_idx + 10, len(df))):
        row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
        if any(marker in row_str for marker in ['current/notice/vacant', 'current residents', 'occupied']):
            data_start_idx = idx + 1
            logger.info(f"Found data section marker at row {idx}: {row_str[:100]}")
            break
    
    # Find data end - look for summary/total markers
    data_end_idx = len(df)
    for idx in range(data_start_idx, len(df)):
        row_values = df.iloc[idx].values
        row_str = ' '.join(str(val).lower() for val in row_values if pd.notna(val))
        
        # Check for end markers
        if any(marker in row_str for marker in ['summary groups', 'future residents', 'totals', 'grand total']):
            data_end_idx = idx
            logger.info(f"Found end marker at row {idx}")
            break
        
        # Also check if first column contains 'total' or similar
        first_col = str(row_values[0]).lower() if len(row_values) > 0 else ''
        if first_col in ['total', 'totals', 'subtotal', 'grand total']:
            data_end_idx = idx
            logger.info(f"Found total row at {idx}")
            break
    
    # Extract data
    data_df = df.iloc[data_start_idx:data_end_idx].copy()
    data_df.columns = combined_header[:len(data_df.columns)]
    
    logger.info(f"Extracted {len(data_df)} data rows")
    
    # Normalize column names
    data_df.columns = [col.lower().strip().replace(' ', '_') for col in data_df.columns]
    
    # Standard column mappings for Yardi
    column_mappings = {
        'unit_sq_ft': 'sq_ft',
        'unit_sqft': 'sq_ft',
        'name': 'resident_name',
        'resident': 'resident_code'
    }
    
    for old, new in column_mappings.items():
        if old in data_df.columns and new not in data_df.columns:
            data_df = data_df.rename(columns={old: new})
    
    # CRITICAL: Remove rows where unit is 'nan', empty, or invalid
    if 'unit' in data_df.columns:
        # Convert to string and clean
        data_df['unit'] = data_df['unit'].astype(str).str.strip()
        
        # Filter out invalid units
        invalid_units = ['nan', 'NaN', 'NAN', 'null', 'NULL', 'None', 'NONE', '', ' ']
        initial_len = len(data_df)
        data_df = data_df[~data_df['unit'].isin(invalid_units)]
        data_df = data_df[data_df['unit'].notna()]
        
        # Also remove any rows where unit contains 'total'
        data_df = data_df[~data_df['unit'].str.lower().str.contains('total', na=False)]
        
        removed = initial_len - len(data_df)
        if removed > 0:
            logger.info(f"Removed {removed} rows with invalid/empty units")
    
    logger.info(f"Final shape: {data_df.shape}")
    logger.info(f"Final columns: {list(data_df.columns)}")
    
    # Log sample of the data
    if not data_df.empty:
        logger.info(f"First few units: {data_df['unit'].head(10).tolist() if 'unit' in data_df.columns else 'No unit column'}")
        if 'charge_code' in data_df.columns and 'amount' in data_df.columns:
            sample = data_df[['unit', 'charge_code', 'amount']].head(20)
            logger.info(f"Sample charge data:\n{sample}")
    
    return data_df


def find_header_and_data_start_csv(file_buffer: io.BytesIO) -> Tuple[int, int, List[str]]:
    """
    CSV processing function - identifies the header rows and data section.
    """
    file_buffer.seek(0)
    lines = file_buffer.readlines()
    file_buffer.seek(0)

    header_start_idx = -1
    data_start_idx = -1

    # Find header row
    for i, line_bytes in enumerate(lines):
        line = line_bytes.decode('utf-8', errors='ignore').strip().lower()
        if 'unit' in line and 'type' in line and 'resident' in line:
            header_start_idx = i
            logger.info(f"Found CSV header at line {i}")
            break

    if header_start_idx == -1:
        raise ValueError("Could not find the header row in the CSV file")

    # Parse the two header rows
    header1 = lines[header_start_idx].decode('utf-8', errors='ignore').strip().split(',')
    header2 = []
    if header_start_idx + 1 < len(lines):
        header2 = lines[header_start_idx + 1].decode('utf-8', errors='ignore').strip().split(',')

    # Combine headers
    combined_header = []
    max_cols = max(len(header1), len(header2) if header2 else 0)
    
    for i in range(max_cols):
        h1 = header1[i].strip() if i < len(header1) else ''
        h2 = header2[i].strip() if i < len(header2) else ''
        
        if h1 and h2:
            combined_header.append(f"{h1} {h2}".strip())
        elif h1:
            combined_header.append(h1)
        elif h2:
            combined_header.append(h2)
        else:
            combined_header.append(f'column_{i}')

    # Find data start
    for i in range(header_start_idx + 2, len(lines)):
        line = lines[i].decode('utf-8', errors='ignore').strip().lower()
        if 'current/notice/vacant' in line or 'current residents' in line:
            data_start_idx = i + 1
            logger.info(f"Found CSV data start at line {i + 1}")
            break

    if data_start_idx == -1:
        data_start_idx = header_start_idx + 2
        logger.warning("Could not find data section marker, using default")

    return header_start_idx, data_start_idx, combined_header


def load_and_prepare_dataframe(file_buffer: io.BytesIO, filename: str) -> pd.DataFrame:
    """
    Main entry point - loads CSV or Excel file.
    NO FORMAT DETECTION - assumes Yardi format.
    """
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if file_extension in ['xlsx', 'xls']:
        logger.info(f"Loading Excel file: {filename}")
        
        # Read Excel without headers
        df = pd.read_excel(file_buffer, header=None, engine='openpyxl')
        logger.info(f"Raw Excel shape: {df.shape}")
        
        # Process as Yardi Excel
        return find_header_and_data_start_excel(df)
        
    else:  # Assume CSV
        logger.info(f"Loading CSV file: {filename}")
        
        # Get header info
        header_start_idx, data_start_idx, combined_header = find_header_and_data_start_csv(file_buffer)
        
        # Find footer
        file_buffer.seek(0)
        lines = file_buffer.readlines()
        footer_start_idx = len(lines)
        
        for i in range(data_start_idx, len(lines)):
            line = lines[i].decode('utf-8', errors='ignore').strip().lower()
            if any(marker in line for marker in ['summary groups', 'future residents', 'total', 'grand total']):
                footer_start_idx = i
                logger.info(f"Found CSV footer at line {i}")
                break

        # Read CSV
        file_buffer.seek(0)
        df = pd.read_csv(
            file_buffer,
            header=None,
            skiprows=data_start_idx,
            nrows=footer_start_idx - data_start_idx,
            names=combined_header,
            engine='python'
        )

        # Normalize column names
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        
        # Standard mappings
        column_mappings = {
            'unit_sq_ft': 'sq_ft',
            'unit_sqft': 'sq_ft',
            'name': 'resident_name',
            'resident': 'resident_code'
        }
        
        df = df.rename(columns=column_mappings)
        
        # Remove invalid units
        if 'unit' in df.columns:
            df['unit'] = df['unit'].astype(str).str.strip()
            invalid_units = ['nan', 'NaN', 'NAN', 'null', 'NULL', 'None', 'NONE', '', ' ']
            df = df[~df['unit'].isin(invalid_units)]
            df = df[df['unit'].notna()]
            df = df[~df['unit'].str.lower().str.contains('total', na=False)]
        
        logger.info(f"CSV final shape: {df.shape}")
        logger.info(f"CSV columns: {list(df.columns)}")

        return df
