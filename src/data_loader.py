"""
Data loader module for Yardi rent roll processing.
Reverted to simpler working approach.
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
    Based on the working version - simpler and more direct.
    """
    # Find the header row
    header_row_idx = -1
    for idx in range(min(20, len(df))):  # Check first 20 rows
        row_values = df.iloc[idx].values
        row_str = ' '.join(str(val).lower() for val in row_values if pd.notna(val))
        
        # Look for the characteristic Yardi header pattern
        if all(keyword in row_str for keyword in ['unit', 'type', 'resident']):
            header_row_idx = idx
            logger.info(f"Found header row at index {idx}")
            break
    
    if header_row_idx == -1:
        raise ValueError("Could not find the primary header row in the Excel file.")
    
    # Get the two header rows (Yardi uses two-row headers)
    header1 = df.iloc[header_row_idx].fillna('')
    header2 = df.iloc[header_row_idx + 1].fillna('') if header_row_idx + 1 < len(df) else pd.Series([''] * len(df.columns))
    
    # Combine headers
    combined_header = []
    for i in range(len(df.columns)):
        h1 = str(header1.iloc[i]).strip() if i < len(header1) else ''
        h2 = str(header2.iloc[i]).strip() if i < len(header2) else ''
        
        # Clean up 'nan' strings
        h1 = '' if h1.lower() in ['nan', 'none'] else h1
        h2 = '' if h2.lower() in ['nan', 'none'] else h2
        
        if h1 and h2:
            combined_header.append(f"{h1} {h2}".strip())
        elif h1:
            combined_header.append(h1)
        elif h2:
            combined_header.append(h2)
        else:
            combined_header.append(f'column_{i}')
    
    logger.info(f"Combined headers: {combined_header[:15]}")  # Log first 15 headers
    
    # Find data start (usually after "Current/Notice/Vacant Residents" marker)
    data_start_idx = header_row_idx + 2  # Default to right after headers
    
    for idx in range(header_row_idx + 2, min(header_row_idx + 10, len(df))):
        row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
        if 'current/notice/vacant' in row_str or 'current residents' in row_str:
            data_start_idx = idx + 1
            logger.info(f"Found data section marker at row {idx}")
            break
    
    # Find data end (look for summary markers)
    data_end_idx = len(df)
    for idx in range(data_start_idx, len(df)):
        row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
        if any(marker in row_str for marker in ['summary groups', 'future residents', 'totals', 'grand total']):
            data_end_idx = idx
            logger.info(f"Found end marker at row {idx}")
            break
    
    # Extract the data
    new_df = df.iloc[data_start_idx:data_end_idx].reset_index(drop=True)
    
    # Apply the combined headers
    new_df.columns = combined_header[:len(new_df.columns)]
    
    # Normalize column names
    new_df.columns = [col.lower().strip().replace(' ', '_') for col in new_df.columns]
    
    # Handle common column name variations
    column_mappings = {
        'unit_sq_ft': 'sq_ft',
        'unit_sqft': 'sq_ft',
        'name': 'resident_name',
        'resident': 'resident_code',
        'charge_code': 'charge_code',  # Ensure this stays as is
        'amount': 'amount'  # Ensure this stays as is
    }
    
    for old_name, new_name in column_mappings.items():
        if old_name in new_df.columns and new_name not in new_df.columns:
            new_df = new_df.rename(columns={old_name: new_name})
    
    logger.info(f"Final columns: {list(new_df.columns)[:15]}")  # Log first 15 columns
    logger.info(f"Data shape: {new_df.shape}")
    
    # Log sample of charge data if available
    if 'charge_code' in new_df.columns and 'amount' in new_df.columns:
        sample = new_df[['unit', 'charge_code', 'amount']].head(20)
        logger.info(f"Sample charge data:\n{sample}")
    
    return new_df


def find_header_and_data_start_csv(file_buffer: io.BytesIO) -> Tuple[int, int, List[str]]:
    """
    Original CSV processing function - identifies the header rows and data section.
    """
    file_buffer.seek(0)
    lines = file_buffer.readlines()
    file_buffer.seek(0)

    header_start_idx = -1
    data_start_idx = -1

    # Find header row
    for i, line_bytes in enumerate(lines):
        line = line_bytes.decode('utf-8', errors='ignore').strip().lower()
        if all(keyword in line for keyword in ['unit,', 'type,', 'resident,']):
            header_start_idx = i
            logger.info(f"Found CSV header at line {i}")
            break

    if header_start_idx == -1:
        raise ValueError("Could not find the primary header row in the CSV file.")

    # Parse the two header rows
    header1 = lines[header_start_idx].decode('utf-8', errors='ignore').strip().split(',')
    header2 = lines[header_start_idx + 1].decode('utf-8', errors='ignore').strip().split(',') if header_start_idx + 1 < len(lines) else []

    # Combine headers
    combined_header = []
    max_cols = max(len(header1), len(header2))
    
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
        data_start_idx = header_start_idx + 2  # Default to right after headers
        logger.warning("Could not find data section marker, using default")

    return header_start_idx, data_start_idx, combined_header


def load_and_prepare_dataframe(file_buffer: io.BytesIO, filename: str) -> pd.DataFrame:
    """
    Loads the file (CSV or Excel) and prepares it for processing.
    Simplified version based on working code.
    """
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if file_extension in ['xlsx', 'xls']:
        logger.info(f"Processing Excel file: {filename}")
        
        # Read Excel without headers
        df = pd.read_excel(file_buffer, header=None, engine='openpyxl')
        logger.info(f"Raw Excel shape: {df.shape}")
        
        # Process Excel
        df = find_header_and_data_start_excel(df)
        
        return df
        
    else:  # Assume CSV
        logger.info(f"Processing CSV file: {filename}")
        
        # Get header info
        header_start_idx, data_start_idx, combined_header = find_header_and_data_start_csv(file_buffer)
        
        # Find footer/end
        file_buffer.seek(0)
        lines = file_buffer.readlines()
        footer_start_idx = len(lines)
        
        for i in range(data_start_idx, len(lines)):
            line = lines[i].decode('utf-8', errors='ignore').strip().lower()
            if any(marker in line for marker in ['summary groups', 'future residents', 'totals']):
                footer_start_idx = i
                logger.info(f"Found CSV footer at line {i}")
                break

        # Read CSV with proper boundaries
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
        
        # Handle common column name variations
        column_mappings = {
            'unit_sq_ft': 'sq_ft',
            'unit_sqft': 'sq_ft',
            'name': 'resident_name',
            'resident': 'resident_code'
        }
        
        df = df.rename(columns=column_mappings)
        
        logger.info(f"CSV shape after loading: {df.shape}")
        logger.info(f"CSV columns: {list(df.columns)[:15]}")

        return df
