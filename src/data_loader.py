"""
Data loader module for rent roll processing.
Handles both CSV and Excel file formats with complex header structures.
Adapts parsing based on detected format.
"""

import pandas as pd
import io
import logging
import csv
from typing import List, Tuple
from src.format_detector import detect_format, get_date_parser
from src.config import FORMAT_PROFILES, DATA_END_MARKERS, DEFAULT_FORMAT_PROFILE

logger = logging.getLogger(__name__)


def find_header_and_data_start_excel(df: pd.DataFrame, format_info: dict = None) -> pd.DataFrame:
    """
    Identifies header rows and data section in an Excel DataFrame.
    RESTORING ORIGINAL LOGIC that searches for headers.
    """
    header_row_idx = -1
    
    # Log what we're looking at
    logger.info(f"Searching for headers in Excel file with {len(df)} rows")
    for idx in range(min(10, len(df))):
        row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
        logger.info(f"Row {idx}: {row_str[:100]}...")
    
    # ORIGINAL LOGIC: Find the header row containing key fields
    for idx, row in df.iterrows():
        row_str = ' '.join(str(val).lower() for val in row.values if pd.notna(val))
        # Check for the pattern that indicates headers
        if all(keyword in row_str for keyword in ['unit', 'unit type', 'resident']):
            header_row_idx = idx
            logger.info(f"Found header at row {idx}")
            break
    
    if header_row_idx == -1:
        logger.error("Could not find header row in Excel file")
        raise ValueError("Could not find the primary header row in the Excel file.")
    
    # Get two header rows (ORIGINAL LOGIC)
    header1 = df.iloc[header_row_idx].fillna('')
    header2 = df.iloc[header_row_idx + 1].fillna('') if header_row_idx + 1 < len(df) else pd.Series()
    
    # Combine headers (ORIGINAL LOGIC)
    combined_header = []
    for i in range(len(header1)):
        h1 = str(header1.iloc[i]).strip()
        h2 = str(header2.iloc[i]).strip() if i < len(header2) else ''
        if h1 and h2:
            combined_header.append(f"{h1} {h2}".strip())
        elif h1:
            combined_header.append(h1)
        else:
            combined_header.append(h2 if h2 else f'column_{i}')
    
    # Find data start (after section marker)
    data_start_idx = header_row_idx + 2
    for idx in range(header_row_idx + 2, len(df)):
        row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
        if 'current/notice/vacant residents' in row_str:
            data_start_idx = idx + 1
            logger.info(f"Found data section marker at row {idx}")
            break
    
    # Find data end (before summary sections)
    data_end_idx = len(df)
    for idx in range(data_start_idx, len(df)):
        row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
        if 'summary groups' in row_str or 'future residents/applicants' in row_str:
            data_end_idx = idx
            break
    
    # Extract data section
    new_df = df.iloc[data_start_idx:data_end_idx].reset_index(drop=True)
    new_df.columns = combined_header[:len(new_df.columns)]
    
    # Normalize column names
    new_df = normalize_column_names(new_df, format_info)
    
    logger.info(f"Extracted {len(new_df)} rows from Excel file")
    return new_df


def find_header_and_data_start_csv(file_buffer: io.BytesIO, format_info: dict = None) -> Tuple[int, int, List[str]]:
    """
    Identifies header rows and data section in a CSV file.
    THIS IS THE ORIGINAL WORKING LOGIC - searches through lines to find headers.
    """
    file_buffer.seek(0)
    lines = file_buffer.readlines()
    file_buffer.seek(0)

    header_start_idx = -1
    data_start_idx = -1

    # Log first few lines to debug
    logger.info("Searching for headers in CSV file...")
    for i in range(min(10, len(lines))):
        line = lines[i].decode('utf-8', errors='ignore').strip()
        logger.info(f"Line {i}: {line[:100]}...")

    # ORIGINAL LOGIC: Find the line containing "Unit,Unit Type,Unit,Resident..."
    for i, line_bytes in enumerate(lines):
        line = line_bytes.decode('utf-8', errors='ignore').strip()
        # Check for the exact pattern that worked before
        if all(keyword in line.lower() for keyword in ['unit,', 'unit type,', 'resident,']):
            header_start_idx = i
            logger.info(f"Found header at line {i}")
            break

    if header_start_idx == -1:
        logger.error("Could not find header row with pattern ['unit,', 'unit type,', 'resident,']")
        raise ValueError("Could not find the primary header row in the file.")

    # ORIGINAL LOGIC: The header is composed of two lines
    header1 = lines[header_start_idx].decode('utf-8', errors='ignore').strip().split(',')
    header2 = lines[header_start_idx + 1].decode('utf-8', errors='ignore').strip().split(',') if header_start_idx + 1 < len(lines) else []

    logger.info(f"Header row 1 has {len(header1)} columns. First 5: {header1[:5]}")
    logger.info(f"Header row 2 has {len(header2)} columns. First 5: {header2[:5]}")

    # ORIGINAL LOGIC: Combine the two header lines into one
    combined_header = []
    for i in range(len(header1)):
        h1 = header1[i].strip()
        h2 = header2[i].strip() if i < len(header2) else ''
        # Combine if h1 is a primary field, otherwise use h2 if it's more specific
        if h1 and h2:
            combined_header.append(f"{h1} {h2}".strip())
        elif h1:
            combined_header.append(h1)
        else:
            combined_header.append(h2)

    logger.info(f"Combined header has {len(combined_header)} columns. First 5: {combined_header[:5]}")

    # ORIGINAL LOGIC: Data starts after the "Current/Notice/Vacant Residents" line
    for i in range(header_start_idx + 2, len(lines)):
        line = lines[i].decode('utf-8', errors='ignore').strip().lower()
        if 'current/notice/vacant residents' in line:
            data_start_idx = i + 1
            logger.info(f"Found data start marker at line {i}")
            break

    if data_start_idx == -1:
        # Fallback if marker not found
        data_start_idx = header_start_idx + 2
        logger.warning("No section marker found, assuming data starts after headers")

    return header_start_idx, data_start_idx, combined_header

    return header_start_idx, data_start_idx, combined_header


def normalize_column_names(df: pd.DataFrame, format_info: dict = None) -> pd.DataFrame:
    """
    Normalizes column names based on the detected format.
    Maps format-specific column names to standard names.
    """
    # First, lowercase and clean all column names
    df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
    
    # Log the columns before mapping
    logger.info(f"Columns before normalization: {list(df.columns)[:10]}")
    
    # Apply format-specific mappings if available
    if format_info and format_info.get('profile') and 'column_mappings' in format_info['profile']:
        mapping = format_info['profile']['column_mappings']
        # Only rename columns that exist
        mapping_to_apply = {k: v for k, v in mapping.items() if k in df.columns}
        if mapping_to_apply:
            df = df.rename(columns=mapping_to_apply)
            logger.info(f"Applied column mappings: {mapping_to_apply}")
    else:
        # Apply default mappings
        df = df.rename(columns={'unit_sq_ft': 'sq_ft', 'unit_sqft': 'sq_ft'})
    
    # Log the columns after mapping
    logger.info(f"Columns after normalization: {list(df.columns)[:10]}")
    
    # Ensure 'unit' column exists (critical column)
    if 'unit' not in df.columns:
        # Check if there's a column that might be 'unit' with different name
        possible_unit_cols = [col for col in df.columns if 'unit' in col.lower() or 'apt' in col.lower() or 'apartment' in col.lower()]
        if possible_unit_cols:
            logger.warning(f"'unit' column not found, but found similar: {possible_unit_cols}. Using first match.")
            df = df.rename(columns={possible_unit_cols[0]: 'unit'})
        else:
            logger.error(f"No 'unit' column found. Available columns: {list(df.columns)}")
    
    return df


def load_and_prepare_dataframe(file_buffer: io.BytesIO, filename: str) -> pd.DataFrame:
    """
    Main entry point for loading rent roll files.
    Detects format and routes to appropriate parser.
    """
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if file_extension in ['xlsx', 'xls']:
        logger.info("Processing Excel file")
        # For Excel, read first then detect format from the actual data
        df = pd.read_excel(file_buffer, header=None, engine='openpyxl')
        
        # Now detect format from the Excel DataFrame
        format_info = detect_format(
            df=df,
            filename=filename
        )
        
        logger.info(f"Using format profile: {format_info['format']} (confidence: {format_info['confidence']}%)")
        return find_header_and_data_start_excel(df, format_info)
    else:
        logger.info("Processing CSV file")
        # For CSV, we can read a sample for format detection
        file_buffer.seek(0)
        sample_content = file_buffer.read(10000).decode('utf-8', errors='ignore')
        file_buffer.seek(0)
        
        format_info = detect_format(
            file_content=sample_content,
            filename=filename
        )
        
        logger.info(f"Using format profile: {format_info['format']} (confidence: {format_info['confidence']}%)")
        
        header_start_idx, data_start_idx, combined_header = find_header_and_data_start_csv(file_buffer, format_info)
        
        # Find data end (before summary sections)
        file_buffer.seek(0)
        lines = file_buffer.readlines()
        footer_start_idx = len(lines)
        
        for i in range(data_start_idx, len(lines)):
            line = lines[i].decode('utf-8', errors='ignore').strip().lower()
            if any(marker in line for marker in DATA_END_MARKERS):
                footer_start_idx = i
                break

        # Read data section
        file_buffer.seek(0)
        df = pd.read_csv(
            file_buffer,
            header=None,
            skiprows=data_start_idx,
            nrows=footer_start_idx - data_start_idx,
            names=combined_header,
            engine='python'
        )

        # Normalize column names based on format
        df = normalize_column_names(df, format_info)

        logger.info(f"Extracted {len(df)} rows from CSV file")
        return df
