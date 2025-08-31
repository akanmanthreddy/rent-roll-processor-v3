"""
Enhanced Data loader module for rent roll processing.
Improved format detection for Excel files.
"""

import pandas as pd
import io
import logging
import csv
from typing import List, Tuple, Optional
from src.format_detector import detect_format, get_date_parser, clean_text_for_comparison
from src.config import FORMAT_PROFILES, DATA_END_MARKERS, DEFAULT_FORMAT_PROFILE

logger = logging.getLogger(__name__)


def find_header_and_data_start_excel(df: pd.DataFrame, format_info: dict = None) -> pd.DataFrame:
    """
    Enhanced Excel header detection with better Yardi support.
    """
    header_row_idx = -1
    
    # Log initial DataFrame info
    logger.info(f"Analyzing Excel file with {len(df)} rows and {len(df.columns)} columns")
    
    # First, try to detect format from the DataFrame if not already done
    if not format_info or format_info.get('format') == 'generic':
        logger.info("Running format detection on Excel DataFrame...")
        format_info = detect_format(df=df, debug=True)
        logger.info(f"Format detection result: {format_info['format']} (confidence: {format_info['confidence']}%)")
    
    # Log first 10 rows for debugging
    logger.info("First 10 rows of Excel file:")
    for idx in range(min(10, len(df))):
        row_values = df.iloc[idx].values
        non_null_values = [str(val) for val in row_values if pd.notna(val)][:10]  # First 10 non-null values
        logger.info(f"Row {idx}: {non_null_values}")
    
    # Strategy 1: Look for Yardi-specific header pattern
    if format_info and format_info.get('format') == 'yardi':
        logger.info("Using Yardi-specific header detection...")
        
        # Yardi often has headers containing these key fields
        yardi_header_indicators = [
            ['unit', 'type', 'resident'],  # Most common
            ['unit', 'type', 'sq', 'resident'],  # With sq ft
            ['unit', 'resident', 'name'],  # Alternative
        ]
        
        for idx in range(min(20, len(df))):  # Check more rows
            row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
            
            # Check for any of the Yardi patterns
            for indicators in yardi_header_indicators:
                if all(indicator in row_str for indicator in indicators):
                    header_row_idx = idx
                    logger.info(f"Found Yardi header at row {idx} using pattern: {indicators}")
                    break
            
            if header_row_idx != -1:
                break
    
    # Strategy 2: Generic header search
    if header_row_idx == -1:
        logger.info("Using generic header detection...")
        
        # Look for rows containing common header keywords
        header_keywords = ['unit', 'resident', 'tenant', 'apartment', 'rent']
        
        for idx in range(min(20, len(df))):
            row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
            
            # Count how many keywords appear
            keyword_count = sum(1 for keyword in header_keywords if keyword in row_str)
            
            # If we find enough keywords, likely a header
            if keyword_count >= 2:
                header_row_idx = idx
                logger.info(f"Found header at row {idx} with {keyword_count} keywords")
                break
    
    if header_row_idx == -1:
        logger.error("Could not find header row in Excel file")
        # Try to use the first row as header as fallback
        header_row_idx = 0
        logger.warning("Using first row as header (fallback)")
    
    # Handle two-row headers (common in Yardi)
    has_two_row_header = format_info.get('profile', {}).get('has_two_row_header', True)
    
    if has_two_row_header and header_row_idx + 1 < len(df):
        logger.info("Processing two-row header...")
        
        header1 = df.iloc[header_row_idx].fillna('')
        header2 = df.iloc[header_row_idx + 1].fillna('')
        
        # Check if second row actually contains header info
        # (sometimes it's already data)
        row2_str = ' '.join(str(val).lower() for val in header2.values if pd.notna(val) and str(val).strip())
        is_row2_data = any(char.isdigit() for char in row2_str[:50])  # Check if row2 has numbers early
        
        if is_row2_data:
            logger.info("Second row appears to be data, using single row header")
            combined_header = [str(h).strip() if pd.notna(h) else f'column_{i}' 
                             for i, h in enumerate(header1)]
            data_start_idx = header_row_idx + 1
        else:
            # Combine two header rows
            combined_header = []
            for i in range(len(header1)):
                h1 = str(header1.iloc[i]).strip() if i < len(header1) else ''
                h2 = str(header2.iloc[i]).strip() if i < len(header2) else ''
                
                # Smart combination based on content
                if h1 and h2 and h1 != h2:
                    # Both have values and are different
                    if h1.lower() in ['unit', 'resident', 'charge'] and h2.lower() not in ['', 'nan']:
                        # Primary field with subfield
                        combined_header.append(f"{h1} {h2}".strip())
                    else:
                        # Prefer h1 unless it's generic
                        combined_header.append(h1 if h1.lower() != 'nan' else h2)
                elif h1 and h1.lower() != 'nan':
                    combined_header.append(h1)
                elif h2 and h2.lower() != 'nan':
                    combined_header.append(h2)
                else:
                    combined_header.append(f'column_{i}')
            
            data_start_idx = header_row_idx + 2
    else:
        # Single row header
        header_row = df.iloc[header_row_idx]
        combined_header = [str(val).strip() if pd.notna(val) else f'column_{i}' 
                          for i, val in enumerate(header_row)]
        data_start_idx = header_row_idx + 1
    
    logger.info(f"Combined header (first 10 cols): {combined_header[:10]}")
    
    # Find data section start (look for section markers)
    actual_data_start = data_start_idx
    
    if format_info and format_info.get('profile'):
        section_markers = format_info['profile'].get('section_markers', [])
        
        for idx in range(data_start_idx, min(data_start_idx + 10, len(df))):
            row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
            
            for marker in section_markers:
                if clean_text_for_comparison(marker) in clean_text_for_comparison(row_str):
                    actual_data_start = idx + 1
                    logger.info(f"Found section marker '{marker}' at row {idx}")
                    break
            
            if actual_data_start != data_start_idx:
                break
    
    # Find data end (before summary sections)
    data_end_idx = len(df)
    
    for idx in range(actual_data_start, len(df)):
        row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values if pd.notna(val))
        
        # Check for end markers
        for marker in DATA_END_MARKERS:
            if marker in row_str:
                data_end_idx = idx
                logger.info(f"Found end marker '{marker}' at row {idx}")
                break
        
        if data_end_idx != len(df):
            break
    
    # Extract data section
    new_df = df.iloc[actual_data_start:data_end_idx].reset_index(drop=True)
    
    # Apply headers
    if len(combined_header) >= len(new_df.columns):
        new_df.columns = combined_header[:len(new_df.columns)]
    else:
        logger.warning(f"Header length mismatch: {len(combined_header)} headers for {len(new_df.columns)} columns")
        # Pad with generic names
        new_df.columns = combined_header + [f'column_{i}' for i in range(len(combined_header), len(new_df.columns))]
    
    # Normalize column names
    new_df = normalize_column_names(new_df, format_info)
    
    logger.info(f"Extracted {len(new_df)} data rows from Excel file")
    logger.info(f"Final columns (first 10): {list(new_df.columns)[:10]}")
    
    return new_df


def find_header_and_data_start_csv(file_buffer: io.BytesIO, format_info: dict = None) -> Tuple[int, int, List[str]]:
    """
    CSV header detection with format awareness.
    """
    file_buffer.seek(0)
    lines = file_buffer.readlines()
    file_buffer.seek(0)

    header_start_idx = -1
    data_start_idx = -1

    # Log first few lines for debugging
    logger.info("Analyzing CSV file...")
    for i in range(min(10, len(lines))):
        line = lines[i].decode('utf-8', errors='ignore').strip()
        logger.info(f"Line {i}: {line[:100]}...")

    # Try format-specific detection first
    if format_info and format_info.get('format') != 'generic':
        profile = format_info.get('profile', {})
        header_markers = profile.get('header_markers', [])
        
        for i, line_bytes in enumerate(lines[:20]):  # Check first 20 lines
            line = line_bytes.decode('utf-8', errors='ignore').strip().lower()
            
            # Check if line contains expected headers
            marker_count = sum(1 for marker in header_markers if marker in line)
            if marker_count >= 2:  # At least 2 markers found
                header_start_idx = i
                logger.info(f"Found header at line {i} using format-specific markers")
                break
    
    # Fallback to generic detection
    if header_start_idx == -1:
        for i, line_bytes in enumerate(lines[:20]):
            line = line_bytes.decode('utf-8', errors='ignore').strip().lower()
            
            # Generic header detection
            if 'unit' in line and ('resident' in line or 'tenant' in line):
                header_start_idx = i
                logger.info(f"Found header at line {i} using generic detection")
                break

    if header_start_idx == -1:
        logger.error("Could not find header row in CSV file")
        raise ValueError("Could not find the header row in the CSV file.")

    # Handle two-row headers if needed
    has_two_row_header = format_info.get('profile', {}).get('has_two_row_header', True) if format_info else True
    
    if has_two_row_header and header_start_idx + 1 < len(lines):
        header1 = lines[header_start_idx].decode('utf-8', errors='ignore').strip().split(',')
        header2 = lines[header_start_idx + 1].decode('utf-8', errors='ignore').strip().split(',')
        
        # Check if second row is actually data
        is_row2_data = any(char.isdigit() for char in header2[0][:10] if header2[0])
        
        if is_row2_data:
            combined_header = header1
            data_start_idx = header_start_idx + 1
        else:
            # Combine headers
            combined_header = []
            for i in range(max(len(header1), len(header2))):
                h1 = header1[i].strip() if i < len(header1) else ''
                h2 = header2[i].strip() if i < len(header2) else ''
                
                if h1 and h2:
                    combined_header.append(f"{h1} {h2}".strip())
                elif h1:
                    combined_header.append(h1)
                else:
                    combined_header.append(h2)
            
            data_start_idx = header_start_idx + 2
    else:
        header1 = lines[header_start_idx].decode('utf-8', errors='ignore').strip().split(',')
        combined_header = header1
        data_start_idx = header_start_idx + 1

    logger.info(f"Combined header has {len(combined_header)} columns. First 5: {combined_header[:5]}")

    # Look for section markers to find actual data start
    if format_info and format_info.get('profile'):
        section_markers = format_info['profile'].get('section_markers', [])
        
        for i in range(data_start_idx, min(data_start_idx + 10, len(lines))):
            line = lines[i].decode('utf-8', errors='ignore').strip().lower()
            
            for marker in section_markers:
                if clean_text_for_comparison(marker) in clean_text_for_comparison(line):
                    data_start_idx = i + 1
                    logger.info(f"Found section marker '{marker}' at line {i}")
                    break

    return header_start_idx, data_start_idx, combined_header


def normalize_column_names(df: pd.DataFrame, format_info: dict = None) -> pd.DataFrame:
    """
    Enhanced column normalization with better format-specific handling.
    """
    # First, clean all column names
    df.columns = [col.lower().strip().replace('  ', ' ').replace(' ', '_') for col in df.columns]
    
    logger.info(f"Columns before normalization: {list(df.columns)[:10]}")
    
    # Apply format-specific mappings if available
    if format_info and format_info.get('profile') and 'column_mappings' in format_info['profile']:
        mapping = format_info['profile']['column_mappings']
        
        # Create reverse mapping for better matching
        reverse_mapping = {}
        for original, standard in mapping.items():
            # Try exact match
            if original in df.columns:
                reverse_mapping[original] = standard
            # Try with underscores
            elif original.replace('_', ' ').replace(' ', '_') in df.columns:
                for col in df.columns:
                    if col.replace('_', ' ') == original.replace('_', ' '):
                        reverse_mapping[col] = standard
                        break
        
        if reverse_mapping:
            df = df.rename(columns=reverse_mapping)
            logger.info(f"Applied column mappings: {reverse_mapping}")
    
    # Standard mappings that apply to all formats
    standard_mappings = {
        'unit_sq_ft': 'sq_ft',
        'unit_sqft': 'sq_ft',
        'square_feet': 'sq_ft',
        'square_footage': 'sq_ft',
        'apartment': 'unit',
        'apt': 'unit',
        'unit_number': 'unit',
        'tenant': 'resident_name',
        'tenant_name': 'resident_name',
        'name': 'resident_name',
        'monthly_rent': 'market_rent',
        'rent_amount': 'market_rent',
        'move_in_date': 'move_in',
        'movein_date': 'move_in',
        'lease_end': 'lease_expiration',
        'lease_end_date': 'lease_expiration',
        'expiration': 'lease_expiration'
    }
    
    for old, new in standard_mappings.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    
    # Ensure critical columns exist
    critical_columns = ['unit']
    
    for critical_col in critical_columns:
        if critical_col not in df.columns:
            # Try to find a similar column
            possible_cols = [col for col in df.columns if critical_col in col.lower()]
            if possible_cols:
                logger.warning(f"'{critical_col}' not found, using '{possible_cols[0]}' instead")
                df = df.rename(columns={possible_cols[0]: critical_col})
            else:
                logger.error(f"Critical column '{critical_col}' not found and no alternative found")
    
    logger.info(f"Columns after normalization: {list(df.columns)[:10]}")
    
    return df


def load_and_prepare_dataframe(file_buffer: io.BytesIO, filename: str) -> pd.DataFrame:
    """
    Enhanced main entry point with better format detection.
    """
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if file_extension in ['xlsx', 'xls']:
        logger.info(f"Processing Excel file: {filename}")
        
        # Read Excel file without headers first
        try:
            df = pd.read_excel(file_buffer, header=None, engine='openpyxl')
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise ValueError(f"Could not read Excel file: {e}")
        
        # Run format detection on the DataFrame
        format_info = detect_format(df=df, filename=filename, debug=True)
        
        logger.info(f"Detected format: {format_info['format']} (confidence: {format_info['confidence']}%)")
        
        # Log debug info if available
        if format_info.get('debug_info'):
            debug_info = format_info['debug_info']
            logger.info(f"Debug - Best match: {debug_info.get('best_match')} with score: {debug_info.get('best_score')}")
            if debug_info.get('patterns_found'):
                logger.info(f"Debug - Patterns found: {debug_info['patterns_found'][:5]}")
        
        # Process with detected format
        return find_header_and_data_start_excel(df, format_info)
        
    else:  # CSV processing
        logger.info(f"Processing CSV file: {filename}")
        
        # Read sample for format detection
        file_buffer.seek(0)
        sample_content = file_buffer.read(10000).decode('utf-8', errors='ignore')
        file_buffer.seek(0)
        
        # Detect format
        format_info = detect_format(file_content=sample_content, filename=filename, debug=True)
        
        logger.info(f"Detected format: {format_info['format']} (confidence: {format_info['confidence']}%)")
        
        # Process with detected format
        header_start_idx, data_start_idx, combined_header = find_header_and_data_start_csv(file_buffer, format_info)
        
        # Find data end
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

        # Normalize column names
        df = normalize_column_names(df, format_info)

        logger.info(f"Extracted {len(df)} rows from CSV file")
        return df
