"""
Enhanced Format Detection Module with debugging capabilities.
Improved detection for Yardi and other property management systems.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union
import re

logger = logging.getLogger(__name__)

# Format profiles for different property management systems
FORMAT_PROFILES = {
    'yardi': {
        'identifiers': ['yardi', 'voyager', 'rent roll report'],
        'header_markers': ['unit', 'unit type', 'resident'],
        'section_markers': [
            'current/notice/vacant residents', 
            'current/notice/vacant',
            'current residents',
            'current / notice / vacant residents'  # Handle spaces in markers
        ],
        'specific_patterns': [
            # CSV patterns
            'unit,unit type,sq ft,resident,name',
            'unit,unit type,unit,resident,name',  # Sometimes unit appears twice
            # Excel patterns (might appear with spaces or tabs)
            r'unit\s+unit\s+type\s+sq\s+ft\s+resident',
            r'unit.*unit\s+type.*resident',  # More flexible pattern
            # Section markers
            'current/notice/vacant residents',
            'current / notice / vacant residents',
            'summary groups',
            'future residents/applicants',
            'future residents / applicants',
            # Report titles
            'rent roll with lease charges',
            'rent roll report',
            # Yardi-specific terms
            'charge code',
            'resident deposit',
            'other deposit',
            'move-in',  # Yardi uses hyphen
            'lease expiration'
        ],
        'column_patterns': [  # New: patterns to look for in column names
            'charge_code',
            'resident_deposit',
            'other_deposit',
            'move_in',
            'lease_expiration'
        ],
        'date_format': '%m/%d/%Y',
        'currency_format': 'parentheses',
        'has_subtotals': True,
        'has_two_row_header': True,
        'column_mappings': {
            'unit_sq_ft': 'sq_ft',
            'unit_sqft': 'sq_ft',
            'name': 'resident_name',
            'resident': 'resident_code'
        }
    },
    'realpage': {
        'identifiers': ['realpage', 'onesite', 'site name', 'property name'],
        'header_markers': ['unit', 'resident', 'lease'],
        'section_markers': [
            'resident list',
            'current residents',
            'occupied units'
        ],
        'specific_patterns': [
            'unit,floorplan,resident,lease start,lease end',
            'unit,floor plan,resident',
            'property summary report',
            'resident aging summary',
            'unit status report',
            'lease start',
            'lease end',
            'floor plan'
        ],
        'column_patterns': [
            'floorplan',
            'floor_plan',
            'lease_start',
            'lease_end'
        ],
        'date_format': '%Y-%m-%d',
        'currency_format': 'minus',
        'has_subtotals': True,
        'has_two_row_header': True,
        'column_mappings': {
            'apartment': 'unit',
            'tenant': 'resident_name',
            'lease_start': 'move_in',
            'lease_end': 'lease_expiration',
            'square_feet': 'sq_ft',
            'floor_plan': 'unit_type',
            'floorplan': 'unit_type'
        }
    },
    'appfolio': {
        'identifiers': ['appfolio', 'property management', 'buildium'],
        'header_markers': ['unit', 'tenant', 'rent'],
        'section_markers': [
            'occupied units',
            'vacant units',
            'all units'
        ],
        'specific_patterns': [
            'unit,tenant,rent,security deposit',
            'unit,tenant,lease start,lease end,rent',
            'unit,building,tenant,rent',
            'security deposit',
            'pet deposit',
            'monthly rent'
        ],
        'column_patterns': [
            'tenant',
            'security_deposit',
            'pet_deposit',
            'monthly_rent'
        ],
        'date_format': '%m/%d/%Y',
        'currency_format': 'minus',
        'has_subtotals': False,
        'has_two_row_header': False,
        'column_mappings': {
            'tenant': 'resident_name',
            'rent': 'market_rent',
            'monthly_rent': 'market_rent',
            'security_deposit': 'resident_deposit',
            'lease_start': 'move_in',
            'lease_end': 'lease_expiration'
        }
    },
    'entrata': {
        'identifiers': ['entrata', 'lease summary', 'property solutions'],
        'header_markers': ['apartment', 'resident', 'lease term'],
        'section_markers': [
            'current residents',
            'notice residents',
            'vacant apartments'
        ],
        'specific_patterns': [
            'apartment,resident,lease from,lease to',
            'apartment,floor plan,resident',
            'unit status report',
            'resident summary',
            'lease from',
            'lease to',
            'notice date'
        ],
        'column_patterns': [
            'apartment',
            'lease_from',
            'lease_to',
            'notice_date'
        ],
        'date_format': '%m/%d/%Y',
        'currency_format': 'parentheses',
        'has_subtotals': True,
        'has_two_row_header': True,
        'column_mappings': {
            'apartment': 'unit',
            'resident': 'resident_name',
            'lease_from': 'move_in',
            'lease_to': 'lease_expiration',
            'unit_sqft': 'sq_ft'
        }
    },
    'mri': {
        'identifiers': ['mri', 'management reports', 'mri software'],
        'header_markers': ['unit code', 'tenant name', 'lease from'],
        'section_markers': [
            'residential units',
            'commercial units',
            'retail units'
        ],
        'specific_patterns': [
            'unit code,tenant name,lease from,lease to',
            'unit code,unit type,tenant',
            'property rent roll',
            'tenant name',
            'unit code'
        ],
        'column_patterns': [
            'unit_code',
            'tenant_name',
            'lease_from',
            'lease_to'
        ],
        'date_format': '%d-%b-%Y',
        'currency_format': 'minus',
        'has_subtotals': True,
        'has_two_row_header': False,
        'column_mappings': {
            'unit_code': 'unit',
            'tenant_name': 'resident_name',
            'lease_from': 'move_in',
            'lease_to': 'lease_expiration',
            'unit_area': 'sq_ft'
        }
    }
}


def clean_text_for_comparison(text: str) -> str:
    """
    Clean and normalize text for comparison.
    Handles various formatting issues that might prevent detection.
    """
    if pd.isna(text):
        return ''
    
    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s/\-,]', '', text)
    
    return text


def extract_text_from_dataframe(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Extract searchable text from DataFrame for format detection.
    Handles both header rows and data rows.
    """
    search_text_parts = []
    
    # Add column names (might be default like 0, 1, 2 if no headers)
    if df.columns is not None:
        col_text = ' '.join([clean_text_for_comparison(str(col)) for col in df.columns])
        search_text_parts.append(col_text)
    
    # Add first N rows - these often contain the real headers
    for idx in range(min(max_rows, len(df))):
        row_values = df.iloc[idx].values
        # Join non-null values
        row_text = ' '.join([clean_text_for_comparison(str(val)) for val in row_values if pd.notna(val)])
        search_text_parts.append(row_text)
        
        # Also create comma-separated version (for CSV pattern matching)
        row_csv = ','.join([clean_text_for_comparison(str(val)) for val in row_values if pd.notna(val)])
        search_text_parts.append(row_csv)
    
    return '\n'.join(search_text_parts)


def debug_detection(search_text: str, filename: str = None) -> Dict[str, Any]:
    """
    Debug function to show what patterns are being found.
    Returns detailed information about what was detected.
    """
    debug_info = {
        'filename': filename,
        'search_text_length': len(search_text),
        'search_text_preview': search_text[:500],
        'formats_checked': {},
        'patterns_found': []
    }
    
    # Check each format
    for format_name, profile in FORMAT_PROFILES.items():
        format_debug = {
            'score': 0,
            'matches': {
                'identifiers': [],
                'header_markers': [],
                'section_markers': [],
                'specific_patterns': [],
                'column_patterns': []
            }
        }
        
        # Check identifiers
        for identifier in profile.get('identifiers', []):
            if identifier in search_text:
                format_debug['matches']['identifiers'].append(identifier)
                format_debug['score'] += 3
        
        # Check header markers
        for marker in profile.get('header_markers', []):
            if marker in search_text:
                format_debug['matches']['header_markers'].append(marker)
                format_debug['score'] += 1
        
        # Check section markers
        for marker in profile.get('section_markers', []):
            clean_marker = clean_text_for_comparison(marker)
            if clean_marker in search_text:
                format_debug['matches']['section_markers'].append(marker)
                format_debug['score'] += 4
        
        # Check specific patterns
        for pattern in profile.get('specific_patterns', []):
            clean_pattern = clean_text_for_comparison(pattern)
            if clean_pattern in search_text:
                format_debug['matches']['specific_patterns'].append(pattern)
                format_debug['score'] += 5
                debug_info['patterns_found'].append(f"{format_name}: {pattern}")
        
        # Check column patterns
        for pattern in profile.get('column_patterns', []):
            if pattern in search_text:
                format_debug['matches']['column_patterns'].append(pattern)
                format_debug['score'] += 2
        
        debug_info['formats_checked'][format_name] = format_debug
    
    # Find best match
    best_format = max(debug_info['formats_checked'].items(), 
                     key=lambda x: x[1]['score'])
    debug_info['best_match'] = best_format[0]
    debug_info['best_score'] = best_format[1]['score']
    
    return debug_info


def detect_format(
    file_content: str = None, 
    df: pd.DataFrame = None, 
    filename: str = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Enhanced format detection with better Yardi recognition.
    
    Args:
        file_content: Raw file content as string
        df: DataFrame with parsed data (for Excel files)
        filename: Name of the uploaded file
        debug: If True, return detailed debugging information
    
    Returns:
        Dictionary with detected format, confidence score, and debug info
    """
    detection_result = {
        'format': 'generic',
        'confidence': 0,
        'profile': None,
        'detected_markers': [],
        'debug_info': None
    }
    
    # Build search text from all available sources
    search_text_parts = []
    
    # Add filename
    if filename:
        search_text_parts.append(clean_text_for_comparison(filename))
    
    # Add file content (for CSV files)
    if file_content:
        # Clean and add first portion of file content
        content_preview = file_content[:10000] if len(file_content) > 10000 else file_content
        search_text_parts.append(clean_text_for_comparison(content_preview))
    
    # Add DataFrame content (for Excel files)
    if df is not None and not df.empty:
        df_text = extract_text_from_dataframe(df, max_rows=30)  # Look at more rows
        search_text_parts.append(df_text)
    
    # Combine all search text
    search_text = '\n'.join(search_text_parts).lower()
    
    # Debug mode
    if debug:
        debug_info = debug_detection(search_text, filename)
        detection_result['debug_info'] = debug_info
        logger.info(f"Debug detection info: {debug_info}")
    
    # Track scores for each format
    format_scores = {}
    
    for format_name, profile in FORMAT_PROFILES.items():
        score = 0
        markers_found = []
        
        # 1. Check specific patterns FIRST (highest weight - most reliable)
        for pattern in profile.get('specific_patterns', []):
            clean_pattern = clean_text_for_comparison(pattern)
            # For regex patterns
            if pattern.startswith('r'):
                try:
                    if re.search(pattern[1:], search_text):
                        score += 5
                        markers_found.append(f"pattern: {pattern}")
                except:
                    pass
            # For exact patterns
            elif clean_pattern in search_text:
                score += 5
                markers_found.append(f"pattern: {pattern}")
        
        # 2. Check column patterns (good indicators)
        for col_pattern in profile.get('column_patterns', []):
            if col_pattern in search_text:
                score += 2
                markers_found.append(f"column: {col_pattern}")
        
        # 3. Check section markers (format-specific)
        for marker in profile.get('section_markers', []):
            clean_marker = clean_text_for_comparison(marker)
            if clean_marker in search_text:
                score += 4
                markers_found.append(f"section: {marker}")
        
        # 4. Check identifiers (sometimes in metadata)
        for identifier in profile.get('identifiers', []):
            if identifier in search_text:
                score += 3
                markers_found.append(f"identifier: {identifier}")
        
        # 5. Check header markers (generic - lowest weight)
        for marker in profile.get('header_markers', []):
            clean_marker = clean_text_for_comparison(marker)
            if clean_marker in search_text:
                score += 1
                markers_found.append(f"header: {marker}")
        
        format_scores[format_name] = {
            'score': score,
            'markers': markers_found
        }
    
    # Find best match
    best_match = max(format_scores.items(), key=lambda x: x[1]['score'])
    best_format = best_match[0]
    best_score = best_match[1]['score']
    best_markers = best_match[1]['markers']
    
    # Set threshold for detection
    MIN_DETECTION_SCORE = 3
    
    if best_score >= MIN_DETECTION_SCORE:
        detection_result['format'] = best_format
        detection_result['profile'] = FORMAT_PROFILES[best_format]
        detection_result['detected_markers'] = best_markers
        # Calculate confidence (normalize to 0-100)
        detection_result['confidence'] = min(100, int(best_score * 5))
    else:
        # If no format detected with confidence, still return the best guess
        # but with low confidence
        if best_score > 0:
            detection_result['format'] = best_format
            detection_result['profile'] = FORMAT_PROFILES[best_format]
            detection_result['detected_markers'] = best_markers
            detection_result['confidence'] = min(30, int(best_score * 5))
    
    # Log the detection results
    logger.info(f"Format detection result: {detection_result['format']} (confidence: {detection_result['confidence']}%)")
    logger.info(f"Markers found: {detection_result['detected_markers'][:5]}")  # Show first 5 markers
    
    # Log all format scores for debugging
    logger.info("All format scores:")
    for fmt, data in format_scores.items():
        logger.info(f"  {fmt}: score={data['score']}, markers={len(data['markers'])}")
    
    return detection_result


def detect_format_from_columns(columns: List[str]) -> Dict[str, Any]:
    """
    Alternative detection method based solely on column names.
    Useful when file content isn't available.
    """
    # Clean column names
    clean_columns = [clean_text_for_comparison(col) for col in columns]
    columns_text = ' '.join(clean_columns)
    
    format_scores = {}
    
    for format_name, profile in FORMAT_PROFILES.items():
        score = 0
        
        # Check if column mappings match
        for original, mapped in profile.get('column_mappings', {}).items():
            if original in columns_text:
                score += 2
        
        # Check column patterns
        for pattern in profile.get('column_patterns', []):
            if pattern in columns_text:
                score += 3
        
        format_scores[format_name] = score
    
    best_format = max(format_scores.items(), key=lambda x: x[1])
    
    if best_format[1] > 0:
        return {
            'format': best_format[0],
            'confidence': min(100, best_format[1] * 10),
            'profile': FORMAT_PROFILES[best_format[0]]
        }
    
    return {
        'format': 'generic',
        'confidence': 0,
        'profile': None
    }


def get_date_parser(format_name: str):
    """
    Returns the appropriate date parser for the detected format.
    """
    if format_name in FORMAT_PROFILES:
        date_format = FORMAT_PROFILES[format_name]['date_format']
        return lambda x: pd.to_datetime(x, format=date_format, errors='coerce')
    return lambda x: pd.to_datetime(x, errors='coerce', infer_datetime_format=True)


def get_currency_cleaner(format_name: str):
    """
    Returns the appropriate currency cleaning function for the detected format.
    """
    if format_name in FORMAT_PROFILES:
        currency_format = FORMAT_PROFILES[format_name].get('currency_format', 'minus')
        
        if currency_format == 'parentheses':
            # Handle (100) as -100
            def clean_currency(value):
                if pd.isna(value):
                    return value
                value_str = str(value).strip()
                value_str = value_str.replace('$', '').replace(',', '')
                if value_str.startswith('(') and value_str.endswith(')'):
                    return -float(value_str[1:-1])
                try:
                    return float(value_str) if value_str else 0
                except:
                    return 0
            return clean_currency
    
    # Default cleaner
    def default_clean(value):
        if pd.isna(value):
            return value
        value_str = str(value).strip()
        value_str = value_str.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
        try:
            return float(value_str) if value_str else 0
        except:
            return 0
    
    return default_clean


# Test function for debugging
def test_yardi_detection():
    """
    Test function to verify Yardi detection is working.
    """
    # Sample Yardi headers
    test_cases = [
        "Unit,Unit Type,Unit,Sq Ft,Resident,Name,Market Rent",
        "Current/Notice/Vacant Residents",
        "unit unit type sq ft resident name",
        "Rent Roll with Lease Charges"
    ]
    
    for test in test_cases:
        result = detect_format(file_content=test)
        print(f"Test: {test[:50]}...")
        print(f"Result: {result['format']} (confidence: {result['confidence']}%)")
        print()
