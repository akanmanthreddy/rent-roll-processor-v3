"""
Format detection module for different property management systems.
Identifies the source system and applies appropriate parsing rules.
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Format profiles for different property management systems
FORMAT_PROFILES = {
    'yardi': {
        'identifiers': ['yardi', 'voyager', 'rent roll report', 'resident ar'],
        'header_markers': ['unit', 'unit type', 'resident'],
        'section_markers': ['current/notice/vacant residents', 'current residents'],
        'date_format': '%m/%d/%Y',
        'currency_format': 'parentheses',  # Negatives shown as (100)
        'has_subtotals': True
    },
    'realpage': {
        'identifiers': ['realpage', 'onesite', 'site name', 'property summary'],
        'header_markers': ['unit', 'resident', 'lease'],
        'section_markers': ['resident list', 'current residents'],
        'date_format': '%Y-%m-%d',
        'currency_format': 'minus',  # Negatives shown as -100
        'has_subtotals': True
    },
    'appfolio': {
        'identifiers': ['appfolio', 'property management', 'rent roll'],
        'header_markers': ['unit', 'tenant', 'rent'],
        'section_markers': ['occupied units', 'vacant units'],
        'date_format': '%m/%d/%Y',
        'currency_format': 'minus',
        'has_subtotals': False
    },
    'entrata': {
        'identifiers': ['entrata', 'lease summary', 'resident summary'],
        'header_markers': ['apartment', 'resident', 'lease term'],
        'section_markers': ['current residents', 'notice residents'],
        'date_format': '%m/%d/%Y',
        'currency_format': 'parentheses',
        'has_subtotals': True
    },
    'mri': {
        'identifiers': ['mri', 'management reports', 'rent roll'],
        'header_markers': ['unit code', 'tenant name', 'lease from'],
        'section_markers': ['residential units', 'commercial units'],
        'date_format': '%d-%b-%Y',
        'currency_format': 'minus',
        'has_subtotals': True
    }
}


def detect_format(file_content: str = None, df: pd.DataFrame = None, filename: str = None) -> Dict[str, Any]:
    """
    Detects the property management system format.
    
    Args:
        file_content: Raw file content as string
        df: DataFrame with parsed data
        filename: Name of the uploaded file
    
    Returns:
        Dictionary with detected format and confidence score
    """
    detection_result = {
        'format': 'generic',
        'confidence': 0,
        'profile': None,
        'detected_markers': []
    }
    
    # Convert everything to lowercase for comparison
    search_text = ""
    
    if file_content:
        search_text += file_content.lower()[:5000]  # Check first 5000 chars
    
    if df is not None and not df.empty:
        # Add column names
        search_text += ' '.join(df.columns.astype(str).str.lower())
        # Add first few rows
        if len(df) > 0:
            search_text += ' '.join(df.head(10).astype(str).values.flatten())
    
    if filename:
        search_text += ' ' + filename.lower()
    
    # Check each format profile
    best_match = None
    best_score = 0
    
    for format_name, profile in FORMAT_PROFILES.items():
        score = 0
        markers_found = []
        
        # Check identifiers (weighted heavily)
        for identifier in profile['identifiers']:
            if identifier in search_text:
                score += 3
                markers_found.append(identifier)
        
        # Check header markers
        for marker in profile['header_markers']:
            if marker in search_text:
                score += 1
                markers_found.append(marker)
        
        # Check section markers
        for marker in profile['section_markers']:
            if marker in search_text:
                score += 2
                markers_found.append(marker)
        
        if score > best_score:
            best_score = score
            best_match = format_name
            detection_result['detected_markers'] = markers_found
    
    # Set results based on best match
    if best_match and best_score >= 3:  # Minimum threshold
        detection_result['format'] = best_match
        detection_result['profile'] = FORMAT_PROFILES[best_match]
        # Calculate confidence (max score would be about 15-20)
        detection_result['confidence'] = min(100, int(best_score * 10))
    
    logger.info(f"Detected format: {detection_result['format']} (confidence: {detection_result['confidence']}%)")
    
    return detection_result


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
                return float(value_str) if value_str else 0
            return clean_currency
    
    # Default cleaner
    def default_clean(value):
        if pd.isna(value):
            return value
        value_str = str(value).strip()
        value_str = value_str.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
        return float(value_str) if value_str else 0
    
    return default_clean