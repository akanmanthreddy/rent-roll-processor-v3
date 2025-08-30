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
        'identifiers': ['yardi', 'voyager', 'rent roll report'],
        'header_markers': ['unit', 'unit type', 'resident'],
        'section_markers': ['current/notice/vacant residents', 'current residents'],
        'specific_patterns': [
            'unit,unit type,sq ft,resident,name',  # Exact Yardi CSV header
            'current/notice/vacant residents',  # Unique to Yardi
            'summary groups',  # Yardi summary section
            'future residents/applicants'  # Yardi future section
        ],
        'date_format': '%m/%d/%Y',
        'currency_format': 'parentheses',
        'has_subtotals': True
    },
    'realpage': {
        'identifiers': ['realpage', 'onesite', 'site name'],
        'header_markers': ['unit', 'resident', 'lease'],
        'section_markers': ['resident list', 'current residents'],
        'specific_patterns': [
            'unit,floorplan,resident,lease start,lease end',  # RealPage pattern
            'property summary report',
            'resident aging summary',
            'unit,resident,move-in,move-out,lease'
        ],
        'date_format': '%Y-%m-%d',
        'currency_format': 'minus',
        'has_subtotals': True
    },
    'appfolio': {
        'identifiers': ['appfolio', 'property management'],
        'header_markers': ['unit', 'tenant', 'rent'],
        'section_markers': ['occupied units', 'vacant units'],
        'specific_patterns': [
            'unit,tenant,rent,security deposit',  # AppFolio pattern
            'unit,tenant,lease start,lease end,rent',
            'occupied units',
            'vacant units',
            'unit,building,tenant,rent'
        ],
        'date_format': '%m/%d/%Y',
        'currency_format': 'minus',
        'has_subtotals': False
    },
    'entrata': {
        'identifiers': ['entrata', 'lease summary'],
        'header_markers': ['apartment', 'resident', 'lease term'],
        'section_markers': ['current residents', 'notice residents'],
        'specific_patterns': [
            'apartment,resident,lease from,lease to',  # Entrata pattern
            'unit status report',
            'resident summary',
            'apartment,floorplan,resident'
        ],
        'date_format': '%m/%d/%Y',
        'currency_format': 'parentheses',
        'has_subtotals': True
    },
    'mri': {
        'identifiers': ['mri', 'management reports'],
        'header_markers': ['unit code', 'tenant name', 'lease from'],
        'section_markers': ['residential units', 'commercial units'],
        'specific_patterns': [
            'unit code,tenant name,lease from,lease to',  # MRI pattern
            'property rent roll',
            'unit code,unit type,tenant',
            'residential units'
        ],
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
        
        # Check specific patterns FIRST (highest weight - these are most reliable)
        if 'specific_patterns' in profile:
            for pattern in profile['specific_patterns']:
                if pattern in search_text:
                    score += 5  # High weight for exact patterns
                    markers_found.append(f"pattern: {pattern}")
        
        # Check identifiers (medium weight - sometimes present)
        for identifier in profile['identifiers']:
            if identifier in search_text:
                score += 3
                markers_found.append(identifier)
        
        # Check section markers (medium-high weight - format-specific)
        for marker in profile['section_markers']:
            if marker in search_text:
                score += 4
                markers_found.append(f"section: {marker}")
        
        # Check header markers (low weight - too generic)
        for marker in profile['header_markers']:
            if marker in search_text:
                score += 1
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
