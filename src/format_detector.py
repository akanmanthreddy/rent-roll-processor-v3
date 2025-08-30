"""
Format detection module for different property management systems.
Identifies the source system and applies appropriate parsing rules.
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional
from src.config import FORMAT_PROFILES, DETECTION_WEIGHTS, MIN_DETECTION_SCORE

logger = logging.getLogger(__name__)


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
                    score += DETECTION_WEIGHTS['specific_pattern']
                    markers_found.append(f"pattern: {pattern}")
        
        # Check section markers (format-specific)
        for marker in profile['section_markers']:
            if marker in search_text:
                score += DETECTION_WEIGHTS['section_marker']
                markers_found.append(f"section: {marker}")
        
        # Check identifiers (sometimes present)
        for identifier in profile['identifiers']:
            if identifier in search_text:
                score += DETECTION_WEIGHTS['identifier']
                markers_found.append(identifier)
        
        # Check header markers (generic)
        for marker in profile['header_markers']:
            if marker in search_text:
                score += DETECTION_WEIGHTS['header_marker']
                markers_found.append(marker)
        
        if score > best_score:
            best_score = score
            best_match = format_name
            detection_result['detected_markers'] = markers_found
    
    # Set results based on best match
    if best_match and best_score >= MIN_DETECTION_SCORE:
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
        if date_format:
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
