"""
Data validation module for rent roll processing.
Identifies data quality issues and anomalies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def validate_rent_roll(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates rent roll data and returns issues found.
    
    Returns:
        Dictionary with 'errors' (critical issues) and 'warnings' (potential issues)
    """
    validation_results = {
        'errors': [],
        'warnings': [],
        'statistics': {},
        'data_quality_score': 100  # Start at 100, deduct for issues
    }
    
    # Check for required columns
    required_cols = ['unit', 'unit_type', 'sq_ft', 'market_rent']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation_results['errors'].append(f"Missing required columns: {', '.join(missing_cols)}")
        validation_results['data_quality_score'] -= 25
    
    # Check for data completeness
    if 'unit' in df.columns:
        if df['unit'].isna().any():
            validation_results['errors'].append("Some rows have missing unit numbers")
            validation_results['data_quality_score'] -= 20
    
    # Validate numeric fields
    if 'market_rent' in df.columns:
        negative_market_rent = df[df['market_rent'] < 0]
        if not negative_market_rent.empty:
            validation_results['warnings'].append(
                f"Units with negative market rent: {negative_market_rent['unit'].tolist()}"
            )
            validation_results['data_quality_score'] -= 10
            
        # Check for unrealistic rents (over $50,000/month)
        high_rent = df[df['market_rent'] > 50000]
        if not high_rent.empty:
            validation_results['warnings'].append(
                f"Units with unusually high rent (>$50k): {high_rent['unit'].tolist()}"
            )
            validation_results['data_quality_score'] -= 5
    
    # Validate square footage
    if 'sq_ft' in df.columns:
        # Check for negative or zero sq ft
        invalid_sqft = df[(df['sq_ft'] <= 0) | (df['sq_ft'].isna())]
        if not invalid_sqft.empty:
            validation_results['warnings'].append(
                f"Units with invalid sq ft: {invalid_sqft['unit'].tolist()}"
            )
            validation_results['data_quality_score'] -= 10
            
        # Check for unrealistic sq ft (over 10,000 for residential)
        huge_units = df[df['sq_ft'] > 10000]
        if not huge_units.empty:
            validation_results['warnings'].append(
                f"Units over 10,000 sq ft: {huge_units['unit'].tolist()}"
            )
    
    # Date validation
    date_cols = ['move_in', 'move_out', 'lease_expiration']
    for col in date_cols:
        if col in df.columns:
            # Check for future move-in dates (might be applicants)
            if col == 'move_in':
                future_movein = df[df[col] > pd.Timestamp.now()]
                if not future_movein.empty:
                    validation_results['warnings'].append(
                        f"Units with future move-in dates: {future_movein['unit'].tolist()}"
                    )
            
            # Check for move-out before move-in
            if 'move_in' in df.columns and 'move_out' in df.columns:
                date_mismatch = df[
                    df['move_out'].notna() & 
                    df['move_in'].notna() & 
                    (df['move_out'] < df['move_in'])
                ]
                if not date_mismatch.empty:
                    validation_results['errors'].append(
                        f"Units with move-out before move-in: {date_mismatch['unit'].tolist()}"
                    )
                    validation_results['data_quality_score'] -= 15
    
    # Check for duplicate units
    if 'unit' in df.columns:
        duplicates = df[df.duplicated(subset=['unit'], keep=False)]
        if not duplicates.empty:
            dup_units = duplicates['unit'].unique().tolist()
            validation_results['warnings'].append(
                f"Duplicate unit numbers found: {dup_units[:10]}..."  # Show first 10
            )
            validation_results['data_quality_score'] -= 5
    
    # Calculate statistics
    if 'occupancy_status' in df.columns:
        total_units = int(len(df))
        vacant_units = int((df['occupancy_status'] == 'Vacant').sum())
        occupied_units = int((df['occupancy_status'] == 'Occupied').sum())
        
        validation_results['statistics'] = {
            'total_units': total_units,
            'occupied_units': occupied_units,
            'vacant_units': vacant_units,
            'occupancy_rate': round(occupied_units / total_units * 100, 2) if total_units > 0 else 0
        }
    
    # Add rent statistics if available
    if 'market_rent' in df.columns:
        validation_results['statistics']['avg_market_rent'] = float(round(df['market_rent'].mean(), 2))
        validation_results['statistics']['total_market_rent'] = float(round(df['market_rent'].sum(), 2))
    
    # Ensure score doesn't go below 0
    validation_results['data_quality_score'] = max(0, validation_results['data_quality_score'])
    
    logger.info(f"Validation complete. Score: {validation_results['data_quality_score']}/100")
    
    return validation_results


def generate_validation_summary(validation_results: Dict[str, Any]) -> str:
    """
    Generates a human-readable summary of validation results.
    """
    summary = []
    
    summary.append(f"Data Quality Score: {validation_results['data_quality_score']}/100")
    
    if validation_results['errors']:
        summary.append("\n⚠️ CRITICAL ISSUES:")
        for error in validation_results['errors']:
            summary.append(f"  • {error}")
    
    if validation_results['warnings']:
        summary.append("\n⚡ WARNINGS:")
        for warning in validation_results['warnings']:
            summary.append(f"  • {warning}")
    
    if validation_results['statistics']:
        summary.append("\n📊 STATISTICS:")
        for key, value in validation_results['statistics'].items():
            label = key.replace('_', ' ').title()
            if 'rate' in key:
                summary.append(f"  • {label}: {value}%")
            elif 'rent' in key:
                summary.append(f"  • {label}: ${value:,.2f}")
            else:
                summary.append(f"  • {label}: {value}")
    
    return '\n'.join(summary)
