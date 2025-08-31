"""
Data validation module for rent roll processing.
Fixed to handle missing columns gracefully.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def validate_rent_roll(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates rent roll data and returns issues found.
    Now handles missing columns gracefully.
    
    Returns:
        Dictionary with 'errors' (critical issues) and 'warnings' (potential issues)
    """
    validation_results = {
        'errors': [],
        'warnings': [],
        'statistics': {},
        'data_quality_score': 100  # Start at 100, deduct for issues
    }
    
    penalties = VALIDATION_THRESHOLDS['data_quality_penalties']
    
    # Log available columns for debugging
    logger.info(f"Columns available for validation: {list(df.columns)[:20]}")
    
    # Check for required columns (but don't fail if missing)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        validation_results['warnings'].append(f"Missing expected columns: {', '.join(missing_cols)}")
        # Reduce penalty for missing columns since Yardi format might name them differently
        validation_results['data_quality_score'] -= min(penalties['missing_required_columns'], 10)
        logger.warning(f"Missing columns: {missing_cols}")
    
    # Check for data completeness
    if 'unit' in df.columns:
        if df['unit'].isna().any():
            validation_results['errors'].append("Some rows have missing unit numbers")
            validation_results['data_quality_score'] -= penalties['missing_unit_numbers']
        
        # Check for duplicate units
        duplicates = df[df.duplicated(subset=['unit'], keep=False)]
        if not duplicates.empty:
            dup_units = duplicates['unit'].unique().tolist()
            validation_results['warnings'].append(
                f"Duplicate unit numbers found: {dup_units[:10]}..."  # Show first 10
            )
            validation_results['data_quality_score'] -= penalties['duplicate_units']
    else:
        validation_results['errors'].append("No 'unit' column found")
        validation_results['data_quality_score'] -= penalties['missing_required_columns']
    
    # Validate numeric fields - ONLY if they exist
    if 'market_rent' in df.columns:
        try:
            negative_market_rent = df[df['market_rent'] < 0]
            if not negative_market_rent.empty:
                validation_results['warnings'].append(
                    f"Units with negative market rent: {negative_market_rent['unit'].tolist() if 'unit' in negative_market_rent.columns else 'Multiple units'}"
                )
                validation_results['data_quality_score'] -= penalties['negative_market_rent']
            
            # Check for unrealistic rents
            high_rent = df[df['market_rent'] > VALIDATION_THRESHOLDS['max_reasonable_rent']]
            if not high_rent.empty:
                validation_results['warnings'].append(
                    f"Units with unusually high rent (>${VALIDATION_THRESHOLDS['max_reasonable_rent']}): {high_rent['unit'].tolist()[:5] if 'unit' in high_rent.columns else 'Multiple units'}"
                )
                validation_results['data_quality_score'] -= penalties['unrealistic_values']
        except Exception as e:
            logger.warning(f"Error validating market_rent: {e}")
    
    # Validate square footage - ONLY if column exists
    if 'sq_ft' in df.columns:
        try:
            # Convert to numeric first if needed
            if df['sq_ft'].dtype == 'object':
                df['sq_ft'] = pd.to_numeric(df['sq_ft'], errors='coerce')
            
            # Check for invalid sq ft
            invalid_sqft = df[
                (df['sq_ft'] < VALIDATION_THRESHOLDS['min_reasonable_sqft']) | 
                (df['sq_ft'].isna())
            ]
            if not invalid_sqft.empty and len(invalid_sqft) > 0:
                unit_list = invalid_sqft['unit'].tolist()[:5] if 'unit' in invalid_sqft.columns else []
                validation_results['warnings'].append(
                    f"Units with invalid sq ft (<{VALIDATION_THRESHOLDS['min_reasonable_sqft']}): {unit_list}"
                )
                validation_results['data_quality_score'] -= min(penalties['invalid_sqft'], 5)
            
            # Check for unrealistic sq ft
            huge_units = df[df['sq_ft'] > VALIDATION_THRESHOLDS['max_reasonable_sqft']]
            if not huge_units.empty:
                unit_list = huge_units['unit'].tolist()[:5] if 'unit' in huge_units.columns else []
                validation_results['warnings'].append(
                    f"Units over {VALIDATION_THRESHOLDS['max_reasonable_sqft']} sq ft: {unit_list}"
                )
        except Exception as e:
            logger.warning(f"Error validating sq_ft: {e}")
    else:
        logger.info("sq_ft column not found, skipping square footage validation")
    
    # Date validation - with error handling
    date_cols = ['move_in', 'move_out', 'lease_expiration']
    for col in date_cols:
        if col in df.columns:
            try:
                # Check for future move-in dates (might be applicants)
                if col == 'move_in':
                    future_movein = df[df[col] > pd.Timestamp.now()]
                    if not future_movein.empty:
                        unit_list = future_movein['unit'].tolist()[:5] if 'unit' in future_movein.columns else []
                        validation_results['warnings'].append(
                            f"Units with future move-in dates: {unit_list}"
                        )
                
                # Check for move-out before move-in
                if 'move_in' in df.columns and 'move_out' in df.columns:
                    date_mismatch = df[
                        df['move_out'].notna() & 
                        df['move_in'].notna() & 
                        (df['move_out'] < df['move_in'])
                    ]
                    if not date_mismatch.empty:
                        unit_list = date_mismatch['unit'].tolist()[:5] if 'unit' in date_mismatch.columns else []
                        validation_results['errors'].append(
                            f"Units with move-out before move-in: {unit_list}"
                        )
                        validation_results['data_quality_score'] -= penalties['date_mismatch']
            except Exception as e:
                logger.warning(f"Error validating date column {col}: {e}")
    
    # Calculate statistics
    try:
        total_units = int(len(df))
        
        # Determine occupancy based on available columns
        if 'occupancy_status' in df.columns:
            vacant_units = int((df['occupancy_status'] == 'Vacant').sum())
            occupied_units = int((df['occupancy_status'] == 'Occupied').sum())
        elif 'resident_name' in df.columns:
            # If no occupancy_status, check resident_name
            vacant_units = int(df['resident_name'].isna().sum())
            occupied_units = total_units - vacant_units
        elif 'resident_code' in df.columns:
            # Or check resident_code
            vacant_units = int(df['resident_code'].isna().sum())
            occupied_units = total_units - vacant_units
        else:
            # Can't determine occupancy
            vacant_units = 0
            occupied_units = 0
        
        validation_results['statistics'] = {
            'total_units': total_units,
            'occupied_units': occupied_units,
            'vacant_units': vacant_units,
            'occupancy_rate': round(occupied_units / total_units * 100, 2) if total_units > 0 else 0
        }
        
        # Add rent statistics if available
        if 'market_rent' in df.columns:
            try:
                validation_results['statistics']['avg_market_rent'] = float(round(df['market_rent'].mean(), 2))
                validation_results['statistics']['total_market_rent'] = float(round(df['market_rent'].sum(), 2))
            except:
                pass
        
        # Add actual rent if available (from charge codes)
        if 'rent' in df.columns:
            try:
                validation_results['statistics']['avg_actual_rent'] = float(round(df['rent'].mean(), 2))
                validation_results['statistics']['total_actual_rent'] = float(round(df['rent'].sum(), 2))
            except:
                pass
        
        # Add square footage statistics if available
        if 'sq_ft' in df.columns:
            try:
                sq_ft_numeric = pd.to_numeric(df['sq_ft'], errors='coerce')
                validation_results['statistics']['avg_sq_ft'] = float(round(sq_ft_numeric.mean(), 2))
                validation_results['statistics']['total_sq_ft'] = float(round(sq_ft_numeric.sum(), 2))
            except:
                pass
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
    
    # Ensure score doesn't go below 0
    validation_results['data_quality_score'] = max(0, validation_results['data_quality_score'])
    
    # Add column info for debugging
    validation_results['available_columns'] = list(df.columns)
    validation_results['row_count'] = len(df)
    
    logger.info(f"Validation complete. Score: {validation_results['data_quality_score']}/100")
    logger.info(f"Total units: {validation_results['statistics'].get('total_units', 'Unknown')}")
    
    return validation_results


def generate_validation_summary(validation_results: Dict[str, Any]) -> str:
    """
    Generates a human-readable summary of validation results.
    """
    summary = []
    
    summary.append(f"Data Quality Score: {validation_results['data_quality_score']}/100")
    summary.append(f"Total Rows Processed: {validation_results.get('row_count', 'Unknown')}")
    
    if validation_results['errors']:
        summary.append("\n⚠️ CRITICAL ISSUES:")
        for error in validation_results['errors']:
            summary.append(f"  • {error}")
    
    if validation_results['warnings']:
        summary.append("\n⚡ WARNINGS:")
        for warning in validation_results['warnings'][:10]:  # Limit to first 10
            summary.append(f"  • {warning}")
        if len(validation_results['warnings']) > 10:
            summary.append(f"  • ... and {len(validation_results['warnings']) - 10} more warnings")
    
    if validation_results['statistics']:
        summary.append("\n📊 STATISTICS:")
        for key, value in validation_results['statistics'].items():
            label = key.replace('_', ' ').title()
            if 'rate' in key or 'pct' in key:
                summary.append(f"  • {label}: {value}%")
            elif 'rent' in key or 'price' in key or 'cost' in key:
                summary.append(f"  • {label}: ${value:,.2f}")
            elif isinstance(value, float):
                summary.append(f"  • {label}: {value:,.2f}")
            else:
                summary.append(f"  • {label}: {value}")
    
    # Add column info if available
    if 'available_columns' in validation_results:
        summary.append(f"\n📋 COLUMNS FOUND ({len(validation_results['available_columns'])} total):")
        # Show first 10 columns
        cols_to_show = validation_results['available_columns'][:10]
        summary.append(f"  {', '.join(cols_to_show)}")
        if len(validation_results['available_columns']) > 10:
            summary.append(f"  ... and {len(validation_results['available_columns']) - 10} more columns")
    
    return '\n'.join(summary)
