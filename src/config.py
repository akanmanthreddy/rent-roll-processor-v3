"""
Central configuration module for rent roll processor.
All configurable settings and mappings in one place.
"""

# ============================================================================
# CORE COLUMN DEFINITIONS
# ============================================================================

# Columns that define a primary unit record
PRIMARY_RECORD_COLS = [
    'unit', 'unit_type', 'sq_ft', 'resident_code', 'resident_name',
    'market_rent', 'resident_deposit', 'other_deposit', 'move_in',
    'lease_expiration', 'move_out', 'balance'
]

# Columns that should NOT be forward-filled
NO_FORWARD_FILL_COLS = ['move_out']

# Date columns for parsing
DATE_COLUMNS = ['move_in', 'move_out', 'lease_expiration']

# Currency columns for cleaning
CURRENCY_COLUMNS = ['market_rent', 'amount', 'balance', 'resident_deposit', 'other_deposit']

# ============================================================================
# FORMAT DETECTION PROFILES
# ============================================================================

FORMAT_PROFILES = {
    'yardi': {
        'identifiers': ['yardi', 'voyager', 'rent roll report'],
        'header_markers': ['unit', 'unit type', 'resident'],
        'section_markers': ['current/notice/vacant residents', 'current residents'],
        'specific_patterns': [
            'unit,unit type,sq ft,resident,name',
            'current/notice/vacant residents',
            'summary groups',
            'future residents/applicants'
        ],
        'date_format': '%m/%d/%Y',
        'currency_format': 'parentheses',
        'has_subtotals': True,
        'has_two_row_header': True,
        'column_mappings': {
            'unit_sq_ft': 'sq_ft',
            'unit_sqft': 'sq_ft',
            'resident': 'resident_code',
            'name': 'resident_name',
            'lease_from': 'move_in',
            'lease_to': 'lease_expiration'
        }
    },
    'realpage': {
        'identifiers': ['realpage', 'onesite', 'site name'],
        'header_markers': ['unit', 'resident', 'lease'],
        'section_markers': ['resident list', 'current residents'],
        'specific_patterns': [
            'unit,floorplan,resident,lease start,lease end',
            'property summary report',
            'resident aging summary'
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
            'square_feet': 'sq_ft'
        }
    },
    'appfolio': {
        'identifiers': ['appfolio', 'property management'],
        'header_markers': ['unit', 'tenant', 'rent'],
        'section_markers': ['occupied units', 'vacant units'],
        'specific_patterns': [
            'unit,tenant,rent,security deposit',
            'unit,tenant,lease start,lease end,rent',
            'occupied units',
            'vacant units'
        ],
        'date_format': '%m/%d/%Y',
        'currency_format': 'minus',
        'has_subtotals': False,
        'has_two_row_header': False,
        'column_mappings': {
            'tenant': 'resident_name',
            'rent': 'market_rent',
            'security_deposit': 'resident_deposit',
            'lease_start': 'move_in',
            'lease_end': 'lease_expiration'
        }
    },
    'entrata': {
        'identifiers': ['entrata', 'lease summary'],
        'header_markers': ['apartment', 'resident', 'lease term'],
        'section_markers': ['current residents', 'notice residents'],
        'specific_patterns': [
            'apartment,resident,lease from,lease to',
            'unit status report',
            'resident summary'
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
        'identifiers': ['mri', 'management reports'],
        'header_markers': ['unit code', 'tenant name', 'lease from'],
        'section_markers': ['residential units', 'commercial units'],
        'specific_patterns': [
            'unit code,tenant name,lease from,lease to',
            'property rent roll',
            'residential units'
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

# Default format profile for unrecognized systems
DEFAULT_FORMAT_PROFILE = {
    'identifiers': [],
    'header_markers': ['unit', 'resident', 'rent'],
    'section_markers': ['current', 'vacant'],
    'specific_patterns': [],
    'date_format': None,  # Will infer
    'currency_format': 'minus',
    'has_subtotals': False,
    'has_two_row_header': True,
    'column_mappings': {}
}

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

VALIDATION_THRESHOLDS = {
    'max_reasonable_rent': 50000,  # Flag rents over $50k/month
    'max_reasonable_sqft': 10000,  # Flag units over 10,000 sq ft
    'min_reasonable_sqft': 100,    # Flag units under 100 sq ft
    'data_quality_penalties': {
        'missing_required_columns': 25,
        'missing_unit_numbers': 20,
        'negative_market_rent': 10,
        'invalid_sqft': 10,
        'date_mismatch': 15,
        'duplicate_units': 5,
        'unrealistic_values': 5
    }
}

# Required columns for validation
REQUIRED_COLUMNS = ['unit', 'unit_type', 'sq_ft', 'market_rent']

# ============================================================================
# EXPORT SETTINGS
# ============================================================================

EXPORT_SETTINGS = {
    'excel': {
        'engine': 'openpyxl',
        'include_index': False,
        'sheets': {
            'main': 'Rent Roll',
            'validation': 'Validation',
            'issues': 'Issues'
        }
    },
    'csv': {
        'index': False,
        'encoding': 'utf-8'
    },
    'json': {
        'orient': 'records',
        'indent': 2,
        'date_format': 'iso'
    }
}

# ============================================================================
# PARSING SETTINGS
# ============================================================================

# End-of-data markers (indicate summary sections)
DATA_END_MARKERS = [
    'summary groups',
    'future residents/applicants',
    'totals',
    'grand total',
    'property totals',
    'portfolio summary'
]

# Rows to filter out during processing
FILTER_PATTERNS = [
    r',,,,,',  # Visual separator rows
    'total',   # Subtotal rows
    'summary'  # Summary rows
]

# ============================================================================
# DETECTION SCORING WEIGHTS
# ============================================================================

DETECTION_WEIGHTS = {
    'specific_pattern': 5,  # Exact pattern matches
    'section_marker': 4,    # Section headers
    'identifier': 3,        # System names
    'header_marker': 1      # Generic headers
}

# Minimum score to consider a format detected
MIN_DETECTION_SCORE = 3

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'