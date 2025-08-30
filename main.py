import sys
print(f"Python version: {sys.version}", file=sys.stderr)
print("Starting imports...", file=sys.stderr)

import functions_framework
import json
import io
import logging
from flask import make_response
from datetime import datetime
import pandas as pd

print("Basic imports successful", file=sys.stderr)

# Import our modularized functions
try:
    from src.data_loader import load_and_prepare_dataframe
    from src.processing import process_rent_roll_vectorized
    print("Module imports successful", file=sys.stderr)
except Exception as e:
    print(f"Module import failed: {e}", file=sys.stderr)
    raise
from src.processing import process_rent_roll_vectorized

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Main HTTP Function ---

@functions_framework.http
def process_rent_roll_http(request):
    """
    HTTP Cloud Function to process an uploaded rent roll file (CSV or Excel).
    This function now acts as an orchestrator, calling specialized modules.
    """
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}

    try:
        if 'file' not in request.files:
            return (json.dumps({'error': 'No file part in the request'}), 400, headers)

        file = request.files['file']
        if file.filename == '':
            return (json.dumps({'error': 'No file selected for uploading'}), 400, headers)

        logger.info(f"Processing file: {file.filename}")
        file_buffer = io.BytesIO(file.read())
        
        # Log file size
        file_size = file_buffer.getbuffer().nbytes
        logger.info(f"File size: {file_size} bytes")
        
        if file_size == 0:
            return (json.dumps({'error': 'Empty file uploaded'}), 400, headers)

        # --- Main Processing Pipeline ---
        # Each step now calls a function from a dedicated module
        raw_df = load_and_prepare_dataframe(file_buffer, file.filename)
        logger.info(f"Raw DataFrame shape: {raw_df.shape}")
        logger.info(f"Raw DataFrame columns: {list(raw_df.columns)[:10]}")  # Log first 10 columns
        
        if raw_df.empty:
            return (json.dumps({'error': 'No data found in file after parsing'}), 400, headers)
        
        processed_df = process_rent_roll_vectorized(raw_df)
        logger.info(f"Processed DataFrame shape: {processed_df.shape}")
        # --- End of Pipeline ---

        if processed_df.empty:
            return (json.dumps({'error': 'No units found after processing', 'details': 'The file was parsed but no valid unit data was found'}), 400, headers)

        logger.info(f"Successfully processed {len(processed_df)} units.")

        # Prepare JSON response
        # Convert datetime columns to ISO format string for JSON serialization
        for col in processed_df.select_dtypes(include=['datetime64[ns]']).columns:
            processed_df[col] = processed_df[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)

        # Check if DataFrame has data before converting to JSON
        if len(processed_df) == 0:
            result_json = "[]"
        else:
            result_json = processed_df.to_json(orient='records', indent=2)

        response = make_response(result_json)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'  # FIXED: Proper header setting
        response.headers.update(headers)  # FIXED: Use update() instead of assignment
        return response

    except ValueError as ve:
        logger.error(f"Validation Error: {ve}", exc_info=True)
        return (json.dumps({'error': 'File format error', 'details': str(ve)}), 400, headers)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return (json.dumps({'error': 'Internal Server Error', 'details': str(e)}), 500, headers)
