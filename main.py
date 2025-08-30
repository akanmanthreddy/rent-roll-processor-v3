import functions_framework
import json
import io
import logging
from flask import make_response
import pandas as pd

from src.data_loader import load_and_prepare_dataframe
from src.processing import process_rent_roll_vectorized
from src.validator import validate_rent_roll, generate_validation_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@functions_framework.http
def process_rent_roll_http(request):
    """
    HTTP Cloud Function to process an uploaded rent roll file.
    Accepts CSV or Excel files and returns processed JSON data with validation.
    """
    
    # Handle CORS preflight requests
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
        # Check if this is a validation-only request
        validate_only = request.args.get('validate_only', 'false').lower() == 'true'
        include_validation = request.args.get('include_validation', 'true').lower() == 'true'
        
        # Validate request has a file
        if 'file' not in request.files:
            return (json.dumps({'error': 'No file part in the request'}), 400, headers)

        file = request.files['file']
        if file.filename == '':
            return (json.dumps({'error': 'No file selected for uploading'}), 400, headers)

        logger.info(f"Processing file: {file.filename}")
        
        # Read file into memory
        file_buffer = io.BytesIO(file.read())
        
        # Process the file
        raw_df = load_and_prepare_dataframe(file_buffer, file.filename)
        processed_df = process_rent_roll_vectorized(raw_df)
        
        if processed_df.empty:
            return (json.dumps({'error': 'No valid data found after processing'}), 400, headers)
        
        # Run validation
        validation_results = validate_rent_roll(processed_df)
        
        # If validation-only mode, return just the validation results
        if validate_only:
            validation_summary = generate_validation_summary(validation_results)
            response_data = {
                'validation': validation_results,
                'summary': validation_summary
            }
            response = make_response(json.dumps(response_data, indent=2))
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            response.headers.update(headers)
            return response
        
        logger.info(f"Successfully processed {len(processed_df)} units.")

        # Convert datetime columns for JSON serialization
        for col in processed_df.select_dtypes(include=['datetime64[ns]']).columns:
            processed_df[col] = processed_df[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)

        # Prepare response
        if include_validation:
            # Include both data and validation results
            response_data = {
                'data': json.loads(processed_df.to_json(orient='records')),
                'validation': validation_results,
                'row_count': len(processed_df),
                'column_count': len(processed_df.columns)
            }
        else:
            # Just return the data (original behavior)
            response_data = json.loads(processed_df.to_json(orient='records'))
        
        response = make_response(json.dumps(response_data, indent=2))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.headers.update(headers)
        return response

    except ValueError as ve:
        logger.error(f"Validation Error: {ve}", exc_info=True)
        return (json.dumps({'error': 'File format error', 'details': str(ve)}), 400, headers)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return (json.dumps({'error': 'Internal Server Error', 'details': str(e)}), 500, headers)
