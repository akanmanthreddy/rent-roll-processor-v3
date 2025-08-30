import functions_framework
import json
import io
import logging
from flask import make_response
import pandas as pd

from src.data_loader import load_and_prepare_dataframe
from src.processing import process_rent_roll_vectorized
from src.validator import validate_rent_roll, generate_validation_summary
from src.format_detector import detect_format

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
    Accepts CSV or Excel files and returns processed data in multiple formats.
    
    Query parameters:
    - format: json (default), csv, excel
    - validate_only: true/false (default false)
    - include_validation: true/false (default true)
    - detect_format: true/false (default true)
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
        # Parse query parameters
        export_format = request.args.get('format', 'json').lower()
        validate_only = request.args.get('validate_only', 'false').lower() == 'true'
        include_validation = request.args.get('include_validation', 'true').lower() == 'true'
        detect_format_flag = request.args.get('detect_format', 'true').lower() == 'true'
        
        # Validate export format
        if export_format not in ['json', 'csv', 'excel']:
            return (json.dumps({'error': f'Invalid format: {export_format}. Use json, csv, or excel'}), 400, headers)
        
        # Validate request has a file
        if 'file' not in request.files:
            return (json.dumps({'error': 'No file part in the request'}), 400, headers)

        file = request.files['file']
        if file.filename == '':
            return (json.dumps({'error': 'No file selected for uploading'}), 400, headers)

        logger.info(f"Processing file: {file.filename}")
        
        # Read file into memory
        file_buffer = io.BytesIO(file.read())
        
        # Detect format if requested
        format_info = None
        if detect_format_flag:
            # Read a bit of the file for format detection
            file_buffer.seek(0)
            sample_content = file_buffer.read(10000).decode('utf-8', errors='ignore')
            file_buffer.seek(0)
            
            format_info = detect_format(
                file_content=sample_content,
                filename=file.filename
            )
            logger.info(f"Detected format: {format_info['format']} (confidence: {format_info['confidence']}%)")
        
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
            if format_info:
                response_data['detected_format'] = format_info
            
            response = make_response(json.dumps(response_data, indent=2))
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            response.headers.update(headers)
            return response
        
        logger.info(f"Successfully processed {len(processed_df)} units.")

        # Handle different export formats
        if export_format == 'csv':
            # Export as CSV
            csv_buffer = io.StringIO()
            processed_df.to_csv(csv_buffer, index=False)
            
            response = make_response(csv_buffer.getvalue())
            response.headers['Content-Type'] = 'text/csv; charset=utf-8'
            response.headers['Content-Disposition'] = f'attachment; filename=rent_roll_processed.csv'
            response.headers.update(headers)
            return response
            
        elif export_format == 'excel':
            # Export as Excel
            excel_buffer = io.BytesIO()
            
            # Create Excel writer with multiple sheets if validation is included
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Main data sheet
                processed_df.to_excel(writer, sheet_name='Rent Roll', index=False)
                
                # Add validation sheet if requested
                if include_validation:
                    # Create validation summary DataFrame
                    val_summary = pd.DataFrame([
                        {'Metric': 'Data Quality Score', 'Value': f"{validation_results['data_quality_score']}/100"},
                        {'Metric': 'Total Units', 'Value': validation_results.get('statistics', {}).get('total_units', 'N/A')},
                        {'Metric': 'Occupancy Rate', 'Value': f"{validation_results.get('statistics', {}).get('occupancy_rate', 0)}%"}
                    ])
                    val_summary.to_excel(writer, sheet_name='Validation', index=False)
                    
                    # Add errors and warnings
                    if validation_results['errors'] or validation_results['warnings']:
                        issues_df = pd.DataFrame({
                            'Type': ['Error'] * len(validation_results['errors']) + ['Warning'] * len(validation_results['warnings']),
                            'Issue': validation_results['errors'] + validation_results['warnings']
                        })
                        issues_df.to_excel(writer, sheet_name='Issues', index=False)
            
            excel_buffer.seek(0)
            response = make_response(excel_buffer.getvalue())
            response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            response.headers['Content-Disposition'] = f'attachment; filename=rent_roll_processed.xlsx'
            response.headers.update(headers)
            return response
            
        else:  # JSON format (default)
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
                if format_info:
                    response_data['detected_format'] = format_info
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
