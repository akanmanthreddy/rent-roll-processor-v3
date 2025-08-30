import pandas as pd
import io
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def find_header_and_data_start_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the header rows and the start of the data section in an Excel DataFrame.
    Returns a cleaned DataFrame with proper headers.
    """
    header_row_idx = -1
    for idx, row in df.iterrows():
        row_str = ' '.join(str(val).lower() for val in row.values)
        if all(keyword in row_str for keyword in ['unit', 'unit type', 'resident']):
            header_row_idx = idx
            break
    
    if header_row_idx == -1:
        raise ValueError("Could not find the primary header row in the Excel file.")
    
    header1 = df.iloc[header_row_idx].fillna('')
    header2 = df.iloc[header_row_idx + 1].fillna('') if header_row_idx + 1 < len(df) else pd.Series()
    
    combined_header = []
    for i in range(len(header1)):
        h1 = str(header1.iloc[i]).strip()
        h2 = str(header2.iloc[i]).strip() if i < len(header2) else ''
        if h1 and h2:
            combined_header.append(f"{h1} {h2}".strip())
        elif h1:
            combined_header.append(h1)
        else:
            combined_header.append(h2 if h2 else f'column_{i}')
    
    data_start_idx = header_row_idx + 2
    for idx in range(header_row_idx + 2, len(df)):
        row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values)
        if 'current/notice/vacant residents' in row_str:
            data_start_idx = idx + 1
            break
    
    data_end_idx = len(df)
    for idx in range(data_start_idx, len(df)):
        row_str = ' '.join(str(val).lower() for val in df.iloc[idx].values)
        if 'summary groups' in row_str or 'future residents/applicants' in row_str:
            data_end_idx = idx
            break
    
    new_df = df.iloc[data_start_idx:data_end_idx].reset_index(drop=True)
    new_df.columns = combined_header[:len(new_df.columns)]
    
    new_df.columns = [col.lower().strip().replace(' ', '_') for col in new_df.columns]
    new_df = new_df.rename(columns={'unit_sq_ft': 'sq_ft'})
    
    return new_df

def find_header_and_data_start_csv(file_buffer: io.BytesIO) -> Tuple[int, int, List[str]]:
    """
    Original CSV processing function - identifies the header rows and data section.
    """
    file_buffer.seek(0)
    lines = file_buffer.readlines()
    file_buffer.seek(0)

    header_start_idx = -1
    data_start_idx = -1

    for i, line_bytes in enumerate(lines):
        line = line_bytes.decode('utf-8', errors='ignore').strip()
        if all(keyword in line.lower() for keyword in ['unit,', 'unit type,', 'resident,']):
            header_start_idx = i
            break

    if header_start_idx == -1:
        raise ValueError("Could not find the primary header row in the file.")

    header1 = lines[header_start_idx].decode('utf-8', errors='ignore').strip().split(',')
    header2 = lines[header_start_idx + 1].decode('utf-8', errors='ignore').strip().split(',')

    combined_header = []
    for i in range(len(header1)):
        h1 = header1[i].strip()
        h2 = header2[i].strip() if i < len(header2) else ''
        if h1 and h2:
            combined_header.append(f"{h1} {h2}".strip())
        elif h1:
            combined_header.append(h1)
        else:
            combined_header.append(h2)

    for i in range(header_start_idx + 2, len(lines)):
        line = lines[i].decode('utf-8', errors='ignore').strip().lower()
        if 'current/notice/vacant residents' in line:
            data_start_idx = i + 1
            break

    if data_start_idx == -1:
        raise ValueError("Could not find the start of the data section.")

    return header_start_idx, data_start_idx, combined_header

def load_and_prepare_dataframe(file_buffer: io.BytesIO, filename: str) -> pd.DataFrame:
    """
    Loads the file (CSV or Excel) and prepares it for processing.
    """
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if file_extension in ['xlsx', 'xls']:
        logger.info("Processing Excel file")
        df = pd.read_excel(file_buffer, header=None, engine='openpyxl')
        df = find_header_and_data_start_excel(df)
        return df
    else:
        logger.info("Processing CSV file")
        header_start_idx, data_start_idx, combined_header = find_header_and_data_start_csv(file_buffer)
        
        file_buffer.seek(0)
        lines = file_buffer.readlines()
        footer_start_idx = len(lines)
        for i in range(data_start_idx, len(lines)):
            line = lines[i].decode('utf-8', errors='ignore').strip().lower()
            if 'summary groups' in line or 'future residents/applicants' in line:
                footer_start_idx = i
                break

        file_buffer.seek(0)
        df = pd.read_csv(
            file_buffer,
            header=None,
            skiprows=data_start_idx,
            nrows=footer_start_idx - data_start_idx,
            names=combined_header,
            engine='python'
        )

        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        df = df.rename(columns={'unit_sq_ft': 'sq_ft'})

        return df
