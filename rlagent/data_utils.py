# rlagent/data_utils.py

import pandas as pd

def check_and_process_data(input_path, output_path):
    try:
        data = pd.read_csv(input_path)

        required_columns = {"ligand", "label", "rna_sequence", "region_mask"}

        if not required_columns.issubset(data.columns):
            print(f"Missing required columns! Expected: {required_columns}, Found: {list(data.columns)}")
            return None

        # Example processing (you can add more if needed)
        # For now, just save the loaded data to processed file
        data.to_csv(output_path, index=False)

        print(f"Loaded {len(data)} rows from {input_path}, saved to {output_path}")
        return data

    except Exception as e:
        print(f"Error processing data: {e}")
        return None
