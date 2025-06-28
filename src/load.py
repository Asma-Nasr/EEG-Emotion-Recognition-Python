import pandas as pd

def extract_columns(dataframe, *column_indices):
    """
    Extract specified columns from a DataFrame. If no column indices are provided, extract all columns.

    Parameters:
    - dataframe: pd.DataFrame - The DataFrame from which to extract columns.
    - column_indices: int - Column indices to extract (optional).

    Returns:
    - extracted_data: list - A list of extracted Series for each specified column or all columns if no indices are provided.
    """
    dataframe = pd.DataFrame(dataframe)  # Ensure input is a DataFrame
    if column_indices:
        extracted_data = [dataframe.iloc[:, index] for index in column_indices if index < dataframe.shape[1]]
    else:
        extracted_data = [dataframe.iloc[:, index] for index in range(dataframe.shape[1])]
    return extracted_data

