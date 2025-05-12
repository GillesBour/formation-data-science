import pandas as pd

def clean_data(filepath, separator, y_name, columns_to_drop=None):
    """Clean data for future processing (regression, ridge...).

    Parameters:
    -----------
    filepath (string): relative (or absolute) path to the data file
    separator (string): separator used in the data file
    y_name (string): name of the y column
    columns_to_drop (list(string)): list of columns to drop from the data
        (default is None, which means no columns will be dropped)

    Returns:
    --------
    data (pandas DataFrame): the cleaned dataframe
    """
    data = pd.read_csv(filepath, header=0, sep=separator)
    # Rename the response variable
    data.rename(columns={y_name: "Y"}, inplace=True)
    # Drop specified columns if any
    if columns_to_drop is not None:
        data = data.drop(columns_to_drop, axis=1)
    return data