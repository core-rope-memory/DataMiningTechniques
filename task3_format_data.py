import numpy as np
from pandas import DataFrame, read_csv

used_auto_data_path = "../../task_3/UsedAutoRELEVATEfirst10000-noLatLong.csv"
exp_rel_custom_path = "../../task_3/EXP REL Custom.xls"


def auto_csv_to_df(csv_path: str) -> DataFrame:
    """
    Convert used auto data CSV to a pandas DataFrame.
    """
    # pandas can't handle % chars in the header row so bring in header row first, remove % chars, skip for read_csv
    # then manually add the clean header back in.
    with open(csv_path) as f:
        header_string = f.readline()
    header_string = header_string.replace('%', 'pct')
    col_names = header_string.split(',')

    data_df = read_csv(csv_path, skiprows=1, header=None, names=col_names, low_memory=False)

    # remove rows and columns that contain all nan values.
    data_df = data_df.dropna(how='all')
    data_df = data_df.dropna(axis=1, how='all')

    return data_df


def reformat_auto_data_vals(auto_df: DataFrame) -> DataFrame:
    """
    Convert truncated numerical values back to their original values.
    """
    auto_df = auto_df.drop(['Customer_Link'], axis=1)

    auto_df = auto_df.apply(lambda x: x * 1000 if x.name in [
        'Home Purchase Price',
        'Home Improvement Value',
        'Home Land Value',
        'Home Total Value',
        'Investment - Mortgage Amount',
        'Home Land Square Footage',
        'Investment - Equity Amount',
        'Owner Occupied - Refinance Amount',
        'Investment - Purchase Amount',
        'Mortgage amount in thousands',
        'Investment - Equity Amount',
        'Investment - Refinance Amount',
        'Owner Occupied - Equity Amount',
        'Estimated Equity - Amount in thousands'
    ] else x)

    auto_df = auto_df.apply(lambda x: x * 100 if x.name in [
        'Home Base Square Footage',
        'Home Building Square Footage'
    ] else x)

    return auto_df


def column_value_lists(auto_df: DataFrame) -> dict:
    """
    Takes in the auto data frame and returns a dictionary with keys of columns names and values of the list of unique
    values that appear in that column.
    """
    the_dict = {}
    for i in auto_df.columns:
        temp = auto_df[i].unique()
        the_dict[i] = temp
    return the_dict


def column_percent_nan(auto_df: DataFrame) -> dict:
    """
    Takes in the auto data frame and returns a dictionary with the keys of column names and values of percent nan in
    that column.
    """
    the_dict = {}
    for i in auto_df.columns:
        the_dict[i] = auto_df[i].isnull().mean()
    return the_dict


def column_perc_nan_less_than(perc_dict: dict, percentage: float) -> dict:
    """
    Returns the names and percentags of nans in a column for columns that have percentages of nans less than the
    value passed in.
    """
    return {k: v for (k, v) in perc_dict.items() if v < percentage}


def df_perc_nan_less_than(auto_df: DataFrame, percentage: float) -> DataFrame:
    """
    Returns a filtered version of the used auto DataFrame which only included columns where the percentage of nan values
    is less than the percentage passed in.
    """
    return auto_df[auto_df.columns[auto_df.isnull().mean() < percentage]]


def categorical_to_numerical(auto_df: DataFrame) -> DataFrame:
    """
    Returns a version of the used auto DataFrame where nan values have been replaced with values of -1 and the string
    and character valued categorical values have been replaced with numerical values.
    """
    # fill nan values with -1
    auto_df.fillna(-1, inplace=True)

    for i in auto_df.columns:
        unique_col_vals = auto_df[i].unique()
        if unique_col_vals.dtype != np.dtype('float64'):
            unique_val_list = list(unique_col_vals)
            for indx, val in enumerate(unique_val_list):
                if val != -1:
                    auto_df[i].replace(val, indx+1, inplace=True)
    return auto_df


if __name__ == '__main__':
    used_auto_df = auto_csv_to_df(used_auto_data_path)
    used_auto_df = reformat_auto_data_vals(used_auto_df)

    col_val_lists = column_value_lists(used_auto_df)
    for i in col_val_lists:
        if len(i) < 20:
            print(i)

    len_val_20 = [i for i in col_val_lists if len(i) < 20]
    len_val_40 = [i for i in col_val_lists if len(i) < 40]

    col_perc_nan = column_percent_nan(used_auto_df)
    col_perc_nan_less = column_perc_nan_less_than(col_perc_nan, 0.5)
    filtered_df = df_perc_nan_less_than(used_auto_df, 0.5)

    used_auto_df = categorical_to_numerical(used_auto_df)
