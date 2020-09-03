import pandas

training_data_path = "../../task_1/training.txt"
training_label_path = "../../task_1/label_training.txt"

testing_data_path = "../../task_1/test_data_sample.txt"
testing_label_path = "../../task_1/test_label_sample.txt"


def data_txt_to_wide_df(txt_path: str):
    """
    Reads in a three column training data txt file and returns a DataFrame that has been widened using pandas pivot
    function. Now there is only one row per 'information id' and 'feature id' are now attributes with values 'feature
    value'.

    :param txt_path: path to txt file.
    :return: DataFrame
    """

    data_df = pandas.read_table(txt_path, sep=" ", names=['info_id', 'feature_id', 'feature_val'])
    data_df = data_df.pivot(index='info_id', columns='feature_id', values='feature_val')
    data_df = data_df.fillna(value=0)

    return data_df


def data_txt_to_df(txt_path: str):
    """
    Reads in a three column training data txt file and returns a three column DataFrame.

    :param txt_path: path to txt file.
    :return: DataFrame
    """

    data_df = pandas.read_table(txt_path, sep=" ", names=['info_id', 'feature_id', 'feature_val'])

    return data_df


def label_txt_to_df(txt_path: str):
    """
    Reads in a single column label data txt file and returns a single column DataFrame.

    :param txt_path: path to csv file.
    :return: DataFrame
    """
    label_df = pandas.read_table(txt_path, header=None)

    return label_df


def get_data_attribute_union(x_train: pandas.DataFrame, x_test: pandas.DataFrame) -> set:
    """
    Takes two DataFrames and returns a set that is the union of the attribute ids for the two DataFrames.

    :param x_train: The training data.
    :param x_test: The testing data.
    :return: set of column ids.
    """
    x_train_attr_ids = set(x_train.columns)
    x_test_attr_ids = set(x_test.columns)

    return x_train_attr_ids.union(x_test_attr_ids)


def add_zero_cols_to_df(df: pandas.DataFrame, cols_union: set) -> pandas.DataFrame:
    """
    Adds zero filled columns of needed column ids to the dataframe, then sorts by column ids.

    :param df: The training data.
    :param cols_union: The set of column ids that is the union of training and testing data column ids.
    :return: DataFrame with added columns that are present in other data df, filled with zeros, and sorted.
    """
    columns_present = set(df.columns)
    columns_needed = cols_union - columns_present

    for i in columns_needed:
        df[i] = [0.0] * len(df.index)

    return df.sort_index(axis=1)


def write_df_to_csv(df: pandas.DataFrame, csv_path: str):
    """
    Writes a DataFrame to a .csv file in the specified file path.

    :param df: DataFrame to export.
    :param csv_path: Path to save the .csv file to.
    :return: None
    """
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    """
    Running as a script will output .csv files of the widened data and label information for both training and testing.
    """
    df_train_data_wide = data_txt_to_wide_df(training_data_path)
    df_train_label = label_txt_to_df(training_label_path)
    write_df_to_csv(df_train_data_wide, "../../task_1/x_train.csv")
    write_df_to_csv(df_train_label, "../../task_1/y_train.csv")

    df_test_data_wide = data_txt_to_wide_df(testing_data_path)
    df_test_label = label_txt_to_df(testing_label_path)
    write_df_to_csv(df_test_data_wide, "../../task_1/x_test.csv")
    write_df_to_csv(df_test_label, "../../task_1/y_test.csv")

