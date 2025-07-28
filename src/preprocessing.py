from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarray
        val : np.ndarray
        test : np.ndarray
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # Identify categorical columns with dtype object
    categorical_cols = working_train_df.select_dtypes(include=['object']).columns

    # Divide binary and multiclass columns
    binary_cols = [col for col in categorical_cols if working_train_df[col].nunique() == 2]
    multicategory_cols = [col for col in categorical_cols if working_train_df[col].nunique() > 2]

    # Encode binary columns using OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(working_train_df[binary_cols])

    working_train_df[binary_cols] = ordinal_encoder.transform(working_train_df[binary_cols])
    working_val_df[binary_cols] = ordinal_encoder.transform(working_val_df[binary_cols])
    working_test_df[binary_cols] = ordinal_encoder.transform(working_test_df[binary_cols])

    # Encode columns with more than 2 categories using OneHotEncoder
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    one_hot_encoder.fit(working_train_df[multicategory_cols])

    # Transform using the OneHotEncoder
    ohe_train = one_hot_encoder.transform(working_train_df[multicategory_cols])
    ohe_val = one_hot_encoder.transform(working_val_df[multicategory_cols])
    ohe_test = one_hot_encoder.transform(working_test_df[multicategory_cols])

    # Create new DataFrames with the columns One-Hot encoded and suitable names
    ohe_columns = one_hot_encoder.get_feature_names_out(multicategory_cols)

    # Conversion to DataFrame for concatenate with the original datasets
    ohe_train_df = pd.DataFrame(ohe_train, columns=ohe_columns, index=working_train_df.index)
    ohe_val_df = pd.DataFrame(ohe_val, columns=ohe_columns, index=working_val_df.index)
    ohe_test_df = pd.DataFrame(ohe_test, columns=ohe_columns, index=working_test_df.index)

    # Delete multiclass original columns (already encoded) and concatenate the new ones
    working_train_df.drop(columns=multicategory_cols, inplace=True)
    working_val_df.drop(columns=multicategory_cols, inplace=True)
    working_test_df.drop(columns=multicategory_cols, inplace=True)

    working_train_df = pd.concat([working_train_df, ohe_train_df], axis=1)
    working_val_df = pd.concat([working_val_df, ohe_val_df], axis=1)
    working_test_df = pd.concat([working_test_df, ohe_test_df], axis=1)

    # Create an instance of SimpleImputer using the median as 'strategy'
    imputer = SimpleImputer(strategy='median')

    # Adjust (fit) the imputer using only the train dataset
    imputer.fit(working_train_df)

    # Transform the datasets using the imputer fitted previously
    working_train_df[:] = imputer.transform(working_train_df)
    working_val_df[:] = imputer.transform(working_val_df)
    working_test_df[:] = imputer.transform(working_test_df)

    # Create instance of Min-Max Scaler
    scaler = MinMaxScaler()

    # Adjust the scaler only on the training data
    scaler.fit(working_train_df)

    # Transform the 3 datasets (train, val and test)
    working_train_df[:] = scaler.transform(working_train_df)
    working_val_df[:] = scaler.transform(working_val_df)
    working_test_df[:] = scaler.transform(working_test_df)

    return (
        working_train_df.to_numpy(),
        working_val_df.to_numpy(),
        working_test_df.to_numpy()
    )
