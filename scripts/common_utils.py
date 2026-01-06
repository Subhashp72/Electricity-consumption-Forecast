
import pandas as pd 
import numpy as np
import re 
from pandas import ExcelWriter
### function to get basic data summary###########
def get_data_summary(data):
    """A function to get the basic data summary

    Args:
        data (DataFrame): Any pandas data frame

    Returns:
        DataFrame: A Data Frame with % missing values, unique values, no of total data points, first row etc.
    """
    data = data.reset_index(drop=True)
    data_summary = data.agg(["dtypes", "nunique"]).T.reset_index()
    data_summary.columns = ["Var_name", "Var_type", "unique_values"]
    data_summary["# of observations"] = list(data.apply(len))
    data_summary["missing_values"] = list(data.isnull().sum())
    data_summary["missing %"] = (
        data_summary["missing_values"] / data_summary["# of observations"]
    )
    data_summary["first_row"] = list(data.apply(lambda x: x[0]))
    date_cols = [col for col in data.columns if (re.search("DT|DATE$", col))]
    data_summary["flag"] = np.where(
        data_summary["Var_name"].isin(date_cols),
        "Date",
        np.where(
            data_summary["unique_values"] + data_summary["missing_values"]
            >= data_summary["# of observations"],
            "ID",
            np.nan,
        ),
    )
    # data_summary.head()
    return data_summary


##### function to get numeric variable summary########
def get_numeric_summary(data):
    """A function to get the numeric data summary

    Args:
        data (DataFrame): Any pandas dataframe with numeric variables

    Returns:
        DataFrame: A Dataframe containing data summary like % missing values, No of observations, quantile values etc.
    """
    data= data.reset_index(drop=True)
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    data_summary = pd.DataFrame()
    if len(numeric_cols) > 0:
        data = data[numeric_cols]
        data_summary = data.agg(
            ["dtypes", "nunique", np.std, "mean"]).T.reset_index()
        data_summary.columns = ["Var_name",
                                "Var_type", "unique_values", "std", "mean"]
        data_summary["# of observations"] = list(data.apply(len))
        data_summary["missing_values"] = list(data.isnull().sum())
        data_summary["missing %"] = list(
            data_summary["missing_values"] / data_summary["# of observations"]
        )
        data_summary["first_row"] = list(data.apply(lambda x: x[0]))
        ###
        qt_df = data.quantile([0, 0.25, 0.5, 0.75, 1]).T.round(2).reset_index()
        qt_df.columns = ["Var_name", "min", "25%", "median", "75%", "max"]
        #
        data_summary = data_summary.merge(qt_df, on="Var_name", how="left")
    return data_summary


### function to get categorical summary#######
def get_categorical_summary(data):
    """A function to get the categorical summary like frequency count, missing%, no of levels etc.

    Args:
        data (DataFrame): Any pandas data frame

    Returns:
        DataFrame: A data frame with frequency count, missing%, no of levels etc.
    """
    data = data.reset_index(drop=True)
    character_cols = data.select_dtypes(include=["object"]).columns.tolist()
    cat_summary = pd.DataFrame()
    data = data[character_cols]
    data_summary = data[character_cols].agg(["nunique"]).T.reset_index()
    data_summary.columns = ["Var_name", "# of levels"]
    cond = data_summary["# of levels"] < 100
    character_cols = list(data_summary["Var_name"][cond])
    if len(character_cols) > 0:
        # print(character_cols)
        data = data[character_cols]
        data_summary = data_summary[cond].reset_index(drop=True)
        data_summary["# of observations"] = list(data.apply(len))
        data_summary["missing_values"] = list(data.isnull().sum())
        data_summary["missing %"] = list(
            data_summary["missing_values"] / data_summary["# of observations"]
        )
        ######
        cat_summary = data.stack().reset_index(level=1)
        cat_summary.columns = ["Var_name", "level"]
        cat_summary["level"] = cat_summary["level"].apply(
            lambda x: re.sub("-", " ", str(x))
        )
        cat_summary["level"] = cat_summary["level"].str.title()
        cat_summary = (
            cat_summary.groupby(["Var_name", "level"])
            .size()
            .rename("frequency")
            .reset_index()
        )
        cat_summary = cat_summary.merge(
            data_summary, on="Var_name", how="left")
        cat_summary["share %"] = (
            cat_summary["frequency"] / cat_summary["# of observations"]
        )
        cat_summary["level"] = np.where(
            (cat_summary["share %"] < 0.05), "others", cat_summary["level"]
        )
        cat_summary["share %"] = cat_summary.groupby(["Var_name", "level"])[
            "share %"
        ].transform(lambda x: sum(x))
        cat_summary["frequency"] = cat_summary.groupby(["Var_name", "level"])[
            "frequency"
        ].transform(lambda x: sum(x))
        cat_summary = cat_summary.drop_duplicates().reset_index(drop=True)
        cat_summary = cat_summary.sort_values(
            ["Var_name", "frequency"], ascending=False
        )
        cat_summary["most_frequent"] = cat_summary.groupby(["Var_name"])[
            "level"
        ].transform(lambda x: list(x)[0])
    #
    return cat_summary
#########################
# function to save multiple dataframes as excel
def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer, 'sheet%s' % n)
##############################
def sanity_check(df):
    if df.shape[0] == 0:
        raise ValueError('Dataframe is empty')
    if df.shape[1] == 0:
        raise ValueError('Dataframe has no columns')
    if df.duplicated().sum() > 0:
        raise ValueError('Dataframe has duplicated rows')
    if df.columns.duplicated().sum()>0:
        raise ValueError('Dataframe has duplicated columns')
    if df.isnull().sum().sum() > 0:
        raise Warning('Dataframe has null values')
    if df.select_dtypes(include=['object']).apply(pd.Series.nunique, axis=0).sum() == df.shape[1]:
        raise ValueError('Dataframe has no numerical columns')
    if df.select_dtypes(include=['number']).apply(pd.Series.nunique, axis=0).sum() == df.shape[1]:
        raise ValueError('Dataframe has no categorical columns')