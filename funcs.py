import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class Api:
    def __init__(self):
        return

    def get_fred(self, api_key, series_ids, start_dt='2000-01-01'):
        """
        The function connect to FRED and extract data.
        :param api_key: a string used to authenticate the communication with FRED
        :param series_ids: a list including tickers available on FRED
        :param start_dt: a date specifies the first data point to extract
        :return: a Pandas DataFrame consisting on data extracted from FRED, indexed by date
        """
        fred = Fred(api_key=api_key)
        # Dictionary to store the data
        data_dict = {}

        # Pull data for each series ID
        for series_id in series_ids:
            try:
                data = fred.get_series(series_id, observation_start=start_dt)
                data_dict[series_id] = data
            except Exception as e:
                print(f"Error fetching data for series {series_id}: {e}")

        # Convert the dictionary to a DataFrame
        df_data = pd.DataFrame(data_dict)
        return df_data


class Vis:
    def __init__(self):
        return

    def plot_lines(self, df, n_rows, n_cols, cols=None, fig_size=(15, 10), nm='plots'):
        """
        plot multiple columns in df, one column as one subplot, and save the plot in the directory './data'
        :param df: a Pandas DataFrame indexed by datetime
        :param n_rows: number of subplot in a column
        :param n_cols: number of subplots in a row
        :param cols: a list of column names for plot; if None, plot all columns
        :param fig_size: size of the plot
        :param nm: output file name
        :return: None
        """
        if cols is None:
            cols = list(df.columns)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size)

        for i, ax in enumerate(axs.flatten()):
            if i <= len(cols) - 1:
                c = cols[i]
                ax.plot(df[c])
                ax.set_title(c)
        plt.tight_layout()
        plt.savefig(f"data/{nm} separated.png")
        plt.close

    def plot_lines_in_one(self, df, cols=None, nm='plots'):
        """
        plot multiple columns in df in one coordinate system, and save the plot in the directory './data'
        :param df: a Pandas DataFrame indexed by datetime
        :param cols: a list of column names for plot; if None, plot all columns
        :param nm: output file name
        :return: None
        """
        if cols is None:
            cols = list(df.columns)

        plt.figure(figsize=(15, 15))
        for c in cols:
            plt.plot(df[c], label=c)
        plt.legend()
        plt.savefig(f"data/{nm} all in one.png")
        plt.close()

    def plot_yield_curve(self, df_new, dates, nm='Yield Curve'):
        """
        plot yield curves using yield time series data
        :param df_new: a Pandas DataFrame including yield time series data indexed by date
        :param dates: a list of dates of which yield curves that user wants to plot
        :param nm: output file name
        :return: None
        """
        assert isinstance(dates, list), "dates needs to be a list of dates"
        plt.figure(figsize=(10, 10))
        for date in dates:
            row_to_plot = df_new.loc[date]
            plt.plot(row_to_plot.index, row_to_plot.values, marker='o', linestyle='-', label=date)
            plt.xlabel('Maturity')
            plt.ylabel('Yield (%)')
            plt.legend()
            # plt.title(f'U.S.Yield Curve')
            plt.grid(True)
        plt.savefig(f'data/{nm}.png')
        plt.close()

    def scree_plot(self, exp_var, nm='scree plot'):
        """
        Generate a Scree Plot
        :param exp_var: a Numpy Array
        :param nm: output file name
        :return: None
        """
        plt.figure(figsize=(10, 10))
        plt.plot([f"PC{i}" for i in range(1, 7)], exp_var, color='r')
        plt.bar([f"PC{i}" for i in range(1, 7)], exp_var)
        plt.savefig(f'data/{nm}.png')
        plt.close()

    def heat_map(self, df, nm='corr heatmap'):
        """
        plot a correlation heatmap including all columns in the input dataframe
        :param df: a Pandas DataFrame including the data
        :param nm: output file name
        :return: None
        """
        plt.figure(figsize=(50, 50))
        sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
        # displaying heatmap
        plt.savefig(f'data/{nm}.png')
        plt.close()

class Transform:
    def __init__(self):
        return

    def transform_lvl_chg(self, df, col, s):
        """
        create level change columns
        :param df: a Pandas DataFrame including raw data
        :param col: the column name that user needs to perform the transformation
        :param s: a scalar indicating number of lags that changes are calculated on
        :return: a Pandas series that can be assigned to as a column in a dataframe
        """
        assert isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex), 'Index needs to be Datetime type.'
        df_new = df.sort_index(ascending=True).copy()
        return df_new[col] - df_new[col].shift(s)


class Model:
    def __init__(self):
        return

    def pca(self, df):
        """
        perform Principal Component Analysis
        :param df: a Pandas DataFrame including input data
        :return: a Pandas DataFrame including all PCs; a Numpy array including variance explained of each PC;
                a Numpy array including cumulative variance explained
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # Perform PCA
        pca = PCA(n_components=6)
        principal_components = pca.fit_transform(scaled_data)

        # Create a DataFrame with the principal components
        pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, len(df.columns) + 1)])

        print("\nPCA Components:\n", pca_df)

        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        print(f'\nExplained variance by each component: {explained_variance.astype(float).round(2)}')
        print(f'\nCumulative Explained variance by each component: {explained_variance.cumsum().astype(float).round(2)}')
        print(f'Total explained variance: {sum(explained_variance).astype(float).round(2)}')

        return pca_df.set_index(df.index), explained_variance.astype(float).round(2), explained_variance.cumsum().astype(float).round(2)

    def leave_one_year_out_cv(self, df, start, end):
        """
        perform leave one year out cross validation using PCA
        :param df: a Pandas DataFrame including input data
        :param start: a scalar indicating the first year to leave out
        :param end: a scalar indicating the last year to leave out
        :return: 3 dictionaries including PCs, variance explained and cumulative variance explained
        """
        pca_dfs = {}
        exp_var = {}
        exp_var_cums = {}

        pca_dfs['full'], exp_var['full'], exp_var_cums['full'] = self.pca(df)
        for lo_year in range(start, end):
            df_train = df.loc[df.index.year != lo_year].copy()
            nm = f"leave_out_{lo_year}"
            pca_dfs[nm], exp_var[nm], exp_var_cums[nm] = self.pca(df_train)
        return pca_dfs, exp_var, exp_var_cums

