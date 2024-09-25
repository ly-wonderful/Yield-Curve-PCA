# U.S. Treasury Yield Data Analysis Project
# Lu Yang
# 2024/08/25

from fred_case_study_funcs import *

# initiate classes used to organize functions
api = Api()
vis = Vis()
trans = Transform()
model = Model()

# read in pre-saved txt file containing the FRED API Key
with open('fred_api_key.txt', 'r') as file:
    fred_api_key = file.readlines()[0]

# extract data from FRED
ids = ['GS1', 'GS2', 'GS3', 'GS5', 'GS7', 'GS10']
df_out = api.get_fred(fred_api_key, ids, start_dt='1976-06-01')

# Data Quality Check
# missing values
for c in df_out.columns:
    n_missing = df_out[c].isnull().sum()
    if n_missing > 0:
        print(f"# of missing entries in columns {c}: n_missing")
    else:
        print(f"No missing value in columns {c}")

# detect outliers via visualization
# B/c it's time series data, look at line chart first
vis.plot_lines(df_out, 2, 3, fig_size=(15, 10), nm='plots raw')
vis.plot_lines_in_one(df_out, nm='plots raw')

# visualize some sample yield curves
df_new = df_out.copy()
df_new.rename(columns={old: old[2:] + 'Y' for old in df_new.columns}, inplace=True)
dates = ['1976-12-01', '1990-12-01', '2018-12-01', '2020-12-01', '2022-12-01', '2024-01-01', '2024-07-01']
# dates =list(df_new.index)
vis.plot_yield_curve(df_new, dates)

# Data Transformation - create MoM change
df_out2 = df_out.copy()
for c in df_out2.columns:
    step = 1
    new_col = f"{c}_chg{step}"
    df_out2[new_col] = trans.transform_lvl_chg(df_out2, c, step)

# remove the row with NaNs and keep only MoM Chg columns
chg_cols = [c for c in df_out2.columns if 'chg' in c]
df_clean_full = df_out2[chg_cols].dropna()
print(f"dim before selection: {df_out2.shape}")
print(f"dim after selection: {df_clean_full.shape}")

# select data for pca

df_clean = df_clean_full.loc[df_clean_full.index >= '1976-01-01']

# plot chg variables
vis.plot_lines(df_clean, 2, 3, fig_size=(15, 10), nm='plot mom')
vis.plot_lines_in_one(df_clean, nm='plot mom')

# Covariance Matrix
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(df_clean))
cm = scaled_data.cov()

# reduce dimension
# PCA
pca_df_full, exp_var_full, exp_var_cum_full = model.pca(df_clean)
vis.scree_plot(exp_var_full, nm='scree plot full')
vis.plot_lines(pca_df_full, 2, 3, fig_size=(15, 10), nm='plot pca')

# leave-one-year-out validation
pca_dfs, exp_var, exp_var_cums = model.leave_one_year_out_cv(df_clean, start=1976, end=2024)

df_exp_var = pd.DataFrame(exp_var)
vis.plot_lines_in_one(df_exp_var, nm='plot pca explained variances - leave one year out cross validation')


# Examine CV output
def exam_pc(pca_dfs, pc):
    """
    This function combines specified PC from LOYO CV into one DataFrame, generate plots and a correlation headmap.
    :param pca_dfs: a dictionary generated from model.leave_one_year_out_cv()
    :param pc: column name of a Principal component.
    :return:
    """
    df_pca = pca_dfs['full'][[pc]].copy()
    df_pca.rename(columns={pc: 'full'}, inplace=True)
    for key in pca_dfs.keys():
        if key != 'full':
            df_pca_single = pca_dfs[key][[pc]].copy()
            df_pca_single.rename(columns={pc: key}, inplace=True)
            df_pca = df_pca.join(df_pca_single, how='outer')

    vis.plot_lines(df_pca, 20, 3, fig_size=(15, 100), nm=f'plot {pc} - leave one year out cross validation')
    vis.plot_lines_in_one(df_pca, nm=f'plot {pc} - leave one year out cross validation')
    vis.heat_map(df_pca, nm=f'{pc} corr heatmap')


for pc in ['PC1', 'PC2', 'PC3']:
    exam_pc(pca_dfs, pc)

