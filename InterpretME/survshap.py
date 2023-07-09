import pandas as pd
from pandas import DataFrame
import os
import numpy as np 
import pickle
from tqdm import tqdm
import survshap
import json
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from survshap import SurvivalModelExplainer, PredictSurvSHAP, ModelSurvSHAP
from scipy.integrate import cumtrapz
#plots imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns


def SurvShap_interpretation(X_train, X_test, best_clf, new_sampled_data, survshap_results):
        """Generates SurvShap interpretation results.

    Parameters
    ----------
    X_train : array
        Training dataset used to generate SurvShap interpretation.
    new_sampled_data : dataframe
        Preprocessed dataset containing the important_features.
    best_clf : model
        Best model saved after applying Decision tree.
    ind_test : index
        Testing index.
    X_test : array
        Testing dataset used to generate SurvShap interpretation.
    survshap_results : str
        Path to save SurvShap interpretation results.

    Returns
    -------
    dataframe

    """
        y_train = Surv.from_dataframe("event", "time", X_train)

        # get column names of important features and remove event and time columns
        imp_feature_cols = new_sampled_data.columns.tolist()
        X_train_n = X_train.drop(['time', 'event'], axis=1)
        X_test_n = X_test.drop(['time', 'event'], axis=1)
        # select only important features in X_train and X_test
        x_train_features = X_train.loc[:, X_train.columns.isin(imp_feature_cols)]
        x_test_features = X_test.loc[:, X_test.columns.isin(imp_feature_cols)]
        
        explainer = survshap.SurvivalModelExplainer(best_clf, x_train_features, y_train)

        survshaps = [None]*len(x_test_features)
        features_list = [None]*len(x_test_features)

        pbar = tqdm(total=len(x_test_features), desc='SurvShap explanations')
        for i, obsv in enumerate(x_test_features.values):
            xx = pd.DataFrame(np.atleast_2d(obsv), columns=explainer.data.columns)
            SurvShap_values = survshap.PredictSurvSHAP("sf", "kernel", "integral", "average", 25, None, False)
            SurvShap_values.fit(explainer, xx)
            survshaps[i] = SurvShap_values
            features_list[i] = xx.columns.tolist()
            # convert SurvShap_values to a DataFrame and save as csv
            df_survshap = DataFrame(SurvShap_values.result)
            # Get the directory from the file path & Define the directory for saving CSV files.
            base_directory = os.path.dirname(survshap_results)
            csv_dir_path = os.path.join(base_directory, 'survshap_values')
            # Create the directory if it doesn't exist.
            if not os.path.exists(csv_dir_path):
                os.makedirs(csv_dir_path)
            # Now you can safely write the CSV file.
            csv_file_path = os.path.join(csv_dir_path, f'survshap_{i}.csv')
            df_survshap.to_csv(csv_file_path, index=False)

            pbar.update(1)
        pbar.close()
        # Save results as Pickle file
        with open(survshap_results, "wb") as file:
            pickle.dump(survshaps, file)
        
        return survshaps

def get_local_accuracy_from_shap_explanations(all_explanations, method_label, cluster_label, model_label, last_index=None):
    if last_index is None:
        last_index=len(all_explanations[0].timestamps)
    diffs = []
    preds = []
    for explanation in all_explanations:
        preds.append(explanation.predicted_function[:last_index])
        diffs.append(explanation.predicted_function[:last_index] - explanation.baseline_function[:last_index] - np.array(explanation.result.iloc[:, 5:].sum(axis=0))[:last_index])
    diffs_squared = np.array(diffs)**2
    E_diffs_squared = np.mean(diffs_squared, axis=0)
    preds_squared = np.array(preds)**2
    E_preds_squared = np.mean(preds_squared, axis=0)
    return  pd.DataFrame({"time": all_explanations[0].timestamps[:last_index], "sigma": np.sqrt(E_diffs_squared) / np.sqrt(E_preds_squared), 
     "method": method_label, "cluster": cluster_label, "model": model_label })

def get_feature_orderings_and_ranks_survshap(explanations):
    feature_importance_orderings = []
    feature_importance_ranks = []
    for explanation in explanations:
        result_df = explanation.result.copy()
        cumulative_change = cumtrapz(np.abs(result_df.iloc[:, 5:].values), explanation.timestamps, initial=0, axis=1)[:, -1]
        result_df['aggregated_change'] = cumulative_change
        sorted_df = result_df.sort_values(by='aggregated_change', key=lambda x: -abs(x))
        feature_importance_orderings.append(sorted_df.index.to_list())
        feature_importance_ranks.append(np.abs(sorted_df['aggregated_change']).rank(ascending=False).to_list())
    
    return pd.DataFrame(feature_importance_orderings), pd.DataFrame(feature_importance_ranks)

def prepare_ranking_summary_long(ordering):
    res = pd.DataFrame()

    for i in range(ordering.shape[1]):
        counts = ordering.iloc[:, i].value_counts().reset_index()
        counts.columns = ['variable', 'value']
        counts['importance_ranking'] = i + 1
        res = pd.concat([res, counts])
    
    return res[['importance_ranking', 'variable', 'value']]


# def prepare_ranking_summary_long(ordering):
#     num_cols = ordering.shape[1]  # get number of columns
#     column_names = ["x"+str(i+1) for i in range(num_cols)]  # dynamically create column names
#     res = pd.DataFrame(columns=column_names)

#     for i in range(num_cols):
#         tmp = pd.DataFrame(ordering.iloc[:, i].value_counts().to_dict(), index=[i+1])
#         res = pd.concat([res, tmp])
    
#     res = res.reset_index().rename(columns={i: "x"+str(i+1) for i in range(num_cols)}, index=str)
#     res = res.rename(columns={"index": "importance_ranking"})
    
#     return res.melt(id_vars=["importance_ranking"], value_vars=column_names)

#Plotting 
# Function to make factors in the dataset
def make_factors(data):
    data['importance_ranking'] = pd.Categorical(data['importance_ranking'], 
                                                categories=data['importance_ranking'].unique(),
                                                ordered=True)
    # Make 'variable' a categorical variable with categories based on unique values
    data['variable'] = pd.Categorical(data['variable'], 
                                      categories=data['variable'].unique()[::-1],
                                      ordered=True)
    return data

def barplot_variable_ranking(data, title='', ytitle=''):
    color_palette = ["#9C27B0","#009688","#3F51B5","#FF5733", "#03A9F4", "#FFC300", "#DAF7A6", "#581845",
                 "#C70039", "#FF5733", "#900C3F", "#DAF7A6", "#581845",
                 "#9C27B0", "#673AB7",  "#2196F3", "#900C3F",
                 "#00BCD4", "#4CAF50", "#8BC34A", "#CDDC39"]


    # Create a mapping for y-axis labels
    y_labels = {rank: f"{rank}{suffix}" for rank, suffix in zip(range(1, data['importance_ranking'].nunique() + 1), ['st', 'nd', 'rd'] + ['th'] * 20)}
    data['importance_ranking'] = data['importance_ranking'].replace(y_labels)

    plt.figure(figsize=(8, 4))
    for i, variable in enumerate(data['variable'].cat.categories):
        # Get the subset of data for this variable
        data_subset = data[data['variable'] == variable].groupby(['variable', 'importance_ranking']).sum().reset_index()
        if i == 0:
            # For the first variable, plot the bars normally
            bars = plt.barh(data_subset['importance_ranking'], data_subset['value'], 
                     color=color_palette[i], edgecolor='white', label=variable, height= 0.75)
        else:
            # For subsequent variables, stack the bars on top of the previous ones by specifying the 'left' parameter as the sum of the 'value' field for all previous variables
            prev_data = data[data['variable'].isin(data['variable'].cat.categories[:i])]
            prev_data_sum = prev_data.groupby('importance_ranking')['value'].sum()
            prev_data_sum = prev_data_sum.reindex(data_subset['importance_ranking']).fillna(0)

            bars = plt.barh(data_subset['importance_ranking'], data_subset['value'], 
                     color=color_palette[i], edgecolor='white', 
                     left=prev_data_sum, label=variable, height= 0.75)
        
        # Annotate the value inside each bar segment
        for bar in bars:
            bar_value = bar.get_width()
            if bar_value > 0:  # Only annotate bars with value > 0
                plt.text(bar.get_x() + bar.get_width() / 2, 
                bar.get_y() + bar.get_height() / 2, 
                f"{bar_value:.0f}", 
                verticalalignment='center', 
                horizontalalignment='center', 
                color='black', fontsize=12)
    ax = plt.gca()
    ax.set(title=title, xlabel='Value')
    ax.set_ylabel(ytitle)
    # Reverse the order of y-axis
    ax.invert_yaxis()
    ax.legend(title='Rankins Variable Importance', 
              loc='lower center', bbox_to_anchor=(1, 1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.tight_layout()
    plt.show()
