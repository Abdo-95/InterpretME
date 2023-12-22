import pandas as pd
from pandas import DataFrame
import os
import numpy as np 
import pickle
from tqdm import tqdm
import survshap
import validating_models.stats as stats
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
import glob
time_survshap = stats.get_decorator('PIPE_SURVSHAP')

@time_survshap
def SurvShap_interpretation(X_train, y_train, best_clf, X_test, st, survshap_results):
        """Generates SurvShap interpretation results.

    Parameters
    ----------
    X_train : array
        Training dataset used to generate SurvShap interpretation.
    new_sampled_data : dataframe
        Preprocessed dataset containing the important_features.
    best_clf : model
        Best model saved after applying Decision tree.
    st : int
        Unique identifier.
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
        explainer = survshap.SurvivalModelExplainer(best_clf, X_train, y_train)
        survshaps = [None]*len(X_test)
        features_list = [None]*len(X_test)
        # Check and create the survshap_results directory if it's not None
        if survshap_results is not None:
            if not os.path.exists(survshap_results):
                os.makedirs(survshap_results, exist_ok=True)
                # Create a subdirectory for individual SurvSHAP files
                patients_dir_path = os.path.join(survshap_results, 'SurvSHAP_files')
                os.makedirs(patients_dir_path, exist_ok=True)
            # Define the directory for saving CSV files.
        csv_dir_path = survshap_results
        patients_dir_path = os.path.join(survshap_results, 'SurvSHAP_files')
        if not os.path.exists(csv_dir_path):
            os.makedirs(csv_dir_path, exist_ok=True)
        pbar = tqdm(total=len(X_test), desc='SurvShap explanations')
        for i, (index, obsv) in enumerate(X_test.iterrows()):
            xx = pd.DataFrame(np.atleast_2d(obsv), columns=explainer.data.columns)
            SurvShap_values = survshap.PredictSurvSHAP("sf", "kernel", "integral", "average", 25, None, False)
            SurvShap_values.fit(explainer, xx)
            survshaps[i] = SurvShap_values
            features_list[i] = xx.columns.tolist()
            
            # convert SurvShap_values to a DataFrame and save as csv
            df_survshap = DataFrame(SurvShap_values.result)
            patient_id = index  # Assuming patient_id is the index of X_test
            csv_file_path = os.path.join(patients_dir_path, f'patient_{patient_id}.csv')
            df_survshap.to_csv(csv_file_path, index=False)

            pbar.update(1)
        combine_survshaps_files(patients_dir_path, st)
        pbar.close()
        if survshap_results is not None:
            ## If the path defined in 'survshap_results' does not exist, Create it!
            if not os.path.exists(survshap_results):
                os.makedirs(survshap_results, exist_ok=True)
            # Construct the full file path for the pickle file
            pickle_file_path = os.path.join(survshap_results, "survshap_results.pkl")
            # Save results as Pickle file
            with open(pickle_file_path, "wb") as file:
                pickle.dump(survshaps, file)
        return survshaps

def combine_survshaps_files(survshap_files_path, st):
    """Combine survshaps csv files fa single CSV file.

    Parameters
    ----------
    survshap_files_path : string
        Directory containing the input SurvShap output CSV files
    st : Integer
        Run_ID
    Returns
    -------
    dataframe

    """
    
    all_files = [os.path.join(survshap_files_path, file) for file in os.listdir(survshap_files_path) if file.endswith('.csv')]

    li = []

    for filename in all_files:
        # Read each csv file
        df = pd.read_csv(filename, index_col=None, header=0)

        # Extracting relevant columns
        temp_df = df[['variable_name', 'variable_value', 'aggregated_change']].copy()

        # Rename columns to match the desired format
        temp_df.columns = ['Features', 'Values', 'Aggregated Weights']

        # Extract the filename (without the extension) to use as the 'index' value
        file_name = os.path.basename(filename).split('.')[0]
        temp_df.insert(0, 'index', file_name)

        # Append this dataframe to the list
        li.append(temp_df)

    # Concatenate all the dataframes in the list
    final_df = pd.concat(li, axis=0, ignore_index=True)
    # Extract the numerical part of the 'index' and convert it to integer
    final_df['index_num'] = final_df['index'].str.extract('(\d+)').astype(int)
    # Sort the final dataframe by 'index' and then by 'features'
    final_df = final_df.sort_values(by=['index_num', 'Features'])
    # Drop the 'index_num' column as it's no longer needed
    final_df.drop('index_num', axis=1, inplace=True)
    # Add the 'tool' column with the value 'SurvShap'
    final_df['run_id'] =  st
    final_df['tool'] = 'SurvShap'
    # Save the final dataframe to the desired location
    parent_dir = os.path.dirname(survshap_files_path)
    # Save the final dataframe to the parent directory
    output_path = os.path.join(parent_dir, 'combined_SurvSHAP.csv')
    final_df.to_csv(output_path, index=False)


def survshap_local_accuracy(all_explanations, method_label, cluster_label, model_label, last_index=None):
    
    """ This function estimates the local accuracy of the predictions made using SurvSHAP.
    The function outputs a DataFrame containing the timestamps,
    the local accuracy (sigma) and the labels for the method, the cluster and the model.
    Parameters
    ----------
    all_explanations : list
        List of SHAP explanations for predictions.
    method_label : str
        Label describing the method used for generating SHAP explanations.
    cluster_label : str
        Label describing the cluster the SHAP explanations belong to.
    model_label : str
        Label describing the model used for predictions.
    last_index : int, optional
        Index at which to stop calculating accuracy. If not provided, the full length of the timestamp array is used.

    Returns
    -------
    pd.DataFrame
        A DataFrame with time, sigma (local accuracy), method, cluster, and model information.

    """

    # If last_index is not provided, use the full length of the timestamp array
    
    if last_index is None:
        last_index=len(all_explanations[0].timestamps)
    diffs = []
    preds = []
    # Loop through all explanations and calculate prediction differences
    for explanation in all_explanations:
        preds.append(explanation.predicted_function[:last_index])
        diffs.append(explanation.predicted_function[:last_index] - explanation.baseline_function[:last_index] - np.array(explanation.result.iloc[:, 5:].sum(axis=0))[:last_index])
    diffs_squared = np.array(diffs)**2
    E_diffs_squared = np.mean(diffs_squared, axis=0)
    preds_squared = np.array(preds)**2
    E_preds_squared = np.mean(preds_squared, axis=0)
    return  pd.DataFrame({"time": all_explanations[0].timestamps[:last_index], "sigma": np.sqrt(E_diffs_squared) / np.sqrt(E_preds_squared), 
     "method": method_label, "cluster": cluster_label, "model": model_label })

def calculate_feature_orderings(explanations):
    """Calculates feature rankings and ranks from SurvShap declarations.
    Parameters
    ----------
    explanations : list
        List of SurvShap explanations for predictions.

    Returns
    -------
    pd.DataFrame
        Two DataFrames, one with the feature ranks and one with the feature ranks.

    """
    feature_importance_orderings = []

    for explanation in explanations:
        result_df = explanation.result.copy()
        cumulative_change = cumtrapz(np.abs(result_df.iloc[:, 5:].values), explanation.timestamps, initial=0, axis=1)[:, -1]
        result_df['aggregated_change'] = cumulative_change
        sorted_df = result_df.sort_values(by='aggregated_change', key=lambda x: -abs(x)).reset_index() 
        # reset the index to get the index as a column
        sorted_df.index = sorted_df['variable_name'] 
        # set the index to the 'Variable_name' column values
        feature_importance_orderings.append(sorted_df.index.to_list())

    
    return pd.DataFrame(feature_importance_orderings)


def create_ranking_summary(ordering):
    res = pd.DataFrame()

    for i in range(ordering.shape[1]):
        counts = ordering.iloc[:, i].value_counts().reset_index()
        counts.columns = ['variable', 'value']
        counts['importance_ranking'] = i + 1
        res = pd.concat([res, counts])
    
    return res[['importance_ranking', 'variable', 'value']]



def prepare_ranking_summary_long(ordering):
    """ Provide a summary of the ranking for a given Ordering DataFrame.

    Parameter
    ----------
    ordering : pd.DataFrame
        A DataFrame containing the ordering of the features.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the fields 'importance_ranking', 'variable' and 'value',
        where 'importance_ranking' is the rank of the variable,
        'variable' is the name of the variable
        and 'value' is the number of the variable in the given rank.
    """
    res = pd.DataFrame()
    #Iterate through each column in the Ordering df
    for i in range(ordering.shape[1]):
        ## Calculate value numbers for each variable
        counts = ordering.iloc[:, i].value_counts().reset_index()
        counts.columns = ['variable', 'value']
        # Assign a rank (i + 1) to each variable
        counts['importance_ranking'] = i + 1
        # Add the counts df to the resulting DataFrame
        res = pd.concat([res, counts])
    # Return the resulting DataFrame with selected columns
    return res[['importance_ranking', 'variable', 'value']]





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
def plotting_features_ranking_bars(data, title='', xtitle='', ytitle=''):
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
    ax.set(title=title, xlabel=xtitle)
    ax.set_ylabel(ytitle)
    # Reverse the order of y-axis
    ax.invert_yaxis()
    ax.legend(title='Examined Features', 
              loc='lower center', bbox_to_anchor=(1, 1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.tight_layout()
    plt.show()

def plot_SurvSHAP_values_to_time(survshap_csv_directory):
    # Get a list of all CSV files in the specified path
    search_path = os.path.join(survshap_csv_directory, '*.csv')

    # Get a list of all CSV files in the specified directory
    csv_files = glob.glob(search_path)
    # Initialize an empty DataFrame to store all data
    all_data = pd.DataFrame()

    # Iterate over all CSV files
    for file in csv_files:
        # Load the CSV file
        df = pd.read_csv(file)

        # Get the column names
        cols = df.columns

        # Identify columns to keep
        cols_to_keep = ['variable_name'] + [col for col in cols if col.startswith('t = ')]

        # Keep necessary columns and take absolute value of the 't = ' columns
        df = df[cols_to_keep]
        for col in df.columns:
            if col.startswith('t = '):
                df[col] = df[col].abs()

         # Append the data to the all_data DataFrame
        all_data = pd.concat([all_data, df], ignore_index=True)

    # Group by 'variable_name' and calculate mean
    average_data = all_data.groupby('variable_name').mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    max_time = 0  # Variable to track the maximum time value
    for variable in average_data['variable_name'].unique():
        data_subset = average_data[average_data['variable_name'] == variable]
        data_subset = data_subset.drop('variable_name', axis=1).mean()

        # Remove 't =' prefix from the first row
        data_subset.index = data_subset.index.str.replace('t = ', '')
        time_values = data_subset.index.astype(float)
        # Update the maximum time value
        max_time = max(max_time, max(time_values))  
        plt.plot(time_values, data_subset.values, label=variable)

    plt.xlabel('Time (days)')
    plt.ylabel('Aggregated |SurvSHAP Values|')
    plt.legend(title="Examined Feature")
    plt.grid(True)

    # Customize x-axis and y-axis ticks
    x_ticks = np.arange(0, max_time + 10, 10)
    plt.xticks(x_ticks, x_ticks.astype(int), fontsize=8)
    plt.yticks(fontsize=8)

    plt.show()

def plot_SurvSHAP_values_to_time_for_specific_file(csv_path, patient_id):

     # Construct the full path of the file
    file_path = os.path.join(csv_path, patient_id + '.csv')

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File {patient_id}.csv not found in the specified directory.")
        return

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Get the column names
    cols = df.columns

    # Identify columns to keep
    cols_to_keep = ['variable_name'] + [col for col in cols if col.startswith('t = ')]

    # Keep necessary columns and take absolute value of the 't = ' columns
    df = df[cols_to_keep]
    for col in df.columns:
        if col.startswith('t = '):
            df[col] = df[col].abs()

    # Group by 'variable_name' and calculate mean
    average_data = df.groupby('variable_name').mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    max_time = 0  # Variable to track the maximum time value
    for variable in average_data['variable_name'].unique():
        data_subset = average_data[average_data['variable_name'] == variable]
        data_subset = data_subset.drop('variable_name', axis=1).mean()

        # Remove 't =' prefix from the first row
        data_subset.index = data_subset.index.str.replace('t = ', '')
        time_values = data_subset.index.astype(float)
        # Update the maximum time value
        max_time = max(max_time, max(time_values))  
        plt.plot(time_values, data_subset.values, label=variable)

    plt.xlabel('Time (days)')
    plt.ylabel(f'|SurvSHAP Values| for {patient_id}' )
    plt.legend(title="Examined Feature")
    plt.grid(True)

    # Customize x-axis and y-axis ticks
    x_ticks = np.arange(0, max_time + 10, 10)
    plt.xticks(x_ticks, x_ticks.astype(int), fontsize=8)
    plt.yticks(fontsize=8)

    plt.show()

def plot_SurvSHAP_values_to_time_for_specific_file_and_feature(csv_path, patient_id, feature_to_plot=None):
    # Construct the full path of the file
    file_path = os.path.join(csv_path, patient_id + '.csv')

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File {patient_id}.csv not found in the specified directory.")
        return

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Get the column names
    cols = df.columns

    # Identify columns to keep
    cols_to_keep = ['variable_name'] + [col for col in cols if col.startswith('t = ')]

    # Keep necessary columns and take absolute value of the 't = ' columns
    df = df[cols_to_keep]
    for col in df.columns:
        if col.startswith('t = '):
            df[col] = df[col].abs()

    # Filter for a specific feature if provided
    if feature_to_plot:
        df = df[df['variable_name'] == feature_to_plot]

    # Group by 'variable_name' and calculate mean
    average_data = df.groupby('variable_name').mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    max_time = 0  # Variable to track the maximum time value
    for variable in average_data['variable_name'].unique():
        data_subset = average_data[average_data['variable_name'] == variable]
        data_subset = data_subset.drop('variable_name', axis=1).mean()

        # Remove 't =' prefix from the first row
        data_subset.index = data_subset.index.str.replace('t = ', '')
        time_values = data_subset.index.astype(float)
        # Update the maximum time value
        max_time = max(max_time, max(time_values))  
        plt.plot(time_values, data_subset.values, label=variable, color='brown')
        plt.xlabel('Time (days)')
    plt.ylabel(f'|SurvSHAP Values| for {patient_id}' )
    plt.legend(title="Examined Feature")
    plt.grid(True)

    # Customize x-axis and y-axis ticks
    x_ticks = np.arange(0, max_time + 10, 10)
    plt.xticks(x_ticks, x_ticks.astype(int), fontsize=8)
    plt.yticks(fontsize=8)

    plt.show()


def plotting_features_ranking_bars_x(data, feature_colors, title='', xtitle='', ytitle=''):
    """
    Plot a feature ranking bar chart with specified colors for each feature.

    Parameters:
    data (DataFrame): Input DataFrame containing the data with columns 'variable', 'importance_ranking', and 'value'.
    feature_colors (dict): Dictionary with features as keys and their corresponding colors as values.
    title (str): Title of the plot.
    xtitle (str): Title for the x-axis.
    ytitle (str): Title for the y-axis.
    """
    
    # Create a mapping for y-axis labels
    y_labels = {rank: f"{rank}{suffix}" for rank, suffix in zip(range(1, data['importance_ranking'].nunique() + 1), ['st', 'nd', 'rd'] + ['th'] * 20)}
    data['importance_ranking'] = data['importance_ranking'].replace(y_labels)

    plt.figure(figsize=(8, 4))
    for i, variable in enumerate(data['variable'].cat.categories):
        # Check if the variable has a specified color, default to a gray color if not specified
        color = feature_colors.get(variable, '#808080')

        # Get the subset of data for this variable
        data_subset = data[data['variable'] == variable].groupby(['variable', 'importance_ranking']).sum().reset_index()

        # Calculate the position for this bar based on previous bars
        if i == 0:
            left = None
        else:
            prev_data = data[data['variable'].isin(data['variable'].cat.categories[:i])]
            left = prev_data.groupby('importance_ranking')['value'].sum().reindex(data_subset['importance_ranking']).fillna(0)

        # Plot the bars
        bars = plt.barh(data_subset['importance_ranking'], data_subset['value'], 
                        color=color, edgecolor='white', 
                        left=left, label=variable, height= 0.75)
        
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
    ax.set(title=title, xlabel=xtitle)
    ax.set_ylabel(ytitle)
    ax.invert_yaxis()  # Reverse the order of y-axis
    ax.legend(title='Examined Features', loc='lower center', bbox_to_anchor=(1, 1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.tight_layout()
    plt.show()
