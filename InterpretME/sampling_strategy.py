import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def sampling_strategy(encode_data, encode_target, strategy, results, survival):
    """Sampling strategy to balance the imbalanced data for predictive model.

    Parameters
    ----------
    encode_data : dataframe
        Preprocessed unbalanced dataframe needed to apply sampling.
    encode_target : dataframe
        Preprocessed unbalanced target dataframe needed to apply sampling.
    strategy : str
        Sampling strategy specified by user (undersampling or oversampling).
    results : dict
        To store results of sampling plots.

    Returns
    -------
    (dataframe, dataframe, dict)
        Sampled dataa and results dictionary.

    """
    if strategy == 'undersampling':
        X, y, results = undersampling(encode_data, encode_target, results, survival)
    if strategy == 'oversampling':
        X, y, results = oversampling(encode_data, encode_target, results, survival)
    return X, y, results


def undersampling(encode_data, encode_target, results, survival):
    """Under-Sampling strategy.

    Parameters
    ----------
    encode_data : dataframe
        Preprocessed unbalanced dataframe needed to apply sampling.
    encode_target : dataframe
        Preprocessed unbalanced target dataframe needed to apply sampling.
    results : dict
        To store results of sampling plots.

    Returns
    -------
    (dataframe, dataframe, dict)
        Sampled data and results dictionary.

    """
    sampling_strategy = "not minority"
    X = encode_data
    Y = encode_target
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=123)
    X_res, y_res = rus.fit_resample(X, Y['class'])
    if survival == 1:
        y_res_time = Y['time'].iloc[rus.sample_indices_]
        # Combine 'class' and 'time' back into a dataframe
        y_res = pd.DataFrame({
            'class': y_res,
            'time': y_res_time
        })
    X_res.index = X.index[rus.sample_indices_]
    y_res.index = Y.index[rus.sample_indices_]
    samp = y_res.value_counts()
    results['sampling'] = samp
    y_res = pd.DataFrame(data=y_res)
    print('this is y_res', y_res)
    return X_res, y_res, results


def oversampling(encode_data, encode_target, results, survival):
    """Over-Sampling strategy.

    Parameters
    ----------
    encode_data : dataframe
        Preprocessed unbalanced dataframe needed to apply sampling.
    encode_target : dataframe
        Preprocessed unbalanced target dataframe needed to apply sampling.
    results : dict
        To store results of sampling plots.

    Returns
    -------
    (dataframe, dataframe, dict)
        Sampled dataa and results dictionary.

    """
    sampling_strategy = "not majority"
    X = encode_data
    Y = encode_target
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=123)
    X_res, y_res = rus.fit_resample(X, Y['class'])
    if survival == 1:
        y_res_time = Y['time'].iloc[rus.sample_indices_]
        # Combine 'class' and 'time' back into a dataframe
        y_res = pd.DataFrame({
            'class': y_res,
            'time': y_res_time
        })
    X_res.index = X.index[rus.sample_indices_]
    y_res.index = Y.index[rus.sample_indices_]
    samp = y_res.value_counts()
    results['sampling'] = samp
    y_res = pd.DataFrame(data=y_res)
    print('this is y_res', y_res)
    return X_res, y_res, results

