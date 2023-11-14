import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import validating_models.stats as stats
from sksurv.util import Surv
time_preprocessing = stats.get_decorator('PIPE_PREPROCESSING')


def define_class(classes, dependent_var, annotated_dataset):
    """Define classes specified by user in the dataset extracted from knowledge graphs.

    Parameters
    ----------
    classes : list
        List of classes defined by user in input file.
    dependent_var : dataframe
        Target variable column of dataframe.
    annotated_dataset : dataframe
        The dataframe extracted from knowledge graph.

    Returns
    -------
    dataframe

    """
    cls = {}
    target = annotated_dataset[dependent_var]
    length = len(classes)
    if length >= 2:
        if length == 2:
              class0, class1 = map(str, classes)
              id_0 = target.index[target.iloc[:, 0].astype(str).str.contains(class0)]
              target.loc[(target.index.isin(id_0)), 'class'] = 0
              target.loc[~(target.index.isin(id_0)), 'class'] = 1
              target = target[['class']]
        else:
            for i, j in enumerate(classes):
                cls[j] = i
            target['class'] = target.iloc[:, 0].map(cls)
            if target['class'].isnull().values.any():
                n = len(classes)
                target['class'] = target['class'].replace(np.nan, n-1)
            target = target[['class']]
    else:
        print("Error - less than 2 classes given for classification")

    return target

# def define_class(classes, dependent_var, annotated_dataset):
#     """Define classes specified by user in the dataset extracted from knowledge graphs.

#     Parameters
#     ----------
#     classes : list
#         List of classes defined by user in input file.
#     dependent_var : dataframe
#         Target variable column of dataframe.
#     annotated_dataset : dataframe
#         The dataframe extracted from knowledge graph.

#     Returns
#     -------
#     dataframe

#     """
#     if len(dependent_var) == 2:
#         # If there are two dependent variables (like "time" and "event"), return them as they are
#         target = annotated_dataset[dependent_var]
#         event_col = dependent_var[1]  # Assuming 'event' is the second dependent variable
        
#         # Create a class mapping based on the classes list
#         class_mapping = {str(i): class_name for i, class_name in enumerate(classes)}
        
#         target[event_col] = target[event_col].map(class_mapping)
#         return target
#     else:
#         # Original logic for classifying
#         cls = {}
#         target = annotated_dataset[dependent_var]
#         length = len(classes)
#         if length >= 2:
#             if length == 2:
#                   class0, class1 = map(str, classes)
#                   id_0 = target.index[target.iloc[:, 0].astype(str).str.contains(class0)]
#                   target.loc[(target.index.isin(id_0)), 'class'] = 0
#                   target.loc[~(target.index.isin(id_0)), 'class'] = 1
#                   target = target[['class']]
#             else:
#                 for i, j in enumerate(classes):
#                     cls[j] = i
#                 target['class'] = target.iloc[:, 0].map(cls)
#                 if target['class'].isnull().values.any():
#                     n = len(classes)
#                     target['class'] = target['class'].replace(np.nan, n-1)
#                 target = target[['class']]
#         else:
#             print("Error - less than 2 classes given for classification")
#         return target


def transform_to_binary(data, attribute, val_a, val_b):
    data.loc[data[attribute] == val_a, attribute] = 0
    data.loc[data[attribute] == val_b, attribute] = 1
    return data.rename(columns={attribute: attribute + '_' + val_b})


def hot_encode(data, seed_var):
    """One-hot encoding is method of converting data to prepare it for an algorithm and get a better prediction

    Parameters
    ----------
    data : dataframe
        Dataframe on which one-hot encoding needs to be performed.
    seed_var : str
        Index variable used to identify the entity.

    Returns
    -------
    dataframe

    """
    col_list = []
    count = data.T.apply(lambda x: x.nunique(dropna=False), axis=1)
    for col_name, v in count.items():
        if v == 2:
            col_val = data[col_name].values.ravel()
            unique = pd.unique(col_val)
            val_a, val_b = ["".join(item) for item in unique.astype(str)]
            data = transform_to_binary(data, col_name, val_a, val_b)
        else:
            if col_name != seed_var:
                col_list.append(col_name)
    new_data = pd.get_dummies(data=data, columns=col_list)
    return new_data


@time_preprocessing
def load_data(seed_var,independent_var, dependent_var, classes, annotated_dataset, survival):
    """Preprocessing (one-hot encoding) the dataset extracted from input knowledge graph.

    Parameters
    ----------
    seed_var : str
        Index variable used to identify the entity.
    independent_var : list
        A list of independent variables from input knowledge graph.
    dependent_var : str
        Target variable.
    classes : list
        A list of classes used for classification.
    annotated_dataset : dataframe
        Dataset extracted from input knowledge graph.

    Returns
    -------
    (dataframe, dataframe)
        Returns preprocessed dataset.

    """
    # print("--------- Preprocessing Data --------------")
    # print(annotated_dataset)
    if survival == 0:
        independent_var = [var for var in independent_var if var != seed_var]
        encode_target = define_class(classes, dependent_var, annotated_dataset)
        ann_data = annotated_dataset.drop(dependent_var, axis=1)
        independent_data = ann_data[independent_var]
        encode_data = hot_encode(independent_data, seed_var)

        return encode_data, encode_target
    elif survival==1:
        # Remove specific value 'seed_var' from independent_var list
        independent_var = [var for var in independent_var if var != seed_var]
        ann_data = annotated_dataset.drop(['event', 'time'], axis=1)
        encode_data = ann_data[independent_var]
        encode_target = Surv.from_dataframe("event", "time", annotated_dataset)
        new_dtype = [('event', 'bool'), ('time', 'float64')]
        encode_target = np.array(encode_target, dtype=new_dtype)

        return encode_data, encode_target
