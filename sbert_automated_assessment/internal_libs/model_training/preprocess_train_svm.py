'''
    Contains functions to encode data, create train and test splits, hyperparameter search for SVM to find the best model.
'''
#------------------------------------Load modules--------------------------------------------
import numpy as np
import pandas as pd


from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

#------------------------------------Create functions--------------------------------------------
def encode_and_bind(data:pd.DataFrame, y_var:str, features_to_encode:list) -> pd.DataFrame:
    '''
    One-hot encoding of categorical predictors and label encoding for outcome as [0,1,...n]

    Parameters
    ----------
    data               : Dataset to encode with predictor and outcome variables
    features_to_encode : Feature names to encode
    y_var              : Outcome feature

    Returns
    -------
    data_x_encoded     : Dataset with encoded features
    data_y             : Outcome encoded  

    '''
    # Encode outcome
    data_y = LabelEncoder().fit_transform(data[y_var])
    
    # One-hot encode x variables
    data_x_encoded = data.drop(columns=[y_var])
    
    for feature in features_to_encode:
        features_encoded = pd.get_dummies(data_x_encoded[feature], prefix = feature).reset_index().drop(columns = "index")
        data_x_encoded = pd.concat([data_x_encoded, features_encoded], axis = "columns")
        data_x_encoded = data_x_encoded.drop(columns = feature)
    
    return data_x_encoded, data_y

def create_train_test_data(data_x:pd.DataFrame, data_y:list, answer_embeddings:np.array, train_features_exclude:list, test_size:float) -> pd.DataFrame:
    '''
    Training and test data generation

    Parameters
    ----------
    data_x                 : All input features with categorical variables encoded
    data_y                 : Outcome label encoded
    answer_embeddings      : Array of text embeddings
    train_features_exclude : Predictors to exclude
    test_size              : Test data size as a proportion of dataset

    Returns
    -------
    X_train, X_test        : Train and test set for predictors
    y_train, y_test        : Train and test set for outcome

    '''
    # Exclude features not used for prediction
    data_x = data_x.drop(columns=train_features_exclude)
    # Add text embeddings
    data_x = pd.concat([data_x, pd.DataFrame(answer_embeddings)], axis='columns')
    data_x.columns = data_x.columns.astype(str)
    # Create train test split
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

def parameter_tune_svm_model(X_train:list, y_train:list, param_grid:dict, n_splits:int, n_repeats:int, scoring:str='accuracy') -> GridSearchCV:
    '''
    Tune SVM model

    Parameters
    ----------
    X_train    : Train set of predictors
    y_train    : Train set of outcomes
    param_grid : List of all SVM parameters for gridsearch
    n_splits   : Number of stratified folds (K)
    n_repeats  : Number of times K fold is repeated
    scoring    : Metric to tune parameters for

    Returns
    -------
    svm_gridsearch : Gridsearch outcome with best parameters and scores

    '''
    svm_model = SVC()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)        
    svm_gridsearch = GridSearchCV(svm_model, param_grid=param_grid, scoring=scoring, cv=cv, verbose=3)
    svm_gridsearch.fit(X_train, y_train)
    print("Best SVM params:", svm_gridsearch.best_params_)
    print(f"Best {scoring}: {svm_gridsearch.best_score_}")
    
    return svm_gridsearch