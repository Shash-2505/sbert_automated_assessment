'''
    Used to predict the box numbers for the correct answers provided by the students using machine learning. 
    
    Student answers are compared to the true answers corresponding to each box using cosine similarity
    of the sentence embeddings. A classifier is trained to estimate box numbers using question, student answer sentence embeddings,
    and similarity scores.
'''

#------------------------------------Load modules--------------------------------------------------
import numpy as np
import pandas as pd

from sbert_automated_assessment.internal_libs.data_preprocess import preprocess_answer_data, exclude_box_true_answers, text_to_embeddings, semantic_sim_student_true_answer
from sbert_automated_assessment.internal_libs.multiscorer import MultiScorer
from sbert_automated_assessment.internal_libs.preprocess_train_svm import encode_and_bind, create_train_test_data, parameter_tune_svm_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from statistics import mean

#-----------------------------------Input variables------------------------------------------------
# Path to data folder (Data unavailable due to privacy concerns)
data_folder = './data/input/'

# Path for embeddings folder (Data unavailable due to privacy concerns)
embeddings_folder = './data/embeddings/'

# Path for output
output_folder = './results/'

# Title of sentence transformer used for embedding
sentence_transformer = 'jegormeister/robbert-v2-dutch-base-mqa-finetuned' #SB1 in the paper

# Desar dataset chosen as it has box numbers accurately coded
dataset_name = ['Desar']
# Box numbers 1-4 are for correct answers (0 is for incorrect answers)
box_numbers = [1,2,3,4]

# True answer boxes to exclude
# These boxes are filled for the students
exclude_boxes = {'Beton': 5,
                 'Botox': 1,
                 'Geld' : 5,
                 'Metro': 1,
                 'Muziek': 2,
                 'Suez': 5}

#---------------------------------------Read data--------------------------------------------------
# Training data with text, answers, box number, and right/wrong human coding
answers_coded = pd.read_excel(data_folder + "20230804_student_answers_all_datasets.xlsx", sheet_name="labeled")

# Actual correct answers to each text and corresponding box
true_answers = pd.read_csv(data_folder + "True_Answers.csv", sep=';')

#------------------------------------Create functions--------------------------------------------
def svm_cross_val_score(svm_gridsearch:GridSearchCV, data_x:pd.DataFrame, data_y:list, average_method:str='macro') -> dict:
    '''
    Create 10 fold cross-validated accuracy, precision, recall, and f1 scores for a model on the full dataset

    Parameters
    ----------
    svm_gridsearch : GridSearch output
    data_x         : Predictors
    data_y         : Outcome
    average_metric : Averaging for precision, recall, and f1 if multi-class outcome

    Returns
    -------
    cross_val_results: Cross-valiated prediction scores

    '''
    # Initialize model with best parameters from gridsearch
    model = SVC(C=svm_gridsearch.best_params_['C'], 
                kernel=svm_gridsearch.best_params_['kernel'], 
                gamma=svm_gridsearch.best_params_['gamma'],  
                probability=True)
    
    # Generate scores
    scorer = MultiScorer({                                               
                            'accuracy': (accuracy_score, {}),
                            'precision': (precision_score, {'average': average_method}),    
                            'recall': (recall_score, {'average': average_method}),
                            'f1': (f1_score, {'average': average_method})
                        })
    
    cross_val_score(model, data_x, data_y, scoring=scorer, cv=10)
    cross_val_results = scorer.get_results()
    
    return cross_val_results

def svm_test_pred_score(y_test:list, y_pred:list, average_method:str='macro') -> dict:
    '''
    Create accuracy, precision, recall, and f1 scores for a model on test set prediction

    Parameters
    ----------
    y_test         : True or actual outcome
    y_pred         : Predicted outcome
    average_metric : Averaging for precision, recall, and f1 if multi-class outcome

    Returns
    -------
    performance_metrics : Dataframe with accuracy, precision, recall, f1 and cohen kappa scores

    '''
    
    # Store metrics
    performance_metrics = dict()
    performance_metrics['accuracy'] = accuracy_score(y_test, y_pred)    
    performance_metrics['precision'] = precision_score(y_test, y_pred, average=average_method, zero_division=np.nan)
    performance_metrics['recall'] = recall_score(y_test, y_pred, average=average_method, zero_division=np.nan)
    performance_metrics['f1'] = f1_score(y_test, y_pred, average=average_method, zero_division=np.nan)
    performance_metrics['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)
    
    # Convert to dataframe. Round and return
    performance_metrics = pd.DataFrame(performance_metrics.items(), columns=['metric', 'score'])
    performance_metrics['score'] = performance_metrics['score'].round(4)
    print(performance_metrics)
    
    return performance_metrics

#------------------------------------Data preprocessing--------------------------------------------
# Preprocess student answers data
remove_columns = ['Klas', 'Veldnummer', 'Remarks']
answers_preprocessed = preprocess_answer_data(answers_coded, dataset_name, box_numbers, remove_columns)

# Exclude given answer from true answers
true_answers = exclude_box_true_answers(true_answers, exclude_boxes)

#---------------------------------Generate Sentence embeddings-----------------------------------------
model_sbert_dutch = SentenceTransformer(sentence_transformer)

# Sentence embeddings for student answers from human coded dataset
student_answers = answers_preprocessed['Veld'].to_list()
answer_embeddings = text_to_embeddings(student_answers, model_sbert_dutch, generate=False, save=False,
                                       embeddings_folder=embeddings_folder, embedding_filename='Desar_Correct_Answer_Embedding_SB1')

# Sentence embeddings for true answers
true_answer_embeddings = pd.DataFrame(columns=true_answers.columns)
true_answer_embeddings['TextName'] = true_answers['TextName']
for col in true_answers.columns:
    if col != 'TextName':
        true_answers[col] = true_answers[col].str.lower()
        for i in range(true_answers.shape[0]):
            true_answer_embeddings[col].iloc[i] = text_to_embeddings([true_answers[col].iloc[i]], model_sbert_dutch, generate=True, save=False)
            
#------------------------------Semantic similarity-------------------------------------------------
answers_sim_scores = semantic_sim_student_true_answer(answers_preprocessed, answer_embeddings, true_answer_embeddings)

#------------------------------Create training and testing data-------------------------------------
data_x, data_y = encode_and_bind(answers_sim_scores, 'Verbandnummer', ['Tekstnaam'])

train_features_exclude = ['unique_id', 'Nummer', 'Veld', 'Code', 'Dataset']
test_size = 0.1
X_train, X_test, y_train, y_test = create_train_test_data(data_x, data_y, answer_embeddings, train_features_exclude, test_size)

#--------------------------------Find best parameters for SVM model---------------------------------
param_grid = {'C': [1,10,100], 'gamma': [0.1,0.01,0.001], 'kernel': ['rbf']}
n_splits = 10
n_repeats = 2
scoring = 'accuracy'
svm_gridsearch = parameter_tune_svm_model(X_train, y_train, param_grid, n_splits, n_repeats, scoring)

#Fine tune further
param_grid = {'C': [5,10,15,20], 'gamma': [0.001], 'kernel': ['rbf']}
svm_gridsearch = parameter_tune_svm_model(X_train, y_train, param_grid, n_splits, n_repeats, scoring)
#best params: C=10, gamma=0.001

#--------------------------------Performance metrics----------------------------------------------
# Cross-validated scores
data_x = data_x.drop(columns=train_features_exclude)
data_x = pd.concat([data_x, pd.DataFrame(answer_embeddings)], axis='columns')
data_x.columns = data_x.columns.astype(str)
cross_val_results = svm_cross_val_score(svm_gridsearch, data_x, data_y, average_method='macro') 
for metric in cross_val_results.keys():                                        
        print("%s: %.3f" % (metric, mean(cross_val_results[metric])))

# Test prediction scores
y_pred = svm_gridsearch.predict(X_test)
performance_metrics = svm_test_pred_score(y_test, y_pred)

# Save output to excel
with pd.ExcelWriter(output_folder + "SVM_Test_Prediction_Score_SB1.xlsx") as writer:
    performance_metrics.to_excel(writer, sheet_name="Test_Pred_Scores", index=False)

