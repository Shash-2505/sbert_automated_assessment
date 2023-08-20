'''
    Used to determine the effectiveness of the various strategies used to estimate the box numbers
    for the correct answers provided by the students. 
    
    Student answers are compared to the true answers corresponding to each box using cosine similarity
    of the sentence embeddings. Three approaches are used to identify the box numbers:
        1. Match the student answers to true answers with highest semantic similarity
        2. Match the student answers to true answers with highest semantic similarity with an additional
           constraint that each student answer gets assigned only to one box. If the box corresponding to
           the maximum similarity is already assigned then the next highest similarity is assigned. 
           Assignments among answers are done from highest to least similarity
        3. Matching student answers to true answers is treated as a machine learning classification problem
    The following code is for approach 1 and 2.
'''

#------------------------------------Load modules--------------------------------------------------
import numpy as np
import pandas as pd

from sbert_automated_assessment.internal_libs.data_preprocess import preprocess_answer_data, exclude_box_true_answers, text_to_embeddings, semantic_sim_student_true_answer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
def calculate_box_match_scores(box_numbers_coded:pd.Series, box_numbers_pred:pd.Series, average_method:str='macro') -> pd.DataFrame:
    '''
    Calculate performance for box match prediction.
    Used for approach 1 and 2 where all predictions are used for scoring without cross-validation

    Parameters
    ----------
    box_numbers_coded : Human assigned box numbers from the dataset
    box_numbers_pred  : Predicted box numbers
    average_method    : Averaging for multiclass precision, recall, and F1 scores
                        'micro' = global average. 'macro' = class-wise average
                        'weighted' = weighted average by number of true instances of each class

    Returns
    -------
    performance_metrics : Dataframe with accuracy, precision, recall, and f1 scores

    '''
    
    # Drop any unknown predictions
    box_numbers = pd.concat([box_numbers_coded, box_numbers_pred], axis=1)
    box_numbers.dropna(inplace=True)
    
    # Create list of true and predicted values
    box_numbers_coded = list(box_numbers.iloc[:,0])
    box_numbers_pred = list(box_numbers.iloc[:,1])
    
    # Store metrics
    performance_metrics = dict()
    performance_metrics['accuracy'] = accuracy_score(box_numbers_coded, box_numbers_pred)    
    performance_metrics['precision'] = precision_score(box_numbers_coded, box_numbers_pred, average=average_method, zero_division=np.nan)
    performance_metrics['recall'] = recall_score(box_numbers_coded, box_numbers_pred, average=average_method, zero_division=np.nan)
    performance_metrics['f1'] = f1_score(box_numbers_coded, box_numbers_pred, average=average_method, zero_division=np.nan)
    
    # Convert to dataframe. Round and return
    performance_metrics = pd.DataFrame(performance_metrics.items(), columns=['metric', 'score'])
    performance_metrics['score'] = performance_metrics['score'].round(2)
    print(performance_metrics)
    
    return performance_metrics

#------------------------------------Data preprocessing--------------------------------------------
# Preprocess student answers data
remove_columns = ['Klas', 'Veldnummer', 'Remarks']
answers_preprocessed = preprocess_answer_data(answers_coded, dataset_name, box_numbers, remove_columns)

# Exclude given answer from true answers
true_answers = exclude_box_true_answers(true_answers, exclude_boxes)

#---------------------------------Generate Sentence embeddings-----------------------------------------
# Sentence embeddings for student answers from human coded dataset
model_sbert_dutch = SentenceTransformer(sentence_transformer)
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
       
#--------------------------Estimate box numbers - Approach 1 and 2---------------------------------------
answers_sim_scores[['box_number_1', 'box_number_2']] = np.nan
col_min = answers_sim_scores.columns.get_loc('true_answer_sim_1')
col_max = answers_sim_scores.columns.get_loc('true_answer_sim_4')
# Approach 1 - maximum similarity score for each answer
for i in range(answers_sim_scores.shape[0]):
    answers_sim_scores['box_number_1'].iloc[i] = np.argmax(answers_sim_scores.iloc[i, col_min:col_max+1]) + 1

# Approach 2 - maximum similarity score with a constraint that each box number gets assigned only once per text
for student_id in answers_sim_scores['Nummer']:
    answers_student = answers_sim_scores[answers_sim_scores['Nummer'] == student_id]
    for text_name in answers_student['Tekstnaam']:
        answers_student_text = answers_student[answers_student['Tekstnaam'] == text_name]
        # If only one answer for the text then same as approach 1
        if answers_student_text.shape[0] == 1:
            answers_sim_scores['box_number_2'][answers_sim_scores['unique_id'] == answers_student_text['unique_id'].iloc[0]] = np.argmax(answers_student_text.iloc[0, col_min:col_max+1]) + 1
        else:
            # Check if there are more than 4 answers from a student for a text
            assert answers_student_text.shape[0] < 5, f"More than expected answers for student {student_id}, textname {text_name}"
            # Find max value in each row
            answers_student_text['max_sim'] = answers_student_text.iloc[:,col_min:col_max+1].max(axis=1)
            # Sort so that assignment within each text is from the highest score
            answers_student_text.sort_values(['max_sim'], ascending=False, inplace=True)
            # Boxes is a list to keep track of boxes that are yet to be assigned
            boxes = box_numbers.copy()
            boxes.append(5)
            for i in range(answers_student_text.shape[0]):
                find_match = True
                sim_scores = list(answers_student_text.iloc[i, col_min:col_max+1])
                while find_match:
                    box_number_match = np.argmax(sim_scores) + 1
                    if box_number_match in boxes:
                        find_match = False
                        # Once match found remove it from boxes
                        boxes.remove(box_number_match)
                    else:
                        # If box already assigned then make similarity score 0 so that next most similar box can be matched
                        sim_scores[box_number_match - 1] = 0
                # Assign the match after the match has been found
                answers_sim_scores['box_number_2'][answers_sim_scores['unique_id'] == answers_student_text['unique_id'].iloc[i]] = box_number_match
                
score_approach_1 = calculate_box_match_scores(answers_sim_scores['Verbandnummer'], answers_sim_scores['box_number_1'], average_method='macro')
score_approach_2 = calculate_box_match_scores(answers_sim_scores['Verbandnummer'], answers_sim_scores['box_number_2'], average_method='macro')

# Save output to excel
with pd.ExcelWriter(output_folder + "Similarity_Performance_Score_SB1.xlsx") as writer:
    answers_sim_scores.to_excel(writer, sheet_name="Answers", index=False)
    score_approach_1.to_excel(writer, sheet_name='Metrics_Approach_I', index=False)
    score_approach_2.to_excel(writer, sheet_name='Metrics_Approach_II', index=False)

