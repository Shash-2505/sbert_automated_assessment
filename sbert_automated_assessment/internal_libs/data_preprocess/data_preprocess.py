'''
    Contains functions required for preprocessing student and true answers, generating or loading text embeddings, and 
    creating similarity scores between student and true answers.
'''

#------------------------------------Load modules--------------------------------------------------
import numpy as np
import pandas as pd
import pickle

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


#------------------------------------Create functions--------------------------------------------
def preprocess_answer_data(data:pd.DataFrame, dataset_name:list, box_numbers:list, remove_columns:list) -> pd.DataFrame:
    '''
    Filter for dataset name and preprocess student answer text

    Parameters
    ----------
    data          : Data with text or question type, student answer text, box number, and right/wrong human coding
    dataset_name  : List of the dataset names to use
    box_numbers   : Box numbers to filter from "Verbandnummer"
    remove_columns: Fields to exclude from input data
    
    Returns
    -------
    data_preprocessed : Filtered and student answer text processed dataset.

    '''
    # Check if input data file is correctly loaded
    assert data.shape[0] != 0, 'Empty dataset, check input file'

    # Filter for dataset
    data = data[data['Dataset'].isin(dataset_name)]
    # Choose box numbers
    data = data[data['Verbandnummer'].isin(box_numbers)]
    if dataset_name == ['Desar']:
        assert data.shape[0] == 1563, "Check if Desar dataset is complete in input file"
           
    # Update code field
    data['Code'] = data['Code'].str.replace('f', 'c')
    data['Code'] = data['Code'].str.replace('3', 'g')
    # 'g'=correct and 'c'=wrong answer
    data = data[(data['Code']=='g') | (data['Code']=='c')]
    data = data.reset_index().drop(columns = "index")
    
    # Replace cv with centrale verwarming
    data['Veld'] = data['Veld'].str.lower()
    data['Veld'] = data['Veld'].str.replace(r'\bcv','centrale verwarming', regex=True)
      
    # Sort and reset index
    data.sort_values(['Nummer', 'Tekstnaam'], inplace=True, ignore_index=True)
    
    # Add unique ID per row
    data['index'] = list(data.index.values)
    data['unique_id'] = data['Nummer'] + '_' + data['index'].astype(str)
    
    # Keep only required fields
    remove_columns.append('index')
    data = data.drop(columns=remove_columns)
    
    # Rearrange columns
    data_columns = list(data.columns)
    data_columns.remove('unique_id')
    data_columns.insert(0, 'unique_id')
    data_preprocessed = data[data_columns]
    
    return data_preprocessed

def exclude_box_true_answers(true_answers:pd.DataFrame, exclude_boxes:dict) -> pd.DataFrame:
    '''
    Exclude the box with the answer given to the student from the true answers

    Parameters
    ----------
    true_answers        : Dataframe with all five true answers per text name 
    exclude_boxes       : Box numbers per text name to exclude

    Returns
    -------
    true_answers_4boxes : Dataframe with four true answers for boxes that students had to fill 

    '''
    # Combine all answers to a list
    true_answers['answers_combined'] = true_answers.iloc[:, 1:].values.tolist()
    
    # Join the box numbers to be excluded
    exclude_boxes = pd.DataFrame(exclude_boxes.items(), columns=['TextName', 'exclude_number'])
    true_answers = true_answers.merge(exclude_boxes, how='left', on='TextName') 
    
    # Exclude answers given to the students
    for i in range(true_answers.shape[0]):
        del true_answers['answers_combined'].iloc[i][true_answers['exclude_number'].iloc[i] - 1]
    
    # Expand combined answers to four fields
    field_names = ['Field1', 'Field2', 'Field3', 'Field4']
    true_answers_4boxes = pd.DataFrame(true_answers['answers_combined'].to_list(), columns=field_names)
    
    # Add TextName and rearrange columns
    true_answers_4boxes['TextName'] = true_answers['TextName']
    field_names.insert(0, 'TextName')
    true_answers_4boxes = true_answers_4boxes[field_names]
    
    return true_answers_4boxes

def text_to_embeddings(text_data:list, sentence_transformer_model:SentenceTransformer, generate:bool, save:bool, embeddings_folder:str=None, embedding_filename:str=None) -> np.array:
    '''
    Generate sentence embeddings from text or load saved embeddings from a file

    Parameters
    ----------
    text_data                  : Preprocessed text data in lower case
    sentence_transformer_model : Sentence transformer model used for creating embeddings
    generate                   : Indicator to create embeddings(1) or load from file(0)
    save                       : Indicator to save generated embeddings(1) or not(0)
    embeddings_folder          : Name of folder where embeddings are stored
    embedding_filename         : Name of file with embeddings

    Returns
    -------
    embeddings                 : Sentence embedding array of 768 vectors per text input

    '''
    if generate:
        embeddings = sentence_transformer_model.encode(text_data)
        if save:
            with open(embeddings_folder + embedding_filename, 'wb') as f:
                pickle.dump(embeddings , f)
    else:
        pickle_in = open(embeddings_folder + embedding_filename, 'rb')
        embeddings = pickle.load(pickle_in)
    
    return embeddings

def semantic_sim_student_true_answer(answers_preprocessed:pd.DataFrame, answer_embeddings:np.array, true_answer_embeddings:pd.DataFrame) -> pd.DataFrame:
    '''
    Find semantic similarity between each student answer and true answers for 5 boxes of a text.
    Add the scores as 5 new columns to the dataset.
    Cosine similarity scores are chosen as Sentence BERT models are trained with it. 

    Parameters
    ----------
    answers_preprocessed   : Dataset with student answers and corresponding text name
    answer_embeddings      : Array of sentence embeddings for the student answers 
    true_answer_embeddings : Sentence embeddings of true answers

    Returns
    -------
    answers_sim_scores     : Dataset with similarity scores with 5 true answers

    '''
    # Create a dataset with 4 columns with similarity scores
    answers_sim_scores = answers_preprocessed.copy()
    similarity_column_name = []
    for i in range(1,5):
        col_name = 'true_answer_sim_' + str(i)
        similarity_column_name.append(col_name)
        answers_sim_scores[col_name] = np.nan
    
    # Dataset will contain cosine similarity score for each student answer compared with the 5 true answers for the corresponding text
    true_answer_columns = true_answer_embeddings.columns.tolist()[1:]
    for i in range(np.shape(answer_embeddings)[0]):
        for j in range(len(similarity_column_name)):
            answers_sim_scores[similarity_column_name[j]].iloc[i] = cos_sim(
                answer_embeddings[i].reshape(1,-1), 
                true_answer_embeddings[true_answer_columns[j]][true_answer_embeddings['TextName'] == answers_sim_scores['Tekstnaam'].iloc[i]].to_numpy()[0].reshape(1,-1)).data.cpu().numpy()
    return answers_sim_scores