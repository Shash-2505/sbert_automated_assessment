import pandas as pd
import numpy as np
import os
import pickle
import re
from sentence_transformers import SentenceTransformer

#change directory to where input data, answer embeddings, and model is saved
os.chdir("C://gorilla")

#test input to receive from API
textname = 'Beton'

fields = {"1":"Betonnen gebouwen bouwen",
          "2":"Beton droogt uit",
          "3":"?",
          "4":"Liften blijven hangen",
          "5": "In betonnen gebouwen is vaak liftrenovatie nodig"}


#------------------------identify omissions---------------------------------
fields_out = pd.DataFrame.from_dict(fields, orient='index', columns=['answer'])
fields_out['code'] = np.nan

#text preprocessing to check for various input combinations that could be omissions
#remove spaces if they are the first character in the answer
fields_out['answer'] = fields_out['answer'].str.replace('^\s', "", regex=True)
#if answer is empty then set code as 'o' or omission
#as space is removed if it is the first character, just a space will also be considered as omission
fields_out['code'][fields_out['answer'] == ''] = 'o'
#if answer begins with '?' then set code as 'o' or omission
fields_out['code'][fields_out['answer'].str.match('^\?')] = 'o'
#if answer begins with '/' then set code as 'o' or omission
fields_out['code'][fields_out['answer'].str.match('^/')] = 'o'
#if answer begins with '-' then set code as 'o' or omission
fields_out['code'][fields_out['answer'].str.match('^-')] = 'o'
#if answer is only a number with no text then set code as 'o' or omission
fields_out['code'][fields_out['answer'].str.match('^[0-9]+$')] = 'o'
#the first few conditions ensure any field which has a combination of '?', '/', or '-' is an omission
#if answer contains a word then it is NOT an omission
fields_out['code'][fields_out['answer'].str.contains('[a-zA-Z]')] = np.nan

#--------------prepare input data with non-omitted answers for predictions---------------
#check if at least one answer is a non-omission
if fields_out['code'].isna().sum() > 0:
    #centrale verwarming to cv
    #replace cv with centrale verwarming
    if textname == 'Beton':
        fields_out['answer'] = fields_out['answer'].str.replace(r'\bcv','centrale verwarming')
    
    #get sentence embeddings
    model_snli = SentenceTransformer('jegormeister/bert-base-dutch-cased-snli')
    sentences = fields_out['answer'][fields_out['code'] != 'o'].str.lower().tolist()
    embeddings_snli = model_snli.encode(sentences)
    
    #create a dataframe with input data in the same format as training data
    #one-hot encode text data which is a categorical variable
    text_names = ['Beton', 'Botox', 'Geld', 'Metro', 'Muziek', 'Suez']
    text_encoded = pd.DataFrame(text_names, columns = ['tekstnaam'])
    text_encoded = pd.get_dummies(text_encoded['tekstnaam'], prefix = 'tekstnaam').reset_index().drop(columns = "index")
    #for column which matches the textname input all rows are 1. other textname columns will have 0
    for col in text_encoded.columns:
        if re.findall(textname, col) != []:
            text_encoded[col] = 1
        else:
            text_encoded[col] = 0
    #combine SBERT embeddings and one-hot encoded text name to create input data for prediction
    text_encoded = text_encoded[0:len(embeddings_snli)]
    data_x = pd.concat([text_encoded, pd.DataFrame(embeddings_snli)], axis='columns')
    
    #-------------predict correct or wrong answers and output results--------------
    #load saved model
    filename = 'model_svm.sav'
    model = pickle.load(open(filename, 'rb'))
    y_pred = model.predict(data_x)
    
    #add predictions to output file
    pred_ctr = 0
    for i in range(len(fields_out)):
        if fields_out['code'].iloc[i] != 'o':
            if y_pred[pred_ctr] == 1:
                fields_out['code'].iloc[i] = 'g'
            else:
                fields_out['code'].iloc[i] = 'c'
            pred_ctr+=1


#assigning boxes based on similarity
#assignment done only if at least one answer is 'g'
if 'g' in fields_out['code'].values:
    #load correct answers
    pickle_in = open('Correct_Answers_Embeddings.pickle', 'rb')
    answer_embeddings = pickle.load(pickle_in)
    
    #find similarity score with answers
    from sklearn.metrics.pairwise import cosine_similarity
    sim_score = {'field': [],
                 'box_num':[],
                 'cos_sim':[]}
    #find box number for only for correct answers
    k=0
    for i in range(len(fields_out)):
        if fields_out.iloc[i]['code'] == 'o':
            continue
        elif fields_out.iloc[i]['code'] == 'c':
            k+=1
        else:
            for j in range(answer_embeddings.shape[1]):           
                sim_score['field'].append(fields_out.index[i])
                sim_score['box_num'].append(j+1)
                sim_score['cos_sim'].append(cosine_similarity(embeddings_snli[k].reshape(1,-1),answer_embeddings.loc[textname][j].reshape(1,-1))[0,0])
            k+=1
    
    #algorithm to identify box associated with each answer
    fields_out['box_num'] = np.nan #output that will contain the matched box number
    #if answer is omission or incorrect box_num is 0
    fields_out['box_num'][fields_out['code'] != 'g'] = 0
    sim_score_df = pd.DataFrame(sim_score)
    sim_score_df = sim_score_df.sort_values(by=['cos_sim'], ascending=False).reset_index(drop=True)
    for i in range(len(fields_out)):
        if fields_out['code'].iloc[i] == 'g':
            #take field and box_num of highest entry
            f = sim_score_df.loc[0]['field']
            b = sim_score_df.loc[0]['box_num']
            #assign box_num to field with highest similarity score
            fields_out['box_num'].loc[f] = b
            #delete the box number assigned so that it is not assigned again
            sim_score_df = sim_score_df[sim_score_df['field'] != f]
            sim_score_df = sim_score_df[sim_score_df['box_num'] != b]
            sim_score_df = sim_score_df.reset_index(drop=True)
else:
    fields_out['box_num'] = 0    

#output dictionary to be returned
res = {}
res = fields_out[['code', 'box_num']].to_dict()
print(res)