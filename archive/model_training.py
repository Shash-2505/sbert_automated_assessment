import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

#read data
os.chdir("C://Users//Shashank//OneDrive//Documents//DSDM//Automatic Scoring App//Data//Input")
data = pd.read_excel("Human_Coded_Answers.xlsx", sheet_name="Data")

#keep only required fields
drop_columns = ['Nummer', 'Klas', 'Veldnummer']
data = data.drop(columns=drop_columns)
#update code field
data['Code'] = data['Code'].str.replace('f', 'c')
data['Code'] = data['Code'].str.replace('3', 'g')
data = data[(data['Code']=='g') | (data['Code']=='c')]
data = data.reset_index().drop(columns = "index")
#replace cv with centrale verwarming
data['Veld'] = data['Veld'].str.lower()
data['Veld'] = data['Veld'].str.replace(r'\bcv','centrale verwarming')

'''#embed student answers with sentence transformer
#embed student answers
from sentence_transformers import SentenceTransformer
sentences = data['Veld'].tolist()
model_snli = SentenceTransformer('jegormeister/bert-base-dutch-cased-snli')
embeddings_snli = model_snli.encode(sentences)

#save sentence embeddings
with open('sentence_embeddings_snli', "wb") as f:
    pickle.dump(embeddings_cv_snli, f)
'''
    
#load embeddings
os.chdir("C://Users//Shashank//OneDrive//Documents//DSDM//Automatic Scoring App//Data//Embeddings")
pickle_in = open('sentence_embeddings_processed_snli', 'rb')
embeddings_snli = pickle.load(pickle_in)
    
#create correct answer paragraph for each field
correct_answers = pd.read_csv("True_Answers.csv", sep=';')

#create sentence embeddings for correct answers
import time
from sentence_transformers import SentenceTransformer
model_snli = SentenceTransformer('jegormeister/bert-base-dutch-cased-snli')

print("Time", (time.time()))
sentences = pd.DataFrame(columns=correct_answers.columns)
sentences['TextName'] = correct_answers['TextName']
for col in correct_answers.columns:
    if col != 'TextName':
        sentences[col] = correct_answers[col].str.lower().tolist()
answer_embeddings = pd.DataFrame(columns=correct_answers.columns)
answer_embeddings['TextName'] = correct_answers['TextName']
print("Time", (time.time())) #step 1 takes 0.01562 seconds
for col in correct_answers.columns:
    if col != 'TextName':
        for i in range(len(correct_answers)):
            answer_embeddings[col].iloc[i] = model_snli.encode(sentences[col].iloc[i])
print("Time", (time.time())) #step 2 takes 1.08233 seconds
#step 1 and 2 can be eliminated by pre-loading already saved answer embeddings

#find similarity score with answers
from sentence_transformers.util import cos_sim
data_col = []
for i in range(1,6):
    data_col.append('sim_'+str(i)) 
    data['sim_'+str(i)] = np.nan
answer_col = correct_answers.columns.tolist()[1:]
for i in range(len(data)):
    for j in range(len(data_col)):
        data[data_col[j]].iloc[i] = cos_sim(embeddings_snli[i].reshape(1,-1), answer_embeddings[answer_col[j]][answer_embeddings['TextName'] == data['Tekstnaam'].iloc[i]].to_numpy()[0].reshape(1,-1)).data.cpu().numpy()
print("Time", (time.time())) #step 3 takes 32.47702 seconds. 0.003186 seconds per answer. 0.012745 seconds for 4 answers
#print("Run time in seconds", (time.time() - start_time))

#create dataset for model
def encode_and_bind(input_data, features_to_encode):
    input_data_encoded = input_data
    for feature in features_to_encode:
        dummies = pd.get_dummies(input_data[feature], prefix = feature).reset_index().drop(columns = "index")
        input_data_encoded = pd.concat([input_data_encoded, dummies], axis = "columns")
        input_data_encoded = input_data_encoded.drop(columns = feature)
    return input_data_encoded

#output variable
from sklearn.preprocessing import LabelEncoder
data_y = LabelEncoder().fit_transform(data['Code'])
class_labels = ['wrong', 'goed']

data_x = encode_and_bind(data, ['Tekstnaam'])
data_x = data_x.drop(columns=['Veld', 'Code', 'Verbandnummer','Dataset'])
data_x = pd.concat([data_x, pd.DataFrame(embeddings_snli)], axis='columns')
data_x.columns = data_x.columns.astype(str)

#--------------------------------tune SVM model---------------------------------
from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC

def plot_confusionmatrix(y_train_pred, y_train, class_labels):
    cf = confusion_matrix(y_train_pred, y_train)
    sns.heatmap(cf, annot=True, yticklabels=class_labels,
               xticklabels=class_labels, cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.show()

def svm_tune(data_x, data_y, param_grid, n_splits=10, n_repeats=1, scoring='accuracy', weight=None):
    if(weight == 'balanced'):
        weights = compute_class_weight('balanced', np.unique(data_y), data_y)
        class_weights = {0:weights[0], 1:weights[1]}
        svm_model = SVC(class_weight=class_weights)
    else:
        svm_model = SVC()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)        
    svm_gs = GridSearchCV(svm_model,param_grid, scoring="accuracy", cv=cv, verbose=3)
    svm_gs.fit(data_x, data_y)
    y_pred = svm_gs.predict(data_x)
    plot_confusionmatrix(y_pred, data_y, class_labels)
    print("Best params:", svm_gs.best_params_)
    print(f"Best {scoring}: {svm_gs.best_score_}")
    return svm_gs

from sklearn.metrics import cohen_kappa_score, make_scorer
def svm_cv_score(svm, data_x, data_y, weight=None):
    cohen_kappa = make_scorer(cohen_kappa_score)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42)
    if(weight == 'balanced'):
        weights = compute_class_weight('balanced', np.unique(data_y), data_y)
        class_weights = {0:weights[0], 1:weights[1]}
        model = SVC(class_weight=class_weights, C=svm.best_params_['C'], kernel="rbf", gamma=svm.best_params_['gamma'],  probability=True)
    else:
        model = SVC(C=svm.best_params_['C'], kernel="rbf", gamma=svm.best_params_['gamma'],  probability=True)
    
    scoring = {'acc': 'accuracy',
               'auc': 'roc_auc',
               'precision': 'precision',
               'recall': 'recall',
               "f1": "f1",
               "cohen_kappa": cohen_kappa
              }
    scores_svm = cross_validate(model, data_x, data_y, scoring=scoring, cv=cv)
    for key in scores_svm:
        print(f"{key} {scores_svm[key].mean()}")
    return scores_svm

#input data with embeddings
data_x = pd.concat([data_x, pd.DataFrame(embeddings_snli)], axis='columns')

#----------Model with embeddings and similarity scores----
param_grid = {'C': [1,10,100], 'gamma': [0.1,0.01,0.001],'kernel': ['rbf']}
svm_gs = svm_tune(data_x, data_y, param_grid)
svm_score = svm_cv_score(svm_gs, data_x, data_y)
#best params: C=1, gamma=0.01

#----------Model with weights balanced----
param_grid = {'C': [1,10,100], 'gamma': [0.1,0.01,0.001],'kernel': ['rbf']}
svm_gs = svm_tune(data_x, data_y, param_grid, weight='balanced')
svm_score = svm_cv_score(svm_gs, data_x, data_y, weight='balanced')

#----------Model with embeddings and similarity scores further tuning----
param_grid = {'C': [1, 0.9, 1.1], 'gamma': [0.009, 0.01, 0.02],'kernel': ['rbf']}
svm_gs = svm_tune(data_x, data_y, param_grid)
svm_run2_results = pd.DataFrame.from_dict(svm_gs.cv_results_)
svm_run2_results.to_excel('SVM_Best_CV_Results_v2.xlsx', index=False)
svm_score = svm_cv_score(svm_gs, data_x, data_y)

#----------Scores for best parameter---------------
#best parameters C=1, gamma=0.01
# calculate metrics
from sklearn.metrics import cohen_kappa_score, make_scorer
cohen_kappa = make_scorer(cohen_kappa_score)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42)
model = SVC(C=1, kernel="rbf", gamma=0.01,  probability=True)

scoring = {'acc': 'accuracy',
           'auc': 'roc_auc',
           'precision': 'precision',
           'recall': 'recall',
           "f1": "f1",
           "cohen's kappa": cohen_kappa
          }

scores_svm = cross_validate(model, data_x, data_y, scoring=scoring, cv=cv)
for key in scores_svm:
    print(f"{key} {scores_svm[key].mean()}")

model.fit(data_x, data_y)
y_pred = model.predict(data_x)
plot_confusionmatrix(y_pred, data_y, class_labels)

#-----------------------Repeat experiment for Neural Network-------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

#train NN
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

#directory to save model
#read data
os.chdir("C://Users//Shashank//OneDrive//Documents//DSDM//Automatic Scoring App//Results//Clever//NN")

#define model
def build_nn(X_train, hidden_layer_size, num_layers=1):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=X_train.shape[1], activation='relu'))
    if num_layers > 1:
        for i in range(num_layers - 1):
            model.add(Dense(hidden_layer_size, input_dim=hidden_layer_size, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    return model


#compile and fit
def nn_fit(model, X_train, y_train, X_val, y_val, batch_size, epochs, save_model):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(save_model, monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
    #compile model
    hist = model.fit(X_train, y_train, 
                         validation_data=(X_val, y_val),
                         batch_size = batch_size,
                         epochs = epochs,
                         verbose=1,
                         callbacks=[es,mc])
    return hist

#plot training metrics
def plot_training_metrics(hist):   
    acc = hist.history['accuracy']
    val = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    #plot train and val accuracy
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    ax[0].plot()
    ax[0].plot(epochs, acc, '-', label='Training accuracy')
    ax[0].plot(epochs, val, ':', label='Validation accuracy')
    ax[0].set_title('Training and Validation Accuracy')
    #ax[0].xlabel('Epoch')
    #ax[0].ylabel('Accuracy')
    ax[0].legend(loc='lower right')
    
    #plot train and val loss
    ax[1].plot()
    ax[1].plot(epochs, loss, '-', label='Training loss')
    ax[1].plot(epochs, val_loss, ':', label='Validation loss')
    ax[1].set_title('Training and Validation Loss')
    #ax[1].xlabel('Epoch')
    #ax[1].ylabel('Loss')
    ax[1].legend(loc='lower right')
    plt.show()
    
#evaluate model on test set
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, cohen_kappa_score

def evaluate_NN(save_model, X_test, y_test):
    model = load_model(save_model)
    scoring = dict()
    y_pred = (model.predict(X_test) > 0.5).astype('int32')
    scoring['accuracy'] = accuracy_score(y_test, y_pred)    
    scoring['roc_auc'] = roc_auc_score(y_test, y_pred)
    scoring['precision'] = precision_score(y_test, y_pred)
    scoring['recall'] = recall_score(y_test, y_pred)
    scoring['f1'] = f1_score(y_test, y_pred)
    scoring['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)
    print(scoring)
    return scoring


model = build_nn(X_train, 10, 1)
model_name = 'NN_SBERT2_Clever_1Layer_hid10_16'
hist = nn_fit(model, X_train, y_train, X_val, y_val, 
       batch_size=16, epochs=100, save_model=model_name+'.h5')
plot_training_metrics(hist)


#evaluate all models and compare scores
model_list = ['NN_1Layer_hid10_16', 'NN_SBERT2_1Layer_hid10_16']
acc_scores = {'model':[],
              'accuracy':[]}
for model_name in model_list:
    score = evaluate_NN(model_name+'.h5', X_test, y_test)
    acc_scores['model'].append(model_name)
    acc_scores['accuracy'].append(score['accuracy'])

#write scores to output file
#pd.DataFrame(acc_scores).to_csv('output.csv')
score = evaluate_NN('NN_SBERT2_Clever_1Layer_hid10_16'+'.h5', X_test, y_test)
pd.DataFrame(score, index=[0]).to_csv('output.csv')