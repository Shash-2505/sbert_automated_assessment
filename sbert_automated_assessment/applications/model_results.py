import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

#read data
os.chdir("C://Users//Shashank//OneDrive//Documents//DSDM//Automatic Scoring App//Data")
data = pd.read_excel("input_data.xlsx", sheet_name="Data")

#keep only required fields
drop_columns = ['Nummer', 'Klas', 'Veldnummer','Verbandnummer', 'Dataset']
data = data.drop(columns=drop_columns)
#update code
data['Code'] = data['Code'].str.replace('f', 'c')
data['Code'] = data['Code'].str.replace('3', 'g')
#filter only good or correct answers
data = data[(data['Code']=='g') | (data['Code']=='c')]
data = data.reset_index().drop(columns = "index")
#replace cv with centrale verwarming
data['Veld'] = data['Veld'].str.lower()
data['Veld'] = data['Veld'].str.replace(r'\bcv','centrale verwarming')

#load embeddings
pickle_in = open('sentence_embeddings_processed_snli', 'rb')
embeddings_snli = pickle.load(pickle_in)

#create dataset for SVM
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
data_x = data_x.drop(columns=['Veld', 'Code'])
data_x = pd.concat([data_x, pd.DataFrame(embeddings_snli)], axis='columns')


#build SVM model
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

def plot_confusionmatrix(y_train_pred, y_train, class_labels):
    cf = confusion_matrix(y_train_pred, y_train)
    sns.heatmap(cf, annot=True, yticklabels=class_labels,
               xticklabels=class_labels, cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.show()

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
import time
start_time = time.time()
y_pred = model.predict(data_x)
print("Run time in seconds", (time.time() - start_time))
plot_confusionmatrix(y_pred, data_y, class_labels)

# save the model to disk
filename = 'Final_SVM.sav'
pickle.dump(model, open(filename, 'wb'))

#load saved model and test
start_time = time.time()
model = pickle.load(open(filename, 'rb'))
start_time = time.time()
y_pred = model.predict(data_x)
print("Run time in seconds", (time.time() - start_time))
plot_confusionmatrix(y_pred, data_y, class_labels)
