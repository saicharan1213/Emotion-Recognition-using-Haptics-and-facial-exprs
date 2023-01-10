import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle


scratch = pd.read_csv('scratch_clean.csv')
scratch = scratch.iloc[:124]
tap = pd.read_csv('tap_clean.csv')
smooth = pd.read_csv('smooth_clean.csv')
data = pd.concat([scratch,tap,smooth],ignore_index = True)

Y = data['500']
X = data.drop(columns=['500'])
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

'''Intialization of ANN instance using alpha as 1e-04, 2 hidden layers with 20 and 3 neurons respectively 
solver as LBFGS, total iteration of 200

To obtain the results we oobtained please run ann_model_pickle.py'''
ann = MLPClassifier(alpha=1e-04, hidden_layer_sizes=(20,3), random_state=1,solver='lbfgs')
y_train = np.asarray(y_train)
ann.fit(x_train,y_train)
y_pred_ann = ann.predict(x_test)
y_test = np.asarray(y_test)
print(accuracy_score(y_test,y_pred_ann))
print(classification_report(y_test,y_pred_ann))