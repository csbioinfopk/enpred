import csv
import pickle
import numpy as np
import seaborn as sns
from sklearn import decomposition
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

np.random.seed(7)
sns.set()
inputSize = 134
outputCol = inputSize + 1
Random_Clf = RandomForestClassifier(n_estimators=25)
EnPred_dataset = np.genfromtxt("./FVs.csv", delimiter=",", dtype=float)
X = EnPred_dataset[:, 0:inputSize]
Y = EnPred_dataset[:, inputSize:outputCol]
std_scale = StandardScaler().fit(X)
X = std_scale.transform(X)
Random_Clf.fit(X, Y.ravel())
pred = np.round(Random_Clf.predict(X))
cm = confusion_matrix(Y, pred, labels=[1, 0]).ravel()
print(cm)
tp, fp, fn, tn = confusion_matrix(Y, pred, labels=[1, 0]).ravel()
np.random.seed(7)
acc = np.round(((tn + tp) / (tn + fp + fn + tp)) * 100, 2)
sp = np.round((tn / (fp + tn)) * 100, 2)
sn = np.round((tp / (tp + fn)) * 100, 2)
mcc = np.round(matthews_corrcoef(Y, pred), 5)
print([tp, fp, tn, fn, acc, sp, sn, mcc])
enpred_scores=[]
enpred_scores.append([tp, fp, tn, fn, acc, sp, sn, mcc])

pickle.dump(Random_Clf, open('./enpred_1Model.pkl', 'wb'))
pickle.dump(std_scale, open('./enpred_Scale.pkl', 'wb'))

print('\n\nResults are Saved in EnPred-Results.csv')
with open('./EnPred-Results.csv', 'w', newline='') as csvfile:
    resultwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
    resultwriter.writerow(['Confusion Matrix'])
    resultwriter.writerow(
        ['True Positive', 'False Positive', 'True Negative', 'False Negative', 'Accuracy', 'Specificity', 'Sensitivity',
         'MCC'])
    resultwriter.writerow(enpred_scores[0])
 