!pip install BioPython
import csv
import numpy as np
import pandas as pd
from numpy import interp
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

random_state = np.random.RandomState(5)
inputSize = 134
outputCol = inputSize + 1

# 10 Fold Part
print('\nk-fold Cross-Validation')
results = pd.DataFrame(columns=['training_score', 'test_score'])
fprs, tprs, scores = [], [], []
dataset = np.genfromtxt("./FVs.csv", delimiter=",", dtype=float)
X = dataset[:, 0:inputSize]
Y = dataset[:, inputSize:outputCol]
std_scale = StandardScaler().fit(X)
X = std_scale.transform(X)
cvscores = []

# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5) # for 5Fold and k=10 for 10Fold 
classifier = RandomForestClassifier(n_estimators=25)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, Y):
    print('Fold : ' + str(i))
    classifier.fit(X[train], Y[train].ravel())
    pred = np.round(classifier.predict(X[test]))
    tp, fn, fp, tn = confusion_matrix(Y[test], pred, labels=[1, 0]).ravel()
    acc = np.round(((tn + tp) / (tn + fp + fn + tp)) * 100, 2)
    sp = np.round((tn / (fp + tn)) * 100, 2)
    sn = np.round((tp / (tp + fn)) * 100, 2)
    mcc = np.round(matthews_corrcoef(Y[test].ravel(), pred), 5)
    cvscores.append([tp, fp, tn, fn, acc, sp, sn, mcc])
    print([tp, fp, tn, fn, acc, sp, sn, mcc])
    probas_ = classifier.predict_proba(X[test])
    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = metrics.roc_curve(Y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.2,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Curve')
plt.legend(loc="lower right")
# plt.legend('')
plt.show()

print('\n\nResults are Saved in Cross-Validation-Results.csv')
with open('./Cross-Validation-Results-5-fold-SVM.csv', 'w', newline='') as csvfile:
    resultwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
    resultwriter.writerow(['5-Fold Cross-Validation'])
    resultwriter.writerow(
        ['True Positive', 'False Positive', 'True Negative', 'False Negative', 'Accuracy', 'Specificity', 'Sensitivity',
         'MCC'])
    for i in range(cvscores.__len__()):
        resultwriter.writerow(cvscores[i])
