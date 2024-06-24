import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import DetCurveDisplay
from sklearn.metrics import classification_report
from sklearn.utils import resample
import joblib
import matplotlib.pyplot as plt

rebalancemultiplier = 2

#training definition (mixed bag)
def SupervisedClass(x_train, y_train, x_final):
 estrf1 = RandomForestClassifier(n_estimators=250,max_features=None)
 estrf = OneVsRestClassifier(estrf1)
 estnn1 = MLPClassifier(max_iter=2000, hidden_layer_sizes=(150,150,150) )
 estnn = OneVsRestClassifier(estnn1)
 estgb1 = GradientBoostingClassifier(n_estimators=100)
 estgb = OneVsRestClassifier(estgb1)
 estkn1 = KNeighborsClassifier(n_neighbors=5)
 estkn = OneVsRestClassifier(estkn1)
 estsv1 = SVC()
 estsv = OneVsRestClassifier(estsv1)
 estgp1 = GaussianProcessClassifier(max_iter_predict=1000)
 estgp = OneVsRestClassifier(estgp1)
 estet1 = ExtraTreesClassifier(n_estimators=100)
 estet = OneVsRestClassifier(estet1)
 estad1 = AdaBoostClassifier(n_estimators=100)
 estad = OneVsRestClassifier(estad1)
 estlr1 = LogisticRegression(C=1.0, solver='saga',  max_iter=10000)
 estlr = OneVsRestClassifier(estlr1)
 estpa1 = PassiveAggressiveClassifier(C=1.0, max_iter=10000, early_stopping=True)
 estpa = OneVsRestClassifier(estpa1)
 estimators = [ ('rf', BaggingClassifier(estimator=estrf, n_estimators=3, bootstrap=True, bootstrap_features=True, n_jobs=1) ),
            ('nn', BaggingClassifier(estimator=estnn, n_estimators=3, bootstrap=True, bootstrap_features=True, n_jobs=1) ),
            ('gb',BaggingClassifier(estimator=estgb, n_estimators=3, bootstrap=True, bootstrap_features=True, n_jobs=1) ),
            ('kn',BaggingClassifier(estimator=estkn, n_estimators=3, bootstrap=True, bootstrap_features=True, n_jobs=1) ),
            ('sv',BaggingClassifier(estimator=estsv, n_estimators=3, bootstrap=True, bootstrap_features=True, n_jobs=1) ),
            ('gp',BaggingClassifier(estimator=estgp, n_estimators=3, bootstrap=True, bootstrap_features=True, n_jobs=1) ),
            ('et',BaggingClassifier(estimator=estet, n_estimators=3, bootstrap=True, bootstrap_features=True, n_jobs=1) ),
            ('ad',BaggingClassifier(estimator=estad, n_estimators=3, bootstrap=True, bootstrap_features=True, n_jobs=1) ),
            ('lr',BaggingClassifier(estimator=estlr, n_estimators=3, bootstrap=True, bootstrap_features=True, n_jobs=1) ),
            ('pa',BaggingClassifier(estimator=estpa, n_estimators=3, bootstrap=True, bootstrap_features=True, n_jobs=1) ),
  ]
 clf = StackingClassifier(estimators=estimators, stack_method='predict_proba', final_estimator=RandomForestClassifier(n_estimators=200,max_features=None))
 x_train1, x_score, y_train1, y_score = train_test_split(x_train, y_train, test_size=0.2)
 modscore = clf.fit(x_train1, y_train1).score(x_score, y_score)
 probs = pd.DataFrame(clf.predict_proba(x_final), columns=clf.classes_)
 modelname = "supervised_" 
 joblib.dump(clf, modelname)
 return probs, modscore

#execute
probs1, score1 = SupervisedClass(x_train = x_train.to_numpy(),
                         y_train = y_train.to_numpy().ravel(),
                         x_final = x_final.to_numpy())


#inference
def predicting(finaleval_x):
 model = joblib.load('supervised_')
 pred = pd.DataFrame(model.predict_proba(finaleval_x.to_numpy()), columns=model.classes_)
 return pred
output = predicting(finaleval_x=eval_x)

#ROC curves
models = ['supervised_']
fig, [ax_roc, ax_det] = plt.subplots(1,2,figsize=(15,7))
for stm in models:
  RocCurveDisplay.from_estimator(joblib.load(stm), finaleval_x.to_numpy(), finaleval_actual.to_numpy(), ax=ax_roc, name=stm)
  DetCurveDisplay.from_estimator(joblib.load(stm), finaleval_x.to_numpy(), finaleval_actual.to_numpy(), ax=ax_det, name=stm)
ax_roc.set_title("ROC curves")
ax_det.set_title("DET curves")
plt.legend()
plt.show()