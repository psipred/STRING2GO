import numpy as np
import math
import glob
import operator
from itertools import product
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

np.random.seed(1337)  # for reproducibility

F1Score=make_scorer(f1_score)
fileListFeatureLargePart=glob.glob("./Training_STRING2GO_Functional_Representation.txt")
fileListLabelsTrainingSVM=glob.glob("./Training_Label/*")
fileListFeaturesTestingSVM=glob.glob("./Testing_STRING2GO_Functional_Representation.txt")
ProteinList=glob.glob("./Testing_ProteinList.txt")
outputPrediction = open("./Prediction.txt", 'w')

with open(fileListFeatureLargePart[0], 'r') as infile:
    MatrixFeaturesLargePart1=[list(x.split(",")) for x in infile]
    MatrixFeaturesLargePart2=[line[0:800]for line in MatrixFeaturesLargePart1[:]]
    MatrixFeaturesLargePart=[[float(y) for y in x] for x in MatrixFeaturesLargePart2]
    infile.close()

with open(ProteinList[0], 'r') as infile:
     Protein = infile.read().splitlines()
     infile.close()

with open("./GOTerms.txt") as GONameFile:
    GOName1=GONameFile.readlines()

GOName2=[]
for ss in range(0,len(GOName1)):
    GOName2.append("GO"+GOName1[ss].split(":")[1].rstrip())
print(GOName2[0:])

with open(fileListFeaturesTestingSVM[0], 'r') as infileFeatureTestingSVM:
    MatrixFeaturesTestingSVM1 = [list(x.split(",")) for x in infileFeatureTestingSVM]
    MatrixFeaturesTestingSVM2 = [line[0:800] for line in MatrixFeaturesTestingSVM1[:]]
    MatrixFeaturesTestingSVM = [[float(y) for y in x] for x in MatrixFeaturesTestingSVM2]
    infile.close()

for a in range(0,len(GOName2)):
  for c in range(0,len(fileListLabelsTrainingSVM)):
              if GOName2[a] in fileListLabelsTrainingSVM[c] and "Labels" in fileListLabelsTrainingSVM[c]:
                  with open(fileListLabelsTrainingSVM[c], 'r') as infileClassSVMTraining:
                       TempReadMatrixclassTrainingSVM = infileClassSVMTraining.read().splitlines()
                       infileClassSVMTraining.close()

  for rIndex in range(len(TempReadMatrixclassTrainingSVM)):
      RealMatrixClass.append(float(TempReadMatrixclassTrainingSVM[rIndex]))

  param_grid = [
     {'C': [0.1,1,10.0,100.0],
       'gamma': [0.001,0.01,0.1,1.0]}
  ]

  rbf_svc = svm.SVC(kernel='rbf')
  clf = GridSearchCV(estimator=rbf_svc, param_grid=param_grid, scoring=F1Score)
  clf.fit(MatrixFeaturesLargePart, RealMatrixClass)
  para = clf.best_params_
  CValue = float(para['C'])
  GammerValue = float(para['gamma'])

  rbf_svc_trained = svm.SVC(kernel='rbf', gamma=GammerValue, C=CValue, probability=True).fit(MatrixFeaturesLargePart, RealMatrixClass)
  prob_preds = rbf_svc_trained.predict_proba(MatrixFeaturesTestingSVM)
  for rowIndex in range(len(prob_preds)):
      outputPrediction.write(GOName2[a])
      outputPrediction.write(",")
      outputPrediction.write(Protein[rowIndex])
      outputPrediction.write(",")
      outputPrediction.write(str(prob_preds[rowIndex,1]))
      outputPrediction.write("\n")
      print(str(prob_preds[rowIndex,1]))

outputPrediction.flush()
