import numpy as np
import csv
from itertools import product
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn import svm
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import PredefinedSplit
from sklearn.neighbors import NearestNeighbors
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.layers.core import Dropout, Dense, Activation
from sklearn.model_selection import KFold, LeaveOneOut
import pandas as pd
from keras import regularizers
import math
import glob
import numpy as np
np.random.seed(1337)
from keras import optimizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import MaxoutDense
from keras.layers import LSTM
from sklearn.model_selection import GridSearchCV
import itertools
import operator
from keras.models import model_from_json


MCCValue=make_scorer(matthews_corrcoef)

fileListFeature=glob.glob(".../BPProteinVectorCombinedscoreNetworkVector-Training-MashupDMNN.txt")
fileListLabel=glob.glob(".../Label-BP-LargePart-MashupDMNN.txt")
fileListFeatureTest=glob.glob(".../CombinedscoreNetwork-Features-Heldout-Mashup.txt")
outputResultsFile = open(".../Results-Combinedscore-MashupDMNN.txt", 'w')
ProteinListTest=glob.glob(".../Heldout-Proteins.txt")

with open(fileListFeature[0], 'r') as infile:
    MatrixFeatures1=[list(x.split(",")) for x in infile]
    MatrixFeatures2=[line[0:800]for line in MatrixFeatures1[:]]
    MatrixFeatures=[[float(y) for y in x] for x in MatrixFeatures2]
    infile.close()
with open(fileListLabel[0], 'r') as infile:
    MatrixClass1=[list(x.split(",")) for x in infile]
    MatrixClass2=[line[0:204]for line in MatrixClass1[:]]
    MatrixClass=[[float(y) for y in x] for x in MatrixClass2]
    infile.close()

with open(fileListFeatureTest[0], 'r') as infile:
    MatrixFeaturesTest1=[list(x.split(",")) for x in infile]
    MatrixFeaturesTest2=[line[0:800]for line in MatrixFeaturesTest1[:]]
    MatrixFeaturesTest=[[float(y) for y in x] for x in MatrixFeaturesTest2]
    infile.close()


with open(ProteinListTest[0], 'r') as infile:
    Protein = infile.read().splitlines()
    infile.close()

AllMCCCV = [[0 for x in range(len(MatrixClass[0]))] for x in range(1)]
AllF1CV=[[0 for x in range(len(MatrixClass[0]))] for x in range(1)]
AllTPCV=[[0 for x in range(len(MatrixClass[0]))] for x in range(1)]
AllTNCV=[[0 for x in range(len(MatrixClass[0]))] for x in range(1)]
AllPRECV=[[0 for x in range(len(MatrixClass[0]))] for x in range(1)]
AllRECCV=[[0 for x in range(len(MatrixClass[0]))] for x in range(1)]

combinations = []
parameters = [
    ['Adagrad'],
    [0.05],
    [500],
    [700],
    [800],
    ['tanh'],
    ['tanh'],
    ['tanh']
]

for element in itertools.product(*parameters):
    combinations.append(element)


EachMCCCV = [[]]
EachF1CV = [[]]
EachTPCV = [[]]
EachTNCV= [[]]
EachPRECV = [[]]
EachRECCV = [[]]



for element in itertools.product(*parameters):
        if element[0]=='Adam':
                optim = optimizers.Adam(lr=element[1])
        if element[0] == 'Adagrad':
                optim = optimizers.Adagrad(lr=element[1])
        if element[0] == 'SGD':
                optim = optimizers.SGD(lr=0.01, momentum=0.01, nesterov=True)

        model = Sequential()
        model.add(MaxoutDense(element[2], nb_feature=3, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_shape=(len(MatrixFeatures[0]),)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(MaxoutDense(element[3], nb_feature=3, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=element[2]))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(MaxoutDense(element[4], nb_feature=3, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_shape=(len(MatrixFeatures[0]),)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(len(MatrixClass[0]), activation='sigmoid'))
        model.compile(optimizer=optim, loss='binary_crossentropy')
        model.summary()
        model.fit(MatrixFeatures[0:], MatrixClass[0:], batch_size=100, nb_epoch=150, verbose=0)

        model_json = model.to_json()
        with open(".../model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(".../model.h5")

        json_file = open('.../model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(".../model.h5")

        preds = loaded_model.predict(MatrixFeaturesTest[0:])

        for indexRow in range(len(preds)):
            outputResultsFile.write(Protein[indexRow])
            outputResultsFile.write(",")
            for indexColumn in range(len(preds[0])):
                outputResultsFile.write(str(round(preds[indexRow][indexColumn], 2)))
                outputResultsFile.write(",")
            outputResultsFile.write("\n")

outputResultsFile.flush()
outputResultsFile.close()


 
