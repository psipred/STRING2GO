import numpy as np
import math
import glob
import itertools
import operator
from itertools import product
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import matthews_corrcoef, make_scorer, f1_score
from sklearn.cross_validation import PredefinedSplit
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.layers.core import Dropout, Dense, Activation
from keras import optimizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import MaxoutDense
from keras.models import model_from_json
from sklearn.metrics import f1_score
np.random.seed(1337) 

fileListFeatureLargePart=glob.glob("./Training_Mashup_Embeddings.txt")
fileListLabelLargePart=glob.glob("./Training_Label.txt")

with open(fileListFeatureLargePart[0], 'r') as infile:
    MatrixFeaturesLargePart1=[list(x.split(",")) for x in infile]
    MatrixFeaturesLargePart2=[line[0:800]for line in MatrixFeaturesLargePart1[:]]
    MatrixFeaturesLargePart=[[float(y) for y in x] for x in MatrixFeaturesLargePart2]
    infile.close()
with open(fileListLabelLargePart[0], 'r') as infile:
    MatrixClassLargePart1=[list(x.split(",")) for x in infile]
    MatrixClassLargePart2=[line[0:204]for line in MatrixClassLargePart1[:]]
    MatrixClassLargePart=[[float(y) for y in x] for x in MatrixClassLargePart2]
    infile.close()

combinations = []
parameters = [
    ['Adagrad'],
    [0.05],
    [700],
    [700],
    [800],
]

for element in itertools.product(*parameters):
    combinations.append(element)

    model = Sequential()
    model.add(MaxoutDense(element[2], nb_feature=3, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_shape=(len(MatrixFeaturesLargePart[0]),)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(MaxoutDense(element[3], nb_feature=3, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=element[2]))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(MaxoutDense(element[4], nb_feature=3, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=element[3]))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(len(MatrixClassLargePart[0]), activation='sigmoid'))
    model.compile(optimizer='AdaGrad', loss='binary_crossentropy')
    model.summary()
    model.fit(MatrixFeaturesLargePart[:], MatrixClassLargePart[:], batch_size=100, nb_epoch=150, verbose=1)

model2 = Sequential()
model2.add(MaxoutDense(700, nb_feature=3, weights=model.layers[0].get_weights(),input_shape=(len(MatrixFeaturesLargePart[0]),)))
model2.add(BatchNormalization(weights=model.layers[1].get_weights()))
model2.add(Dropout(0.5))
model2.add(MaxoutDense(700, nb_feature=3, weights=model.layers[3].get_weights(), input_dim=700))
model2.add(BatchNormalization(weights=model.layers[4].get_weights()))
model2.add(Dropout(0.5))
model2.add(MaxoutDense(800, nb_feature=3, weights=model.layers[6].get_weights(), input_dim=700))


fileListFeaturesTestingSVM=glob.glob("./Testing_Mashup_Embeddings.txt")
outputDMNNGeneratedFeaturesTesting = open("./Testing_STRING2GO_Functional_Representations.txt", 'w')

with open("./BPTerms.txt") as GONameFile:
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

LearnedFeaturesTesting = model2.predict(MatrixFeaturesTestingSVM)

for index1 in range(0,len(LearnedFeaturesTesting)):
    for index2 in range(0,len(LearnedFeaturesTesting[0])):
        outputDMNNGeneratedFeaturesTesting.write(str(LearnedFeaturesTesting[index1][index2]))
        outputDMNNGeneratedFeaturesTesting.write(",")
    outputDMNNGeneratedFeaturesTesting.write("\n")
outputDMNNGeneratedFeaturesTesting.flush()
outputDMNNGeneratedFeaturesTesting.close() 
