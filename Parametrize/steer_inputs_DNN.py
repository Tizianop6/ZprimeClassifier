#!/usr/bin/env python3

import numpy as np
from numpy import inf
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import math
import pickle
from Training import *
from Plotting import *
from GetInputs import *
from RankNetworks import *
from PredictExternal import *
from functions import *
#from TrainModelOnPredictions import *
#from TrainSecondNetwork import *
#from TrainThirdNetwork import *
from ExportModel import *

# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

#variations = ['NOMINAL','JEC_up','JEC_down','JER_up','JER_down']
variations = ['NOMINAL']
merged_str = 'Merged'
parameters = {
    'layers':[512,512],
    'batchsize': 32768,
    'classes':{0: ['TTbar'], 1: ['ST'], 2:['WJets','DY'], 3 : ['RSGluonToTT']},#['RSGluonToTT']},
    'masseslist':[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000],
    'exclmasseslist': [500,1000,1500,2000,2500,3000,3500,4000,4500],
    'regmethod': 'dropout',
    'regrate':0.50,
    'batchnorm': False,
    'epochs':50,
    'learningrate': 0.00050,
    'runonfraction': 1.0,
    'eqweight': False,
    'preprocess': 'StandardScaler',
    'sigma': 1.0, #sigma for Gaussian prior (BNN only)
    #'inputdir': '../Inputs_UL18_muon/',
    'bkginputdir': '../Inputs_UL17_muon_lite/',
    'siginputdir': '../Inputs_UL17_muon_signals/',
    #        'systvar': variations[ivars],
    'inputsubdir': 'MLInput/', #path to input files: inputdir + systvar + inputsubdir
    'prepreprocess': 'RAW' #for inputs with systematics don't do preprocessing before merging all inputs on one,     #FixME: add prepreprocessing in case one does not need to merge inputs
}
#testmasseslist=['RSGluonToTT_M-6000','RSGluonToTT_M-500','RSGluonToTT_M-3000']
#testmasseslist=['RSGluonToTT_M-500']#,'RSGluonToTT_M-3000']


parameters['classes']={0: ['TTbar'], 1: ['ST'], 2:['WJets','DY'], 3 : ['RSGluonToTT']}
tag_trained=dict_to_str(parameters)
#classtag_orig = get_classes_tag(parameters)

#for masslabel in testmasseslist:
    #print('lavorando su: ', masslabel)
    #parameters['classes']={0: ['TTbar'], 1: ['ST'], 2:['WJets','DY'], 3 : [masslabel]}
tag = dict_to_str(parameters)
print('tag: ', tag)
classtag = get_classes_tag(parameters)
print('classtag: ', classtag)

########## GetInputs ########
merged_str = 'Merged'
for ivars in range(len(variations)):
     merged_str = merged_str+'__'+variations[ivars]
     parameters['systvar'] = variations[ivars]
     # # # # # # Get all the inputs
     # # # # # # # # # ==================
     inputfolder = 'output/'+parameters['inputsubdir']+parameters['systvar']+'/'+parameters['prepreprocess']+'/'+ classtag
     GetInputsDifferentMassesParametrized(parameters)
     PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder='Plots/'+parameters['prepreprocess']+'/InputDistributions/'+parameters['systvar']+'/' + classtag)
  
parameters['inputdir']='output/'
#outputfolder_orig_applyprep=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag_orig

MixInputsDifferentMasses(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, variations=variations, filepostfix='')
SplitInputsDifferentMasses(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')

FitPrepocessingParametrized(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='train')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='test')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='val')

# ApplyPrepocessingTest(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='train',outputfolder_orig=outputfolder_orig_applyprep)
# ApplyPrepocessingTest(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='test',outputfolder_orig=outputfolder_orig_applyprep)
# ApplyPrepocessingTest(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='val',outputfolder_orig=outputfolder_orig_applyprep)
#ApplySignalPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')

inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag
outputfolder='output/'+parameters['preprocess']+'/'+merged_str+'/' + classtag+'/DNN_'+tag
plotfolder = 'Plots/'+parameters['preprocess']
PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder=plotfolder+'/InputDistributions/'+merged_str+'/' + classtag)

# #####

# # DNN 

TrainNetworkExclMass(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag)

PredictExternal(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='')
PlotPerformance(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='', plotfolder='Plots_performance_all/'+parameters['preprocess']+'/DNN_'+tag, use_best_model=True, usesignals=[2,4])

#PredictExternalExclMass(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='')

#PlotPerformanceExclMass(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='', plotfolder='Plots/'+parameters['preprocess']+'/DNN_'+tag, use_best_model=True, usesignals=[2,4])


masseslist=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]

PredictExternalMass(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='',masses=masseslist)
for i in range(len(masseslist)):
    PlotPerformanceMassOnlyROCs(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='', plotfolder='Plots/'+parameters['preprocess']+'/DNN_'+tag+'/mass_'+str(masseslist[i]), use_best_model=True, usesignals=[2,4],mass=masseslist[i])

# PredictExternalAlreadyTrained(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='',outputfolder_trained='output/'+parameters['preprocess']+'/DNN_'+tag_trained)
# PlotPerformanceTest(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='', plotfolder='Plots/'+parameters['preprocess']+'/DNN_'+tag, use_best_model=True, usesignals=[2,4],outputfolder_orig='output/'+parameters['preprocess']+'/DNN_'+tag_trained)
# #ExportModel(parameters, inputfolder='input/', outputfolder='output/', use_best_model=True)
