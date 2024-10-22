import numpy as np
from numpy import inf
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, auc,precision_recall_fscore_support, balanced_accuracy_score, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics, regularizers

#from ROOT import TCanvas, TFile, TH1F, TH2F, gROOT, kRed, kBlue, kGreen, kMagenta, kCyan, gStyle
#from ROOT import gErrorIgnoreLevel, kInfo, kWarning, kError
import ROOT
import math
import pickle
import sys
import os
from functions import *
from constants import *
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import h5py
import pandas as pd

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def _prior_normal_fn(sigma, dtype, shape, name, trainable, add_variable_fn):
    """Normal prior with mu=0 and sigma=sigma. Can be passed as an argument to                                                                                               
    the tpf.layers                                                                                                                                                           
    """
    del name, trainable, add_variable_fn
    dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(sigma))
    batch_ndims = tf.size(input=dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)


def PlotMetrics(parameters, tag):
    auc_list=[]
    wa_list=[]
    f1_list=[]
    gmean_list=[]
    pr_auc_list=[]
    masseslistplot=[]
    base_path='Plots/'+parameters['preprocess']+'/DNN_'+tag
    file_name="metrics.pkl"
    for mass in range(500, 6001, 500):
        folder_name = f"mass_{mass}"
        file_path = os.path.join(base_path, folder_name, file_name)
        
        # Controlla se il file esiste
        if os.path.exists(file_path):
            masseslistplot.append(mass)
            # Apri il file e carica il dizionario
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                #dictionaries.append(data)
                print(data)
                wa_list.append(data['wa'])
                f1_list.append(data['F1'][3])
                gmean_list.append(data['gmean'])
                pr_auc_list.append(data['pr_aucs'][3])
                #auc_list.append(data["RSGluonToTT"])
        else:
            print(f"File {file_path} non trovato.")

    np_auc_list=np.array(auc_list,dtype=np.float32)
    np_wa=np.array(wa_list,dtype=np.float32)
    np_f1=np.array(f1_list,dtype=np.float32)
    np_gmean=np.array(gmean_list,dtype=np.float32)
    np_pr_auc=np.array(pr_auc_list,dtype=np.float32)
    np_masses=np.array(masseslistplot,dtype=np.float32)

    canvases=[]
    graphs=[]

    canvases.append(ROOT.TCanvas("c1","c1",1000,700))
    graphs.append(ROOT.TGraph(len(masseslistplot),np_masses,np_wa))

    graphs[-1].SetTitle("Weighted accuracy; Signal mass [GeV]; Accuracy")
    graphs[-1].SetLineColor(ROOT.kRed)
    graphs[-1].SetMarkerColor(ROOT.kRed)
    graphs[-1].SetMarkerStyle(20)
    graphs[-1].SetMarkerSize(1)
    graphs[-1].SetLineWidth(2)
    graphs[-1].Draw("ALP")
    canvases[-1].SaveAs(base_path+"/weighted_accuracy.pdf")


    canvases.append(ROOT.TCanvas("c2","c2",1000,700))
    graphs.append(ROOT.TGraph(len(masseslistplot),np_masses,np_f1))
    
    graphs[-1].SetTitle("F1 score; Signal mass [GeV]; F1 score")
    graphs[-1].SetLineColor(ROOT.kBlue)
    graphs[-1].SetMarkerColor(ROOT.kBlue)
    graphs[-1].SetMarkerStyle(20)
    graphs[-1].SetMarkerSize(1)
    graphs[-1].SetLineWidth(2)
    graphs[-1].Draw("ALP")
    
    canvases[-1].SaveAs(base_path+"/f1.pdf")



    canvases.append(ROOT.TCanvas("c3","c3",1000,700))
    graphs.append(ROOT.TGraph(len(masseslistplot),np_masses,np_gmean))
    
    graphs[-1].SetTitle("G-mean score; Signal mass [GeV]; G-mean score")
    graphs[-1].SetLineColor(ROOT.kViolet)
    graphs[-1].SetMarkerColor(ROOT.kViolet)
    graphs[-1].SetMarkerStyle(20)
    graphs[-1].SetMarkerSize(1)
    graphs[-1].SetLineWidth(2)
    graphs[-1].Draw("ALP")
    canvases[-1].SaveAs(base_path+"/gmean.pdf")


    canvases.append(ROOT.TCanvas("c4","c4",1000,700))
    graphs.append(ROOT.TGraph(len(masseslistplot),np_masses,np_pr_auc))
    
    graphs[-1].SetTitle("Precision-Recall AUC; Signal mass [GeV]; Precision-Recall AUC")
    graphs[-1].SetLineColor(ROOT.kTeal-7)
    graphs[-1].SetMarkerColor(ROOT.kTeal-7)
    graphs[-1].SetMarkerStyle(20)
    graphs[-1].SetMarkerSize(1)
    graphs[-1].SetLineWidth(2)
    graphs[-1].Draw("ALP")
    canvases[-1].SaveAs(base_path+"/pr_aucs.pdf")


    c2=ROOT.TCanvas("cmg","cmg",1000,700)
    mg= ROOT.TMultiGraph()
    mg.SetTitle("Metric values for different masses; Signal mass [GeV]; Metric values")

    for g in graphs:
        mg.Add(g)
    #mg.Add(graphs[0])
    #mg.Add(graphs[1])
    #mg.Add(graphs[2])

    c2.cd()
    mg.Draw("ALP")
    c2.BuildLegend()
    c2.SaveAs(base_path+"/mg.png")
    c2.SaveAs(base_path+"/mg.pdf")


    outfile_root=ROOT.TFile(base_path+"/graphs.root", "recreate")
    for g in graphs:
        g.Write()
    mg.Write()
    outfile_root.Close()




def PlotPerformance(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0]):
    print('Now plotting the performance')
    #gErrorIgnoreLevel = kWarning

    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)

    # Get model and its history
    model = tf.keras.models.load_model(outputfolder+'/model.h5')
    with open(outputfolder+'/model_history.pkl', 'rb') as f:
        model_history = pickle.load(f)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)

    predpostfix = ''
    if use_best_model:
        predpostfix = '_best'
    pred_train, pred_test, pred_val, pred_signals = load_predictions(outputfolder=outputfolder, filepostfix=predpostfix) 



    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)


    log_model_performance(parameters=parameters, model_history=model_history, outputfolder=outputfolder)
    plot_loss(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_accuracy(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    plot_model(model, show_shapes=True, to_file=plotfolder+'/Model.pdf')
    plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, use_best_model=use_best_model)


    pred_trains, weights_trains, normweights_trains, lumiweights_trains, pred_vals, weights_vals, normweights_vals, lumiweights_vals, pred_tests, weights_tests, normweights_tests, lumiweights_tests = get_data_dictionaries(parameters=parameters, eventweights_train=eventweights_train, sample_weights_train=sample_weights_train, pred_train=pred_train, labels_train=labels_train, eventweights_val=eventweights_val, sample_weights_val=sample_weights_val, pred_val=pred_val, labels_val=labels_val, eventweights_test=eventweights_test, sample_weights_test=sample_weights_test, pred_test=pred_test, labels_test=labels_test)
    plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)

    plot_outputs_1d_classes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, use_best_model=use_best_model)
    # plot_outputs_2d(parameters=parameters, plotfolder=plotfolder, pred_vals=pred_vals, lumiweights_vals=lumiweights_vals, use_best_model=use_best_model)
    # best_cuts = cut_iteratively(parameters=parameters, outputfolder=outputfolder, pred_val=pred_val, labels_val=labels_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals)
    # plot_cuts(parameters=parameters, outputfolder=outputfolder, plotfolder=plotfolder, best_cuts=best_cuts, pred_vals=pred_vals, labels_val=labels_val, lumiweights_vals=lumiweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    # apply_cuts(parameters=parameters, outputfolder=outputfolder, best_cuts=best_cuts, input_train=input_train, input_val=input_val, input_test=input_test, labels_train=labels_train, labels_val=labels_val, labels_test=labels_test, sample_weights_train=sample_weights_train, sample_weights_val=sample_weights_val, sample_weights_test=sample_weights_test, eventweights_train=eventweights_train, eventweights_val=eventweights_val, eventweights_test=eventweights_test, pred_train=pred_train, pred_val=pred_val, pred_test=pred_test, signals=signals, eventweight_signals=eventweight_signals, pred_signals=pred_signals, signal_identifiers=signal_identifiers, use_best_model=use_best_model)


    # for cl in range(labels_train.shape[1]):
    #     # 'cl' is the output node number
    #     nbins = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    #     y_trains = {}
    #     y_vals = {}
    #     y_tests = {}
    #     ytots = {}
    #
    #
    #
    #     for i in range(labels_train.shape[1]):
    #         # 'i' is the true class (always the first index)
    #         y_trains[i], dummy = np.histogram(pred_trains[i][cl], bins=nbins, weights=lumiweights_trains[i][cl])
    #         y_vals[i], dummy = np.histogram(pred_vals[i][cl], bins=nbins, weights=lumiweights_vals[i][cl])
    #         y_tests[i], dummy = np.histogram(pred_tests[i][cl], bins=nbins, weights=lumiweights_tests[i][cl])
    #         ytots[i] = y_trains[i] + y_vals[i] + y_tests[i]
    #         print "node %i, class %i" % (cl, i)
    #         print ytots[i]
    #         print 'sum: %f' % (ytots[i].sum())
    #Store model as JSON file for usage in UHH2
    arch = model.to_json()
    # save the architecture string to a file somehow, the below will work
    with open(outputfolder+'/architecture.json', 'w') as arch_file:
        arch_file.write(arch)
    # now save the weights as an HDF5 file
    model.save_weights(outputfolder+'/weights.h5')
    print("--- END of DNN Plotting ---")

def PlotPerformanceExclMass(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0]):
    print('Now plotting the performance')
    #gErrorIgnoreLevel = kWarning

    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    masseslist= parameters['masseslist']
    exclmasseslist= parameters['exclmasseslist']
    
    # Get model and its history
    model = tf.keras.models.load_model(outputfolder+'/model.h5')
    with open(outputfolder+'/model_history.pkl', 'rb') as f:
        model_history = pickle.load(f)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data_excl_mass(parameters, inputfolder=inputfolder, filepostfix=filepostfix,exclmasslist=exclmasseslist)

    predpostfix = ''
    if use_best_model:
        predpostfix = '_best'
    pred_train, pred_test, pred_val, pred_signals = load_predictions(outputfolder=outputfolder, filepostfix=predpostfix) 



    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)


    log_model_performance(parameters=parameters, model_history=model_history, outputfolder=outputfolder)
    plot_loss(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_accuracy(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    plot_model(model, show_shapes=True, to_file=plotfolder+'/Model.pdf')
    plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, use_best_model=use_best_model)


    pred_trains, weights_trains, normweights_trains, lumiweights_trains, pred_vals, weights_vals, normweights_vals, lumiweights_vals, pred_tests, weights_tests, normweights_tests, lumiweights_tests = get_data_dictionaries(parameters=parameters, eventweights_train=eventweights_train, sample_weights_train=sample_weights_train, pred_train=pred_train, labels_train=labels_train, eventweights_val=eventweights_val, sample_weights_val=sample_weights_val, pred_val=pred_val, labels_val=labels_val, eventweights_test=eventweights_test, sample_weights_test=sample_weights_test, pred_test=pred_test, labels_test=labels_test)
    plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)

    plot_outputs_1d_classes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, use_best_model=use_best_model)
    # plot_outputs_2d(parameters=parameters, plotfolder=plotfolder, pred_vals=pred_vals, lumiweights_vals=lumiweights_vals, use_best_model=use_best_model)
    # best_cuts = cut_iteratively(parameters=parameters, outputfolder=outputfolder, pred_val=pred_val, labels_val=labels_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals)
    # plot_cuts(parameters=parameters, outputfolder=outputfolder, plotfolder=plotfolder, best_cuts=best_cuts, pred_vals=pred_vals, labels_val=labels_val, lumiweights_vals=lumiweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    # apply_cuts(parameters=parameters, outputfolder=outputfolder, best_cuts=best_cuts, input_train=input_train, input_val=input_val, input_test=input_test, labels_train=labels_train, labels_val=labels_val, labels_test=labels_test, sample_weights_train=sample_weights_train, sample_weights_val=sample_weights_val, sample_weights_test=sample_weights_test, eventweights_train=eventweights_train, eventweights_val=eventweights_val, eventweights_test=eventweights_test, pred_train=pred_train, pred_val=pred_val, pred_test=pred_test, signals=signals, eventweight_signals=eventweight_signals, pred_signals=pred_signals, signal_identifiers=signal_identifiers, use_best_model=use_best_model)


    # for cl in range(labels_train.shape[1]):
    #     # 'cl' is the output node number
    #     nbins = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    #     y_trains = {}
    #     y_vals = {}
    #     y_tests = {}
    #     ytots = {}
    #
    #
    #
    #     for i in range(labels_train.shape[1]):
    #         # 'i' is the true class (always the first index)
    #         y_trains[i], dummy = np.histogram(pred_trains[i][cl], bins=nbins, weights=lumiweights_trains[i][cl])
    #         y_vals[i], dummy = np.histogram(pred_vals[i][cl], bins=nbins, weights=lumiweights_vals[i][cl])
    #         y_tests[i], dummy = np.histogram(pred_tests[i][cl], bins=nbins, weights=lumiweights_tests[i][cl])
    #         ytots[i] = y_trains[i] + y_vals[i] + y_tests[i]
    #         print "node %i, class %i" % (cl, i)
    #         print ytots[i]
    #         print 'sum: %f' % (ytots[i].sum())
    #Store model as JSON file for usage in UHH2
    arch = model.to_json()
    # save the architecture string to a file somehow, the below will work
    with open(outputfolder+'/architecture.json', 'w') as arch_file:
        arch_file.write(arch)
    # now save the weights as an HDF5 file
    model.save_weights(outputfolder+'/weights.h5')
    print("--- END of DNN Plotting ---")


def PlotROCsMassAlternative(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0],mass=0):
    print('Now plotting the performance for mass', mass)
    #gErrorIgnoreLevel = kWarning

    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    masseslist=parameters['masseslist']    
    # Get model and its history
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals, masslabels_train, masslabels_test, masslabels_val = load_data_differentmasses(parameters, inputfolder=inputfolder, filepostfix=filepostfix)
    #input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data_excl_mass(parameters, inputfolder=inputfolder, filepostfix=filepostfix,exclmasslist=list_excluded_masses)

    predpostfix = ''
    if use_best_model:
        predpostfix = '_best'
    
    outputfolder_pred=outputfolder+'/m_'+str(mass)
    
    pred_train, pred_test, pred_val, pred_signals = load_predictions(outputfolder=outputfolder_pred, filepostfix=predpostfix) 
    
    idxtodel=[]
    for i in range(len(masslabels_val)):
        if (masslabels_val[i]!=mass and masslabels_val[i]!=-1):
            #print('idx: ',i, "masslabels_val[i] ", masslabels_val[i])
            idxtodel.append(i)
    
    #print('')
    new_input_val=np.copy(input_val)
    new_labels_val=np.copy(labels_val)
    print('before deleting new_input_val.shape ', new_input_val.shape)
    new_input_val=np.delete(new_input_val,idxtodel,axis=0)
    new_labels_val=np.delete(new_labels_val,idxtodel,axis=0)
    new_sample_weights_val=np.delete(sample_weights_val,idxtodel,axis=0)
    new_eventweights_val=np.delete(eventweights_val,idxtodel,axis=0)
    new_masslabels_val=np.delete(masslabels_val,idxtodel,axis=0)
    
    print('shapes:',input_val.shape,labels_val.shape,sample_weights_val.shape,eventweights_val.shape)
    print('after deleting:',new_input_val.shape,new_labels_val.shape,new_sample_weights_val.shape,new_eventweights_val.shape)
    print("pred_val shapes", pred_val.shape)
    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)
    if not os.path.isdir(plotfolder+"/TestPlots/"): os.makedirs(plotfolder+"/TestPlots/")

    print("pred_val:\n",pred_val )
    print("labels_val:\n",labels_val)

    list_nobkg=np.array([ pred_val[i] for i in range(len(pred_val)) if new_masslabels_val[i]!=-1 ])
    list_nobkg_wt=np.array([ new_eventweights_val[i] for i in range(len(new_eventweights_val)) if new_masslabels_val[i]!=-1 ])
    
    list_bkg=np.array([ pred_val[i] for i in range(len(pred_val)) if new_masslabels_val[i]==-1 ])
    list_bkg_wt=np.array([ new_eventweights_val[i] for i in range(len(new_eventweights_val)) if new_masslabels_val[i]==-1 ])
    

    labels_nobkg=np.array([ new_labels_val[i] for i in range(len(new_labels_val)) if new_masslabels_val[i]!=-1 ])


    labels_check=[]
    for i in range(new_labels_val.shape[0]):
        if (new_labels_val[i,3]==1):
            labels_check.append(new_labels_val[i,3])

    print("len labels check ", len(labels_check) )
    print("labels_nobkg.shape: ",labels_nobkg.shape,"labels_nobkg\n",labels_nobkg)
    
    c1=ROOT.TCanvas()
    h1=ROOT.TH1F("h","Output score, node 3, for true class RSGluon;Score;Weighted counts",100,0.,1.)
    h2=ROOT.TH1F("h2","Output score, node 3, for background classes;Score;Weighted counts",100,0.,1.)
    for i in range(len(list_nobkg)):
        h1.Fill(list_nobkg[i,3],list_nobkg_wt[i])

    for i in range((len(list_bkg))):
        h2.Fill(list_bkg[i,3],list_bkg_wt[i])
    
    #plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=new_labels_val, sample_weights_val=new_sample_weights_val, eventweights_val=new_eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    aucs_list=plot_PR_curves(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=new_labels_val, sample_weights_val=new_eventweights_val, eventweights_val=new_eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)


    outfile_root=ROOT.TFile(plotfolder+"/file1.root", "recreate")
    h1.Write()
    h2.Write()

    outfile_root.Close()

    c1.cd()
    c1.SetLogy()
    h1.Draw("L")
    c1.SaveAs(plotfolder+"/TestPlots/test"+str(mass)+".pdf")
    h2.Draw("L")
    c1.SetLogy()
    c1.SaveAs(plotfolder+"/TestPlots/testbkg"+str(mass)+".pdf")
    
    #print("list_nobkg.shape ", list_nobkg.shape)
    #print("list_nobkg\n ", list_nobkg)

    max_value_index_pred = np.argmax(pred_val, axis=1)
    max_value_index_labels = np.argmax(new_labels_val, axis=1)

    for j in range(4):
        roc_sklearn=roc_auc_score(new_labels_val[:,j],pred_val[:,j], average=None)#,multi_class='ovr')
        print("class no. ", j , "roc auc score: " , roc_sklearn)
    PRscore=precision_recall_fscore_support(max_value_index_labels,max_value_index_pred,beta=1.0,sample_weight=new_eventweights_val)
    gmeanscore=1
    for l in range(len(PRscore[1])):
        gmeanscore=gmeanscore*PRscore[1][l]
    gmeanscore=math.sqrt(gmeanscore)
    weighted_accuracy=balanced_accuracy_score(max_value_index_labels, max_value_index_pred, sample_weight=new_eventweights_val)

    dict_metrics={}
    dict_metrics["wa"]=weighted_accuracy
    dict_metrics["gmean"]=gmeanscore
    dict_metrics["prscore"]=PRscore
    dict_metrics["F1"]=PRscore[2]
    dict_metrics["pr_aucs"]=aucs_list
    print("m=",mass ," dictionary", dict_metrics)
    
    #print("rok_sklearn\n",roc_sklearn,"\n AUCS old way:\n ", AUCS_dict)
    with open(plotfolder+'/PrecisionRecall.pkl', 'wb') as f:
        pickle.dump(PRscore, f)
    with open(plotfolder+'/metrics.pkl', 'wb') as f:
        pickle.dump(dict_metrics, f)

    print("multiclass: ",roc_auc_score(new_labels_val, pred_val,multi_class='ovr'))
    #roc_curve(new_labels_val[:,3],pred_val[:,3])
    
    #plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, use_best_model=use_best_model)

    plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=new_labels_val, sample_weights_val=new_sample_weights_val, eventweights_val=new_eventweights_val, use_best_model=use_best_model)

    



def PlotPerformanceMass(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0],mass=0):
    print('Now plotting the performance for mass', mass)
    #gErrorIgnoreLevel = kWarning

    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    masseslist=parameters['masseslist']    
    # Get model and its history
    model = tf.keras.models.load_model(outputfolder+'/model.h5')
    with open(outputfolder+'/model_history.pkl', 'rb') as f:
        model_history = pickle.load(f)

    # Get inputs
    
    list_excluded_masses=[]

    for i in range(len(masseslist)):
        if(masseslist[i]!= mass):
            list_excluded_masses.append(masseslist[i])    

    #=masseslist.remove(mass)


    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals, masslabels_train, masslabels_test, masslabels_val = load_data_differentmasses(parameters, inputfolder=inputfolder, filepostfix=filepostfix)
    #input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data_excl_mass(parameters, inputfolder=inputfolder, filepostfix=filepostfix,exclmasslist=list_excluded_masses)

    predpostfix = ''
    if use_best_model:
        predpostfix = '_best'
    
    outputfolder_pred=outputfolder+'/m_'+str(mass)
    
    pred_train, pred_test, pred_val, pred_signals = load_predictions(outputfolder=outputfolder_pred, filepostfix=predpostfix) 
    
    idxtodel=[]
    for i in range(len(masslabels_val)):
        if (masslabels_val[i]!=mass and masslabels_val[i]!=-1):
            #print('idx: ',i, "masslabels_val[i] ", masslabels_val[i])
            idxtodel.append(i)
    
    #print('')
    new_input_val=np.copy(input_val)
    new_labels_val=np.copy(labels_val)
    print('before deleting new_input_val.shape ', new_input_val.shape)
    new_input_val=np.delete(new_input_val,idxtodel,axis=0)
    new_labels_val=np.delete(new_labels_val,idxtodel,axis=0)
    new_sample_weights_val=np.delete(sample_weights_val,idxtodel,axis=0)
    new_eventweights_val=np.delete(eventweights_val,idxtodel,axis=0)
    print('shapes:',input_val.shape,labels_val.shape,sample_weights_val.shape,eventweights_val.shape)
    print('after deleting:',new_input_val.shape,new_labels_val.shape,new_sample_weights_val.shape,new_eventweights_val.shape)
    
    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)


    log_model_performance(parameters=parameters, model_history=model_history, outputfolder=outputfolder)
    #plot_loss(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    #plot_accuracy(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    

    plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=new_labels_val, sample_weights_val=new_sample_weights_val, eventweights_val=new_eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    #plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    
    #plot_model(model, show_shapes=True, to_file=plotfolder+'/Model.pdf')
    #plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, use_best_model=use_best_model)

    #plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=new_labels_val, sample_weights_val=new_sample_weights_val, eventweights_val=new_eventweights_val, use_best_model=use_best_model)


    pred_trains, weights_trains, normweights_trains, lumiweights_trains, pred_vals, weights_vals, normweights_vals, lumiweights_vals, pred_tests, weights_tests, normweights_tests, lumiweights_tests = get_data_dictionaries(parameters=parameters, eventweights_train=eventweights_train, sample_weights_train=sample_weights_train, pred_train=pred_train, labels_train=labels_train, eventweights_val=new_eventweights_val, sample_weights_val=new_sample_weights_val, pred_val=pred_val, labels_val=new_labels_val, eventweights_test=eventweights_test, sample_weights_test=sample_weights_test, pred_test=pred_test, labels_test=labels_test)
    '''
    new_weights_vals=np.delete(weights_vals,idxtodel,axis=0)
    new_normweights_vals=np.delete(normweights_vals,idxtodel,axis=0)
    new_lumiweight_vals=np.delete(lumiweight_vals,idxtodel,axis=0)
    
    print('lumi shapes before:',weights_vals.shape,normweights_vals.shape,lumiweight_vals.shape)
    print('after deleting:',new_weights_vals.shape,new_normweights_vals.shape,new_lumiweight_vals.shape)
    '''
    
    #plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=new_labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    
    #plot_outputs_1d_classes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=new_labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, use_best_model=use_best_model)
    # plot_outputs_2d(parameters=parameters, plotfolder=plotfolder, pred_vals=pred_vals, lumiweights_vals=lumiweights_vals, use_best_model=use_best_model)
    # best_cuts = cut_iteratively(parameters=parameters, outputfolder=outputfolder, pred_val=pred_val, labels_val=labels_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals)
    # plot_cuts(parameters=parameters, outputfolder=outputfolder, plotfolder=plotfolder, best_cuts=best_cuts, pred_vals=pred_vals, labels_val=labels_val, lumiweights_vals=lumiweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    # apply_cuts(parameters=parameters, outputfolder=outputfolder, best_cuts=best_cuts, input_train=input_train, input_val=input_val, input_test=input_test, labels_train=labels_train, labels_val=labels_val, labels_test=labels_test, sample_weights_train=sample_weights_train, sample_weights_val=sample_weights_val, sample_weights_test=sample_weights_test, eventweights_train=eventweights_train, eventweights_val=eventweights_val, eventweights_test=eventweights_test, pred_train=pred_train, pred_val=pred_val, pred_test=pred_test, signals=signals, eventweight_signals=eventweight_signals, pred_signals=pred_signals, signal_identifiers=signal_identifiers, use_best_model=use_best_model)


    # for cl in range(labels_train.shape[1]):
    #     # 'cl' is the output node number
    #     nbins = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    #     y_trains = {}
    #     y_vals = {}
    #     y_tests = {}
    #     ytots = {}
    #
    #
    #
    #     for i in range(labels_train.shape[1]):
    #         # 'i' is the true class (always the first index)
    #         y_trains[i], dummy = np.histogram(pred_trains[i][cl], bins=nbins, weights=lumiweights_trains[i][cl])
    #         y_vals[i], dummy = np.histogram(pred_vals[i][cl], bins=nbins, weights=lumiweights_vals[i][cl])
    #         y_tests[i], dummy = np.histogram(pred_tests[i][cl], bins=nbins, weights=lumiweights_tests[i][cl])
    #         ytots[i] = y_trains[i] + y_vals[i] + y_tests[i]
    #         print "node %i, class %i" % (cl, i)
    #         print ytots[i]
    #         print 'sum: %f' % (ytots[i].sum())
    #Store model as JSON file for usage in UHH2
    arch = model.to_json()
    # save the architecture string to a file somehow, the below will work
    with open(outputfolder+'/architecture.json', 'w') as arch_file:
        arch_file.write(arch)
    # now save the weights as an HDF5 file
    model.save_weights(outputfolder+'/weights.h5')
    print("--- END of DNN Plotting ---")



def PlotPerformanceTest(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0],outputfolder_orig=''):
    print('Now plotting the performance')
    #gErrorIgnoreLevel = kWarning

    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)

    # Get model and its history
    model = tf.keras.models.load_model(outputfolder_orig+'/model.h5')
    with open(outputfolder_orig+'/model_history.pkl', 'rb') as f:
        model_history = pickle.load(f)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)

    predpostfix = ''
    if use_best_model:
        predpostfix = '_best'
    pred_train, pred_test, pred_val, pred_signals = load_predictions(outputfolder=outputfolder, filepostfix=predpostfix) 



    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)


    log_model_performance(parameters=parameters, model_history=model_history, outputfolder=outputfolder)
    plot_loss(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_accuracy(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    plot_model(model, show_shapes=True, to_file=plotfolder+'/Model.pdf')
    plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, use_best_model=use_best_model)


    pred_trains, weights_trains, normweights_trains, lumiweights_trains, pred_vals, weights_vals, normweights_vals, lumiweights_vals, pred_tests, weights_tests, normweights_tests, lumiweights_tests = get_data_dictionaries(parameters=parameters, eventweights_train=eventweights_train, sample_weights_train=sample_weights_train, pred_train=pred_train, labels_train=labels_train, eventweights_val=eventweights_val, sample_weights_val=sample_weights_val, pred_val=pred_val, labels_val=labels_val, eventweights_test=eventweights_test, sample_weights_test=sample_weights_test, pred_test=pred_test, labels_test=labels_test)
    plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)

    plot_outputs_1d_classes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, use_best_model=use_best_model)
    # plot_outputs_2d(parameters=parameters, plotfolder=plotfolder, pred_vals=pred_vals, lumiweights_vals=lumiweights_vals, use_best_model=use_best_model)
    # best_cuts = cut_iteratively(parameters=parameters, outputfolder=outputfolder, pred_val=pred_val, labels_val=labels_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals)
    # plot_cuts(parameters=parameters, outputfolder=outputfolder, plotfolder=plotfolder, best_cuts=best_cuts, pred_vals=pred_vals, labels_val=labels_val, lumiweights_vals=lumiweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    # apply_cuts(parameters=parameters, outputfolder=outputfolder, best_cuts=best_cuts, input_train=input_train, input_val=input_val, input_test=input_test, labels_train=labels_train, labels_val=labels_val, labels_test=labels_test, sample_weights_train=sample_weights_train, sample_weights_val=sample_weights_val, sample_weights_test=sample_weights_test, eventweights_train=eventweights_train, eventweights_val=eventweights_val, eventweights_test=eventweights_test, pred_train=pred_train, pred_val=pred_val, pred_test=pred_test, signals=signals, eventweight_signals=eventweight_signals, pred_signals=pred_signals, signal_identifiers=signal_identifiers, use_best_model=use_best_model)


    # for cl in range(labels_train.shape[1]):
    #     # 'cl' is the output node number
    #     nbins = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    #     y_trains = {}
    #     y_vals = {}
    #     y_tests = {}
    #     ytots = {}
    #
    #
    #
    #     for i in range(labels_train.shape[1]):
    #         # 'i' is the true class (always the first index)
    #         y_trains[i], dummy = np.histogram(pred_trains[i][cl], bins=nbins, weights=lumiweights_trains[i][cl])
    #         y_vals[i], dummy = np.histogram(pred_vals[i][cl], bins=nbins, weights=lumiweights_vals[i][cl])
    #         y_tests[i], dummy = np.histogram(pred_tests[i][cl], bins=nbins, weights=lumiweights_tests[i][cl])
    #         ytots[i] = y_trains[i] + y_vals[i] + y_tests[i]
    #         print "node %i, class %i" % (cl, i)
    #         print ytots[i]
    #         print 'sum: %f' % (ytots[i].sum())
    #Store model as JSON file for usage in UHH2
    arch = model.to_json()
    # save the architecture string to a file somehow, the below will work
    with open(outputfolder+'/architecture.json', 'w') as arch_file:
        arch_file.write(arch)
    # now save the weights as an HDF5 file
    model.save_weights(outputfolder+'/weights.h5')
    print("--- END of DNN Plotting ---")


def PlotBayesianPerformance(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0]):
    print(' ---- Now plotting the performance (PlotBayesianPerformance) ---- ')
    gErrorIgnoreLevel = kWarning

    # Get parameters
    # runonfullsample = parameters['runonfullsample']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    # classes = parameters['classes']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)
    layers=parameters['layers']

    # Get model and its history
    regrate=parameters['regrate']
    #sigma=1.0 #FixME, read https://arxiv.org/pdf/1904.10004.pdf to choose right value           
    sigma=parameters['sigma']         
    prior = lambda x, y, z, w, k: _prior_normal_fn(sigma, x, y, z, w, k)                                                                                                     
#    method = lambda d: d.mean()
    method = lambda d: d.sample()
    layer_hd = []
    batchnorm_hd = []
    dropout_hd = []
    inputs = tf.keras.layers.Input(shape=(input_train.shape[1],))


    #### Without DropOut
    #layer_hd.append(tf.layers.Dense(layers[0], activation=tf.nn.relu)(inputs))
    layer_hd.append(tfp.layers.DenseFlipout(layers[0], activation=tf.nn.relu, kernel_prior_fn=prior, kernel_posterior_tensor_fn=method)(inputs))
    k=1
    for i in layers[1:len(layers)+1]:
        print(("current k:",k))
        label=str(k+1)
        #layer_hd.append(tfp.layers.DenseFlipout(i, activation=tf.nn.relu, kernel_prior_fn=prior, kernel_posterior_tensor_fn=method)(layer_hd[k-1]))
        #layer_hd.append(tf.layers.Dense(i, activation=tf.nn.relu)(layer_hd[k-1]))
        layer_hd.append(tfp.layers.DenseFlipout(i, activation=tf.nn.relu, kernel_prior_fn=prior, kernel_posterior_tensor_fn=method)(layer_hd[k-1]))
        k = k+1
    print(("total number of hidden layers:",k))
    last_layer = tfp.layers.DenseFlipout(labels_train.shape[1], activation='softmax', kernel_prior_fn=prior, kernel_posterior_tensor_fn=method)(layer_hd[k-1])
    #last_layer = tf.layers.Dense(labels_train.shape[1], activation='softmax')(layer_hd[k-1])

    print('Number of output classes: %i' % (labels_train.shape[1]))
    model = tf.keras.models.Model(inputs=inputs,outputs=last_layer)
    opt  = tf.train.AdamOptimizer()
    mymetrics = [metrics.categorical_accuracy]
    file = h5py.File(outputfolder+'/model_weights.h5', 'r')
    weight = []
    for i in range(len(list(file.keys()))):
        weight.append(file['weight' + str(i)][:])
    model.set_weights(weight)
    print(model.summary())

    model_history = pd.read_hdf(outputfolder+'/summary.h5')

    predpostfix = ''
    if use_best_model:
        predpostfix = '_best'
    pred_train_all, pred_test_all, pred_val_all, pred_signals_all = load_predictions(outputfolder=outputfolder, filepostfix=predpostfix) #load predictions sampled N times


    pred_signals = {}
    pred_signals_std = {}
#    for i in range(len(signal_identifiers)):
#        pred_signals[i] = np.median(pred_signals_all[i],axis=0)
#        pred_signals_std[i] = np.std(pred_signals_all[i],axis=0)

    pred_train = np.median(pred_train_all,axis=0)
    pred_test = np.median(pred_test_all,axis=0)
    pred_val = np.median(pred_val_all,axis=0)

    pred_train_std = np.std(pred_train_all,axis=0)
    pred_test_std = np.std(pred_test_all,axis=0)
    pred_val_std = np.std(pred_val_all,axis=0)
 
    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)
    for eventid in range(100):
        plot_prediction_samples(parameters=parameters,plotfolder=plotfolder+'/ResponseSamples',pred_train_all=pred_train_all,labels_train=labels_train,eventID=eventid)

    log_model_performance(parameters=parameters, model_history=model_history, outputfolder=outputfolder) #OK
    plot_loss(parameters=parameters, plotfolder=plotfolder, model_history=model_history) #OK
    plot_accuracy(parameters=parameters, plotfolder=plotfolder, model_history=model_history) #OK
#    plot_model(model, show_shapes=True, to_file=plotfolder+'/Model.pdf') #OK
    plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)

    plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, use_best_model=use_best_model)

    #print "--- mean ---"
    pred_trains, weights_trains, normweights_trains, lumiweights_trains = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_train, sample_weights=sample_weights_train, pred=pred_train, labels=labels_train)
    pred_vals, weights_vals, normweights_vals, lumiweights_vals = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_val, sample_weights=sample_weights_val, pred=pred_val, labels=labels_val)
    pred_tests, weights_tests, normweights_tests, lumiweights_tests = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_test, sample_weights=sample_weights_test, pred=pred_test, labels=labels_test)

   # plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    #print "--- std ---"
    pred_trains_std, weights_trains_std, normweights_trains_std, lumiweights_trains_std = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_train, sample_weights=sample_weights_train, pred=pred_train_std, labels=labels_train)
    pred_vals_std, weights_vals_std, normweights_vals_std, lumiweights_vals_std = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_val, sample_weights=sample_weights_val, pred=pred_val_std, labels=labels_val)
    pred_tests_std, weights_tests_std, normweights_tests_std, lumiweights_tests_std = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_test, sample_weights=sample_weights_test, pred=pred_test_std, labels=labels_test)
    # #plot classifier output with error from std of the output
    # plot_outputs_1d_nodes_with_stderror(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains,pred_trains_std=pred_train_std, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals,pred_vals_std=pred_vals_std, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, pred_signals_std=pred_signals_std,eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    # #plot std of classifier output
    #print("aaaaa pred_trains_std[0] = ",pred_trains_std[0])
    plot_outputs_1d_nodes_std(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains,pred_trains_std=pred_trains_std, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals,pred_vals_std=pred_vals_std, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, pred_signals_std=pred_signals_std,eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)

    plot_outputs_1d_classes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, use_best_model=use_best_model)

    # # #FixME:
    # # #TypeError: ('Not JSON Serializable:', <function _fn at 0x2b8475af1668>)
    # print("... Store model architecture in PlotBayesianPerformance ...")
    # #Store model as JSON file for usage in UHH2
    # arch = model.to_json()
    # # save the architecture string to a file somehow, the below will work
    # with open(outputfolder+'/architecture.json', 'w') as arch_file:
    #     arch_file.write(arch)
    # # now save the weights as an HDF5 file
    # model.save_weights(outputfolder+'/weights.h5')
    print("End of PlotBayesianPerformance")
   
def PlotDeepPerformance(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0]):
    print(' ---- Now plotting the performance (PlotDeepPerformance) ---- ')
    gErrorIgnoreLevel = kWarning

    # Get parameters
    # runonfullsample = parameters['runonfullsample']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    # classes = parameters['classes']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)
    layers=parameters['layers']

    # Get model and its history
    regrate=parameters['regrate']
    #sigma=1 #FixME, read https://arxiv.org/pdf/1904.10004.pdf to choose right value   
    sigma=parameters['sigma']                    
    prior = lambda x, y, z, w, k: _prior_normal_fn(sigma, x, y, z, w, k)                                                                                                     
    method = lambda d: d.mean()
    layer_hd = []
    batchnorm_hd = []
    dropout_hd = []
    inputs = tf.keras.layers.Input(shape=(input_train.shape[1],))
    layer_hd.append(tf.layers.Dense(layers[0], activation=tf.nn.relu)(inputs))
    dropout_hd.append(tf.layers.dropout(layer_hd[0], rate=regrate)) #FixME: regmethod might be different
    #batchnorm_hd.append(tf.layers.batch_normalization(dropout_hd[0]))
    k=1
    for i in layers[1:len(layers)+1]:
        print(("current k:",k))
        label=str(k+1)
        #layer_hd.append(tfp.layers.DenseFlipout(i, activation=tf.nn.relu)(batchnorm_hd[k-1])) 
        layer_hd.append(tf.layers.Dense(i, activation=tf.nn.relu)(dropout_hd[k-1]))
        dropout_hd.append(tf.layers.dropout(layer_hd[k], rate=regrate))
        #batchnorm_hd.append(tf.layers.batch_normalization(dropout_hd[k]))
        k = k+1
    print(("total number of hidden layers:",k))
    #last_layer = tf.layers.Dense(labels_train.shape[1], activation=tf.nn.relu)(dropout_hd[k-1])
    last_layer = tf.layers.Dense(labels_train.shape[1], activation='softmax')(dropout_hd[k-1])
    print('Number of output classes: %i' % (labels_train.shape[1]))
    model = tf.keras.models.Model(inputs=inputs,outputs=last_layer) 
    file = h5py.File('output/DNN_'+tag+'/model.h5', 'w')
    weight = []
    for i in range(len(list(file.keys()))):
        weight.append(file['weight' + str(i)][:])
    model.set_weights(weight)                                                                                                                             
    model.summary()

    
    with open(outputfolder+'/model_history.pkl', 'rb') as f:
        model_history = pickle.load(f)


    predpostfix = ''
    if use_best_model:
        predpostfix = '_best'
    pred_train, pred_test, pred_val, pred_signals = load_predictions(outputfolder=outputfolder, filepostfix=predpostfix)

    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)

    log_model_performance(parameters=parameters, model_history=model_history, outputfolder=outputfolder)
    plot_loss(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_accuracy(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    plot_model(model, show_shapes=True, to_file=plotfolder+'/Model.pdf')
    plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, use_best_model=use_best_model)


    pred_trains, weights_trains, normweights_trains, lumiweights_trains, pred_vals, weights_vals, normweights_vals, lumiweights_vals, pred_tests, weights_tests, normweights_tests, lumiweights_tests = get_data_dictionaries(parameters=parameters, eventweights_train=eventweights_train, sample_weights_train=sample_weights_train, pred_train=pred_train, labels_train=labels_train, eventweights_val=eventweights_val, sample_weights_val=sample_weights_val, pred_val=pred_val, labels_val=labels_val, eventweights_test=eventweights_test, sample_weights_test=sample_weights_test, pred_test=pred_test, labels_test=labels_test)
    plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)

    plot_outputs_1d_classes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, use_best_model=use_best_model)
    



def PlotInputs(parameters, inputfolder, filepostfix, plotfolder):

    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    classtag = get_classes_tag(parameters)
    tag = dict_to_str(parameters)
    eqweight=parameters['eqweight']

    if not os.path.isdir(plotfolder):
        os.makedirs(plotfolder)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)

    with open(inputfolder+'/variable_names.pkl', 'rb') as f:
        variable_names = pickle.load(f)

    print("GET for plotting: input_train[0] = ", input_train[0])

    # Divide into classes
    input_train_classes = {}
    input_test_classes = {}
    input_val_classes = {}
    weights_train_classes = {}
    weights_test_classes = {}
    weights_val_classes = {}
    for i in range(labels_train.shape[1]):
        input_train_classes[i] = input_train[labels_train[:,i] == 1]
        input_test_classes[i] = input_test[labels_test[:,i] == 1]
        input_val_classes[i] = input_val[labels_val[:,i] == 1]
        weights_train_classes[i] = sample_weights_train[labels_train[:,i] == 1]
        weights_test_classes[i] = sample_weights_test[labels_test[:,i] == 1]
        weights_val_classes[i] = sample_weights_val[labels_val[:,i] == 1]

        if not eqweight: 
          weights_train_classes[i] = eventweights_train[labels_train[:,i] == 1]
          weights_test_classes[i] =  eventweights_test[labels_test[:,i] == 1]
          weights_val_classes[i] =   eventweights_val[labels_val[:,i] == 1]

    # Create class-title dictionary
    classes = parameters['classes']
    classtitles = {}
    for key in list(classes.keys()):
        cl = classes[key]
        title = ''
        for i in range(len(cl)):
            title = title + cl[i]
            if i < len(cl)-1:
                title = title + '+'
        classtitles[key] = title

    matplotlib.style.use('default')
    # print input_train_classes
    nbins = 100
    idx = 0
    for varname in variable_names:
        xmax = max([max(input_train_classes[i][:,idx]) for i in range(len(input_train_classes))])
        xmin = min([min(input_train_classes[i][:,idx]) for i in range(len(input_train_classes))])
        if xmax == xmin: xmax = xmin + 1.
        xmin = min([0,xmin])
        binwidth = (xmax - xmin) / float(nbins)
        bins = np.arange(xmin, xmax + binwidth, binwidth)

        plt.clf()
        fig = plt.figure()
        for i in range(len(input_train_classes)):
            mycolor = 'C'+str(i)
            #print input_train_classes[i][:,idx], len(input_train_classes[i][:,idx]), str(i), colorstr[i], varname
            plt.hist(input_train_classes[i][:,idx], weights=weights_train_classes[i], bins=bins, histtype='step', label='Training sample, '+classtitles[i], color=colorstr[i])
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(varname)
        plt.ylabel('Number of events / bin')
        fig.savefig(plotfolder + '/' + varname + '_'+fraction+'.pdf')
        # if runonfullsample: fig.savefig('Plots/InputDistributions/' + classtag+  '/' + varname + '_full.pdf')
        # else: fig.savefig('Plots/InputDistributions/' + classtag+  '/' + varname + '_part.pdf')
        idx += 1

        sys.stdout.write( '{0:d} of {1:d} plots done.\r'.format(idx, len(variable_names)))
        if not i == len(variable_names): sys.stdout.flush()


