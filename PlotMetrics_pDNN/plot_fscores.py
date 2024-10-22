import os
import pickle
import ROOT
import numpy as np
# Inizializza una lista per contenere i dizionari
dictionaries = []
auc_gluon=[]
masses=[]
# Definisci il percorso base e il nome del file


def readfile(base_path, folder_name=""):
    auc_list=[]
    for mass in range(500, 6001, 500):
        folder_name = f"mass_{mass}"
        file_path = os.path.join(base_path+"/", folder_name, file_name)
        #masses.append(mass)
        
        # Controlla se il file esiste
        if os.path.exists(file_path):
            # Apri il file e carica il dizionario
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                #dictionaries.append(data)
                print(data)
                auc_list.append(data["RSGluonToTT"])
        else:
            print(f"File {file_path} non trovato.")

    np_auc_list=np.array(auc_list,dtype=np.float32)
    return np_auc_list


base_path = ""  # Sostituisci con il percorso appropriato se necessario
file_name = "AUCS.pkl"

nonparam_excl_4000="/home/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__regmethod_dropout__regrate_050000__batchnorm_False__epochs_50__learningrate_000050__runonfraction_100__eqweight_False__preprocess_StandardScaler__priorSigma_100exclM_4000_orig"
nonparam_default="/home/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__regmethod_dropout__regrate_050000__batchnorm_False__epochs_50__learningrate_000050__runonfraction_100__eqweight_False__preprocess_StandardScaler__priorSigma_100exclM_"
nonparam_excl_500_1000="/home/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__regmethod_dropout__regrate_050000__batchnorm_False__epochs_50__learningrate_000050__runonfraction_100__eqweight_False__preprocess_StandardScaler__priorSigma_100exclM_500_1000"

param_all="/home/pauletto/work/tensorflow/parametrize_all/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__regmethod_dropout__regrate_050000__batchnorm_False__epochs_50__learningrate_000050__runonfraction_100__eqweight_False__preprocess_StandardScaler__priorSigma_100exclM_"

param_excl_4000="/home/pauletto/work/tensorflow/parametrize_all/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__regmethod_dropout__regrate_050000__batchnorm_False__epochs_50__learningrate_000050__runonfraction_100__eqweight_False__preprocess_StandardScaler__priorSigma_100exclM_4000"


param_excl_500_1000="/home/pauletto/work/tensorflow/parametrize_all/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__regmethod_dropout__regrate_050000__batchnorm_False__epochs_50__learningrate_000050__runonfraction_100__eqweight_False__preprocess_StandardScaler__priorSigma_100exclM_500_1000"

nonparam_excl_all_exc4000="/home/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_500_1000_1500_2000_2500_3000_3500_4500_5000_5500_6000"

base_path=nonparam_default+"/"
# Itera attraverso i nomi delle cartelle
auc_nonparam_default=[]
for mass in range(500, 6001, 500):
    folder_name = f"mass_{mass}"
    file_path = os.path.join(base_path, folder_name, file_name)
    masses.append(mass)
    
    # Controlla se il file esiste
    if os.path.exists(file_path):
        # Apri il file e carica il dizionario
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            dictionaries.append(data)
            #print(data)
            auc_nonparam_default.append(data["RSGluonToTT"])
    else:
        print(f"File {file_path} non trovato.")

# Ora la lista 'dictionaries' contiene tutti i dizionari caricati
print("Dizionari caricati:", len(dictionaries))
print("dizioni: ", auc_nonparam_default)
print("masses:", masses )


base_path=param_all+"/"
# Itera attraverso i nomi delle cartelle
auc_param_all=[]
for mass in range(500, 6001, 500):
    folder_name = f"mass_{mass}"
    file_path = os.path.join(base_path, folder_name, file_name)
    #masses.append(mass)
    
    # Controlla se il file esiste
    if os.path.exists(file_path):
        # Apri il file e carica il dizionario
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            dictionaries.append(data)
            #print(data)
            auc_param_all.append(data["RSGluonToTT"])
    else:
        print(f"File {file_path} non trovato.")


base_path=param_excl_500_1000+"/"
# Itera attraverso i nomi delle cartelle
auc_param_excl_500_1000=[]
for mass in range(500, 6001, 500):
    folder_name = f"mass_{mass}"
    file_path = os.path.join(base_path, folder_name, file_name)
    #masses.append(mass)
    
    # Controlla se il file esiste
    if os.path.exists(file_path):
        # Apri il file e carica il dizionario
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            dictionaries.append(data)
            #print(data)
            auc_param_excl_500_1000.append(data["RSGluonToTT"])
    else:
        print(f"File {file_path} non trovato.")





base_path=param_excl_4000+"/"
# Itera attraverso i nomi delle cartelle
auc_param_excl_4000=[]
for mass in range(500, 6001, 500):
    folder_name = f"mass_{mass}"
    file_path = os.path.join(base_path, folder_name, file_name)
    #masses.append(mass)
    
    # Controlla se il file esiste
    if os.path.exists(file_path):
        # Apri il file e carica il dizionario
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            dictionaries.append(data)
            #print(data)
            auc_param_excl_4000.append(data["RSGluonToTT"])
    else:
        print(f"File {file_path} non trovato.")



base_path=nonparam_excl_4000+"/"
# Itera attraverso i nomi delle cartelle
auc_nonparam_excl_4000=[]

for mass in range(500, 6001, 500):
    folder_name = f"mass_{mass}"
    file_path = os.path.join(base_path, folder_name, file_name)
    #masses.append(mass)
    
    # Controlla se il file esiste
    if os.path.exists(file_path):
        # Apri il file e carica il dizionario
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            dictionaries.append(data)
            #print(data)
            auc_nonparam_excl_4000.append(data["RSGluonToTT"])
    else:
        print(f"File {file_path} non trovato.")





c1=ROOT.TCanvas("c1","c1",1000,700)
np_masses=np.array(masses,dtype=np.float32)
np_aucs=np.array(auc_nonparam_default,dtype=np.float32)
np_auc_param_all=np.array(auc_param_all,dtype=np.float32)
np_auc_param_excl_500_1000=np.array(auc_param_excl_500_1000,dtype=np.float32)

np_auc_nonparam_excl_4000=np.array(auc_nonparam_excl_4000,dtype=np.float32)

np_auc_param_excl_4000=np.array(auc_param_excl_4000,dtype=np.float32)
print(np_auc_param_excl_500_1000)
print(np_auc_param_excl_4000)

graphs=[]

graphs.append(ROOT.TGraph(len(masses),np_masses,np_aucs))
graphs[-1].SetTitle("AUC for non parametrized DNN; Signal mass [GeV]; AUC")
graphs[-1].SetLineColor(ROOT.kRed)
graphs[-1].SetMarkerColor(ROOT.kRed)
graphs[-1].SetMarkerStyle(20)
graphs[-1].SetMarkerSize(1)
graphs[-1].SetLineWidth(2)


graphs.append(ROOT.TGraph(len(masses),np_masses,np_auc_param_all))
graphs[-1].SetTitle("AUC for parametrized DNN, all masses; Signal mass [GeV]; AUC")
graphs[-1].SetLineColor(ROOT.kBlue)
graphs[-1].SetMarkerColor(ROOT.kBlue)
graphs[-1].SetMarkerStyle(20)
graphs[-1].SetMarkerSize(1)
graphs[-1].SetLineWidth(2)

graphs.append(ROOT.TGraph(len(masses),np_masses,np_auc_param_excl_500_1000))
graphs[-1].SetTitle("AUC for parametrized DNN, excluding 500, 1000 GeV signals; Signal mass [GeV]; AUC")
graphs[-1].SetLineColor(ROOT.kViolet)
graphs[-1].SetMarkerColor(ROOT.kViolet)
graphs[-1].SetMarkerStyle(20)
graphs[-1].SetMarkerSize(1)
graphs[-1].SetLineWidth(2)


graphs.append(ROOT.TGraph(len(masses),np_masses,np_auc_param_excl_4000))
graphs[-1].SetTitle("AUC for parametrized DNN, excluding 4000 GeV signals; Signal mass [GeV]; AUC")
graphs[-1].SetLineColor(ROOT.kTeal-7)
graphs[-1].SetMarkerColor(ROOT.kTeal-7)
graphs[-1].SetMarkerStyle(20)
graphs[-1].SetMarkerSize(1)
graphs[-1].SetLineWidth(2)



graphs.append(ROOT.TGraph(len(masses),np_masses,np_auc_nonparam_excl_4000))
graphs[-1].SetTitle("AUC for nonparametrized DNN, excluding 4000 GeV signals; Signal mass [GeV]; AUC")
graphs[-1].SetLineColor(ROOT.kRed+3)
graphs[-1].SetMarkerColor(ROOT.kRed+3)
graphs[-1].SetMarkerStyle(20)
graphs[-1].SetMarkerSize(1)
graphs[-1].SetLineWidth(2)


graphs.append(ROOT.TGraph(len(masses),np_masses,readfile(nonparam_excl_500_1000)))
graphs[-1].SetTitle("AUC for nonparametrized DNN, excluding 500, 1000 GeV signals; Signal mass [GeV]; AUC")
graphs[-1].SetLineColor(ROOT.kBlue+3)
graphs[-1].SetMarkerColor(ROOT.kBlue+3)
graphs[-1].SetMarkerStyle(20)
graphs[-1].SetMarkerSize(1)
graphs[-1].SetLineWidth(2)


graphs.append(ROOT.TGraph(len(masses),np_masses,readfile(nonparam_excl_all_exc4000)))
graphs[-1].SetTitle("AUC for nonparametrized DNN, excluding all exc 4000 GeV signals; Signal mass [GeV]; AUC")
graphs[-1].SetLineColor(ROOT.kBlue+3)
graphs[-1].SetMarkerColor(ROOT.kBlue+3)
graphs[-1].SetMarkerStyle(20)
graphs[-1].SetMarkerSize(1)
graphs[-1].SetLineWidth(2)


nonparam_excl_all="/home/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_500_1000_1500_2000_2500_3000_3500_4000_4500_5000_5500_6000"
graphs.append(ROOT.TGraph(len(masses),np_masses,readfile(nonparam_excl_all)))
graphs[-1].SetTitle("AUC for nonparametrized DNN, excluding all; Signal mass [GeV]; AUC")
graphs[-1].SetLineColor(ROOT.kBlue+3)
graphs[-1].SetMarkerColor(ROOT.kBlue+3)
graphs[-1].SetMarkerStyle(20)
graphs[-1].SetMarkerSize(1)
graphs[-1].SetLineWidth(2)


param_excl_allbutlast3="/home/pauletto/work/tensorflow/parametrize_all/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_500_1000_1500_2000_2500_3000_3500_4000_4500"

graphs.append(ROOT.TGraph(len(masses),np_masses,readfile(param_excl_allbutlast3)))
graphs[-1].SetTitle("Parametrized DNN, only last 3 masses; Signal mass [GeV]; AUC")
graphs[-1].SetLineColor(ROOT.kViolet-2)
graphs[-1].SetMarkerColor(ROOT.kViolet-2)
graphs[-1].SetMarkerStyle(20)
graphs[-1].SetMarkerSize(1)
graphs[-1].SetLineWidth(2)

nonparam_excl_allbutlast3="/home/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_500_1000_1500_2000_2500_3000_3500_4000_4500"
graphs.append(ROOT.TGraph(len(masses),np_masses,readfile(nonparam_excl_allbutlast3)))
graphs[-1].SetTitle("Nonparametrized DNN, only last 3 masses; Signal mass [GeV]; AUC")
graphs[-1].SetLineColor(ROOT.kRed-3)
graphs[-1].SetMarkerColor(ROOT.kRed-3)
graphs[-1].SetMarkerStyle(20)
graphs[-1].SetMarkerSize(1)
graphs[-1].SetLineWidth(2)

nonparam_only4000="/home/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_500_1000_1500_2000_2500_3000_3500_4500_5000_5500_6000"
graphs.append(ROOT.TGraph(len(masses),np_masses,readfile(nonparam_only4000)))
graphs[-1].SetTitle("Nonparametrized DNN, only 4000GeV; Signal mass [GeV]; AUC")
graphs[-1].SetLineColor(ROOT.kBlue+3)
graphs[-1].SetMarkerColor(ROOT.kBlue+3)
graphs[-1].SetMarkerStyle(20)
graphs[-1].SetMarkerSize(1)
graphs[-1].SetLineWidth(2)




for i in range(len(graphs)):
    canvas=ROOT.TCanvas()
    canvas.cd()
    graphs[i].Draw("ALP")
    canvas.SaveAs("AUCPlots/"+graphs[i].GetTitle()+".png")#plot_aucs.png")
    canvas.SaveAs("AUCPlots/"+graphs[i].GetTitle()+".pdf")#plot_aucs.png")
    
    #c1.SaveAs("plot_aucs.pdf")


graphs[0].SetTitle("Non parametrized; Signal mass [GeV]; AUC")
graphs[1].SetTitle("Parametrized, all masses; Signal mass [GeV]; AUC")
graphs[2].SetTitle("Parametrized without 500,1000 GeV; Signal mass [GeV]; AUC")
graphs[3].SetTitle("Parametrized without 4000 GeV; Signal mass [GeV]; AUC")

graphs[4].SetTitle("Nonparametrized without 4000 GeV; Signal mass [GeV]; AUC")

graphs[5].SetTitle("Nonparametrized without 500,1000 GeV; Signal mass [GeV]; AUC")


c2=ROOT.TCanvas("c1","c1",1000,700)
mg= ROOT.TMultiGraph()
mg.SetTitle("AUC of RSGluon class, for different signal masses; Signal mass [GeV]; AUC")




#mg.Add(graphs[0])
#mg.Add(graphs[1])
#mg.Add(graphs[2])
mg.Add(graphs[3])

mg.Add(graphs[4])

mg.Add(graphs[5])

c2.cd()
mg.Draw("ALP")
c2.BuildLegend()

c2.SaveAs("mg.png")

c2.SaveAs("mg.pdf")
