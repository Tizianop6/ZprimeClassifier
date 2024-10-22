import ROOT

# Percorsi dei file .root
file2_path = "/gfsvol01/cms/users/pauletto/work/tensorflow/parametrize_all/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_500_1000_1500_2000_2500_3000_3500_4000_4500/graphs.root"
file1_path = "/gfsvol01/cms/users/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_500_1000_1500_2000_2500_3000_3500_4000_4500/graphs.root"

# Apri i file ROOT
file1 = ROOT.TFile(file1_path)
file2 = ROOT.TFile(file2_path)
print(file1.ls())
# Estrai i TGraph
tg1 = file1.Get("F1 score")
tg2 = file2.Get("F1 score")
canvas = ROOT.TCanvas("canvas", "MultiGraph Example", 800, 600)
tg1.Draw("APL")
canvas.SaveAs("ScorePlots/F1.pdf")

# Crea un TMultiGraph e aggiungi i TGraph
mg = ROOT.TMultiGraph()
mg.Add(tg1, "PLC")  # "P" per disegnare i punti
mg.Add(tg2, "PLC")

# Crea un canvas per disegnare il TMultiGraph
mg.Draw("A")  # "A" per disegnare gli assi

# Opzionale: Aggiungi titolo e etichette agli assi
mg.SetTitle("F1 scores;Signal mass [GeV];F1 scores")

# Mantieni la finestra aperta
input("Premi invio per terminare...")

# Chiudi i file ROOT
file1.Close()
file2.Close()
