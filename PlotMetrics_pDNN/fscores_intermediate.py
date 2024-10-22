import ROOT


def readgraph(path):
	file=ROOT.TFile(path)
	graph=file.Get("Graph;2 F1 score")
	return graph


# Percorsi dei file .root
file2_path = "/gfsvol01/cms/users/pauletto/work/tensorflow/parametrize_all/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_500_1000_1500_2000_2500_3000_3500_4000_4500/graphs.root"
file1_path = "/gfsvol01/cms/users/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_500_1000_1500_2000_2500_3000_3500_4000_4500/graphs.root"
nonparam_all_p="/gfsvol01/cms/users/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_/graphs.root"
param_all_p="/gfsvol01/cms/users/pauletto/work/tensorflow/parametrize_all/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_/graphs.root"
nonparam_excl_500_1000_p="/gfsvol01/cms/users/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_500_1000/graphs.root"
param_excl_500_1000_p="/gfsvol01/cms/users/pauletto/work/tensorflow/parametrize_all/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_500_1000/graphs.root"
param_excl_4000_p="/gfsvol01/cms/users/pauletto/work/tensorflow/parametrize_all/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_4000/graphs.root"
nonparam_excl_4000_p="/gfsvol01/cms/users/pauletto/work/tensorflow/nonparametrized/Plots/StandardScaler/DNN_layers_512_512__batchsize_32768__classes_4_TTbar_ST_WJets+DY_RSGluonToTT__epochs_50__eqweight_False__preprocess_StandardScalerexclM_4000/graphs.root"

# Apri i file ROOT
file1 = ROOT.TFile(file1_path)
file2 = ROOT.TFile(file2_path)

nonparam_all=ROOT.TFile(nonparam_all_p)
param_all=ROOT.TFile(param_all_p)
print(file1.ls())
# Estrai i TGraph
tg1 = file1.Get("Graph;2 F1 score")
tg2 = file2.Get("Graph;2 F1 score")

nonparam_all_graph=nonparam_all.Get("Graph;2 F1 score")
param_all_graph = param_all.Get("Graph;2 F1 score")
nonparam_excl_500_1000=readgraph(nonparam_excl_500_1000_p)
param_excl_500_1000=readgraph(param_excl_500_1000_p)
nonparam_excl_4000=readgraph(nonparam_excl_4000_p)
param_excl_4000=readgraph(param_excl_4000_p)


tg1.SetTitle("Non parametrized, only last 3 masses")
tg2.SetTitle("Parametrized, only last 3 masses")
nonparam_all_graph.SetTitle("Non parametrized, trained on all masses")
param_all_graph.SetTitle("Parametrized,  trained on all masses")
nonparam_excl_500_1000.SetTitle("Non parametrized, excluding 500,1000 GeV sigs")
param_excl_500_1000.SetTitle("Parametrized, excluding 500,1000 GeV sigs")
nonparam_excl_4000.SetTitle("Non parametrized, trained excluding 4000 GeV sigs")
param_excl_4000.SetTitle("Parametrized, trained excluding 4000 GeV sigs")


tg2.SetLineColor(ROOT.kRed)
tg2.SetMarkerColor(ROOT.kRed)
nonparam_all_graph.SetLineColor(ROOT.kTeal-7)
nonparam_all_graph.SetMarkerColor(ROOT.kTeal-7)
#nonparam_all_graph.SetLineStyle(ROOT.kDotted)

param_all_graph.SetLineColor(ROOT.kViolet)
param_all_graph.SetMarkerColor(ROOT.kViolet)
param_all_graph.SetLineStyle(ROOT.kDotted)

param_excl_500_1000.SetLineColor(ROOT.kRed)
param_excl_500_1000.SetMarkerColor(ROOT.kRed)


nonparam_excl_4000.SetLineStyle(ROOT.kDotted)


param_excl_4000.SetLineColorAlpha(ROOT.kRed,0.8)
param_excl_4000.SetLineStyle(ROOT.kDotted)
param_excl_4000.SetMarkerColorAlpha(ROOT.kRed,0.8)



canvas = ROOT.TCanvas("canvas", "MultiGraph Example", 800, 600)

# Crea un TMultiGraph e aggiungi i TGraph
mg = ROOT.TMultiGraph()
mg.SetTitle("F1 scores, comparing the DNNs trained on all masses with DNNs trained excluding the 4000 GeV sample;Signal mass [GeV];F1 scores")

#mg.Add(tg1)  # "P" per disegnare i punti
#mg.Add(tg2)
#mg.Add(nonparam_all_graph)
#mg.Add(param_all_graph)
#mg.Add(param_excl_500_1000)
#mg.Add(nonparam_excl_500_1000)

mg.Add(param_excl_4000)
mg.Add(nonparam_excl_4000)

# Crea un canvas per disegnare il TMultiGraph
mg.Draw("APL")  # "A" per disegnare gli assi
canvas.BuildLegend()
canvas.SaveAs("ScorePlots/F1_param_nonparam_excl4000.pdf")
"""
canvas3 = ROOT.TCanvas("canvas3", "MultiGraph Example", 800, 600)
canvas3.cd()
mg.GetXaxis().SetLimits(0,1300);

mg.SetMinimum(0)
mg.SetMaximum(0.014)
mg.GetYaxis().SetLimits(0,0.014);
mg.Draw("APL")
canvas3.BuildLegend()
canvas3.Modified()
canvas3.SaveAs("ScorePlots/F1_param_nonparam_first2zoom.pdf")
"""

# Opzionale: Aggiungi titolo e etichette agli assi

# Chiudi i file ROOT
file1.Close()
file2.Close()
