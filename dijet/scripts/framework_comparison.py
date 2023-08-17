import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import uproot
import pickle
import hist

cf_store_local = "/nfs/dust/cms/user/paaschal/WorkingArea/Analysis/JERC/DiJet/data/dijet_store/"  # $CF_STORE_LOCAL

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", type=str, nargs="+")
parser.add_argument("-t", "--task", type=str, nargs="+")
parser.add_argument("-c", "--category", type=str, nargs="+")
args = parser.parse_args()
version = args.version
task = args.task
category = args.category

def create_empty_histogram_like(existing_hist):
    # Create an empty histogram with the same shape
    return hist.Hist(*existing_hist.axes)

def cf_get_data():
    output_path = cf_store_local+"/analysis_dijet/cf.MergeHistograms/config_2017_limited/data_jetht_e/nominal/calib__skip_jecunc/sel__default/prod__default/v0/"

    file_name = "hist__dijets_asymmetry.pickle"
    pickle_file = os.path.join(output_path, file_name)

    with open(pickle_file, "rb") as f:
        histograms = pickle.load(f)

    # print(type(histograms))
    # print(histograms.axes)
    # for hist_name, hist_data in histograms.items():
    #     print(f"Axis names for histogram '{hist_name}':")
    #     for axis_name in hist_data["axes"]:
    #         print(axis_name)

    data_h = histograms[{
        "category": [hist.loc(0)],
    }]
    print(data_h.values())
    return data_h


def root_get_data(year, study, corr, coll, run, hname):
    fname = "histograms_data_incl_full.root"
    dpath = "/nfs/dust/cms/user/paaschal/UHH2_DiJet/CMSSW_10_6_28/src/UHH2/DiJetJERC/JERSF_Analysis/"
    hpath = "hist_preparation/data/wide_eta_bin/file/"
    fpath = f"{study}/{year}/{corr}/{coll}/{run}_{year}/"
    path = dpath + hpath + fpath + fname
    with uproot.open(path) as uhh2_file:
        # keys = uhh2_file.keys()
        # for key in keys:
        #     print(key)
        uhh2_hist = uhh2_file[hname].to_hist()
    print(type(uhh2_hist))
    return uhh2_hist


if __name__ == "__main__":

    cf_tt = cf_get_data()
    cf_x = cf_tt.axes[3].edges
    print(11111)
    print()
    print(cf_tt.axes)
    print()
    print("------------------  NULLER")
    print(cf_tt.axes[0])
    print()
    print(cf_tt.axes[0].edges)
    print()
    print("------------------  Threeer")
    print(cf_tt.axes[3])
    print()
    print(cf_tt.axes[3].edges)
    print()
    print("------------------  XXXXER")
    print(cf_x)
    print(len(cf_x))
    cf_y = cf_tt.values()
    cf_y = cf_y[-1, -1, -1]
    print("------------------  WHYYYYER")
    print(cf_y)
    print(len(cf_y))
    cf_bin_centers = (cf_x[:-1] + cf_x[1:]) / 2
    print()
    print(cf_bin_centers)
    print(len(cf_bin_centers))
    print(f"cf: {len(cf_x)} x {len(cf_y)}")

    study = "eta_common_fine1_finealpha_prescale"
    hname = "asymm_FE_reference_eta2_pt10_alpha11"
    uhh2_tt = root_get_data("UL17", study, "Summer20UL17_V2", "AK4Puppi", "RunBCDEF", hname)

    uhh2_x = uhh2_tt.axes[0].edges
    uhh2_y = uhh2_tt.values()
    uhh2_y = np.append(uhh2_y, uhh2_y[-1])
    uhh2_bin_centers = (uhh2_x[:-1] + uhh2_x[1:]) / 2
    print(f"uhh2: {len(uhh2_x)} x {len(uhh2_y)}")

    fig_root, ax_root = plt.subplots(figsize=(8, 8))
    ax_root.step(uhh2_x[:], uhh2_y / np.sum(uhh2_y), label="uhh2", where="post")
    ax_root.step(cf_bin_centers[:], cf_y / np.sum(cf_y), label="cf", where="post")

    ax_root.grid()
    ax_root.legend(loc="upper right")
    ax_root.set_title("Just give me a title")
    ax_root.set_xlabel("jet pt")
    ax_root.set_ylabel("event counts")
    fig_root.savefig("/nfs/dust/cms/user/paaschal/WorkingArea/Analysis/JERC/DiJet/dijet/scripts/plots/comparison/dijet_asym.pdf")
    fig_root.clear()
    print("Created plots for jet1_pt")
