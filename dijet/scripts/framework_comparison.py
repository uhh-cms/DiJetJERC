import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import uproot
import pickle
import hist

cf_user = "/nfs/dust/cms/user/paaschal/WorkingArea/Analysis/JERC/DiJet/"
cf_store_local = cf_user + "data/dijet_store/"  # $CF_STORE_LOCAL
uhh_path = "/nfs/dust/cms/user/paaschal/UHH2_DiJet/CMSSW_10_6_28/src/UHH2/DiJetJERC/JERSF_Analysis/"

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", type=str, nargs="+")
parser.add_argument("-t", "--task", type=str, nargs="+")
parser.add_argument("-c", "--category", type=str, nargs="+")
args = parser.parse_args()
version = args.version
task = args.task
category = args.category

cat_list = [0]
cat_list += [
    int(f"1{a:01d}{pp:02d}{nn:02d}")
    for a in range(1,6)
    for pp in range(1,34)
    for nn in range(1,14)
]
# print(cat_list)

cat_dict = {
    "incl": [cat_list[0]],
}

var_dict = {
    'jet1_pt': r'$pT_{jet 1} [GeV]$',
    "dijet_asymmetry": r"A",
}

SM_ttbar_dict = {
    'sl': ('tt_sl_powheg', 1100),
    'dl': ('tt_dl_powheg', 1200),
    'fl': ('tt_fh_powheg', 1300),
}


def create_empty_histogram_like(existing_hist):
    # Create an empty histogram with the same shape
    return hist.Hist(*existing_hist.axes)


def cf_get_data():
    output_path = cf_store_local + "/analysis_dijet/cf.MergeHistograms/config_2017_limited/"
    output_path += "data_jetht_e/nominal/calib__skip_jecunc/sel__default/prod__default/v0/"

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
    # print(data_h.values())
    return data_h


def root_get_data(year, study, corr, coll, run, hname):
    fname = "histograms_data_incl_full.root"
    hpath = "hist_preparation/data/wide_eta_bin/file/"
    fpath = f"{study}/{year}/{corr}/{coll}/{run}_{year}/"
    path = uhh_path + hpath + fpath + fname
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
    cf_y = cf_tt.values()
    cf_y = cf_y[-1, -1, -1]
    cf_bin_centers = (cf_x[:-1] + cf_x[1:]) / 2
    print(f"cf: {len(cf_bin_centers)} x {len(cf_y)}")

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
    fig_root.savefig(cf_user + "dijet/scripts/plots/comparison/dijet_asym.pdf")
    fig_root.clear()
    print("Created plots for jet1_pt")
