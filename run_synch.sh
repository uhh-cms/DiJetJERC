#!/bin/sh

config="run2_2017_JMEnano_v9"
version="v1"

datasets=(
    "qcd_ht100to200_madgraph"
    "data_jetht_d"
)

variables_2d=(
    # -- UHH vs CF
    # dijet quantities
    "s_alpha_uhh2-s_alpha_cf"
    "s_pt_avg_uhh2-s_pt_avg_cf"
    "s_asymmetry_uhh2-s_asymmetry_signflip_cf"
    # leading jet pTs
    "s_jet1_pt_uhh2-s_jet1_pt_cf"
    "s_jet2_pt_uhh2-s_jet2_pt_cf"
    "s_jet3_pt_uhh2-s_jet3_pt_cf"
    "s_jet4_pt_uhh2-s_jet4_pt_cf"
    # number of jets
    "s_n_jet_uhh2-s_n_jet_cf"
    # probe jet
    "s_probe_jet_pt_uhh2-s_probe_jet_pt_cf"
    "s_probe_jet_eta_uhh2-s_probe_jet_eta_cf"
    "s_probe_jet_abseta_uhh2-s_probe_jet_abseta_cf"
    "s_probe_jet_phi_uhh2-s_probe_jet_phi_cf"
    # # reference jet
    "s_reference_jet_pt_uhh2-s_reference_jet_pt_cf"
    "s_reference_jet_eta_uhh2-s_reference_jet_eta_cf"
    "s_reference_jet_abseta_uhh2-s_reference_jet_abseta_cf"
    "s_reference_jet_phi_uhh2-s_reference_jet_phi_cf"
    # CF-only
    "s_reference_jet_eta_cf-s_probe_jet_eta_cf"
    "s_reference_jet_abseta_cf-s_probe_jet_abseta_cf"
    "s_reference_jet_pt_cf-s_probe_jet_pt_cf"
    "s_reference_jet_eta_cf-s_reference_jet_phi_cf"
    "s_probe_jet_eta_cf-s_probe_jet_phi_cf"
    "s_alpha_cf-s_jet3_pt_cf"
    # UHH2-only
    "s_reference_jet_abseta_uhh2-s_probe_jet_abseta_uhh2"
    "s_reference_jet_pt_uhh2-s_probe_jet_pt_uhh2"
    "s_reference_jet_eta_uhh2-s_reference_jet_phi_uhh2"
    "s_probe_jet_eta_uhh2-s_probe_jet_phi_uhh2"
    "s_alpha_uhh2-s_jet3_pt_uhh2"
)
variables_2d_str="$(IFS=","; echo "${variables_2d[*]}")"
variables_1d_str="$(echo "${variables_2d_str}" | tr '-' ',')"

# loop through datasets
for dataset in "${datasets[@]}"; do

args_common=(
    "--config" "${config}"
    "--version" "${version}"
    "--datasets" "${dataset}"
    "--selector" "uhh2"
    "--producers" "default,uhh2,uhh2_categories"
    "--hist-producer" "cf_default"
    "--categories" "incl,uhh2_same_probe_jet"
    "--workflow" htcondor
    "--workers" "15"
    "--file-types" "png,pdf"
)

#
# 1D plots with CF and UHH in same figure
#

args=(
    "${args_common[@]}"
    "--variables" "${variables_2d_str}"
)
echo law run dijet.PlotTwoVariables1D "${args[@]}" "$@"


#
# 2D plots
#

args=(
    "${args_common[@]}"
    "--variables" "${variables_2d_str}"
    "--zscale" "log"
)
echo law run cf.PlotVariables2D "${args[@]}" "$@"


#
# 1D plots with CF and UHH in separate figures
#

args=(
    "${args_common[@]}"
    "--variables" "${variables_1d_str}"
)
echo law run cf.PlotVariables1D "${args[@]}" "$@"

done
