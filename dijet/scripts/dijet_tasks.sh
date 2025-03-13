#!/bin/bash

dijet_selection(){
    law run cf.SelectEvents --config config_2017_limited --version v0 --dataset data_jetht_e --remove-output 0,a,y
}

dijet_plotting(){
    law run cf.PlotVariables1D --config config_2017_limited --version v0 --dataset data_jetht_e --remove-output 2,a,y --variables dijets_asymmetry
}

dijet_alpha(){
    law run dijet.AlphaExtrapolation                                         \
    --config config_2017_limited                                                     \
    --version sync_alpha                                                     \
    --datasets data_jetht_e                                                  \
    --processes data                                                         \
    --variables dijets_alpha-probejet_abseta-dijets_pt_avg-dijets_asymmetry  \
    --remove-output 0,a,y
}

dijet_jer(){
    law run dijet.JER                                                        \
    --config config_2017_limited                                                     \
    --version sync_alpha                                                     \
    --datasets data_jetht_e                                                  \
    --processes data                                                         \
    --variables dijets_alpha-probejet_abseta-dijets_pt_avg-dijets_asymmetry  \
    --remove-output 0,a,y
}

dijet_plot_asym(){
    law run dijet.PlotAsymmetries                                            \
    --config config_2017_limited                                                     \
    --version sync_alpha                                                     \
    --datasets data_jetht_e                                               \
    --processes data                                                    \
    --variables dijets_alpha-probejet_abseta-dijets_pt_avg-dijets_asymmetry  \
    --remove-output 0,a,y
}
