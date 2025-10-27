#!/bin/sh

config="run2_2017_JMEnano_v9"
version="v1"

args=(
    "--config" "${config}"
    "--version" "${version}"
    "--samples" "qcdht,data"
    "--workflow" "htcondor"
    "--workers" "4"
)
echo law run dijet.Asymmetry "${args[@]}" "$@"
echo law run dijet.AlphaExtrapolation "${args[@]}" "$@"
echo law run dijet.JER "${args[@]}" "$@"
echo law run dijet.SF "${args[@]}" "$@"

args+=(
    "--bin-selectors" 'alpha,min=0.3,max=0.3:probejet_abseta,min=0.0,max=0.5:dijets_pt_avg,min=100,max=200'
    #"--bin-selectors" 'alpha,min=0.3,max=0.3:probejet_abseta,min=1.9,max=2.7:dijets_pt_avg,min=100,max=200'
)
echo law run dijet.PlotSF "${args[@]}" "$@"
echo law run dijet.PlotJER "${args[@]}" "$@"
echo law run dijet.PlotAlphaExtrapolation "${args[@]}" "$@"
echo law run dijet.PlotAsymmetry "${args[@]}" "$@"
