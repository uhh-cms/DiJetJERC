#!/bin/sh

config="run3_2024_JMEnano_v15"
#config="run3_2024_JMEnano_v15_limited"
version="v3"

args=(
    "--config" "${config}"
    "--version" "${version}"
    "--samples" "qcdht,data"
    "--postprocessor mctruth"
    "--levels" "gen"
    #"--workflow" "htcondor"
    #"--workers" "4"
)
echo law run dijet.Asymmetry "${args[@]}" "$@"
echo law run dijet.AlphaExtrapolation "${args[@]}" "$@"
echo law run dijet.JER "${args[@]}" "$@"
echo law run dijet.SF "${args[@]}" "$@"

args+=(
    #--bin-selectors" 'alpha,min=0.3,max=0.3:abseta,min=0.0,max=0.5:pt,min=100,max=200'
    "--file-types png,pdf"
    #"--bin-selectors" 'alpha,min=0.3,max=0.3:abseta,min=1.9,max=2.7:pt,min=100,max=200'
)
echo claw run dijet.PlotSF "${args[@]}" "$@"
echo claw run dijet.PlotJER "${args[@]}" "$@"
echo claw run dijet.PlotAlphaExtrapolation "${args[@]}" "$@"
echo claw run dijet.PlotAsymmetry "${args[@]}" "$@"
echo claw run dijet.PlotMCTruthResponse "${args[@]}" "$@"
