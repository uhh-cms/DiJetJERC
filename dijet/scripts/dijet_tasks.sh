#!/bin/bash

dijet_selection(){
    law run cf.SelectEvents --config config_2017_limited --version v0 --dataset data_jetht_e --remove-output 0,a,y
}

dijet_plotting(){
    law run cf.PlotVariables1D --config config_2017_limited --version v0 --dataset data_jetht_e --remove-output 2,a,y --variables dijets_asymmetry
}