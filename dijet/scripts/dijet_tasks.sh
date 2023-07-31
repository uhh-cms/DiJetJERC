#!/bin/bash

dijet_selection(){
    law run cf.SelectEvents --config config_2017_limited --version v0 --dataset data_jetht_e --selector dijet_selection --remove-output 0,a,y
}