#!/usr/bin/python3.6
# coding: utf-8

"""
Script to convert output of dijet.JERtoRoot to a TGraphAsymError.
Keep eye on uprrot PR https://github.com/scikit-hep/uproot5/pull/1144
to directly implement it in the task itself.

Root is not in Sandboxes from CF.
Needs to be run with python 3.6 isntead.

Structure of arguments is based on CF output. Run via
./convert_to_root.py --version <v> --sample <dataset> --<option> <arg> ...
"""

import sys
import argparse
import os
import pickle
import ROOT

# Filter sys.path to remove Python 3.9 specific paths
# Conflicts of 3.9 to 3.6. 
# Second option is to sort sys.path and move paths from 3.9 to bottom
sys.path = [p for p in sys.path if "python3.9" not in p]

# Set up argument parser
parser = argparse.ArgumentParser(description='Read a pickle file with JERs and convert them to root objects.')
parser.add_argument('--config', default='config_2017', help='Configuration name (default: config_2017)')
parser.add_argument('--shift', default='nominal', help='Shift type (default: nominal)')
parser.add_argument('--uncertainty', default='skip_jecunc', help='Uncertainty type (default: skip_jecunc)')
parser.add_argument('--selector', default='default', help='Selector type (default: default)')
parser.add_argument('--producer', default='default', help='Producer type (default: default)')
parser.add_argument('--version', required=True, help='Version, must be specified.')
parser.add_argument('--sample', required=True, help='Sample name, must be specified.')

# Parse the command-line arguments
args = parser.parse_args()

# Print all options and the arguments given
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

# Construct the file path
base_path = "/nfs/dust/cms/user/paaschal/WorkingArea/Analysis/JERC/DiJet/data/dijet_store/analysis_dijet/"
full_path = os.path.join(
    base_path,
    "dijet.JERtoRoot/",
    f"{args.config}/",
    f"{args.shift}/",
    f"calib__{args.uncertainty}/",
    f"sel__{args.selector}",
    f"prod__{args.producer}/",
    f"{args.version}/",
    f"{args.sample}/",
    "jers.pickle",
)

# Check if the file exists
if not os.path.exists(full_path):
    raise Exception(f"The file {full_path} does not exist.")

# Load and display the contents of the pickle file
with open(full_path, 'rb') as file:
    data = pickle.load(file)

# Store root file in output directory from analysis
root_path = full_path.replace("dijet.JERtoRoot", "rootfiles").replace(".pickle", ".root")

# Ensure the directory exists
os.makedirs(os.path.dirname(root_path), exist_ok=True)
root_file = ROOT.TFile(root_path, "RECREATE")

for key, value in data.items():
    # Extract data for TGraphAsymmErrors
    # mask = value['fX'].nonzero()
    n = len(value['fX'])
    x = value['fX']
    y = value['fY']
    xerrup = value['fXerrUp']
    xerrdown = value['fXerrDown']
    yerrup = value['fYerrUp']
    yerrdown = value['fYerrDown']
    
    # Create TGraphAsymmErrors object
    graph = ROOT.TGraphAsymmErrors(n, y, x, yerrdown, yerrup, xerrdown, xerrup)

root_file.Close()
print(f"All TGraphAsymmErrors objects have been stored in {root_path}")
