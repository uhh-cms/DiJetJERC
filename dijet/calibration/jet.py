# coding: utf-8

"""
Custom jet energy calibration methods that disable data uncertainties (for searches).
"""

# from columnflow.calibration import Calibrator
from columnflow.calibration.cms.jets import jec

# custom jec calibrator that only runs nominal correction
jec_nominal_run3 = jec.derive(
    "jec_nominal_run3",
    cls_dict={
        "uncertainty_sources": [],
        "met_name": "PuppiMET",
        "raw_met_name": "RawPuppiMET",
    },
)

jec_nominal_run2 = jec.derive(
    "jec_nominal_run2",
    cls_dict={
        "uncertainty_sources": [],
        "met_name": "MET",
        "raw_met_name": "RawMET",
    },
)

# cannot be set dynamically because the original init of the derived calibrator is not called then anymore
# @jec_nominal.init
# def jec_nominal_init(self: Calibrator):
#     # set met names from config
#     self.met_name = self.config_inst.x.met_name
#     self.raw_met_name = self.config_inst.x.raw_met_name
