# coding: utf-8

"""
Configuration of the DiJet analysis.
"""

import os
import law
import order as od

from columnflow.util import maybe_import

ak = maybe_import("awkward")


#
# the main analysis object
#

analysis_dijet = ana = od.Analysis(
    name="analysis_dijet",
    id=1,
)

# analysis-global versions
ana.x.versions = {}

# files of bash sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf_prod.sh",
    "$CF_BASE/sandboxes/venv_columnar.sh",
]

# files of cmssw sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.cmssw_sandboxes = [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
]

# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("DIJET_BUNDLE_CMSSW", "1")):
    del ana.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
analysis_dijet.set_aux("config_groups", {})

#
# setup configs
#

# an example config is setup below, based on cms NanoAOD v9 for Run2 2017, focussing on
# ttbar and single top MCs, plus single muon data
# update this config or add additional ones to accomodate the needs of your analysis

#
# import campaigns and load configs
#

from dijet.config.config_run2 import add_config
import cmsdb.campaigns.run2_2017_JMEnano_v9

campaign_run2_2017_JMEnano_v9 = cmsdb.campaigns.run2_2017_JMEnano_v9.campaign_run2_2017_JMEnano_v9

# default config
config_2017 = add_config(
    analysis_dijet,
    campaign_run2_2017_JMEnano_v9.copy(),
    config_name="config_2017",
    config_id=2,
)

# config with limited number of files
config_2017_limited = add_config(
    analysis_dijet,
    campaign_run2_2017_JMEnano_v9.copy(),
    config_name="config_2017_limited",
    config_id=12,
    limit_dataset_files=1,
)
