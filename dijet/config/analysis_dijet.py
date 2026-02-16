# coding: utf-8

"""
Configuration of the DiJet analysis.
"""

import os
import law
import order as od

from columnflow.util import maybe_import

from dijet.config.hist_hooks import add_hist_hooks

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
    "$CF_BASE/sandboxes/cf.sh",
    "$DIJET_BASE/sandboxes/venv_columnar.sh",
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

# add hist hooks
add_hist_hooks(analysis_dijet)

#
# setup configs
#

# an example config is setup below, based on cms NanoAOD v9 for Run2 2017, focussing on
# ttbar and single top MCs, plus single muon data
# update this config or add additional ones to accomodate the needs of your analysis

#
# import campaigns and load configs
#

from dijet.config.config_run2 import add_config as add_config_run2
from cmsdb.campaigns.run2_2017_JMEnano_v9 import campaign_run2_2017_JMEnano_v9

from dijet.config.config_run3 import add_config as add_config_run3
from cmsdb.campaigns.run3_2024_JMEnano_v15 import campaign_run3_2024_JMEnano_v15


# default config
cfg_run2_2017_JMEnano_v9 = add_config_run2(
    analysis_dijet,
    campaign_run2_2017_JMEnano_v9.copy(),
    config_name="run2_2017_JMEnano_v9",
    config_id=2_17_9_1,
)

# config with limited number of files
cfg_run2_2017_JMEnano_v9_limited = add_config_run2(
    analysis_dijet,
    campaign_run2_2017_JMEnano_v9.copy(),
    config_name="run2_2017_JMEnano_v9_limited",
    config_id=2_17_9_2,
    limit_dataset_files=1,
)

# default config
cfg_run3_2024_JMEnano_v15 = add_config_run3(
    analysis_dijet,
    campaign_run3_2024_JMEnano_v15.copy(),
    config_name="run3_2024_JMEnano_v15",
    config_id=3_24_15_1,
)

cfg_run3_2024_JMEnano_v15_limited = add_config_run3(
    analysis_dijet,
    campaign_run3_2024_JMEnano_v15.copy(),
    config_name="run3_2024_JMEnano_v15_limited",
    config_id=3_24_15_2,
    limit_dataset_files=1,
)
