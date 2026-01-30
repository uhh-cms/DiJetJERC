# coding: utf-8

"""
Configuration of the 2018 DiJet analysis.
"""

from __future__ import annotations

import math
import os
import re
from typing import Set

import yaml
from scinum import Number
import order as od

from columnflow.util import DotDict
from columnflow.config_util import get_root_processes_from_campaign
from dijet.config.categories import add_categories
from dijet.config.variables import add_variables, add_uhh2_synch_variables
from dijet.config.cutflow_variables import add_cutflow_variables
from dijet.config.datasets import get_dataset_lfns
from dijet.config.analysis_dijet import analysis_dijet

from dijet.plotting.asymmetry import PlotAsymmetry

thisdir = os.path.dirname(os.path.abspath(__file__))


def add_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
) -> od.Config:
    # validations
    assert campaign.x.year in [2016, 2017, 2018]
    if campaign.x.year == 2016:
        assert campaign.x.vfp in ["pre", "post"]

    # gather campaign data
    year = campaign.x.year
    year2 = year % 100
    corr_postfix = f"{campaign.x.vfp}VFP" if year == 2016 else ""

    if year != 2017:
        raise NotImplementedError("[ERROR] Only 2017 campaign is fully implemented, since they are stored locally")

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # create a config by passing the campaign, so id and name will be identical
    cfg = analysis_dijet.add_config(campaign, name=config_name, id=config_id)

    # use custom get_dataset_lfns function
    cfg.x.get_dataset_lfns = get_dataset_lfns

    # add processes we are interested in
    cfg.add_process(procs.n.data)
    cfg.add_process(procs.n.qcd)

    # set color of some processes
    # stylize_processes(cfg)

    # add datasets we need to study
    dataset_names = [
        # DATA
        "data_jetht_b",
        "data_jetht_c",
        "data_jetht_d",
        "data_jetht_e",
        "data_jetht_f",
        # QCD
        "qcd_ht50to100_madgraph",
        "qcd_ht100to200_madgraph",
        "qcd_ht200to300_madgraph",
        "qcd_ht300to500_madgraph",
        "qcd_ht500to700_madgraph",
        "qcd_ht700to1000_madgraph",
        "qcd_ht1000to1500_madgraph",
        "qcd_ht1500to2000_madgraph",
        "qcd_ht2000toinf_madgraph",
    ]
    for dataset_name in dataset_names:
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

        if limit_dataset_files:
            # apply optional limit on the max. number of files per dataset
            for info in dataset.info.values():
                if info.n_files > limit_dataset_files:
                    info.n_files = limit_dataset_files

        # add aux info to datasets
        if dataset.name.startswith("qcd"):
            dataset.x.is_qcd = True

            if (m := re.match(r"qcd_ht(\d+)to(\d+|inf)_madgraph", dataset.name)):
                g = m.groups()
                ht_min = int(g[0])
                ht_max = math.inf if g[1] == "inf" else int(g[1])
                dataset.x.ht_range = (ht_min, ht_max)

        # mark datasets with missing dijet trigger info
        if dataset.name in ("data_jetht_b", "data_jetht_c"):
            dataset.add_tag("missing_dijet_triggers")

    # default calibrator, selector, producer, ml model and inference model
    cfg.x.default_calibrator = "skip_jecunc"
    cfg.x.default_selector = "default"
    cfg.x.default_producer = "default"
    cfg.x.default_reducer = "cf_default"
    cfg.x.default_hist_producer = "all_weights"
    cfg.x.default_postprocessor = "dijet_balance"
    cfg.x.default_inference_model = "default"
    cfg.x.default_categories = ["incl"]
    cfg.x.default_variables = ["jet1_pt"]

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {
        "all": ["*"],
        "data": ["data_*"],
        "mc": ["qcd_*"],
    }

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    cfg.x.dataset_groups = {
        "all": ["*"],
        "data": ["data_*"],
        "mc": ["qcd_*"],
    }

    # sample definition (dijet-analysis specific)
    # named groups of datasets used as a unit for JER SF derivation
    cfg.x.samples = {
        "data": {
            "datasets": "data_*",
            "label": "Data",
            "plot_kwargs": {
                "__default__": {
                    "method": "errorbar",
                    "fmt": "s",
                    "marker": "s",
                    "fillstyle": "full",
                    "color": "k",
                    "label": "Data",
                },
                PlotAsymmetry: {
                    "method": "errorbar",
                    "fmt": "o",
                    "marker": "o",
                    "fillstyle": "full",
                    "color": "k",
                    "label": "Data",
                },
            },
        },
        "qcdht": {
            "datasets": "qcd_ht*",
            "label": "MC",
            "plot_kwargs": {
                "__default__": {
                    "method": "errorbar",
                    "fmt": "o",
                    "marker": "o",
                    "fillstyle": "none",
                    "color": "indianred",
                    "label": "MC",
                },
                PlotAsymmetry: {
                    "method": "bar",
                    "alpha": 0.6,
                    "color": "indianred",
                    "edgecolor": "none",
                    "label": "MC",
                },
            },
        },
        "qcdht_100to200": {
            "datasets": "qcd_ht100to200_madgraph",
            "label": "MC (100-200)",
            "plot_kwargs": {
                "__default__": {
                    "method": "errorbar",
                    "fmt": "o",
                    "marker": "o",
                    "fillstyle": "none",
                    "color": "indianred",
                    "label": "MC (100-200)",
                },
                PlotAsymmetry: {
                    "method": "bar",
                    "alpha": 0.6,
                    "color": "indianred",
                    "edgecolor": "none",
                    "label": "MC (100-200)",
                },
            },
        },
    }

    # category groups for conveniently looping over certain categories
    # (used during plotting)
    cfg.x.category_groups = {
        "default": ["incl"],
        "sm": ["sm"],
        "fe": ["fe"],
    }

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    cfg.x.variable_groups = {
        "default": ["n_jet", "jet1_pt"],
    }

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    cfg.x.shift_groups = {
        "jer": ["nominal", "jer_up", "jer_down"],
    }

    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    cfg.x.selector_step_groups = {
        "default": ["dijet"],
    }

    cfg.x.selector_step_labels = {
        "json": r"JSON",
        "trigger": r"Trigger",
        "met_filter": r"MET filters",
    }

    # plotting settings groups
    cfg.x.general_settings_groups = {
        "default_norm": {"shape_norm": True, "yscale": "log"},
    }
    cfg.x.process_settings_groups = {
        "Jet": r"$N_{jets}^{AK4} \geq 3$",
    }
    # when drawing DY as a line, use a different type of yellow

    cfg.x.variable_settings_groups = {

    }

    # lumi values in inverse pb
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=2#Combination_and_correlations
    if year == 2016:
        cfg.x.luminosity = Number(36310, {
            "lumi_13TeV_2016": 0.01j,
            "lumi_13TeV_correlated": 0.006j,
        })
    elif year == 2017:
        cfg.x.luminosity = Number(41480, {
            "lumi_13TeV_2017": 0.02j,
            "lumi_13TeV_1718": 0.006j,
            "lumi_13TeV_correlated": 0.009j,
        })
    else:  # 2018
        cfg.x.luminosity = Number(59830, {
            "lumi_13TeV_2017": 0.015j,
            "lumi_13TeV_1718": 0.002j,
            "lumi_13TeV_correlated": 0.02j,
        })

    # MET filters
    # TODO: Different Met filters for different years
    # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2?rev=158#2018_2017_data_and_MC_UL
    cfg.x.met_filters = {
        "Flag.goodVertices",
        "Flag.globalSuperTightHalo2016Filter",
        "Flag.HBHENoiseFilter",
        "Flag.HBHENoiseIsoFilter",
        "Flag.EcalDeadCellTriggerPrimitiveFilter",
        "Flag.BadPFMuonFilter",
        "Flag.BadPFMuonDzFilter",
        "Flag.eeBadScFilter",
        "Flag.ecalBadCalibFilter",
    }

    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    cfg.x.validate_dataset_lfns = limit_dataset_files is None

    # jec configuration
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=201
    jerc_postfix = "APV" if year == 2016 and campaign.x.vfp == "post" else ""
    cfg.x.jec = DotDict.wrap({
        "Jet": {
            "campaign": f"Summer19UL{year2}{jerc_postfix}",
            "version": {2016: "V7", 2017: "V5", 2018: "V5"}[year],
            "jet_type": "AK4PFchs",
            "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
            "levels_for_type1_met": ["L1FastJet"],
            "data_per_era": True,
            "uncertainty_sources": [
                "Total",
            ],
        },
    })

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    cfg.x.jer = DotDict.wrap({
        "Jet": {
            "campaign": f"Summer19UL{year2}{jerc_postfix}",
            "version": "JR" + {2016: "V3", 2017: "V2", 2018: "V2"}[year],
            "jet_type": "AK4PFchs",
        },
    })

    # helper to add column aliases for both shifts of a source
    def add_aliases(shift_source: str, aliases: Set[str], selection_dependent: bool):

        for direction in ["up", "down"]:
            shift = cfg.get_shift(od.Shift.join_name(shift_source, direction))
            # format keys and values
            inject_shift = lambda s: re.sub(r"\{([^_])", r"{_\1", s).format(**shift.__dict__)
            _aliases = {inject_shift(key): inject_shift(value) for key, value in aliases.items()}
            alias_type = "column_aliases_selection_dependent" if selection_dependent else "column_aliases"
            # extend existing or register new column aliases
            shift.set_aux(alias_type, shift.get_aux(alias_type, {})).update(_aliases)

    # register shifts
    cfg.add_shift(name="nominal", id=0)
    cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="minbias_xs_up", id=7, type="shape")
    cfg.add_shift(name="minbias_xs_down", id=8, type="shape")
    add_aliases("minbias_xs", {"pu_weight": "pu_weight_{name}"}, selection_dependent=False)

    with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
        all_jec_sources = yaml.load(f, yaml.Loader)["names"]

    for jec_source in cfg.x.jec["Jet"]["uncertainty_sources"]:
        idx = all_jec_sources.index(jec_source)
        cfg.add_shift(name=f"jec_{jec_source}_up", id=5000 + 2 * idx, type="shape")
        cfg.add_shift(name=f"jec_{jec_source}_down", id=5001 + 2 * idx, type="shape")
        add_aliases(
            f"jec_{jec_source}",
            {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"},
            selection_dependent=True,
        )

    def make_jme_filename(jme_aux, sample_type, name, era=None):
        """
        Convenience function to compute paths to JEC files.
        """
        # normalize and validate sample type
        sample_type = sample_type.upper()
        if sample_type not in ("DATA", "MC"):
            raise ValueError(f"invalid sample type '{sample_type}', expected either 'DATA' or 'MC'")

        jme_full_version = "_".join(s for s in (jme_aux.campaign, era, jme_aux.version, sample_type) if s)

        return f"{jme_aux.source}/{jme_full_version}/{jme_full_version}_{name}_{jme_aux.jet_type}.txt"

    # external files
    json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-c3be7e71"
    cfg.x.external_files = DotDict.wrap({
        # jet energy correction
        "jet_jerc": (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/jet_jerc.json.gz", "v1"),

        # jet veto map
        "jet_veto_map": (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/jetvetomaps.json.gz", "v1"),

        # pileup weights from correctionlib
        "pu_sf": (f"{json_mirror}/POG/LUM/{year}{corr_postfix}_UL/puWeights.json.gz", "v1"),

        # electron scale factors
        "electron_sf": (f"{json_mirror}/POG/EGM/{year}{corr_postfix}_UL/electron.json.gz", "v1"),

        # muon scale factors
        "muon_sf": (f"{json_mirror}/POG/MUO/{year}{corr_postfix}_UL/muon_Z.json.gz", "v1"),

        # btag scale factor
        "btag_sf_corr": (f"{json_mirror}/POG/BTV/{year}{corr_postfix}_UL/btagging.json.gz", "v1"),

        # met phi corrector
        "met_phi_corr": (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/met.json.gz", "v1"),
    })

    # external files with more complex year dependence
    # TODO: generalize to different years
    if year != 2017:  # TODO wrong lumis
        raise NotImplementedError("TODO: generalize external files to different years than 2017")

    cfg.x.external_files.update(DotDict.wrap({
        # files from TODO
        "lumi": {
            "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
            "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
        },

        # files from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II # noqa
        "pu": {
            "json": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/pileup_latest.txt", "v1"),  # noqa
            "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/435f0b04c0e318c1036a6b95eb169181bbbe8344/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
            "data_profile": {
                "nominal": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root", "v1"),  # noqa
                "minbias_xs_up": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root", "v1"),  # noqa
                "minbias_xs_down": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root", "v1"),  # noqa
            },
        },
    }))

    # columns to keep after certain steps
    cfg.x.keep_columns = DotDict.wrap({
        "cf.SelectEvents": {"mc_weight"},
        "cf.MergeSelectionMasks": {
            "mc_weight", "normalization_weight", "process_id", "category_ids",
        },
    })

    cfg.x.keep_columns["cf.ReduceEvents"] = (
        {
            # general event information
            "run", "luminosityBlock", "event",
            # average number of pileup interactions
            "Pileup.nTrueInt",
            # columns added during selection, required in general
            "mc_weight", "PV.npvs", "process_id", "category_ids", "deterministic_seed",
            # weight-related columns
            "pu_weight*",
            # produced by 'jet_assignment' producer
            "use_fe", "use_sm",
            # produced by 'alpha' producer
            "alpha",
        } | set(  # Jets
            f"{jet_obj}.{field}"
            for jet_obj in ["Jet", "probe_jet", "reference_jet"]
            # NOTE: if we run into storage troubles, skip Bjet and Lightjet
            for field in ["pt", "eta", "phi", "mass", "genJetIdx"]
        ) | set(  # MET
            f"MET.{field}"
            for field in ["pt", "phi"]
        ) | set(  # MET
            f"GenMET.{field}"
            for field in ["pt", "phi"]
        ) | set(  # GenJets
            f"{gen_jet_obj}.{field}"
            for gen_jet_obj in ["GenJet"]
            for field in ["pt", "eta", "phi", "mass"]
        )
    )

    
    # specify which weights to apply (including variations if applicable)
    # The expected structure of the *event_weights* aux entry is a dictionary
    # with the weight column name as key and a list of shift sources as values.
    # The shift sources are used to declare the shifts that the produced event
    # weight depends on.
    from columnflow.config_util import get_shifts_from_sources
    cfg.x.event_weights = DotDict()
    cfg.x.event_weights["normalization_weight"] = []
    cfg.x.event_weights["pu_weight"] = get_shifts_from_sources(cfg, "minbias_xs")

    
    # Trigger selection
    # TODO: SingleJet triggers for AK8 and some special cases in UL16 & UL17
    cfg.x.triggers = DotDict.wrap({
        "dijet": {
            "central": [
                "DiPFJetAve40",
                "DiPFJetAve60",
                "DiPFJetAve80",
                "DiPFJetAve140",
                "DiPFJetAve200",
                "DiPFJetAve260",
                "DiPFJetAve320",
                "DiPFJetAve400",
                "DiPFJetAve500",
            ],
            "forward": [
                "DiPFJetAve60_HFJEC",
                "DiPFJetAve80_HFJEC",
                "DiPFJetAve100_HFJEC",
                "DiPFJetAve160_HFJEC",
                "DiPFJetAve220_HFJEC",
                "DiPFJetAve300_HFJEC",
            ],
        },
        # TODO: single jet only for AK4 so far
        #       Needed for AK8
        "singlejet": {
            "central": [
                "PFJet40",
                "PFJet60",
                "PFJet80",
                "PFJet140",
                "PFJet200",
                "PFJet260",
                "PFJet320",
                "PFJet400",
                "PFJet500",
            ],
        },
    })

    cfg.x.trigger_thresholds = DotDict.wrap({
        "dijet": {
            "central": (
                [59, 85, 104, 170, 236, 302, 370, 460, 575]
                if campaign.x.year == 2016
                else
                [70, 87, 111, 180, 247, 310, 373, 457, 562]
                if campaign.x.year == 2017
                else
                [66, 93, 118, 189, 257, 325, 391, 478, 585]
                if campaign.x.year == 2018
                else None
            ),
            "forward": (
                [86, 110, 132, 204, 279, 373]
                if campaign.x.year == 2016
                else
                [73, 93, 113, 176, 239, 318]
                if campaign.x.year == 2017
                else
                [93, 116, 142, 210, 279, 379]
                if campaign.x.year == 2018
                else None
            ),
        },
        "singlejet": {
            "central": (
                [70, 87, 111, 180, 247, 310, 373, 457, 562]
                if campaign.x.year == 2017
                else None
            ),
        },
    })

    # Version of required tasks
    cfg.x.versions = {
        "task_cf.CalibrateEvents": "v0",
    }

    # add categories
    add_categories(cfg)

    # add variables
    add_variables(cfg)
    add_cutflow_variables(cfg)
    add_uhh2_synch_variables(cfg)

    # only produce cutflow features when number of dataset_files is limited (used in selection module)
    cfg.x.do_cutflow_features = bool(limit_dataset_files) and limit_dataset_files <= 10

    return cfg
