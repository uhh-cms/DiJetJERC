# coding: utf-8

"""
Configuration of the Run 3 DiJet analysis.
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
from columnflow.cms_util import CATInfo, CATSnapshot
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
    assert campaign.x.year in [2022, 2023, 2024, 2025]

    # gather campaign data
    year = campaign.x.year
    year2 = year % 100
    corr_postfix = ""

    if year != 2024:
        raise NotImplementedError("[ERROR] Only 2024 campaign is fully implemented.")

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
        "data_jetmet_b",
        "data_jetmet_c",
        "data_jetmet_d",
        "data_jetmet_e",
        "data_jetmet_f",
        "data_jetmet_g",
        "data_jetmet_h",
        "data_jetmet_i",
        # QCD
        "qcd_ht100to200_madgraph",
        "qcd_ht200to400_madgraph",
        "qcd_ht400to600_madgraph",
        "qcd_ht600to800_madgraph",
        "qcd_ht800to1000_madgraph",
        "qcd_ht1000to1200_madgraph",
        "qcd_ht1200to1500_madgraph",
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

        # # mark datasets with missing dijet trigger info
        # if dataset.name in ("data_jetht_b", "data_jetht_c"):
        #     dataset.add_tag("missing_dijet_triggers")

    # default calibrator, selector, producer, ml model and inference model
    cfg.x.default_calibrator = "skip_jecunc"
    cfg.x.default_selector = "default"
    cfg.x.default_producer = "default"
    cfg.x.default_reducer = "cf_default"
    cfg.x.default_hist_producer = "all_weights"
    # cfg.x.default_ml_model = "default"
    # cfg.x.default_ml_model = None
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
    # cfg.x.process_groups["dmuch"] = ["data_mu"] + cfg.x.process_groups["much"]
    # cfg.x.process_groups["dech"] = ["data_e"] + cfg.x.process_groups["ech"]

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
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis
    if year == 2022 and campaign.x.EE == "pre":
        cfg.x.luminosity = Number(7_980.4541, {
            "lumi_13p6TeV_2022": 0.014j,
        })
    elif year == 2022 and campaign.x.EE == "post":
        cfg.x.luminosity = Number(26_671.6097, {
            "lumi_13p6TeV_2022": 0.014j,
        })
    elif year == 2023 and campaign.x.BPix == "pre":
        cfg.x.luminosity = Number(18_062.6591, {
            "lumi_13p6TeV_2023": 0.013j,
        })
    elif year == 2023 and campaign.x.BPix == "post":
        cfg.x.luminosity = Number(9_693.1301, {
            "lumi_13p6TeV_2023": 0.013j,
        })
    elif year == 2024:
        cfg.x.luminosity = Number(109_080.0, {  # TODO: update number
            "lumi_13p6TeV_2024": 0.013j,
        })
        # cfg.x.luminosity = Number(995.223558512, {
        #     "lumi_13p6TeV_2024": 0.013j,
        # })
    else:
        raise NotImplementedError(f"Luminosity for year {year} is not defined.")

    # MET filters
    # TODO: Different MET filters for different years?
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#Run_3_recommendations
    cfg.x.met_filters = {
        "Flag.goodVertices",
        "Flag.globalSuperTightHalo2016Filter",
        "Flag.EcalDeadCellTriggerPrimitiveFilter",
        "Flag.BadPFMuonFilter",
        "Flag.BadPFMuonDzFilter",
        "Flag.eeBadScFilter",
        "Flag.ecalBadCalibFilter",
    }

    # minimum bias cross section in mb (milli) for creating PU weights, values from
    # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData?rev=45#Recommended_cross_section
    cfg.x.minbias_xs = Number(69.2, 0.046j)

    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    cfg.x.validate_dataset_lfns = limit_dataset_files is None

    # jec configuration
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=201
    jerc_postfix = ""
    cfg.x.jec = DotDict.wrap({
        "Jet": {
            "campaign": f"Summer24Prompt{year2}{jerc_postfix}",
            "version": {2024: "V1"}[year],
            "jet_type": "AK4PFPuppi",
            "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
            "levels_for_type1_met": ["L1FastJet"],
            "data_per_era": False,
            "uncertainty_sources": [
                "Total",
            ],
        },
    })

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    cfg.x.jer = DotDict.wrap({
        "Jet": {
            "campaign": "Summer23BPixPrompt23_RunD",
            "version": "JR" + {2024: "V1"}[year],
            "jet_type": "AK4PFPuppi",
        },
    })

    # JEC uncertainty sources propagated to btag scale factors
    # (names derived from contents in BTV correctionlib file)
    cfg.x.btag_sf_jec_sources = [
        "",  # total
        "Absolute",
        "AbsoluteMPFBias",
        "AbsoluteScale",
        "AbsoluteStat",
        f"Absolute_{year}",
        "BBEC1",
        f"BBEC1_{year}",
        "EC2",
        f"EC2_{year}",
        "FlavorQCD",
        "Fragmentation",
        "HF",
        f"HF_{year}",
        "PileUpDataMC",
        "PileUpPtBB",
        "PileUpPtEC1",
        "PileUpPtEC2",
        "PileUpPtHF",
        "PileUpPtRef",
        "RelativeBal",
        "RelativeFSR",
        "RelativeJEREC1",
        "RelativeJEREC2",
        "RelativeJERHF",
        "RelativePtBB",
        "RelativePtEC1",
        "RelativePtEC2",
        "RelativePtHF",
        "RelativeSample",
        f"RelativeSample_{year}",
        "RelativeStatEC",
        "RelativeStatFSR",
        "RelativeStatHF",
        "SinglePionECAL",
        "SinglePionHCAL",
        "TimePtEta",
    ]

    # b-tag working points
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16preVFP?rev=6
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16postVFP?rev=8
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=17
    btag_key = "2016post"  # TODO: adapt to 2024
    cfg.x.btag_working_points = DotDict.wrap({
        "deepjet": {
            "loose": {"2016pre": 0.0508, "2016post": 0.0480, 2017: 0.0532, 2018: 0.0490}[btag_key],
            "medium": {"2016pre": 0.2598, "2016post": 0.2489, 2017: 0.3040, 2018: 0.2783}[btag_key],
            "tight": {"2016pre": 0.6502, "2016post": 0.6377, 2017: 0.7476, 2018: 0.7100}[btag_key],
        },
        "deepcsv": {
            "loose": {"2016pre": 0.2027, "2016post": 0.1918, 2017: 0.1355, 2018: 0.1208}[btag_key],
            "medium": {"2016pre": 0.6001, "2016post": 0.5847, 2017: 0.4506, 2018: 0.4168}[btag_key],
            "tight": {"2016pre": 0.8819, "2016post": 0.8767, 2017: 0.7738, 2018: 0.7665}[btag_key],
        },
    })

    # MET to use
    cfg.x.met_name = "PuppiMET"
    cfg.x.raw_met_name = "RawPuppiMET"

    # TODO: check e/mu/btag corrections and implement
    # btag weight configuration
    cfg.x.btag_sf = ("deepJet_shape", cfg.x.btag_sf_jec_sources)

    # names of electron correction sets and working points
    # (used in the electron_sf producer)
    cfg.x.electron_id = "mvaNoIso_WP80"
    cfg.x.electron_sf_names = ("UL-Electron-ID-SF", f"{year}{corr_postfix}", "wp80iso")

    # names of muon correction sets and working points
    # (used in the muon producer)
    cfg.x.muon_id = "tightId"
    cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}{corr_postfix}_UL")

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
    # TODO: make shifts year-dependent
    cfg.add_shift(name="nominal", id=0)
    cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="minbias_xs_up", id=7, type="shape")
    cfg.add_shift(name="minbias_xs_down", id=8, type="shape")
    add_aliases("minbias_xs", {"pu_weight": "pu_weight_{name}"}, selection_dependent=False)
    cfg.add_shift(name="top_pt_up", id=9, type="shape")
    cfg.add_shift(name="top_pt_down", id=10, type="shape")
    add_aliases("top_pt", {"top_pt_weight": "top_pt_weight_{direction}"}, selection_dependent=False)

    cfg.add_shift(name="e_sf_up", id=40, type="shape")
    cfg.add_shift(name="e_sf_down", id=41, type="shape")
    cfg.add_shift(name="e_trig_sf_up", id=42, type="shape")
    cfg.add_shift(name="e_trig_sf_down", id=43, type="shape")
    add_aliases("e_sf", {"electron_weight": "electron_weight_{direction}"}, selection_dependent=False)

    cfg.add_shift(name="mu_sf_up", id=50, type="shape")
    cfg.add_shift(name="mu_sf_down", id=51, type="shape")
    cfg.add_shift(name="mu_trig_sf_up", id=52, type="shape")
    cfg.add_shift(name="mu_trig_sf_down", id=53, type="shape")
    add_aliases("mu_sf", {"muon_weight": "muon_weight_{direction}"}, selection_dependent=False)

    btag_uncs = [
        "hf", "lf", f"hfstats1_{year}", f"hfstats2_{year}",
        f"lfstats1_{year}", f"lfstats2_{year}", "cferr1", "cferr2",
    ]
    for i, unc in enumerate(btag_uncs):
        cfg.add_shift(name=f"btag_{unc}_up", id=100 + 2 * i, type="shape")
        cfg.add_shift(name=f"btag_{unc}_down", id=101 + 2 * i, type="shape")
        # add_aliases(
        #     f"btag_{unc}",
        #     {
        #         "normalized_btag_weight": f"normalized_btag_weight_{unc}_" + "{direction}",
        #         "normalized_njet_btag_weight": f"normalized_njet_btag_weight_{unc}_" + "{direction}",
        #     },
        #     selection_dependent=False,
        # )

    cfg.add_shift(name="mur_up", id=201, type="shape")
    cfg.add_shift(name="mur_down", id=202, type="shape")
    cfg.add_shift(name="muf_up", id=203, type="shape")
    cfg.add_shift(name="muf_down", id=204, type="shape")
    cfg.add_shift(name="murf_envelope_up", id=205, type="shape")
    cfg.add_shift(name="murf_envelope_down", id=206, type="shape")
    cfg.add_shift(name="pdf_up", id=207, type="shape")
    cfg.add_shift(name="pdf_down", id=208, type="shape")

    for unc in ["mur", "muf", "murf_envelope", "pdf"]:
        # add_aliases(unc, {f"{unc}_weight": f"{unc}_weight_" + "{direction}"}, selection_dependent=False)
        add_aliases(
            unc,
            {f"normalized_{unc}_weight": f"normalized_{unc}_weight_" + "{direction}"},
            selection_dependent=False,
        )

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

    cfg.add_shift(name="jer_up", id=6000, type="shape", tags={"selection_dependent"})
    cfg.add_shift(name="jer_down", id=6001, type="shape", tags={"selection_dependent"})
    add_aliases("jer", {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"}, selection_dependent=True)

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

    #
    # external files
    #

    cfg.x.external_files = DotDict()

    # helper
    def add_external(name, value):
        if isinstance(value, dict):
            value = DotDict.wrap(value)
        cfg.x.external_files[name] = value

    # external files with more complex year dependence
    # TODO: generalize to different years
    if year != 2024:  # TODO wrong lumis
        raise NotImplementedError("TODO: generalize external files to different years than 2024")

    # prepare run/era/nano meta data info to determine files in the CAT metadata structure
    # see https://cms-analysis-corrections.docs.cern.ch
    cat_info = {
        (2022, "", 12): CATInfo(
            run=3,
            vnano=12,
            era="22CDSep23-Summer22",
            pog_directories={"dc": "Collisions22"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-04-15", jme="2025-09-23", lum="2024-01-31", muo="2025-08-14", tau="2025-10-01"),  # noqa: E501
        ),
        (2022, "EE", 12): CATInfo(
            run=3,
            vnano=12,
            era="22EFGSep23-Summer22EE",
            pog_directories={"dc": "Collisions22"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-04-15", jme="2025-10-07", lum="2024-01-31", muo="2025-08-14", tau="2025-10-01"),  # noqa: E501
        ),
        (2023, "", 12): CATInfo(
            run=3,
            vnano=12,
            era="23CSep23-Summer23",
            # pog_eras={"tau": "23CSep23-Summer22"},  # TODO: remove once typo in CAT repo is fixed
            pog_directories={"dc": "Collisions23"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-04-15", jme="2025-10-07", lum="2024-01-31", muo="2025-08-14", tau="2025-10-01"),  # noqa: E501
        ),
        (2023, "BPix", 12): CATInfo(
            run=3,
            vnano=12,
            era="23DSep23-Summer23BPix",
            pog_directories={"dc": "Collisions23"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-04-15", jme="2025-10-07", lum="2024-01-31", muo="2025-08-14", tau="2025-10-01"),  # noqa: E501
        ),
        (2024, "", 15): CATInfo(
            run=3,
            vnano=15,
            era="24CDEReprocessingFGHIPrompt-Summer24",
            pog_directories={"dc": "Collisions24"},
            # TODO: tau and lum not yet available (11.11.25)
            snapshot=CATSnapshot(btv="2025-08-19", dc="2025-07-25", egm="2025-10-22", jme="2025-07-17", muo="2025-10-17"),  # noqa: E501
        ),
    }[(year, campaign.x.postfix, 15)]
    cfg.x.cat_info = cat_info

    # common files
    # (versions in the end are for hashing in cases where file contents changed but paths did not)
    add_external("lumi", {
        "golden": {
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2022
            2022: (cat_info.get_file("dc", "Cert_Collisions2022_355100_362760_Golden.json"), "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2023
            2023: (cat_info.get_file("dc", "Cert_Collisions2023_366442_370790_Golden.json"), "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=180#Year_2024
            # not yet available at CAT space
            # 2024: (cat_info.get_file("dc", "Cert_Collisions2024_378981_386951_Golden.json"), "v1"),
            2024: ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions24/Cert_Collisions2024_378981_386951_Golden.json", "v1"),  # noqa: E501
        }[year],
        "normtag": {
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2022
            2022: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2023
            2023: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=180#Year_2024
            2024: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
        }[year],
    })

    # pileup weight corrections
    if year != 2024:  # TODO: not yet available, see https://cms-analysis-corrections.docs.cern.ch
        add_external("pu_sf", (cat_info.get_file("lum", "puWeights.json.gz"), "v1"))
    elif year == 2024:
        # private preliminary file for 2024 for now
        # https://mattermost.web.cern.ch/cms-ppd/pl/c8j6m64dbinhuc3jxwo185pqac
        # add_external("pu_sf", f"{local_path}config/run3/puWeights_2024_mm.json.gz")
        add_external("pu_sf", ("https://ceballos.web.cern.ch/random/puWeights_2024.json.gz", "v1"))

    # jet energy correction
    add_external("jet_jerc", (cat_info.get_file("jme", "jet_jerc.json.gz"), "v1"))

    # fat jet energy correction
    # FIXME no FatJet corrections available in CAT for 2024 yet
    if year != 2024:
        add_external("fat_jet_jerc", (cat_info.get_file("jme", "fat_jet_jerc.json.gz"), "v1"))

    # jet veto map
    add_external("jet_veto_map", (cat_info.get_file("jme", "jetvetomaps.json.gz"), "v1"))

    # updated jet id
    add_external("jet_id", (cat_info.get_file("jme", "jetid.json.gz"), "v1"))

    # met phi correction
    if year != 2024:  # TODO: not yet available for 2024
        add_external("met_phi_corr", (cat_info.get_file("jme", f"met_xyCorrections_{year}_{year}{campaign.x.postfix}.json.gz"), "v1"))  # noqa: E501

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
            # columns added during selection, required in general
            "mc_weight", "PV.npvs", "process_id", "category_ids", "deterministic_seed",
            # weight-related columns
            "pu_weight*", "pdf_weight*",
            "murf_envelope_weight*", "mur_weight*", "muf_weight*",
            "btag_weight*",
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
            f"{met}.{field}"
            for field in ["pt", "phi"]
            for met in ["MET", "PFMET", "PuppiMET"]
        ) | set(  # GenMET
            f"GenMET.{field}"
            for field in ["pt", "phi"]
        ) | set(  # GenJets
            f"{gen_jet_obj}.{field}"
            for gen_jet_obj in ["GenJet"]
            for field in ["pt", "eta", "phi", "mass"]
        )
    )

    # event weight columns as keys in an ordered dict, mapped to shift instances they depend on
    get_shifts = lambda *keys: sum(([cfg.get_shift(f"{k}_up"), cfg.get_shift(f"{k}_down")] for k in keys), [])
    cfg.x.event_weights = DotDict()

    cfg.x.event_weights["normalization_weight"] = []

    # NOTE: which to use, njet_btag_weight or btag_weight?
    # cfg.x.event_weights["normalized_btag_weight"] = get_shifts(*(f"btag_{unc}" for unc in btag_uncs))
    # TODO: fix pu_weight; takes way too large values (from 0 to 160)
    # cfg.x.event_weights["normalized_pu_weight"] = get_shifts("minbias_xs")
    for dataset in cfg.datasets:
        dataset.x.event_weights = DotDict()
        if not dataset.x("is_qcd", False):
            # pdf/scale weights for all non-qcd datasets
            dataset.x.event_weights["normalized_murf_envelope_weight"] = get_shifts("murf_envelope")
            dataset.x.event_weights["normalized_mur_weight"] = get_shifts("mur")
            dataset.x.event_weights["normalized_muf_weight"] = get_shifts("muf")
            dataset.x.event_weights["normalized_pdf_weight"] = get_shifts("pdf")

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
                "DiPFJetAve260_HFJEC",
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
                "PFJet110",
                "PFJet140",
                "PFJet200",
                "PFJet260",
                "PFJet320",
                "PFJet400",
                "PFJet450",
                "PFJet500",
                "PFJet550",
            ],
        },
    })

    cfg.x.trigger_thresholds = DotDict.wrap({
        "dijet": {
            "central": (
                # TODO: mostly taken from 2018, remeasure for 2024
                # 40, 60,  80, 140, 200, 260, 320, 400, 500
                [66, 93, 118, 189, 257, 325, 391, 478, 585]
            ),
            "forward": (
                # TODO: mostly taken from 2018, remeasure for 2024
                # 60,  80, 100, 160, 220, 260, 300
                [93, 116, 142, 210, 279, 340, 379]
            ),
        },
        "singlejet": {
            "central": (
                # TODO: mostly taken from 2017, remeasure for 2024
                # 40, 60,  80, 110, 140, 200, 260, 320, 400, 450, 500, 550
                [70, 87, 111, 150, 180, 247, 310, 373, 457, 520, 562, 630]
            ),
        },
    })

    # dev_version = "v0"
    # prod_version = "prod1"

    # def reduce_version(cls, inst, params):
    #     version = dev_version
    #     if params.get("selector") == "default":
    #         version = prod_version

    #     return version

    # Version of required tasks
    cfg.x.versions = {
        "task_cf.CalibrateEvents": "v0",
        # "task_cf.SelectEvents": reduce_version,
        # "task_cf.MergeSelectionStats": reduce_version,
        # "task_cf.MergeSelectionMasks": reduce_version,
        # "task_cf.ReduceEvents": reduce_version,
        # "task_cf.MergeReductionStats": reduce_version,
        # "task_cf.MergeReducedEvents": reduce_version,
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
