# coding: utf-8

"""
Custom base tasks.
"""

import luigi
import law

from columnflow.tasks.framework.base import BaseTask, AnalysisTask, DatasetTask, wrapper_factory
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorMixin, ProducersMixin,
    VariablesMixin, DatasetsProcessesMixin, CategoriesMixin,
)
from columnflow.config_util import get_datasets_from_process
from columnflow.util import dev_sandbox, DotDict

from dijet.tasks.base import DiJetTask


class ExternalFileTask(
    law.ExternalTask,
):
    fname = law.Parameter("filename")

    def output(self):
        return law.LocalFileTarget(self.fname)


class UHH2ToParquet(
    DiJetTask,
    DatasetTask,
    law.tasks.TransferLocalFile,
):
    """
    Base task to produce *.parquet versions of UHH2 ntuples.
    """
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # add nested sibling directories to output path
    output_collection_cls = law.NestedSiblingFileCollection

    #
    # parameters
    #
    version = None  # deactivate
    shift = None  # deactivate

    # other class variables

    uhh2_ntuples_base_dir = "/nfs/dust/cms/user/paaschal/sframe_all/DiJetJERC_DiJetHLT"
    uhh2_ntuples_dir = f"{uhh2_ntuples_base_dir}/UL17/eta_common/Summer20UL17_V2/AK4CHS"

    # note: UL17-specific
    uhh2_dataset_file_map = {
        "data_jetht_b":              f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.DATA.DATA_RunB_UL17.root",
        "data_jetht_c":              f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.DATA.DATA_RunC_UL17.root",
        "data_jetht_d":              f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.DATA.DATA_RunD_UL17.root",
        "data_jetht_e":              f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.DATA.DATA_RunE_UL17.root",
        "data_jetht_f":              f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.DATA.DATA_RunF_UL17.root",
        "qcd_ht50to100_madgraph":    f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.MC.QCDHT50to100_UL17.root",
        "qcd_ht100to200_madgraph":   f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.MC.QCDHT100to200_UL17.root",
        "qcd_ht200to300_madgraph":   f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.MC.QCDHT200to300_UL17.root",
        "qcd_ht300to500_madgraph":   f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.MC.QCDHT300to500_UL17.root",
        "qcd_ht500to700_madgraph":   f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.MC.QCDHT500to700_UL17.root",
        "qcd_ht700to1000_madgraph":  f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.MC.QCDHT700to1000_UL17.root",
        "qcd_ht1000to1500_madgraph": f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.MC.QCDHT1000to1500_UL17.root",
        "qcd_ht1500to2000_madgraph": f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.MC.QCDHT1500to2000_UL17.root",
        "qcd_ht2000toinf_madgraph":  f"{uhh2_ntuples_dir}/uhh2.AnalysisModuleRunner.MC.QCDHT2000toInf_UL17.root",
    }

    uhh2_branches = [
        #"B",
        #"MET",
        #"Nelectron",
        "Ngenjet",
        "Njet",
        #"Nmuon",
        #"Nptcl",
        "alpha",
        "asymmetry",
        "barrelgenjet_eta",
        "barrelgenjet_phi",
        "barrelgenjet_pt",
        #"barreljet_chEmEF",
        #"barreljet_chHadEF",
        #"barreljet_dRminParton",
        "barreljet_eta",
        #"barreljet_muonEF",
        #"barreljet_neutEmEF",
        #"barreljet_neutHadEF",
        "barreljet_phi",
        #"barreljet_photonEF",
        "barreljet_pt",
        "barreljet_ptRaw",
        #"barreljet_ptptcl",
        #"barreljet_status_ptcl",
        #"dMET",
        #"dR_GenJet_GenParticle_barrel_matched",
        #"dR_jet3_barreljet",
        #"dR_jet3_probejet",
        #"electron_pt",
        "eventID",
        #"flavor3rdjet",
        #"flavorBarreljet",
        #"flavorLeadingjet",
        #"flavorProbejet",
        #"flavorSubleadingjet",
        #"genB",
        #"genMET",
        #"gen_PUpthat",
        #"gen_alpha",
        "gen_asymmetry",
        "gen_pt_ave",
        "gen_pthat",
        "gen_weight",
        "genjet1_eta",
        "genjet1_phi",
        "genjet1_pt",
        "genjet2_eta",
        "genjet2_phi",
        "genjet2_pt",
        "genjet3_eta",
        "genjet3_phi",
        "genjet3_pt",
        "genjet4_pt",
        #"genjet5_pt",
        #"genjet6_pt",
        #"genjet7_pt",
        #"instantaneous_lumi",
        #"integrated_lumi",
        #"integrated_lumi_in_bin",
        "is_JER_SM",
        #"jet1_genID",
        "jet1_pt",
        "jet1_ptRaw",
        #"jet1_pt_onoff_Resp",
        #"jet2_genID",
        "jet2_pt",
        "jet2_ptRaw",
        #"jet2_pt_onoff_Resp",
        "jet3_eta",
        #"jet3_genID",
        "jet3_phi",
        "jet3_pt",
        "jet3_ptRaw",
        #"jet3_ptptcl",
        #"jet3_status_ptcl",
        #"jet4_genID",
        "jet4_pt",
        #"jet5_genID",
        #"jet5_pt",
        #"jet6_genID",
        #"jet6_pt",
        #"jet7_pt",
        "leadingjet_response",
        "lumi_sec",
        #"lumibin",
        "matchJetId_0",
        "matchJetId_1",
        #"mpf_r",
        #"muon_pt",
        #"nPU",
        #"no_mc_spikes",
        #"nvertices",
        #"partonFlavor",
        #"prefire",
        #"prefire_down",
        #"prefire_up",
        #"prescale",
        #"prescale_L1max",
        #"prescale_L1min",
        "probegenjet_eta",
        "probegenjet_phi",
        "probegenjet_pt",
        #"probejet_chEmEF",
        #"probejet_chHadEF",
        #"probejet_dRminParton",
        "probejet_eta",
        #"probejet_muonEF",
        #"probejet_neutEmEF",
        #"probejet_neutHadEF",
        "probejet_phi",
        #"probejet_photonEF",
        "probejet_pt",
        "probejet_ptRaw",
        #"probejet_ptptcl",
        #"probejet_status_ptcl",
        "pt_ave",
        #"pv0X",
        #"pv0Y",
        #"pv0Z",
        "rel_r",
        "response3rdjet",
        "responseBarreljet_genp",
        "responseProbejet_genp",
        "rho",
        "run",
        "subleadingjet_response",
        #"sum_jets_pt",
        "trigger100_HFJEC",
        "trigger140",
        "trigger140_HFJEC",
        "trigger160_HFJEC",
        "trigger200",
        "trigger200_HFJEC",
        "trigger220_HFJEC",
        "trigger260",
        "trigger260_HFJEC",
        "trigger300_HFJEC",
        "trigger320",
        "trigger320_HFJEC",
        "trigger40",
        "trigger400",
        "trigger400_HFJEC",
        "trigger40_HFJEC",
        "trigger450",
        "trigger500",
        "trigger60",
        "trigger60_HFJEC",
        "trigger80",
        "trigger80_HFJEC",
        "weight",
        #"weight_fsr_2_down",
        #"weight_fsr_2_up",
        #"weight_fsr_4_down",
        #"weight_fsr_4_up",
        #"weight_fsr_sqrt2_down",
        #"weight_fsr_sqrt2_up",
        #"weight_isr_2_down",
        #"weight_isr_2_up",
        #"weight_isr_4_down",
        #"weight_isr_4_up",
        #"weight_isr_sqrt2_down",
        #"weight_isr_sqrt2_up",
        #"weight_isrfsr_2_down",
        #"weight_isrfsr_2_up",
        #"weight_isrfsr_4_down",
        #"weight_isrfsr_4_up",
        #"weight_isrfsr_sqrt2_down",
        #"weight_isrfsr_sqrt2_up",
    ]

    def requires(self) -> dict:
        """
        Requires UHH2 preselection output ROOT files.
        """
        fname = self.uhh2_dataset_file_map[self.dataset]
        return {
            "uhh2": ExternalFileTask(fname=fname),
        }

    def single_output(self):
        """
        Produces one parquet file per dataset.
        """
        return self.target("uhh2.parquet")

    def run(self):
        import awkward as ak
        import uproot

        fname = self.input()["uhh2"].path

        print(f"Loading UHH2 ROOT file: {fname}")
        f = uproot.open(fname)

        print(f"Selecting branches ({len(self.uhh2_branches)})")
        t = f["AnalysisTree"]
        events_uhh2 = t.arrays(self.uhh2_branches)

        print("Writing output file")
        tmp = law.LocalFileTarget(is_tmp=True)
        ak.to_parquet(events_uhh2, tmp.path)
        self.transfer(tmp)


UHH2ToParquetWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=UHH2ToParquet,
    enable=["configs", "skip_configs", "datasets", "skip_datasets"],
    attributes={"version": None, "task_namespace": UHH2ToParquet.task_namespace},
    docs="""
Wrapper task to get LFNs for multiple datasets.

:enables: ["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"]
:overwrites: attribute ``version`` with None
""",
)
