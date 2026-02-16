# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.jets import jec  # , jer
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT

from dijet.calibration.jet import jec_nominal_run2, jec_nominal_run3

ak = maybe_import("awkward")
np = maybe_import("numpy")


@calibrator(
    uses={deterministic_seeds},
    produces={deterministic_seeds},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:

    run_calibrators = {
        2: jec_nominal_run2,
        3: jec_nominal_run3,
    }
    run = self.config_inst.campaign.x.run

    # check for unphysical values in RawPuppiMET.pt
    raw_puppi = events.RawPuppiMET
    raw_puppi = ak.with_field(
        raw_puppi,
        ak.where(
            raw_puppi.pt == np.float32("inf"),
            EMPTY_FLOAT,
            raw_puppi.pt,
        ),
        "pt",
    )
    events = ak.with_field(events, raw_puppi, "RawPuppiMET")

    events = self[deterministic_seeds](events, **kwargs)

    if self.dataset_inst.is_data:
        events = self[run_calibrators[run]](events, **kwargs)
    else:
        events = self[jec](events, **kwargs)
        # events = self[jer](events, **kwargs)

    return events


@default.init
def default_init(self: Calibrator) -> None:

    run_calibrators = {
        2: jec_nominal_run2,
        3: jec_nominal_run3,
    }
    run = self.config_inst.campaign.x.run

    if self.dataset_inst.is_data:
        calibrators = {run_calibrators[run]}
    else:
        calibrators = {jec}

    self.uses |= calibrators
    self.produces |= calibrators


@calibrator(
    uses={deterministic_seeds},
    produces={deterministic_seeds},
)
def skip_jecunc(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """ only uses jec_nominal for test purposes """

    run_calibrators = {
        2: jec_nominal_run2,
        3: jec_nominal_run3,
    }
    run = self.config_inst.campaign.x.run

    # check for unphysical values in RawPuppiMET.pt !JANKY implementation! FIXME
    raw_puppi = events.RawPuppiMET
    raw_puppi = ak.with_field(
        raw_puppi,
        ak.where(
            raw_puppi.pt == np.float32("inf"),
            EMPTY_FLOAT,
            raw_puppi.pt,
        ),
        "pt",
    )
    events = ak.with_field(events, raw_puppi, "RawPuppiMET")

    events = self[deterministic_seeds](events, **kwargs)

    if self.dataset_inst.is_data:
        events = self[run_calibrators[run]](events, **kwargs)
    else:
        events = self[run_calibrators[run]](events, **kwargs)
        # events = self[jer](events, **kwargs)

    return events


@skip_jecunc.init
def skip_jecunc_init(self: Calibrator) -> None:

    # different MET names for Run2 and Run3, this could maybe be implemented better
    run_calibrators = {
        2: jec_nominal_run2,
        3: jec_nominal_run3,
    }
    run = self.config_inst.campaign.x.run

    if self.dataset_inst.is_data:
        calibrators = {run_calibrators[run]}
    else:
        calibrators = {run_calibrators[run]}

    self.uses |= calibrators
    self.produces |= calibrators
