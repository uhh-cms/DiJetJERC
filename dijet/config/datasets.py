# coding: utf-8

import order as od

from columnflow.tasks.external import GetDatasetLFNs


def get_dataset_lfns(
    dataset_inst: od.Dataset,
    shift_inst: od.Shift,
    dataset_key: str,
) -> list[str]:
    """
    Custom method to obtain custom NanoAOD datasets
    """

    return GetDatasetLFNs.get_dataset_lfns_dasgoclient(
        GetDatasetLFNs, dataset_inst=dataset_inst, shift_inst=shift_inst, dataset_key=dataset_key,
    )
