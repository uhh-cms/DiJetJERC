# coding: utf-8

"""
Custom base tasks for plotting workflow steps from JER SF measurement.
"""

import law


from columnflow.tasks.framework.base import ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorMixin, ProducersMixin,
    VariablesMixin, DatasetsProcessesMixin, CategoriesMixin,
)
from columnflow.config_util import get_datasets_from_process
from columnflow.util import dev_sandbox
from dijet.tasks.base import DiJetTask


class PlottingBaseTask(
    DiJetTask,
    DatasetsProcessesMixin,
    CategoriesMixin,
    VariablesMixin,
    ProducersMixin,
    SelectorMixin,
    CalibratorsMixin,
    ShiftTask,
    law.LocalWorkflow,
):
    """
    Base task to plot histogram from each step of the JER SF workflow.
    An example implementation of how to handle the inputs in a run method can be
    found in columnflow/tasks/histograms.py
    """
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # Add nested sibling directories to output path
    output_collection_cls = law.NestedSiblingFileCollection

    # Category ID for methods
    LOOKUP_CATEGORY_ID = {"sm": 1, "fe": 2}

    def get_datasets(self) -> tuple[list[str], bool]:
        """
        Select datasets belonging to the `process` of the current branch task.
        Returns a list of the dataset with requested Runs for data.
        """

        dataset_insts_from_process_data = get_datasets_from_process(
            self.config_inst,
            "data",
            only_first=False,
        )

        # filter to contain only user-supplied datasets from data
        datasets_from_process_data = [d.name for d in dataset_insts_from_process_data]
        datasets_data_filtered = set(self.datasets).intersection(datasets_from_process_data)
        # check that at least one user-supplied dataset matched
        if not datasets_data_filtered:
            raise RuntimeError(
                "no single user-supplied dataset for data matched "
                f"process `{dataset_insts_from_process_data}`",
            )

        # return filtered datasets
        return list(datasets_data_filtered)

    def extract_sample(self):
        datasets = self.get_datasets()
        runs = [dataset.replace("data_jetht_", "").upper() for dataset in datasets]
        sample = "Run" + ("".join(sorted(runs)))
        return sample

    def store_parts(self):
        parts = super().store_parts()
        return parts
