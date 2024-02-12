# coding: utf-8

"""
Custom base tasks.
"""

import law


from columnflow.tasks.framework.base import BaseTask, ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin,
    VariablesMixin, DatasetsProcessesMixin, CategoriesMixin,
)
from columnflow.config_util import get_datasets_from_process
from columnflow.util import dev_sandbox


class DiJetTask(BaseTask):
    task_namespace = "dijet"


class HistogramsBaseTask(
    DiJetTask,
    DatasetsProcessesMixin,
    CategoriesMixin,
    VariablesMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    ShiftTask,
    law.LocalWorkflow,
):
    """
    Base task to load histogram and reduce them to used information.
    An exemplary implementation of how to handle the inputs in a run method can be
    found in columnflow/tasks/histograms.py
    """
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # Add nested sibling directories to output path
    output_collection_cls = law.NestedSiblingFileCollection

    def get_datasets(self):
        # Get a samples from process
        process_sets = get_datasets_from_process(self.config_inst, self.branch_data.process, only_first=False)
        process_names = [item.name for item in process_sets]
        # Get all samples from input belonging to process
        samples = set(self.datasets).intersection(process_names)
        if len(samples) == 0:
            samples = process_names
        return (
            samples,
            self.config_inst.get_dataset(process_sets[0]).is_mc,
        )

    def store_parts(self):
        parts = super().store_parts()
        return parts

    def reduce_histogram(self, histogram, processes, shifts):
        import hist

        def flatten_nested_list(nested_list):
            return [item for sublist in nested_list for item in sublist]

        # transform into lists if necessary
        processes = law.util.make_list(processes)
        shifts = law.util.make_list(shifts)

        # get all sub processes
        process_insts = list(map(self.config_inst.get_process, processes))
        sub_process_insts = set(flatten_nested_list([
            [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        ]))

        # get all shift instances
        shift_insts = [self.config_inst.get_shift(shift) for shift in shifts]

        # work on a copy
        h = histogram.copy()

        # axis selections
        h = h[{
            "process": [
                hist.loc(p.id)
                for p in sub_process_insts
                if p.id in h.axes["process"]
            ],
            "shift": [
                hist.loc(s.id)
                for s in shift_insts
                if s.id in h.axes["shift"]
            ],
        }]

        # TODO: Sum over shift ? Be careful to only take one shift -> Error if more ?
        # axis reductions
        h = h[{"process": sum, "shift": sum, "dijets_alpha": hist.rebin(5), "dijets_asymmetry": hist.rebin(2)}]

        return h
