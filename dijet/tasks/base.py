# coding: utf-8

"""
Custom base tasks.
"""

import law


from columnflow.tasks.framework.base import BaseTask, ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, ProducersMixin,
    VariablesMixin, DatasetsProcessesMixin, CategoriesMixin,
)
from columnflow.config_util import get_datasets_from_process
from columnflow.util import dev_sandbox, DotDict


class DiJetTask(BaseTask):
    task_namespace = "dijet"


class HistogramsBaseTask(
    DiJetTask,
    DatasetsProcessesMixin,
    CategoriesMixin,
    VariablesMixin,
    ProducersMixin,
    CalibratorsMixin,
    ShiftTask,
    law.LocalWorkflow,
):
    """
    Base task to load histogram and reduce them to used information.
    An example implementation of how to handle the inputs in a run method can be
    found in columnflow/tasks/histograms.py
    """
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # Add nested sibling directories to output path
    output_collection_cls = law.NestedSiblingFileCollection

    def get_datasets(self) -> tuple[list[str], bool]:
        """
        Select datasets belonging to the `process` of the current branch task.
        Returns a list of the dataset names and a flag indicating if they are
        data or MC.
        """
        # all datasets in config that are mapped to a process
        dataset_insts_from_process = get_datasets_from_process(
            self.config_inst,
            self.branch_data.process,
            only_first=False,
        )

        # check that at least one config dataset matched
        if not dataset_insts_from_process:
            raise RuntimeError(
                "no single dataset found in config matching "
                f"process `{self.branch_data.process}`"
            )

        # filter to contain only user-supplied datasets
        datasets_from_process = [d.name for d in dataset_insts_from_process]
        datasets_filtered = set(self.datasets).intersection(datasets_from_process)

        # check that at least one user-supplied dataset matched
        if not datasets_filtered:
            raise RuntimeError(
                "no single user-supplied dataset matched "
                f"process `{self.branch_data.process}`"
            )
        
        # set MC flag if any of the filtered datasets
        datasets_filtered_is_mc = set(
            self.config_inst.get_dataset(d).is_mc
            for d in datasets_filtered
        )
        if len(datasets_filtered_is_mc) > 1:
            raise RuntimeError(
                "filtered datasets have mismatched `is_mc` flags"
            )

        # return filtered datasets and is_mc flag
        return (
            datasets_filtered,
            list(datasets_filtered_is_mc)[0]
        )

    def create_branch_map(self):
        """
        Workflow has one branch for each process supplied via `processed`.
        """
        return [
            DotDict({"process": process})
            for process in sorted(self.processes)
        ]

    def store_parts(self):
        parts = super().store_parts()
        return parts

    def extract_sample(self):
        datasets, isMC = self.get_datasets()
        # Define output name
        if isMC:
            sample = "QCDHT"
        else:
            runs = []
            for dataset in datasets:
                runs.append(dataset.replace("data_jetht_", "").upper())
            sample = "Run" + ("".join(sorted(runs)))
        return sample

    def reduce_histogram(self, histogram, processes, shift):
        """
        Reduce away the `shift` and `process` axes of a multidimensional
        histogram by selecting a single shift and summing over the processes.
        """
        import hist

        def flatten_nested_list(nested_list):
            return [item for sublist in nested_list for item in sublist]

        # transform into list if necessary
        processes = law.util.make_list(processes)

        # get all sub processes
        process_insts = list(map(self.config_inst.get_process, processes))
        sub_process_insts = set(flatten_nested_list([
            [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        ]))

        # get shift instance
        shift_inst = self.config_inst.get_shift(shift)
        if shift_inst.id not in histogram.axes["shift"]:
            raise ValueError(f"histogram does not contain shift `{shift}`")

        # work on a copy
        h = histogram.copy()

        # axis selections
        h = h[{
            "process": [
                hist.loc(p.id)
                for p in sub_process_insts
                if p.id in h.axes["process"]
            ],
            "shift": hist.loc(shift_inst.id),
        }]

        # TODO: Sum over shift ? Be careful to only take one shift -> Error if more ?
        # axis reductions
        h = h[{
            "process": sum,
            # TODO: read rebinning factors from config
            "dijets_alpha": hist.rebin(5),
            "dijets_asymmetry": hist.rebin(2),
        }]

        return h
