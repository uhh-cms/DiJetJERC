# coding: utf-8

"""
Custom base tasks.
"""

import law


from columnflow.tasks.framework.base import BaseTask, ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorMixin, ProducersMixin,
    VariablesMixin, DatasetsProcessesMixin, CategoriesMixin,
)
from columnflow.config_util import get_datasets_from_process
from columnflow.util import dev_sandbox, DotDict
from dijet.tasks.base import HistogramBaseTask

class PlottingBaseTask(HistogramBaseTask):
    """
    Base task to plot histogram from each step of the JER SF workflow.
    An example implementation of how to handle the inputs in a run method can be
    found in columnflow/tasks/histograms.py
    """

