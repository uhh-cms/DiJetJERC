# coding: utf-8

"""
Definition of variables.
"""

import order as od

from copy import deepcopy

from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT

np = maybe_import("numpy")
ak = maybe_import("awkward")


from dijet.constants import pt, eta, alpha


def add_feature_variables(config: od.Config) -> None:
    """
    Adds variables to a *config* that are produced as part of the `features` producer.
    """

    # Event properties
    config.add_variable(
        name="n_jet",
        binning=(12, -0.5, 11.5),
        x_title="Number of jets",
        discrete_x=True,
    )

    # jj features
    config.add_variable(
        name="deltaR_jj",
        binning=(40, 0, 5),
        x_title=r"$\Delta R(j_{1},j_{2})$",
    )


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config* that are present after `ReduceEvents`
    without calling any producer
    """

    # (the "event", "run" and "lumi" variables are required for some cutflow plotting task,
    # and also correspond to the minimal set of columns that coffea's nano scheme requires)
    config.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    config.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    config.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )

    #
    # Weights
    #

    # TODO: implement tags in columnflow; meanwhile leave these variables commented out (as they only work for mc)
    config.add_variable(
        name="npvs",
        expression="PV.npvs",
        binning=(51, -.5, 50.5),
        x_title="Number of primary vertices",
        discrete_x=True,
    )

    #
    # Object properties
    #

    config.add_variable(
        name="jets_pt",
        expression="Jet.pt",
        binning=(100, 0, 1000),
        unit="GeV",
        x_title="$p_{T}$ of all jets",
    )

    # Jets (3 pt-leading jets)
    for i in range(3):
        config.add_variable(
            name=f"jet{i+1}_pt",
            expression=f"Jet.pt[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(100, 0., 1000.),
            unit="GeV",
            x_title=r"Jet %i $p_{T}$" % (i + 1),
        )
        config.add_variable(
            name=f"jet{i+1}_eta",
            expression=f"Jet.eta[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(100, -5.0, 5.0),
            x_title=r"Jet %i $\eta$" % (i + 1),
        )
        config.add_variable(
            name=f"jet{i+1}_phi",
            expression=f"Jet.phi[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, -3.2, 3.2),
            x_title=r"Jet %i $\phi$" % (i + 1),
        )
        config.add_variable(
            name=f"jet{i+1}_mass",
            expression=f"Jet.mass[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 200),
            unit="GeV",
            x_title=r"Jet %i mass" % (i + 1),
        )

    #
    # dijet-related variables
    #

    def add_dijet_variable(**kwargs):
        """
        Helper function to add a dijet-related variable, optionally also
        adding a corresponding gen-level variable. if `gen_name` is given.

        If `gen_name` is given, a gen-level variable with an identical binning
        is also added. The optional arguments `gen_expression`, `gen_aux` and
        `gen_x_title` can be used to set the corresponding properties of the
        gen-level variable.

        If a gen-level variable is added, entries `gen_variable` and
        `reco_variable` will be created in the aux data of the main and
        gen-level variable, respectively, containing the name of the
        corresponding variable.
        """

        # pop gen-related kwargs
        gen_name = kwargs.pop("gen_name", None)
        gen_expression = kwargs.pop("gen_expression", None)
        gen_aux = kwargs.pop("gen_aux", None)
        gen_x_title = kwargs.pop("gen_x_title", None)

        # kwargs to pass to add_variable
        var_kwargs = deepcopy(kwargs)

        # handle gen-level variable
        if gen_name is not None:
            # construct default gen-level x axis title if none provided
            if gen_x_title is None:
                x_title = kwargs.get("x_title", None)
                if x_title is not None:
                    gen_x_title = f"{x_title} (gen)"

            # add gen-level variable name to reco variable aux
            var_kwargs.setdefault("aux", {})["gen_variable"] = gen_name

            # kwargs to pass to add_variable for gen-level variable
            gen_var_kwargs = dict(
                kwargs,
                name=gen_name,
                expression=gen_expression,
                x_title=gen_x_title,
            )

            # add reco variable to the aux data
            gen_var_kwargs.setdefault("aux", {})["reco_variable"] = kwargs["name"]

            if gen_aux:
                gen_var_kwargs.setdefault("aux", {}).update(gen_aux)

        # warn if gen_expression but no gen_name
        elif gen_expression is not None:
            print(
                f"[WARNING] `gen_expression` provided for variable "
                f"'{kwargs['name']}', but no `gen_name` -> no gen-level "
                "variable will be added",
            )

        # add the variables to the config
        config.add_variable(**var_kwargs)
        if gen_name is not None:
            config.add_variable(**gen_var_kwargs)

    # alpha variable for additional jet activity
    add_dijet_variable(
        name="dijets_alpha",
        expression="alpha",
        binning=alpha,
        x_title=r"$\alpha$",
        # TODO: implement if needed
        # gen_name="dijets_alpha_gen",
        # gen_expression="alpha_gen",
        # gen_x_title=r"$\alpha$ (gen)",
        aux={
            # options for formatting plain-text string used to
            # represent variable and bin (e.g. in filenames)
            "slug_name": "alpha_lt",
            "bin_slug_format": lambda value: f"{value:1.3f}".replace(".", "p"),
            "bin_slug_func": lambda self, bin_edges: "_".join([
                self.x.slug_name,
                self.x.bin_slug_format(bin_edges[1]),
            ]),
            # options for formatting label used in plots to indicate
            # represent variable and bin
            "bin_label_format": lambda value: f"{value:g}",
            "bin_label_func": lambda self, bin_edges: " < ".join([
                self.x_title,
                self.x.bin_label_format(bin_edges[1]),
            ]),
        },
    )

    # same as above, but with full range (0-1) and finer binning
    add_dijet_variable(
        name="dijets_alpha_fine",
        expression="alpha",
        binning=(100, 0, 1),
        x_title=r"$\alpha$",
        # TODO: implement if needed
        # gen_name="dijets_alpha_gen_fine",
        # gen_expression="alpha_gen",
        # gen_x_title=r"$\alpha$ (gen)",
        aux={
            # options for formatting plain-text string used to
            # represent variable and bin (e.g. in filenames)
            "slug_name": "alpha_lt",
            "bin_slug_format": lambda value: f"{value:1.3f}".replace(".", "p"),
            "bin_slug_func": lambda self, bin_edges: "_".join([
                self.x.slug_name,
                self.x.bin_slug_format(bin_edges[1]),
            ]),
            # options for formatting label used in plots to indicate
            # represent variable and bin
            "bin_label_format": lambda value: f"{value:g}",
            "bin_label_func": lambda self, bin_edges: " < ".join([
                self.x_title,
                self.x.bin_label_format(bin_edges[1]),
            ]),
        },
    )

    #
    # variables for response distributions:
    #

    # pT asymmetry between probe and reference jets
    add_dijet_variable(
        name="dijets_asymmetry",
        expression="dijets.asymmetry",
        binning=(160, -0.8, 0.8),
        x_title=r"Asymmetry",
        gen_name="dijets_asymmetry_gen",
        gen_expression="dijets.asymmetry_gen",
        gen_x_title=r"Asymmetry (gen)",
    )

    # MPF response
    add_dijet_variable(
        name="dijets_mpf",
        expression="dijets.mpf",
        binning=(100, -1, 1),
        x_title=r"MPF",
        gen_name="dijets_mpf_gen",
        gen_expression="dijets.mpf_gen",
        gen_x_title=r"MPF (gen)",
    )

    # MPFx response
    add_dijet_variable(
        name="dijets_mpfx",
        expression="dijets.mpfx",
        binning=(100, -1, 1),
        x_title=r"MPFx",
        gen_name="dijets_mpfx_gen",
        gen_expression="dijets.mpfx_gen",
        gen_x_title=r"MPFx (gen)",
    )

    # average pT of probe and reference jets
    add_dijet_variable(
        name="dijets_pt_avg",
        expression="dijets.pt_avg",
        binning=pt,
        x_title=r"$p_{T}^{avg}$",
        unit="GeV",
        gen_name="dijets_pt_avg_gen",
        gen_expression="dijets.pt_avg_gen",
        gen_x_title=r"$p_{T}^{avg}$ (gen)",
        aux={
            "slug_name": "pt",
            "bin_slug_format": lambda value: f"{int(value):d}",
            "bin_label_format": lambda value: f"{int(value):d}",
        },
    )

    # eta bin is always defined by probe jet
    # SM: Both jets in eta bin
    # FE: Probejet in eta bin
    add_dijet_variable(
        name="probejet_abseta",
        expression=lambda events: abs(events.probe_jet.eta),
        binning=eta,
        x_title=r"$|\eta|$",
        aux={
            "inputs": {"probe_jet.eta"},
            "slug_name": "abseta",
            "bin_slug_format": lambda value: f"{value:1.3f}".replace(".", "p"),
            "bin_label_format": lambda value: f"{value:g}",
        },
        gen_name="probejet_abseta_gen",
        # use `genJetIdx` to get matched gen jet
        gen_expression=lambda events: abs(ak.firsts(events["GenJet"][ak.singletons(events["probe_jet"]["genJetIdx"])].eta)),  # noqa
        gen_x_title=r"$|\eta|$ (gen)",
        gen_aux={
            "inputs": {"GenJet.*", "probe_jet.genJetIdx"},
        },
    )

    # per-event gen/reco ratios

    config.add_variable(
        name="dijets_response_probe",
        expression="dijets.response_probe",
        binning=(100, 0, 2),
        x_title=r"response probe jet",
    )
    config.add_variable(
        name="dijets_response_reference",
        expression="dijets.response_reference",
        binning=(100, 0, 2),
        x_title=r"response reference jet",
    )
