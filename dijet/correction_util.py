# coding: utf-8

from __future__ import annotations

import correctionlib.schemav2 as cs
import correctionlib.convert as convert

from columnflow.util import maybe_import

np = maybe_import("numpy")
hist = maybe_import("hist")


def create_NSC_correctionlib(
    name: str,
    eta_edges: np.ndarray,
) -> cs.Correction():
    """
    Creates an empty correctionset that can be filled in eta bins with the results of the NSC fits.
    """
    corr = cs.Correction(
        name=f"NSC_fit_{name}",
        description="NSC fit to JER for Sample {name}",
        version=1,
        inputs=[
            cs.Variable(
                name="eta",
                type="real",
                description="preusorapdity of the jet",
            ),
            cs.Variable(
                name="pt",
                type="real",
                description="transverse momentum of the jet",
            ),
        ],
        output=cs.Variable(
            name="fitted_JER",
            type="real",
            description="fitted JER value from NSC fit",
        ),
        data=cs.Binning(
            nodetype="binning",
            input="eta",
            edges=eta_edges,
            content=[
                cs.Formula(
                    nodetype="formula",
                    variables=["pt"],
                    parser="TFormula",
                    expression="",
                )
                for _ in range(len(eta_edges) - 1)
            ],
            flow="clamp",
        ),
    )

    return corr


def fill_NSC_correctionlib(
    corr: cs.Correction,
    eta_bin_idx: int,
    fit_params,
) -> None:
    """
    Fills the formula node for the given eta bin index with the fit parameters from the NSC fit.
    """
    N = fit_params["N"]
    S = fit_params["S"]
    C = fit_params["C"]
    d = fit_params["d"]

    if N > 0:
        formula_str = f"sqrt(({N}*{N})/pow(x, 2) + ({S}*{S})/pow(x, {d}) + {C}*{C})"
    else:
        formula_str = f"sqrt(-({N}*{N})/pow(x, 2) + ({S}*{S})/pow(x, {d}) + {C}*{C})"

    corr.data.content[eta_bin_idx].expression = formula_str

    return corr


def create_correctionlib_from_hist(
    h_in: hist.Hist,
    name: str,
    label: str,
    description: str,
) -> cs.Correction:
    """
    Converts a given histogram to a correctionlib correction
    """
    h_in.name = name
    h_in.label = label

    corr = convert.from_histogram(h_in)
    corr.description = description

    return corr


def build_correctionset(
    corrs: list[cs.Correction],
    description: str,
    schema_version: int = 2,
) -> cs.CorrectionSet:
    """
    Builds a correctionlib CorrectionSet from a list of corrections.
    """
    cset = cs.CorrectionSet(
        schema_version=schema_version,
        description=description,
        corrections=corrs,
    )

    return cset
