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
        description=f"NSC fit to JER for Sample {name}",
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
                    # expression="min(sqrt([0]*[0]/(x*x)+[1]*[1]/x+[2]*[2]+pow([3]/x,4)),1.)",
                    expression="sqrt([0]*[0]/(x*x)+[1]*[1]/pow(x, [3])+[2]*[2])",
                    parameters=[0, 0, 0, 0],
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

    # if N > 0:
    #     formula_str = f"sqrt(({N}*{N})/pow(x, 2) + ({S}*{S})/pow(x, {d}) + {C}*{C})"
    # else:
    #     formula_str = f"sqrt(-({N}*{N})/pow(x, 2) + ({S}*{S})/pow(x, {d}) + {C}*{C})"

    corr.data.content[eta_bin_idx].parameters = [N, S, C, d]

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


def build_gen_reco_map(
    pt_edges: np.ndarray,
    eta_edges: np.ndarray,
) -> cs.Correction():
    """
    Creates an empty correctionset that can be filled in eta bins with the results of the NSC fits.
    """
    corr = cs.Correction(
        name="gen_reco_map",
        description="cset to map gen pt to reco pt",
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
            name="mean reco pt",
            type="real",
            description="mean reconstructed pt for given gen pt and eta",
        ),
        data=cs.Binning(
            nodetype="binning",
            input="eta",
            edges=eta_edges,
            content=[
                cs.Binning(
                    nodetype="binning",
                    input="pt",
                    edges=pt_edges,
                    content=[i for i in range(len(pt_edges) - 1)],
                    flow="clamp",
                )
                for _ in range(len(eta_edges) - 1)
            ],
            flow="clamp",
        ),
    )

    return corr
