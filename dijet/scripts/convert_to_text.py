# coding: utf8
"""
Script to convert corrections stored in text files to correctionlib jsons and vice versa.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import gzip
import correctionlib.schemav2 as cs


def convert_to_json(
    lines: list[str],
    output: str | None = None,
):
    """
    Convert text files to correctionlib jsons.
    """
    header = lines[0].split()
    bins = []
    for line in lines[1:]:
        parts = line.split()
        eta_min, eta_max = float(parts[0]), float(parts[1])
        p0, p1, p2, p3 = map(float, parts[-4:])
        bins.append({
            "eta_min": eta_min,
            "eta_max": eta_max,
            "params": [p0, p1, p2, p3],
        })

    corr = cs.Correction(
        name="NSC_fit_from_txt",
        description="NSC fit to JER",
        version=1,
        inputs=[
            cs.Variable(
                name=header[1],
                type="real",
                description="preusorapdity of the jet",
            ),
            cs.Variable(
                name=header[3],
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
            input=header[1],
            edges=[b["eta_min"] for b in bins] + [bins[-1]["eta_max"]],
            content=[
                cs.Formula(
                    nodetype="formula",
                    variables=[header[3]],
                    parser="TFormula",
                    expression=header[-3],
                    parameters=[
                        b["params"][0],
                        b["params"][1],
                        b["params"][2],
                        b["params"][3],
                    ],
                )
                for b in bins
            ],
            flow="clamp",
        ),
    )
    cset = cs.CorrectionSet(
        schema_version=2,
        description=f"Correction converted from {Path(args.infile).stem}",
        corrections=[corr],
    )
    output = args.outfile if args.outfile is not None else f"{Path(args.infile).with_suffix('')}.json.gz"
    print(f"Saving converted file at {output}")
    with gzip.open(f"{output}", "wt") as fout:
        fout.write(cset.json(exclude_unset=True))


def convert_to_text(
    corr_set: str,
    output: str | None = None,
):
    """
    Convert correctionlib jsons to text files.
    """
    cset = cs.CorrectionSet.parse_obj(json.loads(corr_set))
    correction = cset.corrections[0]
    if "eta" not in correction.inputs[0].name or "pt" not in correction.inputs[1].name:
        raise ValueError(
            f"Expected eta and pt as inputs. Correction has inputs {[inp.name for inp in correction.inputs]}",
        )
    output = args.outfile if args.outfile is not None else f"{Path(args.infile).with_suffix('')}.txt"
    print(f"Saving converted file at {output}")
    with open(f"{output}", "w") as fout:
        fout.write(
            "{1     JetEta  2  JetPt   Rho   "
            f"{correction.data.content[0].expression} "
            "Correction L2Relative}\n",
        )
        # Loop over bins
        edges = correction.data.edges
        content = correction.data.content  # list of Formula objects

        for i, formula in enumerate(content):
            eta_min = edges[i]
            eta_max = edges[i + 1]
            params = formula.parameters
            fout.write(
                f"{eta_min:.3f} {eta_max:.3f} 8 5 6800 0 200 "
                f"{params[0]:10.3f} {params[1]:8.4f} {params[2]:8.4f} {params[3]:8.2f}\n",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to-json",
        action="store_true",
        help="If set, the script converts text files to correctionlib jsons. "
        "If not the correctionlib json gets converted to a text file.",
    )
    parser.add_argument(
        "infile",
        nargs="?",
        help="Input file. Should be a text file or a json; depending on the direction of the conversion.",
    )
    parser.add_argument(
        "-o", "--outfile",
        nargs=1,
        help="Output file. If not specified, the output will saved in the same directory as the input.",
    )
    args = parser.parse_args()
    if args.to_json:
        if args.infile is None or not args.infile.endswith(".txt"):
            raise ValueError("Input file missing or invalid format. Must be a text file when converting to json.")

        with open(args.infile, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        convert_to_json(lines, output=args.outfile)
    else:
        if args.infile is None or not args.infile.endswith(".json"):
            raise ValueError("Input file missing or invalid format. Must be a json file when converting to text.")

        with open(args.infile, "r") as f:
            c_set = json.load(f)

        convert_to_text(c_set, output=args.outfile)
