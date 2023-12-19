# coding: utf-8

from __future__ import annotations

# import tabulate
import law

from columnflow.util import maybe_import

np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


def plot_asymmetry(
        asymmetry,
        centers,
        eta, pt, alpha,
        output: law.FileSystemDirectoryTarget,
) -> None:
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots()
    plt.scatter(centers.flatten(), asymmetry.flatten(), marker="s", color="red", where="mid")
    plt.xlim(-0.6, 0.6)
    output["asym"].child(f"asym_e{eta}_p{pt}_a{alpha}.pdf", type="f").dump(plt, formatter="mpl")
