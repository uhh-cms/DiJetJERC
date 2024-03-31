# coding: utf-8

def eta_bin(self, eta_lo, eta_hi):
    return fr"{eta_lo}<$\eta$<{eta_hi}"
                                                                   
def pt_bin(self, pt_lo, pt_hi):
    spt = r"$p_\mathrm{T}$"
    return fr"{pt_lo}<{spt}<{pt_hi}"
                                                                   
def alpha_bin(self, alpha):
    salpha = r"$\alpha$<"
    return fr"{salpha}{alpha}"
                                                                   
def dot_to_p(self, var):
    return f"{var}".replace(".", "p")
                                                                   
def add_text(self, ax, x, y, text, offset=0, ha="left", va="top"):
    ax.text(
        x, y - offset, text,
        transform=ax.transAxes,
        horizontalalignment=ha,
        verticalalignment=va,
    )
