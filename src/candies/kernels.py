"""
GPU kernels for Candies.
"""


from numba import cuda
from candies.utilities import kdm


@cuda.jit
def dedisperse(
    dyn,
    ft,
    nf: int,
    nt: int,
    df: float,
    dt: float,
    fh: float,
    dm: float,
    ffactor: float,
    tfactor: float,
):
    """
    The JIT-compiled GPU kernel for dedispersing the dynamic spectrum via rolling.
    """
    fi, ti = cuda.grid(2)  # type: ignore

    if fi < nf and ti < nt:
        k1 = kdm * dm / dt
        k2 = k1 * fh**-2
        f = fh - fi * df
        dbin = int(round(k1 * f**-2 - k2))
        if dbin >= nt:
            dbin = 0
        xti = ti + dbin
        if xti >= nt:
            xti -= nt
        cuda.atomic.add(dyn, (int(fi / ffactor), int(ti / tfactor)), ft[fi, xti])  # type: ignore


@cuda.jit
def fastdmt(
    dmt,
    ft,
    nf: int,
    nt: int,
    df: float,
    dt: float,
    fh: float,
    ddm: float,
    dmlow: float,
    tfactor: int,
):
    """
    The JIT-compiled GPU kernel for creating the DM v/s time array.
    """
    ti = int(cuda.blockIdx.x)  # type: ignore
    dmi = int(cuda.threadIdx.x)  # type: ignore

    acc = 0.0
    k1 = kdm * (dmlow + dmi * ddm) / dt
    k2 = k1 * fh**-2
    for fi in range(nf):
        f = fh - fi * df
        dbin = int(round(k1 * f**-2 - k2))
        if dbin >= nt:
            dbin = 0
        xti = ti + dbin
        if xti >= nt:
            xti -= nt
        acc += ft[fi, xti]
    dmt[dmi, int(ti / tfactor)] = acc
