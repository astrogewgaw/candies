"""
Process and extract features from candy-date(s).
"""

import numpy as np
from typing import Any
from numba import cuda
from pathlib import Path
from rich.progress import track
from candies.core import Candy, Candies
from candies.utilities import dmt_extent
from candies.kernels import fastdmt, dedisperse


def process_candy(
    candy: Candy,
    ndms: int = 256,
    save: bool = True,
    zoom: bool = True,
    stream: Any | None = None,
):
    """
    Process a single candy-date.
    """
    cuda_params: dict = {}
    if stream is None:
        stream = cuda.stream()
        cuda_params = {"stream": stream}

    data = candy.chop()
    _, candy.nt = data.shape

    if zoom:
        _, _, dmlow, dmhigh = dmt_extent(
            candy.fl,
            candy.fh,
            candy.dt,
            candy.t0,
            candy.dm,
            candy.wbin,
        )
    else:
        dmlow, dmhigh = 0.0, 2.0 * candy.dm

    candy.ndms = ndms
    candy.dmlow = dmlow
    candy.dmhigh = dmhigh
    ddm = (dmhigh - dmlow) / (ndms - 1)

    tfactor = 1 if np.log2(candy.wbin) < 3 else np.log2(candy.wbin) // 2
    ffactor = np.floor(candy.nf // 256).astype(int)
    nfdown = int(candy.nf / ffactor)
    ntdown = int(candy.nt / tfactor)

    ft = cuda.to_device(data, **cuda_params)
    dmtx = cuda.device_array((ndms, ntdown), order="C", **cuda_params)
    dynx = cuda.to_device(np.zeros((nfdown, ntdown), order="C"), **cuda_params)

    dedisperse[  # type: ignore
        (candy.nf // 32, candy.nt // 32),
        (32, 32),
        stream,
    ](
        dynx,
        ft,
        candy.nf,
        candy.nt,
        candy.df,
        candy.dt,
        candy.fh,
        candy.dm,
        ffactor,
        tfactor,
    )

    fastdmt[  # type: ignore
        candy.nt,
        candy.ndms,
        stream,
    ](
        dmtx,
        ft,
        candy.nf,
        candy.nt,
        candy.df,
        candy.dt,
        candy.fh,
        ddm,
        dmlow,
        tfactor,
    )

    dyn = dynx.copy_to_host(**cuda_params)
    dmt = dmtx.copy_to_host(**cuda_params)

    ntmid = int(ntdown / 2)
    nticrop = ntmid - 128
    ntfcrop = ntmid + 128
    dmt = dmt[:, nticrop:ntfcrop]
    dyn = dyn[:, nticrop:ntfcrop]

    candy.dyn = dyn
    candy.dmt = dmt

    if save:
        candy.save()

    if stream is None:
        cuda.close()


def process_candies(
    candidates: str | Path,
    filterbank: str | Path,
    ndms: int = 256,
    device: int = 0,
    save: bool = True,
    zoom: bool = True,
):
    """
    Process candy-dates.
    """
    candies = Candies.wrap(candidates, filterbank)
    cuda.select_device(device)
    stream = cuda.stream()
    for candy in track(candies, description="Processing candy-dates..."):
        process_candy(
            candy,
            ndms,
            save,
            zoom,
            stream,
        )
    cuda.close()
