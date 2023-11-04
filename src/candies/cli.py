import typer
import h5py as h5
import numpy as np
from numba import cuda
from pathlib import Path
from rich.progress import track
from typing_extensions import Annotated
from candies.utilities import dmt_extent
from candies.core import Candies, Filterbank
from candies.kernels import fastdmt, dedisperse

app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")


@app.callback()
def main():
    """
    [b]Sweet, sweet candy-dates![/b]
    """
    pass


@app.command()
def make(
    candidates: Annotated[
        Path,
        typer.Option(
            "-c",
            "--cands",
            help="The list of candidates to process.",
        ),
    ],
    filterbank: Annotated[
        Path,
        typer.Option(
            "-f",
            "--fil",
            help="The filterbank file to process.",
        ),
    ],
    ndms: Annotated[
        int,
        typer.Option(
            help="The number of DMs to use for the DM v/s time array.",
        ),
    ] = 256,
    device: Annotated[
        int,
        typer.Option(
            help="The GPU device ID.",
        ),
    ] = 0,
    save: Annotated[
        bool,
        typer.Option(
            help="If specified, save candidates to disk.",
        ),
    ] = True,
):
    """
    Make features for all candy-dates.
    """
    fil = Filterbank.from_sigproc(filterbank)
    candies = Candies.wrap(candidates)

    cuda.select_device(device)
    stream = cuda.stream()

    for candy in track(candies, description="Processing candy-dates..."):
        data = fil.chop(candy)
        _, nt = data.shape

        _, _, dmlow, dmhigh = dmt_extent(
            fil.fl,
            fil.fh,
            fil.dt,
            candy.t0,
            candy.dm,
            candy.wbin,
        )

        ddm = (dmhigh - dmlow) / (ndms - 1)

        tfactor = 1 if np.log2(candy.wbin) < 3 else np.log2(candy.wbin) // 2
        ffactor = np.floor(fil.nf // 256).astype(int)
        nfdown = int(fil.nf / ffactor)
        ntdown = int(nt / tfactor)

        ft = cuda.to_device(data, stream=stream)
        dmtx = cuda.device_array((ndms, ntdown), order="C", stream=stream)
        dynx = cuda.to_device(np.zeros((nfdown, ntdown), order="C"), stream=stream)

        dedisperse[  # type: ignore
            (fil.nf // 32, nt // 32),
            (32, 32),
            stream,
        ](
            dynx,
            ft,
            fil.nf,
            nt,
            fil.df,
            fil.dt,
            fil.fh,
            candy.dm,
            ffactor,
            tfactor,
        )

        fastdmt[  # type: ignore
            nt,
            ndms,
            stream,
        ](
            dmtx,
            ft,
            fil.nf,
            nt,
            fil.df,
            fil.dt,
            fil.fh,
            ddm,
            dmlow,
            tfactor,
        )

        dyn = dynx.copy_to_host(stream=stream)
        dmt = dmtx.copy_to_host(stream=stream)

        ntmid = int(nt / 2)
        nticrop = ntmid - 128
        ntfcrop = ntmid + 128
        dmt = dmt[:, nticrop:ntfcrop]
        dyn = dyn[:, nticrop:ntfcrop]

        id = f"T{candy.t0:.7f}DM{candy.dm:.5f}SNR{candy.snr:.5f}"

        if save:
            with h5.File(f"{id}.h5", "w") as f:
                f["id"] = id
                f["nt"] = nt
                f["ndms"] = ndms

                f["nf"] = fil.nf
                f["dt"] = fil.dt
                f["df"] = fil.df
                f["fl"] = fil.fl
                f["fh"] = fil.fh

                f["t0"] = candy.t0
                f["dm"] = candy.dm
                f["snr"] = candy.snr
                f["wbin"] = candy.wbin

                dset = f.create_dataset(
                    "dyn",
                    data=dmt,
                    compression="gzip",
                    compression_opts=9,
                )

                dset = f.create_dataset(
                    "dmt",
                    data=dmt,
                    compression="gzip",
                    compression_opts=9,
                )
                dset.dims[0].label = b"dm"
                dset.dims[1].label = b"time"
    cuda.close()


@app.command()
def view():
    """
    View candy-date(s).
    """
    pass


@app.command()
def plot():
    """
    Plot candy-date(s).
    """
    pass