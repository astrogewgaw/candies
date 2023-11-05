"""
Code for the Candies CLI.
"""

import typer
import warnings
import h5py as h5
import numpy as np
from numba import cuda
from pathlib import Path
from rich.progress import track
from typing_extensions import Annotated
from candies.plotting import plot_candy
from candies.utilities import dmt_extent
from candies.core import Candies, Filterbank
from candies.kernels import fastdmt, dedisperse


# Forbidden jutsu: disable all warnings.
warnings.filterwarnings("ignore")


app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="rich",
)


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
        typer.Argument(
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            allow_dash=True,
            help="The list of candidates to process.",
        ),
    ],
    filterbank: Annotated[
        Path,
        typer.Option(
            "-f",
            "--fil",
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            allow_dash=True,
            help="The filterbank file to process.",
        ),
    ],
    ndms: Annotated[
        int,
        typer.Option(
            rich_help_panel="Configure",
            help="The number of DMs to use for the DM v/s time array.",
        ),
    ] = 256,
    device: Annotated[
        int,
        typer.Option(
            help="The GPU device ID.",
            rich_help_panel="Configure",
        ),
    ] = 0,
    save: Annotated[
        bool,
        typer.Option(
            rich_help_panel="Configure",
            help="If specified, save candidates to disk.",
        ),
    ] = True,
    zoom: Annotated[
        bool,
        typer.Option(
            rich_help_panel="Configure",
            help="If specified, employ the DMT zoom feature",
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

        if zoom:
            _, _, dmlow, dmhigh = dmt_extent(
                fil.fl,
                fil.fh,
                fil.dt,
                candy.t0,
                candy.dm,
                candy.wbin,
            )
        else:
            dmlow, dmhigh = 0.0, 2.0 * candy.dm

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

        ntmid = int(ntdown / 2)
        nticrop = ntmid - 128
        ntfcrop = ntmid + 128
        dmt = dmt[:, nticrop:ntfcrop]
        dyn = dyn[:, nticrop:ntfcrop]

        if save:
            with h5.File(f"{candy.id}.h5", "w") as f:
                f.attrs["nt"] = nt
                f.attrs["ndms"] = ndms
                f.attrs["id"] = candy.id
                f.attrs["dmlow"] = dmlow
                f.attrs["dmhigh"] = dmhigh

                f.attrs["nf"] = fil.nf
                f.attrs["dt"] = fil.dt
                f.attrs["df"] = fil.df
                f.attrs["fl"] = fil.fl
                f.attrs["fh"] = fil.fh

                f.attrs["t0"] = candy.t0
                f.attrs["dm"] = candy.dm
                f.attrs["snr"] = candy.snr
                f.attrs["wbin"] = candy.wbin

                dset = f.create_dataset(
                    "dyn",
                    data=dyn,
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
def plot(
    candidates: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            allow_dash=True,
            help="The candidates to plot.",
        ),
    ],
    save: Annotated[
        bool,
        typer.Option(
            help="If specified, save plot to disk.",
        ),
    ] = True,
    show: Annotated[
        bool,
        typer.Option(
            help="If specified, show the plot.",
        ),
    ] = True,
):
    """
    Plot candy-date(s).
    """
    if len(candidates) > 1:
        for candidate in track(candidates, description="Plotting..."):
            plot_candy(candidate, save=save, show=show)
    else:
        plot_candy(candidates[0], save=save, show=show)
