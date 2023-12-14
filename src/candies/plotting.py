"""
Plotting utilities for Candies.
"""

import h5py as h5
import numpy as np
import proplot as pplt
from pathlib import Path
from rich.progress import track

pplt.rc.update(
    {
        "font.size": 20,
        "suptitle.pad": 20,
        "subplots.tight": True,
        "subplots.refwidth": 6.5,
        "subplots.panelpad": 1.5,
    }
)


def plot_candy(
    hf: str | Path,
    save: bool = True,
    show: bool = False,
):
    """
    Plot a candy-date.
    """
    import warnings

    warnings.filterwarnings("ignore")

    with h5.File(hf, "r") as f:
        id = f.attrs["id"]
        nt = f.attrs["nt"]
        dt = f.attrs["dt"]
        fl = f.attrs["fl"]
        fh = f.attrs["fh"]
        ndms = f.attrs["ndms"]
        dmlow = f.attrs["dmlow"]
        dmhigh = f.attrs["dmhigh"]
        dmt = np.asarray(f["data_dm_time"])
        dyn = np.asarray(f["data_freq_time"])

        ntmid = int(nt / 2)  # type: ignore
        nti = ntmid - 128
        ntf = ntmid + 128
        ts = dyn.sum(axis=0)
        f = np.linspace(fl, fh, dyn.shape[0])  # type: ignore
        dms = np.linspace(dmlow, dmhigh, ndms)  # type: ignore
        t = np.linspace(nti * dt, ntf * dt, dyn.shape[1])  # type: ignore

        fig, axs = pplt.subplots(
            nrows=2,
            ncols=1,
            width=10,
            height=15,
            sharey=False,
            sharex="labels",
        )
        px = axs[0].panel_axes("t", width="5em")

        px.plot(t, ts)
        axs[0].pcolormesh(*np.meshgrid(t, f), dyn)
        axs[1].pcolormesh(*np.meshgrid(t, dms), dmt)

        px.format(xlabel="Time (in s)", ylabel="Flux (arb. units)")
        axs[0].format(
            xlabel="Time (in s)",
            ylabel="Frequency (in MHz)",
            title="Dedispersed dynamic spectrum",
        )

        axs[1].format(
            xlabel="Time (in s)",
            ylabel="DM (in pc cm$^-3$)",
            title=f"DMT, from {dmlow:.2f} to {dmhigh:.2f} pc cm$^-3$",
        )

        if save:
            fig.savefig(f"{id}.png", dpi=300)
        if show:
            pplt.show()


def plot_candies(
    hfs: str | Path | list[str | Path],
    save: bool = True,
    show: bool = False,
):
    """
    Plot candy-dates.
    """
    if isinstance(hfs, list):
        if len(hfs) > 1:
            for hf in track(hfs, description="Plotting..."):
                plot_candy(hf, save=save, show=show)
        else:
            plot_candy(hfs[0], save=save, show=show)
    else:
        plot_candy(hfs, save=save, show=show)
