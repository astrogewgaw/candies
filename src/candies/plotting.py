import h5py as h5
import numpy as np
import proplot as pplt
from pathlib import Path

pplt.rc.update(
    {
        "font.size": 12,
        "title.loc": "ul",
        "cycle": "tab20b",
        "suptitle.pad": 20,
        "text.usetex": False,
        "axes.linewidth": 2.5,
        "tick.linewidth": 2.5,
        "subplots.tight": True,
        "subplots.share": True,
        "hatch.linewidth": 2.0,
        "subplots.refwidth": 5.0,
        "subplots.panelpad": 1.5,
        "agg.path.chunksize": 100000,
    }
)


def plot_candy(
    hf: str | Path,
    save: bool = True,
    show: bool = False,
):
    with h5.File(hf, "r") as f:
        id = f.attrs["id"]
        nt = f.attrs["nt"]
        dt = f.attrs["dt"]
        fl = f.attrs["fl"]
        fh = f.attrs["fh"]
        dmlow = f.attrs["dmlow"]
        dmhigh = f.attrs["dmhigh"]
        dyn = np.asarray(f["dyn"])
        dmt = np.asarray(f["dmt"])

        ntmid = int(nt / 2)  # type: ignore
        nti = ntmid - 128
        ntf = ntmid + 128
        ts = dyn.sum(axis=0)
        t = np.linspace(nti * dt, ntf * dt, dyn.shape[1])  # type: ignore

        fig, axs = pplt.subplots(
            nrows=2,
            ncols=1,
            width=25,
            height=15,
            sharey=False,
            sharex="labels",
        )

        px = axs[0].panel_axes("t", width="5em")
        px.plot(t, ts)
        px.format(
            xlabel="Time (in s)",
            ylabel="Flux (arb. units)",
            title="Dedispersed time series",
        )

        axs[0].imshow(
            dyn,
            aspect="auto",
            interpolation="none",
            extent=[t[0], t[-1], fh, fl],
        )

        axs[0].format(
            xlabel="Time (in s)",
            ylabel="Frequency (in MHz)",
            title="Dedispersed dynamic spectrum",
        )

        axs[1].imshow(
            dmt,
            aspect="auto",
            interpolation="none",
            extent=[t[0], t[-1], dmhigh, dmlow],
        )

        axs[1].format(
            xlabel="Time (in s)",
            ylabel="DM (in pc cm$^-3$)",
            title=f"DMT, from {dmlow:.2f} to {dmhigh:.2f} pc cm$^-3$",
        )

        if show:
            pplt.show()
        if save:
            fig.savefig(f"{id}.png", dpi=300)
