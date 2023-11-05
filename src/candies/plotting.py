import h5py as h5
import numpy as np
import proplot as pplt
from pathlib import Path


def plot_candy(hf: str | Path, save: bool = True, show: bool = False):
    with h5.File(hf, "r") as f:
        id = f.attrs["id"]
        nt = f.attrs["nt"]
        nf = f.attrs["nf"]
        dt = f.attrs["dt"]
        df = f.attrs["df"]
        fl = f.attrs["fl"]
        fh = f.attrs["fh"]
        t0 = f.attrs["t0"]
        dm = f.attrs["dm"]
        snr = f.attrs["snr"]
        ndms = f.attrs["ndms"]
        wbin = f.attrs["wbin"]
        dmlow = f.attrs["dmlow"]
        dmhigh = f.attrs["dmhigh"]
        dyn = np.asarray(f["dyn"])
        dmt = np.asarray(f["dmt"])

        ntmid = int(nt / 2)  # type: ignore
        nti = ntmid - 128
        ntf = ntmid + 128
        ts = dyn.sum(axis=0)
        t = np.linspace(nti * dt, ntf * dt, 256)  # type: ignore

        fig = pplt.figure(share=False)
        axs = fig.subplots(nrows=3, ncols=1)

        axs[0].plot(t, ts)

        axs[1].imshow(
            dyn,
            aspect="auto",
            interpolation="none",
            extent=[t[0], t[-1], fh, fl],
        )

        axs[1].format(
            xlabel="Time (in s)",
            title="Dynamic spectrum",
            ylabel="Frequency (in MHz)",
        )

        axs[2].imshow(
            dmt,
            aspect="auto",
            interpolation="none",
            extent=[t[0], t[-1], dmhigh, dmlow],
        )

        axs[2].format(
            xlabel="Time (in s)",
            title="DM v/s t Plot",
            ylabel="DM (in pc cm$^-3$)",
        )

        if show:
            pplt.show()
        if save:
            fig.savefig(f"{id}.png", dpi=300)
