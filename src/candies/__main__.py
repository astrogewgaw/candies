import csv
import typer
import h5py as h5
import numpy as np
import priwo as pw
import proplot as pplt
from numba import cuda


kdm: float = 4.1488064239e3


def dispersive_delay(
    f: float,
    f0: float,
    dm: float,
) -> float:
    """
    Calculates the dispersive delay (in s) for a particular frequency,
    given a value of the dispersion measure (DM, in pc per cm^-3) and
    the reference frequency. Both frequencies must be in MHz.
    """
    return kdm * dm * (f**-2 - f0**-2)


def dmt_extent(
    fl: float,
    fh: float,
    dt: float,
    t0: float,
    dm: float,
    wbin: float,
    fudge: float = 16,
):
    """
    Calculate the extent of a dispersed burst in the DM v/s time plane until
    a certain percentage drop in SNR. The SNR drop is determined via a fudge
    factor = (ΔSNR)**-2; so, for instance, a 25% drop in SNR is equivalent to
    a fudge factor of 16 (which is the default value).
    """
    width = wbin * dt
    tbin = np.round(t0 / dt).astype(int)
    ddm = (fudge * width) / (kdm * (fl**-2 - fh**-2))
    dbin = np.round(kdm * ddm * (fl**-2 - fh**-2) / dt).astype(int)
    t_range = (tbin - dbin, tbin + dbin)
    dm_range = (dm - ddm, dm + ddm) if ddm < dm else (0.0, 2.0 * dm)
    return (*t_range, *dm_range)


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


def main(
    candidates: str,
    ndms: int = 256,
    device: int = 0,
    save: bool = False,
    plot: bool = False,
    showplot: bool = True,
    saveplot: bool = False,
):
    """
    The main Candies program.
    """
    rows = []
    with open(candidates, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            rows.append(row)
    for row in rows:
        (
            fn,
            snr,
            t0,
            wbin,
            dm,
            _,
            _,
            _,
        ) = row

        fn = str(fn)
        t0 = float(t0)
        dm = float(dm)
        snr = float(snr)
        wbin = 2 ** int(wbin)

        meta = pw.readhdr(fn)

        fh = meta["fch1"]
        dt = meta["tsamp"]
        nf = meta["nchans"]
        size = meta["size"]
        df = np.abs(meta["foff"])

        fl = fh - (df * (nf - 1))
        maxdelay = dispersive_delay(fl, fh, dm)
        ti = t0 - maxdelay - (wbin * dt)
        tf = t0 + maxdelay + (wbin * dt)

        _, _, dmlow, dmhigh = dmt_extent(
            fl,
            fh,
            dt,
            t0,
            dm,
            wbin,
        )
        ddm = (dmhigh - dmlow) / (ndms - 1)

        nt = int((tf - ti) / dt)
        with open(fn, "rb") as f:
            f.seek(size)
            data = np.pad(
                np.fromfile(
                    f,
                    dtype=np.uint8,
                    offset=int(0.0 if ti < 0.0 else ti / dt) * nf,
                    count=(int((tf - (0.0 if ti < 0.0 else ti)) / dt) * nf),
                )
                .reshape(-1, nf)
                .T,
                (int(-ti / dt) if ti < 0.0 else 0, 0),
                mode="median",
            )
        nf, nt = data.shape

        tfactor = 1 if np.log2(wbin) < 3 else np.log2(wbin) // 2
        ffactor = np.floor(nf // 256).astype(int)
        nfdown = int(nf / ffactor)
        ntdown = int(nt / tfactor)

        cuda.select_device(device)
        stream = cuda.stream()
        ft = cuda.to_device(data, stream=stream)
        dmtx = cuda.device_array((ndms, ntdown), order="C", stream=stream)
        dynx = cuda.to_device(np.zeros((nfdown, ntdown), order="C"), stream=stream)

        dedisperse[  # type: ignore
            (nf // 32, nt // 32),
            (32, 32),
            stream,
        ](
            dynx,
            ft,
            nf,
            nt,
            df,
            dt,
            fh,
            dm,
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
            nf,
            nt,
            df,
            dt,
            fh,
            ddm,
            dmlow,
            tfactor,
        )

        dyn = dynx.copy_to_host(stream=stream)
        dmt = dmtx.copy_to_host(stream=stream)
        cuda.close()

        ntmid = int(nt / 2)
        nticrop = ntmid - 128
        ntfcrop = ntmid + 128
        dmt = dmt[:, nticrop:ntfcrop]
        dyn = dyn[:, nticrop:ntfcrop]

        cand_id = f"candy_t0{t0:.7f}_dm{dm:.5f}_snr{snr:.5f}"

        if save:
            with h5.File(f"{cand_id}.h5", "w") as f:
                f["nt"] = nt
                f["nf"] = nf
                f["dt"] = dt
                f["df"] = df
                f["fl"] = fl
                f["fh"] = fh
                f["t0"] = t0
                f["dm"] = dm
                f["snr"] = snr
                f["wbin"] = wbin
                f["ndms"] = ndms

                dset = f.create_dataset(
                    "dmt",
                    data=dmt,
                    compression="gzip",
                    compression_opts=9,
                )
                dset.dims[0].label = b"dm"
                dset.dims[1].label = b"time"

        if plot and (saveplot or showplot):
            fig = pplt.figure(share=False)
            axs = fig.subplots(nrows=3, ncols=1)

            ts = dyn.sum(axis=0)
            t = np.linspace(nticrop * dt, ntfcrop * dt, 256)

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

            if saveplot:
                fig.save(f"{cand_id}.png", dpi=300)  # type: ignore
            if showplot:
                pplt.show()


if __name__ == "__main__":
    typer.run(main)
