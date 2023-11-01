import csv
import typer
import h5py as h5
import numpy as np
import priwo as pw
import numba.cuda as cuda
import matplotlib.pyplot as plt


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
    noshow: bool = False,
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

        if np.log2(wbin) < 3:
            tfactor = 1
        else:
            tfactor = np.log2(wbin) // 2

        cuda.select_device(device)
        ft = cuda.to_device(data)
        dmtx = cuda.device_array((ndms, nt), order="C")
        fastdmt[nt, ndms](  # type: ignore
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
        dmt = dmtx.copy_to_host()
        cuda.close()

        ntmid = int(nt / 2)
        ticrop = ntmid - 128
        tfcrop = ntmid + 128
        dmt = dmt[:, ticrop:tfcrop]

        cand_id = f"candy_t0{t0:.7f}_dm{dm:.5f}_snr{snr:.5f}"

        if plot and (saveplot or (not noshow)):
            plt.xlabel("Time (in s)")
            plt.suptitle("DM v/s t Plot")
            plt.ylabel("DM (in pc cm$^-3$)")
            plt.imshow(
                dmt,
                aspect="auto",
                interpolation="none",
                extent=[ticrop, tfcrop, dmhigh, dmlow],
            )
            if saveplot:
                plt.savefig(f"{cand_id}.png", dpi=300)
            if not noshow:
                plt.show()
        else:
            print("Not plotting since `saveplot` is off and `noshow` is on.")

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


if __name__ == "__main__":
    typer.run(main)
