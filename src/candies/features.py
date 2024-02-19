import mmap
import numpy as np
from numba import cuda
from pathlib import Path
from priwo import readhdr
from candies.base import Dedispersed, DMTransform, CandidateList
from candies.utilities import kdm, delay2dm, dm2delay, normalise


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

    acc = 0.0
    if fi < nf and ti < nt:
        k1 = kdm * dm / dt
        k2 = k1 * fh**-2
        f = fh - fi * df
        dbin = int(round(k1 * f**-2 - k2))
        xti = ti + dbin
        if xti >= nt:
            xti -= nt
        acc += ft[fi, xti]
        cuda.atomic.add(dyn, (int(fi / ffactor), int(ti / tfactor)), acc)  # type: ignore


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
    ti = int(cuda.blockIdx.x)  # type: ignore
    dmi = int(cuda.threadIdx.x)  # type: ignore

    acc = 0.0
    k1 = kdm * (dmlow + dmi * ddm) / dt
    k2 = k1 * fh**-2
    for fi in range(nf):
        f = fh - fi * df
        dbin = int(round(k1 * f**-2 - k2))
        xti = ti + dbin
        if xti >= nt:
            xti -= nt
        acc += ft[fi, xti]
    cuda.atomic.add(dmt, (dmi, int(ti / tfactor)), acc)  # type: ignore


def featurize(
    candidates: CandidateList,
    filterbank: str | Path,
    gpuid: int = 0,
    save: bool = True,
    zoom: bool = True,
    zoomfact: int = 512,
):
    cuda.select_device(gpuid)
    stream = cuda.stream()

    meta = readhdr(filterbank)
    fh = meta["fch1"]
    df = meta["foff"]
    dt = meta["tsamp"]
    nf = meta["nchans"]
    mjd = meta["tstart"]
    nskip = meta["size"]

    if df < 0:
        df = abs(df)
        bw = nf * df
        fl = fh - bw + (0.5 * df)
    else:
        fl = fh
        bw = nf * df
        fh = fl + bw - (0.5 * df)

    with open(filterbank, "rb") as f:
        mapped = mmap.mmap(f.fileno(), 0, mmap.MAP_PRIVATE, mmap.PROT_READ)
        nbytes = int(mapped.size()) - nskip
        totalbins = int(nbytes / nf)

        for candidate in candidates:
            maxdelay = dm2delay(fl, fh, candidate.dm)

            binbeg = int((candidate.t0 - maxdelay) / dt) - candidate.wbin
            binend = int((candidate.t0 + maxdelay) / dt) + candidate.wbin
            padbeg = -binbeg if binbeg < 0 else 0
            padend = binend - totalbins if binend > totalbins else 0
            binbeg = 0 if binbeg < 0 else binbeg
            binend = totalbins if binend > totalbins else binend

            ncount = (binend - binbeg) * nf
            noffset = (binbeg * nf) + nskip

            data = np.ascontiguousarray(
                np.pad(
                    np.frombuffer(
                        mapped,
                        count=ncount,
                        offset=noffset,
                        dtype=np.uint8,
                    )
                    .reshape(-1, nf)
                    .T,
                    ((0, 0), (padbeg, padend)),
                    mode="median",
                )
            )

            _, nt = data.shape

            ndms = 256
            if zoom:
                ddm = delay2dm(fl, fh, zoomfact * candidate.wbin * dt)
                dmlow, dmhigh = candidate.dm - ddm, candidate.dm + ddm
            else:
                dmlow, dmhigh = 0.0, 2 * candidate.dm
            ddm = (dmhigh - dmlow) / (ndms - 1)

            downfreq = int(nf / 256)
            downtime = 1 if candidate.wbin < 3 else int(candidate.wbin / 2)

            nfdown = int(nf / downfreq)
            ntdown = int(nt / downtime)

            gpudata = cuda.to_device(data, stream=stream)
            gpudd = cuda.device_array((nfdown, ntdown), order="C", stream=stream)
            gpudmt = cuda.device_array((ndms, ntdown), order="C", stream=stream)

            dedisperse[  # type: ignore
                (int(nf / 32), int(nt / 32)),
                (32, 32),
                stream,
            ](
                gpudd,
                gpudata,
                nf,
                nt,
                df,
                dt,
                fh,
                candidate.dm,
                downfreq,
                downtime,
            )

            fastdmt[  # type: ignore
                nt,
                ndms,
                stream,
            ](
                gpudmt,
                gpudata,
                nf,
                nt,
                df,
                dt,
                fh,
                ddm,
                dmlow,
                downtime,
            )

            ntmid = int(ntdown / 2)

            dedispersed = gpudd.copy_to_host(stream=stream)
            dedispersed = dedispersed[:, ntmid - 128 : ntmid + 128]
            dedispersed = normalise(dedispersed)
            dedispersed = Dedispersed(
                dt=dt,
                fl=fl,
                fh=fh,
                nt=256,
                nf=256,
                dm=candidate.dm,
                data=dedispersed,
                df=(fh - fl) / 256,
            )

            dmtransform = gpudmt.copy_to_host(stream=stream)
            dmtransform = dmtransform[:, ntmid - 128 : ntmid + 128]
            dmtransform = normalise(dmtransform)
            dmtransform = DMTransform(
                dt=dt,
                nt=256,
                ddm=ddm,
                ndms=ndms,
                dmlow=dmlow,
                dmhigh=dmhigh,
                dm=candidate.dm,
                data=dmtransform,
            )

            if save:
                candidate.save(
                    "_".join(
                        [
                            f"MJD{mjd:.7f}",
                            f"T{candidate.t0:.7f}",
                            f"DM{candidate.dm:.5f}",
                            f"SNR{candidate.snr:.5f}",
                        ]
                    )
                    + ".h5"
                )
    cuda.close()
