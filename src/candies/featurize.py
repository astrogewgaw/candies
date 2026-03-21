import math
from dataclasses import dataclass
from multiprocessing.pool import Pool

import numpy as np
from numba import cuda
from scipy.signal import detrend

from candies.interfaces import Interface
from candies.base import Candy, Candies, Dedispersed, DMTransform


@dataclass
class Featurizer:
    interface: Interface
    zoom: bool = True
    gpuid: int = 0

    def __call__(self, candy: Candy) -> Candy:
        cuda.select_device(self.gpuid)
        stream = cuda.stream()

        def znorm(X):
            X = X.astype(np.float32)
            X = np.nan_to_num(X)
            X = detrend(X)
            X = X - np.median(X)
            X = X / np.std(X)
            X = np.nan_to_num(X)
            return X

        @cuda.jit(fastmath=True)
        def crop(Y, X, stride):
            nf = Y.shape[0]
            nt = Y.shape[1]
            ii, jj = cuda.grid(2)  # type: ignore
            if (ii < nf) and (jj < nt):
                Y[ii, jj] = X[ii, jj + stride]

        @cuda.jit(fastmath=True)
        def calcdd(Y, X, dm, shifts, td, fd):
            nf = X.shape[0]
            nt = X.shape[1]
            ii, jj = cuda.grid(2)  # type: ignore
            if (ii < nf) and (jj < nt):
                iiy = ii // fd
                jjy = jj // td
                jjx = jj + int(shifts[ii] * dm + 0.5)
                if jjx >= nt:
                    jjx -= nt
                cuda.atomic.add(Y, (iiy, jjy), X[ii, jjx])  # type: ignore

        @cuda.jit(fastmath=True)
        def calcdmt(Y, X, dms, shifts, td):
            nf = X.shape[0]
            nt = X.shape[1]
            ndms = Y.shape[0]
            ii, jj, kk = cuda.grid(3)  # type: ignore
            if (ii < nf) and (jj < nt) and (kk < ndms):
                jjy = jj // td
                jjx = jj + int(shifts[ii] * dms[kk] + 0.5)
                if jjx >= nt:
                    jjx -= nt
                cuda.atomic.add(Y, (kk, jjy), X[ii, jjx])  # type: ignore

        tbeg, tend, data = self.interface.slice(candy)

        ndms = 256
        fudge = 64
        nf, nt = data.shape
        fh = self.interface.fh
        fl = self.interface.fl
        nf = self.interface.nf
        dt = self.interface.dt
        lodm, hidm = 0.0, 2.0 * candy.dm
        if self.zoom:
            t = fudge * candy.dm * dt
            ddm = t / (4.1488064239e3 * (fl**-2 - fh**-2))
            if ddm < candy.dm:
                lodm, hidm = candy.dm - ddm, candy.dm + ddm
        ddm = (hidm - lodm) / (ndms - 1)
        ff = np.linspace(fh, fl, nf, dtype=np.float32)
        dms = np.linspace(lodm, hidm, ndms, dtype=np.float32)
        perdmshifts = (4.1488064239e3 * (ff**-2 - fh**-2) / dt).astype(np.float32)

        td = 1 if candy.wbin < 3 else int(candy.wbin / 2)
        fd = int(nf / 256)
        nfred = int(nf / fd)
        ntred = int(nt / td)

        dmsdevice = cuda.to_device(dms, stream=stream)
        datadevice = cuda.to_device(data, stream=stream)
        shiftsdevice = cuda.to_device(perdmshifts, stream=stream)
        ddcropped = cuda.device_array((256, 256), order="C", stream=stream, dtype=np.float32)  # type: ignore
        dmtcropped = cuda.device_array((256, 256), order="C", stream=stream, dtype=np.float32)  # type: ignore
        dddevice = cuda.device_array((nfred, ntred), order="C", stream=stream, dtype=np.float32)  # type: ignore
        dmtdevice = cuda.device_array((ndms, ntred), order="C", stream=stream, dtype=np.float32)  # type: ignore

        threads = (32, 32)
        blocks = (math.ceil(nf / threads[0]), math.ceil(nt / threads[1]))
        calcdd[blocks, threads, stream](dddevice, datadevice, candy.dm, shiftsdevice, td, fd)  # type: ignore

        threads = (32, 32)
        blocks = (math.ceil(nfred / threads[0]), math.ceil(ntred / threads[1]))
        crop[blocks, threads, stream](ddcropped, dddevice, ntred // 2 - 128)  # type: ignore

        candy.dedispersed = Dedispersed(
            nt=256,
            nf=256,
            fh=fh,
            fl=fl,
            dt=dt * td,
            dm=candy.dm,
            df=(fh - fl) / 256,
            data=znorm(ddcropped.copy_to_host(stream=stream)),  # type: ignore
        )

        threads = (1, 32, 32)
        blocks = (
            math.ceil(nf / threads[0]),
            math.ceil(nt / threads[1]),
            math.ceil(ndms / threads[2]),
        )
        calcdmt[blocks, threads, stream](dmtdevice, datadevice, dmsdevice, shiftsdevice, td)  # type: ignore

        threads = (32, 32)
        blocks = (math.ceil(ndms / threads[0]), math.ceil(ntred / threads[1]))
        crop[blocks, threads, stream](dmtcropped, dmtdevice, ntred // 2 - 128)  # type: ignore

        candy.dmtransform = DMTransform(
            nt=256,
            ddm=ddm,
            ndms=ndms,
            lodm=lodm,
            hidm=hidm,
            dt=dt * td,
            dm=candy.dm,
            data=znorm(dmtcropped.copy_to_host(stream=stream)),  # type: ignore
        )

        candy.extras = self.interface.extras
        candy.extras["tbeg"] = tbeg
        candy.extras["tend"] = tend

        cuda.close()
        return candy


def featurize(
    candies: Candies,
    interface: Interface,
    njobs: int = 1,
    gpuid: int = 0,
    zoom: bool = True,
) -> Candies:
    featurizer = Featurizer(zoom=zoom, gpuid=gpuid, interface=interface)
    with Pool(processes=njobs) as pool:
        candies = Candies(pool.map(featurizer, candies, chunksize=1))
    return candies
