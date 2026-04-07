import math
from dataclasses import dataclass
from multiprocessing.pool import Pool

import numpy as np
from numba import njit, cuda
from scipy.signal import detrend

from candies.logging import log
from candies.interfaces import Interface
from candies.base import Candy, Candies, Dedispersed, DMTransform, CandiesError


@dataclass
class CPUFeaturizer:
    interface: Interface

    zoom: bool = True
    store: bool = False
    snratio: float = 0.1

    def __call__(self, candy: Candy) -> Candy:
        try:
            sliced = self.interface.slice(candy)
            if self.store:
                candy.sliced = sliced
            nf, nt = sliced.data.shape

            def znorm(X):
                X = X.astype(np.float32)
                X = np.nan_to_num(X)
                X = detrend(X)
                X = X - np.median(X)
                X = X / np.std(X)
                X = np.nan_to_num(X)
                return X

            @njit(cache=True, fastmath=True, boundscheck=False, error_model="numpy")
            def crop(Y, X, stride):
                nf = Y.shape[0]
                nt = Y.shape[1]
                for ii in range(nf):
                    for jj in range(nt):
                        Y[ii, jj] = X[ii, jj + stride]

            @njit(cache=True, fastmath=True, boundscheck=False, error_model="numpy")
            def calcdd(Y, X, shifts, td, fd):
                nf = X.shape[0]
                nt = X.shape[1]
                for ii in range(nf):
                    for jj in range(nt):
                        iiy = ii // fd
                        jjy = jj // td
                        jjx = jj + shifts[ii]
                        if jjx >= nt:
                            jjx -= nt
                        Y[iiy, jjy] += X[ii, jjx]

            @njit(cache=True, fastmath=True, boundscheck=False, error_model="numpy")
            def calcdmt(Y, X, shifts, td):
                nf = X.shape[0]
                nt = X.shape[1]
                ndms = Y.shape[0]
                for jj in range(nt):
                    for kk in range(ndms):
                        acc = 0.0
                        jjy = jj // td
                        for ii in range(nf):
                            shift = shifts[kk, ii]
                            jjx = jj + shift
                            if jjx >= nt:
                                jjx -= nt
                            acc += X[ii, jjx]
                        Y[kk, jjy] += acc

            ndms = 256
            fh = self.interface.fh
            fl = self.interface.fl
            nf = self.interface.nf
            bw = self.interface.bw
            dt = self.interface.dt
            lodm, hidm = 0.0, 2.0 * candy.dm
            if self.zoom:
                fc = 0.5 * (fh + fl)
                if (
                    ddm := np.sqrt(np.pi)
                    * fc**3
                    * (candy.wbin * dt)
                    / (1382 * self.snratio * bw)
                ) < candy.dm:
                    lodm, hidm = candy.dm - ddm, candy.dm + ddm
            ddm = (hidm - lodm) / (ndms - 1)
            ff = np.linspace(fh, fl, nf, dtype=np.float32)
            dms = np.linspace(lodm, hidm, ndms, dtype=np.float32)
            perdmshifts = (4.1488064239e3 * (ff**-2 - fh**-2) / dt).astype(np.float32)

            shifts = (candy.dm * perdmshifts).astype(np.int32)
            allshifts = (dms[:, None] * perdmshifts[None, :]).astype(np.int32)

            td = 1 if candy.wbin < 3 else int(candy.wbin / 2)
            fd = int(nf / 256)
            nfred = int(nf / fd)
            ntred = int(nt / td)

            dd = np.zeros((nfred, ntred), dtype=np.float32)
            dmt = np.zeros((ndms, ntred), dtype=np.float32)
            ddcropped = np.zeros((256, 256), dtype=np.float32)
            dmtcropped = np.zeros((256, 256), dtype=np.float32)

            calcdd(dd, sliced.data, shifts, td, fd)
            crop(ddcropped, dd, ntred // 2 - 128)

            calcdmt(dmt, sliced.data, allshifts, td)
            crop(dmtcropped, dmt, ntred // 2 - 128)

            candy.dedispersed = Dedispersed(
                nt=256,
                nf=256,
                fh=fh,
                fl=fl,
                dt=dt * td,
                dm=candy.dm,
                df=(fh - fl) / 256,
                data=znorm(ddcropped),
            )

            candy.dmtransform = DMTransform(
                nt=256,
                ddm=ddm,
                ndms=ndms,
                lodm=lodm,
                hidm=hidm,
                dt=dt * td,
                dm=candy.dm,
                data=znorm(dmtcropped),
            )

            candy.extras["tbeg"] = sliced.tbeg
            candy.extras["tend"] = sliced.tend
            candy.extras = {**candy.extras, **sliced.extras}

            return candy
        except CandiesError:
            log.error(f"Featurization failed for {candy.id}.")
            return candy


@dataclass
class GPUFeaturizer:
    interface: Interface

    gpuid: int = 0
    zoom: bool = True
    store: bool = False
    snratio: float = 0.1

    def __call__(self, candy: Candy) -> Candy:
        try:
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

            @cuda.jit(cache=True, fastmath=True)
            def crop(Y, X, stride):
                nf = Y.shape[0]
                nt = Y.shape[1]
                ii, jj = cuda.grid(2)  # type: ignore
                if (ii < nf) and (jj < nt):
                    Y[ii, jj] = X[ii, jj + stride]

            @cuda.jit(cache=True, fastmath=True)
            def calcdd(Y, X, shifts, td, fd):
                nf = X.shape[0]
                nt = X.shape[1]
                ii, jj = cuda.grid(2)  # type: ignore
                if (ii < nf) and (jj < nt):
                    iiy = ii // fd
                    jjy = jj // td
                    jjx = jj + shifts[ii]
                    if jjx >= nt:
                        jjx -= nt
                    cuda.atomic.add(Y, (iiy, jjy), X[ii, jjx])  # type: ignore

            @cuda.jit(cache=True, fastmath=True)
            def calcdmt(Y, X, shifts, td):
                nf = X.shape[0]
                nt = X.shape[1]
                ndms = Y.shape[0]
                jj, kk = cuda.grid(2)  # type: ignore
                if jj < nt and kk < ndms:
                    acc = 0.0
                    jjy = jj // td
                    for ii in range(nf):
                        shift = shifts[kk, ii]
                        jjx = jj + shift
                        if jjx >= nt:
                            jjx -= nt
                        acc += X[ii, jjx]
                    cuda.atomic.add(Y, (kk, jjy), acc)  # type: ignore

            sliced = self.interface.slice(candy)
            if self.store:
                candy.sliced = sliced
            nf, nt = sliced.data.shape

            ndms = 256
            fh = self.interface.fh
            fl = self.interface.fl
            nf = self.interface.nf
            bw = self.interface.bw
            dt = self.interface.dt
            lodm, hidm = 0.0, 2.0 * candy.dm
            if self.zoom:
                fc = 0.5 * (fh + fl)
                if (
                    ddm := np.sqrt(np.pi)
                    * fc**3
                    * (candy.wbin * dt)
                    / (1382 * self.snratio * bw)
                ) < candy.dm:
                    lodm, hidm = candy.dm - ddm, candy.dm + ddm
            ddm = (hidm - lodm) / (ndms - 1)
            ff = np.linspace(fh, fl, nf, dtype=np.float32)
            dms = np.linspace(lodm, hidm, ndms, dtype=np.float32)
            perdmshifts = (4.1488064239e3 * (ff**-2 - fh**-2) / dt).astype(np.float32)

            shifts = (candy.dm * perdmshifts).astype(np.int32)
            allshifts = (dms[:, None] * perdmshifts[None, :]).astype(np.int32)

            td = 1 if candy.wbin < 3 else int(candy.wbin / 2)
            fd = int(nf / 256)
            nfred = int(nf / fd)
            ntred = int(nt / td)

            shiftsdevice = cuda.to_device(shifts, stream=stream)
            datadevice = cuda.to_device(sliced.data, stream=stream)
            allshiftsdevice = cuda.to_device(allshifts, stream=stream)
            ddcropped = cuda.device_array((256, 256), order="C", stream=stream, dtype=np.float32)  # type: ignore
            dmtcropped = cuda.device_array((256, 256), order="C", stream=stream, dtype=np.float32)  # type: ignore
            dddevice = cuda.device_array((nfred, ntred), order="C", stream=stream, dtype=np.float32)  # type: ignore
            dmtdevice = cuda.device_array((ndms, ntred), order="C", stream=stream, dtype=np.float32)  # type: ignore

            threads = (32, 32)
            blocks = (math.ceil(nf / threads[0]), math.ceil(nt / threads[1]))
            calcdd[blocks, threads, stream](dddevice, datadevice, shiftsdevice, td, fd)  # type: ignore

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

            threads = (32, 32)
            blocks = (math.ceil(nt / threads[0]), math.ceil(ndms / threads[1]))
            calcdmt[blocks, threads, stream](dmtdevice, datadevice, allshiftsdevice, td)  # type: ignore

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

            candy.extras["tbeg"] = sliced.tbeg
            candy.extras["tend"] = sliced.tend
            candy.extras = {**candy.extras, **sliced.extras}

            cuda.close()
            return candy
        except CandiesError:
            log.error(f"Featurization failed for {candy.id}.")
            return candy


def featurize(
    candies: Candies,
    interface: Interface,
    njobs: int = 1,
    gpuid: int = -1,
    zoom: bool = True,
    store: bool = False,
    snratio: float = 0.1,
) -> Candies:
    with Pool(processes=njobs) as pool:
        candies = Candies(
            pool.map(
                (
                    CPUFeaturizer(
                        zoom=zoom,
                        store=store,
                        snratio=snratio,
                        interface=interface,
                    )
                    if gpuid < 0
                    else GPUFeaturizer(
                        zoom=zoom,
                        gpuid=gpuid,
                        store=store,
                        snratio=snratio,
                        interface=interface,
                    )
                ),
                candies,
                chunksize=1,
            )
        )
    return candies
