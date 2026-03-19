import math
from dataclasses import dataclass
from multiprocessing.pool import Pool

import numpy as np
from numba import cuda

from candies.interfaces import Interface
from candies.functions import K, znorm, delay2dm
from candies.base import Candy, Candies, Dedispersed, DMTransform


@dataclass
class Featurizer:
    interface: Interface
    zoom: bool = True
    gpuid: int = 0

    def __call__(self, candy: Candy) -> Candy:
        cuda.select_device(self.gpuid)
        stream = cuda.stream()

        @cuda.jit
        def crop(Y, X, stride):
            ii, jj = cuda.grid(2)  # type: ignore
            nf, nt = Y.shape
            if (ii < nf) and (jj < nt):
                Y[ii, jj] = X[ii, jj + stride]

        @cuda.jit
        def calcdd(Y, X, dm, fh, df, dt, td, fd):
            ii, jj = cuda.grid(2)  # type: ignore
            nf, nt = X.shape
            if (ii < nf) and (jj < nt):
                cuda.atomic.add(Y, (int(ii / fd), int(jj / td)), X[ii, (jj + int(round(K * dm / dt * ((fh - ii * df) ** -2 - fh**-2)))) % nt])  # type: ignore

        @cuda.jit
        def calcdmt(Y, X, lodm, ddm, fh, df, dt, td):
            ii, jj, kk = cuda.grid(3)  # type: ignore
            nf, nt = X.shape
            ndms, _ = Y.shape
            if (ii < nf) and (jj < nt) and (kk < ndms):
                cuda.atomic.add(Y, (kk, int(jj / td)), X[ii, (jj + int(round(K * (lodm + kk * ddm) / dt * ((fh - ii * df) ** -2 - fh**-2)))) % nt])  # type: ignore

        tbeg, tend, data = self.interface.slice(candy)

        ndms = 256
        fudge = 64
        nf, nt = data.shape
        lodm, hidm = 0.0, 2.0 * candy.dm
        if self.zoom:
            ddm = delay2dm(
                self.interface.fl,
                self.interface.fh,
                fudge * candy.dm * self.interface.dt,
            )
            if ddm < candy.dm:
                lodm, hidm = candy.dm - ddm, candy.dm + ddm
        ddm = (hidm - lodm) / (ndms - 1)

        td = 1 if candy.wbin < 3 else int(candy.wbin / 2)
        fd = int(nf / 256)
        nfred = int(nf / fd)
        ntred = int(nt / td)

        DATADEVICE = cuda.to_device(data, stream=stream)
        DDDEVICE = cuda.device_array(
            (ndms, ntred),
            order="C",
            stream=stream,
            dtype=np.float32,  # type: ignore
        )
        DMTDEVICE = cuda.device_array(
            (nfred, ntred),
            order="C",
            stream=stream,
            dtype=np.float32,  # type: ignore
        )
        DDCROPPED = cuda.device_array(
            (256, 256),
            order="C",
            stream=stream,
            dtype=np.float32,  # type: ignore
        )
        DMTCROPPED = cuda.device_array(
            (256, 256),
            order="C",
            stream=stream,
            dtype=np.float32,  # type: ignore
        )

        nthreads = 32
        nblocksx = math.ceil(nf / nthreads)
        nblocksy = math.ceil(nt / nthreads)
        calcdd[(nblocksx, nblocksy), (nthreads, nthreads), stream](  # type: ignore
            DDDEVICE,
            DATADEVICE,
            candy.dm,
            self.interface.fh,
            self.interface.df,
            self.interface.dt,
            td,
            fd,
        )

        nthreads = 32
        nblocksx = math.ceil(nfred / nthreads)
        nblocksy = math.ceil(ntred / nthreads)
        crop[(nblocksx, nblocksy), (nthreads, nthreads), stream](  # type: ignore
            DDCROPPED,
            DDDEVICE,
            int(int(ntred / 2) - 128),
        )

        candy.dedispersed = Dedispersed(
            nt=256,
            nf=256,
            dm=candy.dm,
            fh=self.interface.fh,
            fl=self.interface.fl,
            dt=self.interface.dt * td,
            df=(self.interface.fh - self.interface.fl) / 256,
            data=znorm(DDCROPPED.copy_to_host(stream=stream)),  # type: ignore
        )

        nthreads = 32
        nblocksx = math.ceil(nf / 1)
        nblocksy = math.ceil(nt / nthreads)
        nblocksz = math.ceil(ndms / nthreads)
        calcdmt[(nblocksx, nblocksy, nblocksz), (1, nthreads, nthreads), stream](  # type: ignore
            DMTDEVICE,
            DATADEVICE,
            lodm,
            ddm,
            self.interface.fh,
            self.interface.df,
            self.interface.dt,
            td,
        )

        nthreads = 32
        nblocksx = math.ceil(ndms / nthreads)
        nblocksy = math.ceil(ntred / nthreads)
        crop[(nblocksx, nblocksy), (nthreads, nthreads), stream](  # type: ignore
            DMTCROPPED,
            DMTDEVICE,
            int(int(ntred / 2) - 128),
        )

        candy.dmtransform = DMTransform(
            nt=256,
            ddm=ddm,
            ndms=ndms,
            lodm=lodm,
            hidm=hidm,
            dm=candy.dm,
            dt=self.interface.dt * td,
            data=znorm(DMTCROPPED.copy_to_host(stream=stream)),  # type: ignore
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
