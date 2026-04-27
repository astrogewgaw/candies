import math
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from numba import njit, cuda
from scipy.signal import detrend
from autoregistry import Registry
from joblib import Parallel, delayed

from candies.base import (
    Slice,
    Candy,
    Candies,
    CandiesError,
    Dedispersed,
    DMTransform,
)
from candies.logging import log
from candies.interfaces import Interface


def znorm(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    X = np.nan_to_num(X)
    X = detrend(X)
    X = X - np.median(X)
    X = X / np.std(X)
    X = np.nan_to_num(X)
    return X


@dataclass
class Featurizer(Registry, recursive=False, suffix="Featurizer"):
    candy: Candy
    gpuid: int = -1
    zoom: bool = True
    store: bool = False
    snratio: float = 0.1
    interface: Interface | None = None

    @property
    def sliced(self) -> Slice:
        if self.candy.sliced is not None:
            return self.candy.sliced
        if self.interface is not None:
            self.interface.slice(self.candy)
            assert self.candy.sliced is not None
            return self.candy.sliced
        else:
            raise CandiesError("NO INTERFACE OR DATA PROVIDED. ABORT.")

    @property
    def data(self) -> np.ndarray:
        return self.sliced.data

    @property
    def nf(self) -> int:
        return self.data.shape[0]

    @property
    def nt(self) -> int:
        return self.data.shape[1]

    @property
    def fh(self) -> float:
        return self.sliced.fh

    @property
    def fl(self) -> float:
        return self.sliced.fl

    @property
    def bw(self) -> float:
        return self.sliced.bw

    @property
    def dt(self) -> float:
        return self.sliced.dt

    @property
    def dm(self) -> float:
        return self.candy.dm

    @property
    def t0(self) -> float:
        return self.candy.t0

    @property
    def wbin(self) -> int:
        return self.candy.wbin

    @property
    def snr(self) -> float:
        return self.candy.snr

    @property
    def ndms(self) -> int:
        return 256

    @property
    def deltadm(self) -> float:
        if self.zoom and (
            (
                deltadm := np.sqrt(np.pi)
                * (0.5 * (self.fh + self.fl)) ** 3
                * (self.wbin * self.dt)
                / (1382 * self.snratio * self.bw)
            )
            < self.dm
        ):
            return deltadm
        return self.dm

    @property
    def lodm(self) -> float:
        return self.dm - self.deltadm

    @property
    def hidm(self) -> float:
        return self.dm + self.deltadm

    @property
    def ddm(self) -> float:
        return (self.hidm - self.lodm) / (self.ndms - 1)

    @property
    def dms(self) -> np.ndarray:
        return np.linspace(self.lodm, self.hidm, self.ndms, dtype=np.float32)

    @property
    def freqs(self) -> np.ndarray:
        return np.linspace(self.fh, self.fl, self.nf, dtype=np.float32)

    @property
    def perdmshifts(self) -> np.ndarray:
        return (4.1488064239e3 * (self.freqs**-2 - self.fh**-2) / self.dt).astype(
            np.float32
        )

    @property
    def shifts(self) -> np.ndarray:
        return (self.dm * self.perdmshifts).astype(np.int32)

    @property
    def allshifts(self) -> np.ndarray:
        return (self.dms[:, None] * self.perdmshifts[None, :]).astype(np.int32)

    @property
    def td(self) -> int:
        return 1 if self.wbin < 3 else int(self.wbin / 2)

    @property
    def fd(self) -> int:
        return int(self.nf / 256)

    @property
    def nfred(self) -> int:
        return int(self.nf / self.fd)

    @property
    def ntred(self) -> int:
        return int(self.nt / self.td)

    @abstractmethod
    def run(self) -> None:
        pass

    def __call__(self) -> None:
        try:
            self.run()
        except Exception as ex:
            log.error(f"Featurization failed for {self.candy.id}. ERROR: {str(ex)}.")
        log.info(f"Featurization succeeded for {self.candy.id}.")


@dataclass
class CPUFeaturizer(Featurizer):
    def run(self):
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

        dd = np.zeros((self.nfred, self.ntred), dtype=np.float32)
        dmt = np.zeros((self.ndms, self.ntred), dtype=np.float32)
        ddcropped = np.zeros((256, 256), dtype=np.float32)
        dmtcropped = np.zeros((256, 256), dtype=np.float32)

        calcdd(dd, self.data, self.shifts, self.td, self.fd)
        crop(ddcropped, dd, self.ntred // 2 - 128)

        calcdmt(dmt, self.data, self.allshifts, self.td)
        crop(dmtcropped, dmt, self.ntred // 2 - 128)

        self.candy.dedispersed = Dedispersed(
            nt=256,
            nf=256,
            dm=self.dm,
            fh=self.fh,
            fl=self.fl,
            dt=self.dt * self.td,
            data=znorm(ddcropped),
            df=(self.fh - self.fl) / 256,
        )

        self.candy.dmtransform = DMTransform(
            nt=256,
            dm=self.dm,
            ddm=self.ddm,
            ndms=self.ndms,
            lodm=self.lodm,
            hidm=self.hidm,
            dt=self.dt * self.td,
            data=znorm(dmtcropped),
        )


@dataclass
class GPUFeaturizer(Featurizer):
    def run(self) -> None:
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

        shiftsdevice = cuda.to_device(self.shifts, stream=stream)
        datadevice = cuda.to_device(self.data, stream=stream)
        allshiftsdevice = cuda.to_device(self.allshifts, stream=stream)
        ddcropped = cuda.device_array((256, 256), order="C", stream=stream, dtype=np.float32)  # type: ignore
        dmtcropped = cuda.device_array((256, 256), order="C", stream=stream, dtype=np.float32)  # type: ignore
        dddevice = cuda.device_array((self.nfred, self.ntred), order="C", stream=stream, dtype=np.float32)  # type: ignore
        dmtdevice = cuda.device_array((self.ndms, self.ntred), order="C", stream=stream, dtype=np.float32)  # type: ignore

        threads = (32, 32)
        blocks = (math.ceil(self.nf / threads[0]), math.ceil(self.nt / threads[1]))
        calcdd[blocks, threads, stream](dddevice, datadevice, shiftsdevice, self.td, self.fd)  # type: ignore

        threads = (32, 32)
        blocks = (
            math.ceil(self.nfred / threads[0]),
            math.ceil(self.ntred / threads[1]),
        )
        crop[blocks, threads, stream](ddcropped, dddevice, self.ntred // 2 - 128)  # type: ignore

        self.candy.dedispersed = Dedispersed(
            nt=256,
            nf=256,
            dm=self.dm,
            fh=self.fh,
            fl=self.fl,
            dt=self.dt * self.td,
            df=(self.fh - self.fl) / 256,
            data=znorm(ddcropped.copy_to_host(stream=stream)),  # type: ignore
        )

        threads = (32, 32)
        blocks = (math.ceil(self.nt / threads[0]), math.ceil(self.ndms / threads[1]))
        calcdmt[blocks, threads, stream](dmtdevice, datadevice, allshiftsdevice, self.td)  # type: ignore

        threads = (32, 32)
        blocks = (math.ceil(self.ndms / threads[0]), math.ceil(self.ntred / threads[1]))
        crop[blocks, threads, stream](dmtcropped, dmtdevice, self.ntred // 2 - 128)  # type: ignore

        self.candy.dmtransform = DMTransform(
            nt=256,
            dm=self.dm,
            ddm=self.ddm,
            ndms=self.ndms,
            lodm=self.lodm,
            hidm=self.hidm,
            dt=self.dt * self.td,
            data=znorm(dmtcropped.copy_to_host(stream=stream)),  # type: ignore
        )
        cuda.close()


def featurize(
    candies: Candies,
    interface: Interface,
    njobs: int = 1,
    gpuid: int = -1,
    zoom: bool = True,
    store: bool = False,
    snratio: float = 0.1,
) -> Candies:
    featurizers = [
        Featurizer["CPU" if gpuid < 0 else "GPU"](
            candy=candy,
            zoom=zoom,
            gpuid=gpuid,
            store=store,
            snratio=snratio,
            interface=interface,
        )
        for candy in candies
    ]
    Parallel(n_jobs=njobs)(delayed(_)() for _ in featurizers)
    return Candies(items=[_.candy for _ in featurizers])


__all__ = ["Featurizer", "CPUFeaturizer", "GPUFeaturizer", "featurize"]
