from pathlib import Path
from functools import partial
from typing import List, Literal

import cyclopts

from candies.classify import classify
from candies.featurize import featurize
from candies.interfaces import Interface
from candies.base import Candy, Candies, CandiesError

app = cyclopts.App()
app["--help"].group = "Admin"
app["--version"].group = "Admin"


@app.command
def list():
    pass


@app.command
def make(
    candidates: str | Path,
    njobs: int = 1,
    gpuid: int = 0,
    zoom: bool = True,
    datafile: str | Path | None = None,
    interface: Literal["splt", "gmrt", "sigproc", "spotlight"] = "sigproc",
):
    if (_ := Interface["file"].get(interface)) is not None:
        maker = partial(featurize, interface=_.load(fn=datafile))
    elif (_ := Interface["live"].get(interface)) is not None:
        maker = partial(featurize, interface=_.load())
    else:
        raise CandiesError("INVALID INTERFACE. ABORT.")
    for candy in maker(
        zoom=zoom,
        njobs=njobs,
        gpuid=gpuid,
        candies=Candies.load(candidates),
    ):
        candy.save()


@app.command
def label(
    candidates: List[Path],
    gpuid: int = 0,
    batchsize: int = 8,
    model: Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"] = "a",
):
    for candy in classify(
        gpuid=gpuid,
        modelid=model,
        batchsize=batchsize,
        candies=Candies([Candy.load(candidate) for candidate in candidates]),
    ):
        candy.save()


@app.command
def wrap(
    candidates: str | Path,
    njobs: int = 1,
    gpuid: int = 0,
    zoom: bool = True,
    batchsize: int = 8,
    datafile: str | Path | None = None,
    interface: Literal["splt", "gmrt", "sigproc", "spotlight"] = "sigproc",
    model: Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"] = "a",
):
    if (_ := Interface["file"].get(interface)) is not None:
        maker = partial(featurize, interface=_.load(fn=datafile))
    elif (_ := Interface["live"].get(interface)) is not None:
        maker = partial(featurize, interface=_.load())
    else:
        raise CandiesError("INVALID INTERFACE. ABORT.")
    for candy in classify(
        gpuid=gpuid,
        modelid=model,
        batchsize=batchsize,
        candies=maker(
            zoom=zoom,
            njobs=njobs,
            gpuid=gpuid,
            candies=Candies.load(candidates),
        ),
    ):
        candy.save()


@app.command
def store():
    pass


@app.command
def plot():
    pass


if __name__ == "__main__":
    app()
