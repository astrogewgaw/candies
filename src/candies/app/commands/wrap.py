from pathlib import Path
from typing import Literal, Annotated

from cyclopts import Parameter
from cyclopts.validators import Number
from cyclopts.types import PositiveInt, ExistingFile, ExistingDirectory

StoreFormats = Literal["fil", "h5"]
InterfaceOptions = Literal["gmrt", "sigproc"]
ModelOptions = Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]


def wrap(
    candidates: ExistingFile,
    datafile: ExistingFile | None = None,
    zoom: Annotated[bool, Parameter(alias="-z")] = True,
    model: Annotated[ModelOptions, Parameter(alias="-m")] = "a",
    storebursts: Annotated[bool, Parameter(alias="-s")] = False,
    wrapsize: Annotated[PositiveInt, Parameter(alias="-w")] = 8,
    batchsize: Annotated[PositiveInt, Parameter(alias="-b")] = 100,
    out: Annotated[ExistingDirectory, Parameter(alias="-o")] = Path.cwd(),
    storeformat: Annotated[StoreFormats, Parameter(alias="-sfmt")] = "fil",
    interface: Annotated[InterfaceOptions, Parameter(alias="-i")] = "sigproc",
    njobs: Annotated[int, Parameter(alias="-n", validator=Number(gte=-1))] = 1,
    gpuid: Annotated[int, Parameter(alias="-g", validator=Number(gte=-1))] = -1,
):
    import numpy as np
    from candies.classify import classify
    from candies.featurize import featurize
    from candies.interfaces import FileInterface
    from candies.base import Candies, CandiesError

    candies = Candies.load(candidates)

    df = candies.pandas
    if datafile is not None:
        df["fn"] = df["fn"].fillna(str(datafile))

    for fn, group in df.groupby("fn"):
        try:
            made = []
            for _, batch in group.groupby(np.arange(len(group) // batchsize)):
                made.append(
                    featurize(
                        zoom=zoom,
                        njobs=njobs,
                        gpuid=gpuid,
                        store=storebursts,
                        candies=Candies.load(group),
                        interface=FileInterface[interface].load(fn=fn),
                    )
                )
        except KeyError:
            raise CandiesError("INVALID INTERFACE. ABORT.")
        for candy in classify(
            gpuid=gpuid,
            modelid=model,
            batchsize=wrapsize,
            candies=Candies(made),
        ):
            candy.save()
            if storebursts:
                candy.sliced.store(f"{candy.id}.highres.{storeformat}")


__all__ = ["wrap"]
