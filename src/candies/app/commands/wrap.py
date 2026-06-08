from typing import Literal, Annotated

from cyclopts import Parameter
from cyclopts.validators import Number
from cyclopts.types import ExistingFile, NonNegativeInt, ExistingCsvPath


def wrap(
    candidates: ExistingCsvPath,
    datafile: ExistingFile | None = None,
    njobs: Annotated[int, Parameter(alias="-n", validator=Number(gte=-1))] = 1,
    gpuid: Annotated[int, Parameter(alias="-g", validator=Number(gte=-1))] = -1,
    zoom: Annotated[bool, Parameter(alias="-z")] = True,
    batchsize: Annotated[NonNegativeInt, Parameter(alias="-b")] = 8,
    model: Annotated[
        Literal[
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
        ],
        Parameter(alias="-m"),
    ] = "a",
    interface: Annotated[
        Literal[
            "gmrt",
            "sigproc",
        ],
        Parameter(alias="-i"),
    ] = "sigproc",
    storebursts: Annotated[bool, Parameter(alias="-s")] = False,
    storeformat: Annotated[
        Literal[
            "fil",
            "h5",
        ],
        Parameter(alias="-sfmt"),
    ] = "fil",
):
    from rich.progress import track
    from candies.classify import classify
    from candies.featurize import featurize
    from candies.interfaces import FileInterface
    from candies.base import Candies, CandiesError

    candies = Candies.load(candidates)

    df = candies.pandas
    if datafile is not None:
        df["fn"] = df["fn"].fillna(str(datafile))

    wrapped = []
    for fn, group in df.groupby("fn"):
        try:
            wrapped.extend(
                classify(
                    gpuid=gpuid,
                    modelid=model,
                    batchsize=batchsize,
                    candies=featurize(
                        zoom=zoom,
                        njobs=njobs,
                        gpuid=gpuid,
                        store=storebursts,
                        candies=Candies.load(group),
                        interface=FileInterface[interface].load(fn=fn),
                    ),
                )
            )
        except KeyError:
            raise CandiesError("INVALID INTERFACE. ABORT.")
    for candy in track(Candies(items=wrapped), description="Saving...", transient=True):
        candy.save()
        if storebursts:
            candy.sliced.store(f"{candy.id}.highres.{storeformat}")


__all__ = ["wrap"]
