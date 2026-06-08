from typing import Literal, Annotated

from cyclopts import Parameter
from cyclopts.validators import Number
from cyclopts.types import ExistingFile, ExistingCsvPath


def make(
    candidates: ExistingCsvPath,
    datafile: ExistingFile | None = None,
    njobs: Annotated[int, Parameter(alias="-n", validator=Number(gte=-1))] = 1,
    gpuid: Annotated[int, Parameter(alias="-g", validator=Number(gte=-1))] = -1,
    zoom: Annotated[bool, Parameter(alias="-z")] = True,
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
    from candies.featurize import featurize
    from candies.interfaces import FileInterface
    from candies.base import Candies, CandiesError

    candies = Candies.load(candidates)

    df = candies.pandas
    if datafile is not None:
        df["fn"] = df["fn"].fillna(str(datafile))

    made = []
    for fn, group in df.groupby("fn"):
        try:
            made.extend(
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
    for candy in track(Candies(items=made), description="Saving...", transient=True):
        candy.save()
        if storebursts:
            candy.sliced.store(f"{candy.id}.highres.{storeformat}")


__all__ = ["make"]
