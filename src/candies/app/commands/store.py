from pathlib import Path
from typing import Literal, Annotated

from cyclopts import Parameter
from cyclopts.types import ExistingFile, ExistingDirectory


def store(
    candidates: list[ExistingFile],
    datafile: ExistingFile | None = None,
    interface: Annotated[
        Literal[
            "gmrt",
            "sigproc",
        ],
        Parameter(alias="-i"),
    ] = "sigproc",
    storeformat: Annotated[
        Literal[
            "fil",
            "h5",
        ],
        Parameter(name=["-fmt", "--format"]),
    ] = "fil",
    storeto: ExistingDirectory = Path.cwd(),
):
    from rich.progress import track
    from candies.interfaces import FileInterface
    from candies.base import Candies, CandiesError

    candies = Candies.load(candidates)

    df = candies.pandas
    if datafile is not None:
        df["fn"] = df["fn"].fillna(str(datafile))

    for fn, group in df.groupby("fn"):
        try:
            for candy in track(
                transient=True,
                description="Storing...",
                sequence=Candies.load(group),
            ):
                candy = FileInterface[interface].load(fn=fn).slice(candy)
                candy.sliced.store(storeto / f"{candy.id}.highres.{storeformat}")
        except KeyError:
            raise CandiesError("INVALID INTERFACE. ABORT.")


__all__ = ["store"]
