from pathlib import Path
from typing import Literal, Annotated

from cyclopts import Parameter
from cyclopts.validators import Number
from cyclopts.types import PositiveInt, ExistingFile, ExistingDirectory

ModelOptions = Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]


def label(
    candidates: list[ExistingFile],
    model: Annotated[ModelOptions, Parameter(alias="-m")] = "a",
    batchsize: Annotated[PositiveInt, Parameter(alias="-b")] = 8,
    out: Annotated[ExistingDirectory, Parameter(alias="-o")] = Path.cwd(),
    gpuid: Annotated[int, Parameter(alias="-g", validator=Number(gte=-1))] = -1,
):
    from rich.progress import track
    from candies.base import Candies
    from candies.classify import classify

    for candy in track(
        classify(
            gpuid=gpuid,
            modelid=model,
            batchsize=batchsize,
            candies=Candies.load(candidates),
        ),
        description="Saving...",
        transient=True,
    ):
        candy.save(fn=out)


__all__ = ["label"]
