from typing import Literal, Annotated

from cyclopts import Parameter
from cyclopts.validators import Number
from cyclopts.types import ExistingFile, NonNegativeInt


def label(
    candidates: list[ExistingFile],
    gpuid: Annotated[int, Parameter(alias="-g", validator=Number(gte=-1))] = -1,
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
):
    from rich.progress import track
    from candies.base import Candies
    from candies.classify import classify

    candies = Candies.load(candidates)
    candies = classify(
        gpuid=gpuid,
        modelid=model,
        candies=candies,
        batchsize=batchsize,
    )
    for candy in track(candies, description="Saving...", transient=True):
        candy.save()


__all__ = ["label"]
