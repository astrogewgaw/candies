from pathlib import Path

from cyclopts.types import ExistingFile, NonNegativeInt, ExistingDirectory


def plot(
    candidates: list[ExistingFile],
    save: bool = True,
    show: bool = False,
    dpi: NonNegativeInt = 96,
    saveto: ExistingDirectory = Path.cwd(),
):
    from rich.progress import track
    from candies.base import Candies

    if not show:
        import matplotlib

        matplotlib.use("agg")
    for candy in track(
        transient=True,
        description="Plotting...",
        sequence=Candies.load(candidates),
    ):
        candy.plot(
            dpi=dpi,
            show=show,
            save=Path(saveto) / f"{candy.id}.png" if save else False,
        )


__all__ = ["plot"]
