from cyclopts.types import ExistingFile, NonExistentCsvPath


def list_(
    candidates: list[ExistingFile],
    show: bool = True,
    save: NonExistentCsvPath | None = None,
):
    from candies.base import Candies

    candies = Candies.load(candidates)
    if show:
        candies.show()
    if save is not None:
        candies.save(save)


__all__ = ["list_"]
