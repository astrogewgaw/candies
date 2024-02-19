import random
import cyclopts
from pathlib import Path
from candies.features import featurize
from candies.base import Candidate, CandidateList

app = cyclopts.App()
app["--help"].group = "Admin"
app["--version"].group = "Admin"


@app.command
def make(
    candlist: str,
    /,
    fil: str | Path,
    gpuid: int = 0,
    save: bool = True,
    zoom: bool = True,
    zoomfact: int = 512,
):
    """
    Make features for all candy-date(s).

    Parameters
    ----------
    gpuid: int
        Specify the GPU to use by its ID.
    save: bool
        Save the candidates to disk after feature creation.
    zoom: bool
        Zoom in to the DMT using a simple, automatic method.
    zoomfact: int
        The factor to zoom in by.
    """
    candidates = CandidateList.fromcsv(candlist)
    random.shuffle(candidates)
    featurize(
        candidates=candidates,
        filterbank=fil,
        gpuid=gpuid,
        save=save,
        zoom=zoom,
        zoomfact=zoomfact,
    )


@app.command
def plot(candidates: list[str | Path], show: bool = False, save: bool = True):
    """
    Plot all candy-date(s).

    Parameters
    ----------
    candidates: list[str | Path]
        List of candy-date files.
    show: bool
        Show plot.
    save: bool
        Save plot to disk.
    """
    for candidate in candidates:
        Candidate.load(candidate).plot(save, show)


if __name__ == "__main__":
    app()
