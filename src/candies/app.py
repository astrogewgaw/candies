"""
The application code for candies.
"""

from pathlib import Path

import cyclopts
from rich.table import Table
from rich.progress import track
from rich.console import Console
from joblib import Parallel, delayed

from candies.features import featurize
from candies.interfaces import SIGPROCFilterbank
from candies.base import Candidate, CandidateList

app = cyclopts.App()
app["--help"].group = "Admin"
app["--version"].group = "Admin"


@app.command
def make(
    candlist: str,
    /,
    fil: str | Path | None = None,
    gpuid: int = 0,
    save: bool = True,
    zoom: bool = True,
    fudging: int = 512,
    show_progress: bool = True,
):
    """
    Make features for all candy-date(s).

    Parameters
    ----------
    candlist: str
        List of candy-dates as a CSV file.
    fil: str | Path, optional
        The name of the file to process.
    gpuid: int, optional
        Specify the GPU to use by its ID.
    save: bool, optional
        Save the candy-dates to disk after feature creation.
    zoom: bool, optional
        Zoom in to the DMT using a simple, automatic method.
    fudging: int, optional
        The factor to zoom in by.
    verbose: bool, optional
        Activate verbose printing. False by default.
    show_progress: bool, optional
        Show the progress bar. True by default.
    """
    candidates = CandidateList.from_csv(candlist)
    groups = candidates.to_df().groupby("file")
    for fname, group in groups:
        featurize(
            CandidateList.from_df(group),
            str(fname) if fil is None else fil,
            save=save,
            zoom=zoom,
            gpuid=gpuid,
            fudging=fudging,
            progressbar=show_progress,
        )


@app.command
def store(
    candlist: str,
    /,
    njobs: int = 1,
    show_progress: bool = True,
    fil: str | Path | None = None,
):
    """
    Store candy-date(s) from a file.

    Parameters
    ----------
    candlist: str
        List of candy-dates as a CSV file.
    fil: str | Path, optional
        The name of the file to extract candy-dates from.
    show_progress: bool, optional
        Show the progress bar. True by default.
    """
    candidates = CandidateList.from_csv(candlist)
    groups = candidates.to_df().groupby("file")
    for fname, group in groups:
        candidates = CandidateList.from_df(group)
        with SIGPROCFilterbank(fil if fil is not None else str(fname)) as f:
            track(
                Parallel(
                    n_jobs=njobs,
                    return_as="generator",
                )(delayed(f.store)(candidate) for candidate in candidates),
                total=len(candidates),
                disable=show_progress,
            )


@app.command
def list_(
    candfiles: list[str | Path],
    /,
    show: bool = True,
    save: bool = False,
    saveto: str = "candidates.csv",
):
    """
    List all candy-date(s).

    Parameters
    ----------
    candfiles: list[str | Path]
        List of candy-date files.
    show: bool, optional
        Show list of candy-dates as a table.
    save: bool, optional
        Save list of candy-dates to a CSV file.
    saveto: str, optional
        Name of the CSV file to save the candy-dates to.
    """

    candidates = CandidateList([Candidate.load(candfile) for candfile in candfiles])

    if show:
        console = Console()
        table = Table(expand=False, padding=(0, 2, 0, 2))
        table.add_column("DM (in pc cm^-3)")
        table.add_column("Arrival time (in s)")
        table.add_column("SNR")
        table.add_column("Width (in bins)")
        for candidate in candidates:
            table.add_row(
                *[
                    f"{candidate.dm:.2f}",
                    f"{candidate.t0:.2f}",
                    f"{candidate.snr:.2f}",
                    f"{candidate.wbin:d}",
                ]
            )
        console.print(table)
    if save:
        candidates.to_csv(saveto)


@app.command
def plot(
    candfiles: list[str | Path],
    /,
    dpi: int = 96,
    save: bool = True,
    show: bool = False,
    saveto: str | None = None,
    show_progress: bool = True,
):
    """
    Plot all candy-date(s).

    Parameters
    ----------
    candfiles: list[str | Path]
        List of candy-date files.
    show: bool, optional
        Show plot.
    save: bool, optional
        Save plot to disk.
    show_progress: bool, optional
        Show the progress bar.
    """
    for candfile in track(
        candfiles,
        disable=(not show_progress),
        description="Plotting...",
    ):
        candidate = Candidate.load(candfile)
        candidate.plot(
            dpi=dpi,
            save=save,
            show=show,
            saveto="".join([str(candidate), ".png"]) if saveto is None else saveto,
        )


if __name__ == "__main__":
    app()
