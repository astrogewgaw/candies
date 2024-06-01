"""
The application code for Candies.
"""

import random
import cyclopts
from pathlib import Path
from typing import Optional
from rich.table import Table
from rich.progress import track
from rich.console import Console
from candies.features import featurize
from candies.base import CandiesError, Candidate, CandidateList

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
    verbose: bool = False,
    show_progress: bool = True,
):
    """
    Make features for all candy-date(s).

    Parameters
    ----------
    candlist: str
        List of candidates as a CSV file.
    fil: str | Path, optional
        The name of the filterbank file to process.
    gpuid: int, optional
        Specify the GPU to use by its ID.
    save: bool, optional
        Save the candidates to disk after feature creation.
    zoom: bool, optional
        Zoom in to the DMT using a simple, automatic method.
    fudging: int, optional
        The factor to zoom in by.
    verbose: bool, optional
        Activate verbose printing. False by default.
    show_progress: bool, optional
        Show the progress bar. True by default.
    """
    candidates = CandidateList.fromcsv(candlist)
    random.shuffle(candidates)

    if fil is None:
        fil, = set([_.fname for _ in candidates])
        if fil is None:
            raise CandiesError("No filterbank file given to process!")

    featurize(
        candidates=candidates,
        filterbank=fil,
        gpuid=gpuid,
        save=save,
        zoom=zoom,
        fudging=fudging,
        verbose=verbose,
        progressbar=show_progress,
    )


@app.command
def list_(
    candfiles: list[str | Path],
    show: bool = False,
    save: bool = True,
    saveto: str = "candidates.csv",
):
    """
    List all candy-date(s).

    Parameters
    ----------
    candfiles: list[str | Path]
        List of candy-date files.
    show: bool, optional
        Show list of candidates as a table.
    save: bool, optional
        Save list of candidates to a CSV file.
    saveto: str, optional
        Name of the CSV file to save the candidates to.
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
        candidates.tocsv(saveto)


@app.command
def plot(
    candfiles: list[str | Path],
    show: bool = False,
    save: bool = True,
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
        mjd = (
            candidate.extras.get("tstart", None)
            if candidate.extras is not None
            else None
        )
        candidate.plot(
            save,
            show,
            saveto="".join(
                [
                    f"MJD{mjd:.7f}_" if mjd is not None else "",
                    f"T{candidate.t0:.7f}_",
                    f"DM{candidate.dm:.5f}_",
                    f"SNR{candidate.snr:.5f}",
                ]
            )
            + ".png",
        )


if __name__ == "__main__":
    app()
