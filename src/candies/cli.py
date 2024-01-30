"""
Code for the Candies CLI.
"""

import typer
from pathlib import Path
from candies.core import Candies
from typing_extensions import Annotated

# The Typer App.
app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def main():
    """
    [b]Sweet, sweet candy-dates![/b]
    """
    pass


@app.command()
def make(
    candidates: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            allow_dash=True,
            help="The list of candidates to process.",
        ),
    ],
    filterbank: Annotated[
        Path,
        typer.Option(
            "-f",
            "--fil",
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            allow_dash=True,
            help="The filterbank file to process.",
        ),
    ],
    ndms: Annotated[
        int,
        typer.Option(
            rich_help_panel="Configure",
            help="The number of DMs to use for the DM v/s time array.",
        ),
    ] = 256,
    device: Annotated[
        int,
        typer.Option(
            help="The GPU device ID.",
            rich_help_panel="Configure",
        ),
    ] = 0,
    save: Annotated[
        bool,
        typer.Option(
            rich_help_panel="Configure",
            help="If specified, save candidates to disk.",
        ),
    ] = True,
    zoom: Annotated[
        bool,
        typer.Option(
            rich_help_panel="Configure",
            help="If specified, employ the DMT zoom feature",
        ),
    ] = True,
    fudge: Annotated[
        float,
        typer.Option(
            help="A fudge factor for zooming into the DMT plane.",
            rich_help_panel="Configure",
        ),
    ] = 16,
):
    """
    Make features for all candy-dates.
    """
    Candies.wrap(candidates, filterbank).process(ndms, device, save, zoom, fudge)


@app.command()
def plot(
    candidates: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            allow_dash=True,
            help="The candidates to plot.",
        ),
    ],
    save: Annotated[
        bool,
        typer.Option(
            help="If specified, save plot to disk.",
        ),
    ] = True,
    show: Annotated[
        bool,
        typer.Option(
            help="If specified, show the plot.",
        ),
    ] = True,
):
    """
    Plot candy-date(s).
    """
    Candies.load(list(candidates)).plot(save, show)
