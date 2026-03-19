from candies.base import OptionalDependencies

from candies.interfaces.gmrt import GMRTFile
from candies.interfaces.sigproc import SIGPROCFile
from candies.interfaces.base import Interface, FileInterface, LiveInterface

__all__ = [
    "GMRTFile",
    "Interface",
    "SIGPROCFile",
    "FileInterface",
    "LiveInterface",
]

if OptionalDependencies.SHAZAM.installed:
    from candies.interfaces.spotlight import SPOTLIGHTLive

    _ = SPOTLIGHTLive
    __all__.append("SPOTLIGHTLive")
