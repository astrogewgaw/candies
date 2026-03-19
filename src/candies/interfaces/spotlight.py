from candies import OptDeps

if OptDeps.SHAZAM.installed:

    from dataclasses import dataclass

    import numpy as np
    from shazam.core import FRBRing
    from typing_extensions import Self

    from candies.base import Candy
    from candies.functions import dm2delay
    from candies.interfaces.base import LiveInterface

    @dataclass
    class SPOTLIGHTLive(LiveInterface):

        ring: FRBRing

        @classmethod
        def load(cls) -> Self:
            ring = FRBRing()
            ring.open("r")
            return cls(
                ring=ring,
                nf=ring.nf,
                df=ring.df,
                dt=ring.dt,
                fh=ring.fh,
                nt=ring.blksamps,
                nbits=ring.nbits,
                extras=ring.header(),
            )

        def slice(self, candy: Candy) -> tuple[float, float, np.ndarray]:
            width = candy.wbin * self.dt
            maxdelay = dm2delay(f=self.fl, f0=self.fh, dm=candy.dm)
            tbeg, tend = candy.t0 - maxdelay - width, candy.t0 + maxdelay + width
            return (
                tbeg,
                tend,
                np.ascontiguousarray(
                    np.asarray(
                        self.ring.getslice(
                            tbeg=tbeg,
                            tend=tend,
                            beam=candy.extras.get("beam", 0),
                        )
                    ).T
                ),
            )
