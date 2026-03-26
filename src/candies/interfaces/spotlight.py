from datetime import timedelta
from candies import OptDeps

if OptDeps.SHAZAM.installed:

    from pathlib import Path
    from dataclasses import dataclass

    import pytz
    import numpy as np
    import astropy.units as uzi
    from astropy.time import Time
    from shazam.core import FRBRing
    from typing_extensions import Self
    from astropy.coordinates import TETE, SkyCoord

    from candies.base import Candy, Slice
    from candies.interfaces.base import LiveInterface

    @dataclass
    class SPOTLIGHTLive(LiveInterface):

        ring: FRBRing

        @classmethod
        def load(cls) -> Self:
            ring = FRBRing()
            ring.open("r")

            hdr = ring.header()
            curblk = ring.curblk
            curtime = ring.timestamps[curblk]
            reftime = curtime - timedelta(seconds=curblk * ring.blktime)
            hdr["mjd"] = Time(
                pytz.timezone("Asia/Kolkata")
                .localize(reftime, is_dst=None)
                .astimezone(pytz.utc)
            ).mjd

            return cls(
                ring=ring,
                df=ring.df,
                dt=ring.dt,
                fh=ring.fh,
                nf=ring.nf,
                extras=hdr,
                nt=ring.blksamps,
                nbits=ring.nbits,
            )

        def slice(self, candy: Candy) -> Slice:
            width = candy.wbin * self.dt
            maxdelay = 4.1488064239e3 * candy.dm * (self.fl**-2 - self.fh**-2)
            tbeg, tend = candy.t0 - maxdelay - width, candy.t0 + maxdelay + width
            data = np.asarray(self.ring.getslice(tbeg=tbeg, tend=tend, beam=candy.beam))
            data = np.ascontiguousarray(data.T)
            nf, nt = data.shape

            hdr = self.extras
            hdr["begmjd"] = hdr["mjd"] + (tbeg * getattr(uzi, "s")).to("day").value
            hdr["endmjd"] = hdr["mjd"] + (tend * getattr(uzi, "s")).to("day").value
            hdr["mjd"] = hdr["mjd"] + (candy.t0 * getattr(uzi, "s")).to("day").value

            radians = getattr(uzi, "rad")
            ra = self.ring.beamras[candy.beam]
            dec = self.ring.beamdecs[candy.beam]
            coords = SkyCoord(
                ra * radians,
                dec * radians,
                frame=TETE(obstime=Time(hdr["mjd"], format="mjd")),
            ).transform_to("icrs")
            rah, ram, ras = getattr(coords.ra, "hms")
            decd, decm, decs = getattr(coords.dec, "dms")
            hdr["raj2000"] = f"{rah}h{ram}m{ras}s"
            hdr["decj2000"] = f"{decd}d{decm}m{decs}s"

            return Slice(
                nf=nf,
                nt=nt,
                tbeg=tbeg,
                tend=tend,
                data=data,
                extras=hdr,
                fh=self.fh,
                fl=self.fl,
                df=self.df,
                dt=self.dt,
                nbits=self.nbits,
                fn=Path(f"{candy.id}.highres.h5"),
            )
