import numpy as np

from candies.base import Candy, Slice
from candies.featurize import Featurizer


def test_cpu(datadir):
    sliced = Slice.load(datadir / "test.highres.h5")
    candy = Candy(
        wbin=1,
        dm=26.8,
        snr=1248.54053,
        t0=sliced.nt // 2 * sliced.dt,
    )
    candy.sliced = sliced
    candy.extras = sliced.extras
    candy = Featurizer["CPU"](candy)()
    refcandy = Candy.load(datadir / "test.h5")

    assert refcandy.dedispersed is not None
    assert refcandy.dmtransform is not None
    assert np.allclose(candy.dedispersed.data, refcandy.dedispersed.data)
    assert np.allclose(candy.dmtransform.data, refcandy.dmtransform.data)
