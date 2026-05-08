from candies.base import Candy
from candies.classify import classify


def test_cpu(datadir):
    candy = Candy.load(datadir / "test.h5")
    candy = classify(candies=candy)[0]
    assert candy.probability > 0.5
    assert candy.label
