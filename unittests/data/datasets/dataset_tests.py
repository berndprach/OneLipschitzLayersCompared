from typing import Protocol, Any


class DatasetTest(Protocol):
    ds: Any


def test_channel_means_in_01(test_cls, ds):
    test_cls.assertEqual(len(ds.channel_means), 3)
    for mean in ds.channel_means:
        test_cls.assertTrue(0. <= mean <= 1.)


def test_shapes(test_cls, ds, goal_x_shape=(3, 32, 32)):
    for partition in ds.partitions:
        x0, y0 = partition[0]
        test_cls.assertEqual(x0.shape, goal_x_shape)
        test_cls.assertIsInstance(y0, int)
