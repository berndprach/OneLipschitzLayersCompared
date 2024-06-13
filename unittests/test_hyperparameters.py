
import unittest
from dataclasses import dataclass
from typing import Any

import yaml

from src.hyperparameters import HP, load_hp_from_dict


@dataclass
class DummyHP(HP):
    d1: int = 1
    d2: float = .1


@dataclass
class OtherHP(HP):
    d1: Any = 1
    d2: Any = .1


@dataclass
class LastHP(HP):
    d1: Any
    d2: Any


class TestHyperparameters(unittest.TestCase):
    def test_default_values(self):
        dhp = DummyHP()
        self.assertEqual(dhp.d1, 1)
        self.assertEqual(dhp.d2, .1)

    def test_changing_values(self):
        dhp = DummyHP()
        dhp.d1 = 2
        self.assertEqual(dhp.d1, 2)

    def test_as_dict_property(self):
        dhp = DummyHP()
        self.assertEqual(dhp.as_dict, {'d1': 1, 'd2': .1})

    def test_typo_safety(self):
        dhp = DummyHP()
        with self.assertRaises(AttributeError):
            dhp.d3 = 3

    def test_nested_hp_to_string(self):
        dhp = DummyHP()
        dhp.d2 = DummyHP()
        goal_str = "DummyHP(d1=1, d2=DummyHP(d1=1, d2=0.1))"
        self.assertEqual(str(dhp), goal_str)

    def test_saving_and_loading(self):
        nested_hp = OtherHP([1, 2, 3], OtherHP(1., "abc"))
        hp_yaml = yaml.dump(nested_hp.as_deep_dict)

        empty_hp = OtherHP(None, OtherHP(None, None))
        self.assertNotEqual(nested_hp, empty_hp)

        hp_dict = yaml.load(hp_yaml, Loader=yaml.SafeLoader)
        load_hp_from_dict(hp_dict, empty_hp)

        self.assertEqual(nested_hp, empty_hp)

        self.assertEqual(empty_hp.d2.d2, "abc")
        self.assertIsInstance(empty_hp.d1, list)
        self.assertIsInstance(empty_hp.d2.d1, float)

    def test_hp_with_args(self):
        hp = LastHP(1, 2)
        self.assertEqual(hp.d1, 1)
        self.assertEqual(hp.d2, 2)
