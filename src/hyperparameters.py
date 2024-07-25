from dataclasses import dataclass, asdict, fields, is_dataclass

import yaml


@dataclass
class Hyperparameters:
    """
    Base class for all hyperparameters classes.
    Prevents typos by not allowing new attributes to be set (after init).
    """

    def __post_init__(self):
        self._attrs_frozen = True

    @property
    def as_dict(self):
        return {
            field.name: getattr(self, field.name) for field in fields(self)
        }  # Shallow copy

    @property
    def as_deep_dict(self):
        return asdict(self)  # Converts nested HPs to dicts as well.

    @property
    def currently_initializing(self):
        return not hasattr(self, "_attrs_frozen") or not self._attrs_frozen

    def __setattr__(self, key, value):
        if self.currently_initializing:
            super().__setattr__(key, value)

        if not hasattr(self, key):
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {key}! "
                f"Options: {[field.name for field in fields(self)]}."
            )
        super().__setattr__(key, value)


def save_hp_to_file(hp: Hyperparameters, path: str):
    hp_dict = asdict(hp)
    with open(path, "w") as f:
        yaml.dump(hp_dict, f)


def load_hp_from_file(path: str, hp: Hyperparameters):
    with open(path, "r") as f:
        hp_dict = yaml.load(f, Loader=yaml.SafeLoader)
    load_hp_from_dict(hp_dict, hp)


def load_hp_from_dict(hp_dict: dict, hp: Hyperparameters):
    for key, value in hp.as_dict.items():
        if key not in hp_dict:
            raise ValueError(f"Missing key {key} in hp_dict.")

        if is_dataclass(value):
            load_hp_from_dict(hp_dict[key], value)
        else:
            setattr(hp, key, hp_dict[key])
