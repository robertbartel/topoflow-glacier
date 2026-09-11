"""A file to hold classes relating to the model's context, or "state". This idea is related to how the internal variables of a model can be imported/exported/saved."""
from __future__ import annotations
from collections.abc import Iterable, Iterator

import numpy as np
from pydantic import BaseModel, ConfigDict

import logging
LOG = logging.getLogger("TFGLACR")

def _ensure(condition: bool, message: str) -> None:
    """
    Assert-like guard that logs a FATAL message before raising AssertionError.
    """
    if not condition:
        LOG.critical(message)  # FATAL log before asserting
        raise AssertionError(message)

class Var(BaseModel):
    """Context variable representation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    unit: str
    value: np.ndarray


class Context:
    """A class to hold a collection of variables representing model conditions"""

    def __init__(self, vars: Iterable[Var]):
        """Initialization Function"""
        self._name_mapping: dict[str, Var] = {var.name: var for var in vars}

    def unit(self, name: str) -> str:
        """Given a variable name, return its unit"""
        return self._name_mapping[name].unit

    def value(self, name: str) -> np.ndarray:
        """Given a variable name, return a value reference"""
        return self._name_mapping[name].value

    def value_at_indices(self, name: str, dest: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Copies the specific pointer values into a destination array"""
        # assert dest.shape[0] >= indices.shape[0], "dest smaller than indices"
        _ensure(dest.shape[0] >= indices.shape[0], "dest smaller than indices")
        src = self.value(name)
        for i in range(indices.shape[0]):
            value_index = indices[i]
            dest[i] = src[value_index]
        return dest

    def set_value(self, name: str, value: np.ndarray):
        """Sets the internal state value"""
        self._name_mapping[name].value[:] = value

    def set_value_at_indices(self, name: str, inds: np.ndarray, src: np.ndarray):
        """Sets the specific pointer values into a destination array"""
        # assert src.shape[0] >= inds.shape[0], "inds larger than src"
        _ensure(src.shape[0] >= inds.shape[0], "inds larger than src")
        arr = self.value(name)
        for i in range(inds.shape[0]):
            arr[inds[i]] = src[i]

    def serializable(self) -> dict[str, np.ndarray]:
        """Create a `dict` of var name-value pairs that is capable of being serialized with `pickle.dumps`.\n
        Currently assumes only the `value` attribute on `Var` needs to be saved."""
        return { name: var.value for name, var in self._name_mapping.items() }

    def load_serialized(self, data: dict[str, np.ndarray]):
        """Read a `dict` of var name-value pairs."""
        for name, value in data.items():
            self.set_value(name, value)

    def names(self) -> Iterable[str]:
        """Returns an iterator for all variables in the model state"""
        yield from self._name_mapping

    def vars(self) -> Iterable[Var]:
        """Returns all variables in the model state"""
        yield from self._name_mapping.values()

    def __contains__(self, name: str) -> bool:
        """Return if the variable name is present in the collection."""
        return name in self._name_mapping

    def __iter__(self) -> Iterator[Var]:
        """Returns an iterable of all variables"""
        return iter(self.vars())

    def __len__(self) -> int:
        """Returns the length of all names in the Model State"""
        return len(self._name_mapping)


def build_context(vars: Iterable[tuple[str, str]]) -> Context:
    """Builds the Context object to save model state

    Parameters
    ----------
    vars: Iterable[tuple[str, str]]
        The input variables with name, unit format

    Returns
    -------
    Context
        The completed state object
    """
    g = (Var(name=name, unit=unit, value=np.array([0.0], dtype=np.float64)) for (name, unit) in vars)
    return Context(vars=g)
