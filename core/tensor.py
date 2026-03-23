import numpy as np


def _to_ndarray(data, dtype=None, copy=False):
    if isinstance(data, Tensor):
        array = data.data
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        elif copy:
            array = array.copy()
        return array.copy() if copy and dtype is not None else array

    array = np.asarray(data, dtype=dtype)
    if copy:
        array = array.copy()
    return array


class Tensor:
    """A thin ndarray wrapper used as the framework's unified tensor type."""

    __array_priority__ = 1000

    def __init__(self, data, dtype=None, copy=False):
        self.data = _to_ndarray(data, dtype=dtype, copy=copy)

    @classmethod
    def ensure(cls, data, dtype=None, copy=False):
        if isinstance(data, cls) and dtype is None and not copy:
            return data
        return cls(data, dtype=dtype, copy=copy)

    @classmethod
    def zeros_like(cls, other):
        return cls(np.zeros_like(cls.ensure(other).data))

    @classmethod
    def ones_like(cls, other):
        return cls(np.ones_like(cls.ensure(other).data))

    @classmethod
    def zeros(cls, shape, dtype=np.float64):
        return cls(np.zeros(shape, dtype=dtype))

    def numpy(self, copy=False):
        return self.data.copy() if copy else self.data

    def item(self):
        return self.data.item()

    def astype(self, dtype, copy=True):
        return Tensor(self.data.astype(dtype, copy=copy))

    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape))

    def transpose(self, *axes):
        return Tensor(self.data.transpose(*axes))

    @property
    def T(self):
        return Tensor(self.data.T)

    def copy(self):
        return Tensor(self.data.copy())

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        value = self.data[key]
        return Tensor(value) if isinstance(value, np.ndarray) else value

    def __setitem__(self, key, value):
        self.data[key] = _to_ndarray(value)

    def __repr__(self):
        return f"Tensor({self.data!r})"

    def __float__(self):
        return float(self.data)

    def __int__(self):
        return int(self.data)

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __add__(self, other):
        return Tensor(self.data + _to_ndarray(other))

    def __radd__(self, other):
        return Tensor(_to_ndarray(other) + self.data)

    def __sub__(self, other):
        return Tensor(self.data - _to_ndarray(other))

    def __rsub__(self, other):
        return Tensor(_to_ndarray(other) - self.data)

    def __mul__(self, other):
        return Tensor(self.data * _to_ndarray(other))

    def __rmul__(self, other):
        return Tensor(_to_ndarray(other) * self.data)

    def __truediv__(self, other):
        return Tensor(self.data / _to_ndarray(other))

    def __rtruediv__(self, other):
        return Tensor(_to_ndarray(other) / self.data)

    def __matmul__(self, other):
        return Tensor(self.data @ _to_ndarray(other))

    def __rmatmul__(self, other):
        return Tensor(_to_ndarray(other) @ self.data)

    def __pow__(self, power):
        return Tensor(self.data ** _to_ndarray(power))

    def __rpow__(self, base):
        return Tensor(_to_ndarray(base) ** self.data)

    def __neg__(self):
        return Tensor(-self.data)

    def __pos__(self):
        return Tensor(+self.data)

    def __eq__(self, other):
        return Tensor(self.data == _to_ndarray(other))

    def __ne__(self, other):
        return Tensor(self.data != _to_ndarray(other))

    def __lt__(self, other):
        return Tensor(self.data < _to_ndarray(other))

    def __le__(self, other):
        return Tensor(self.data <= _to_ndarray(other))

    def __gt__(self, other):
        return Tensor(self.data > _to_ndarray(other))

    def __ge__(self, other):
        return Tensor(self.data >= _to_ndarray(other))
