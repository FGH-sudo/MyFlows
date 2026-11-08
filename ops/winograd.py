"""F(2x2,3x3) cross-correlation and its adjoints. Tiles share the ordinary NCHW input and OIHW weights."""

from dataclasses import dataclass

from ..core.device import xp


def _axis(array, axis, transform):
    v = xp.moveaxis(array, axis, 0)
    if transform == "input":  # B^T
        rows = (v[0] - v[2], v[1] + v[2], v[2] - v[1], v[1] - v[3])
    elif transform == "filter":  # G
        ends = v[0] + v[2]
        rows = (v[0], (ends + v[1]) * 0.5, (ends - v[1]) * 0.5, v[2])
    elif transform == "output":  # A^T
        rows = (v[0] + v[1] + v[2], v[1] - v[2] - v[3])
    elif transform == "output_grad":  # A
        rows = (v[0], v[0] + v[1], v[0] - v[1], -v[1])
    elif transform == "filter_grad":  # G^T
        half = (v[1] + v[2]) * 0.5
        rows = (v[0] + half, (v[1] - v[2]) * 0.5, half + v[3])
    elif transform == "input_grad":  # B
        rows = (v[0], v[1] - v[2] + v[3], v[1] + v[2] - v[0], -v[3])
    else:
        raise ValueError(transform)
    return xp.moveaxis(xp.stack(rows), 0, axis)


def _transform(array, name):
    return _axis(_axis(array, 0, name), 1, name)


@dataclass
class WinogradContext:
    shape: tuple
    padding: tuple
    filters: object
    inputs: object


def forward(x, weight, padding, *, need_dx, need_dw):
    n, ci, h, w = x.shape
    co = weight.shape[0]
    ph, pw = padding
    oh, ow = h + 2 * ph - 2, w + 2 * pw - 2
    if min(oh, ow) < 1:
        raise ValueError("3x3 filter exceeds the padded input")
    th, tw = (oh + 1) // 2, (ow + 1) // 2
    padded = xp.pad(x, ((0, 0), (0, 0), (ph, ph + oh % 2), (pw, pw + ow % 2)))
    patches = xp.stack(
        [xp.stack([padded[:, :, i : i + 2 * th : 2, j : j + 2 * tw : 2] for j in range(4)]) for i in range(4)]
    )
    v = _transform(patches, "input").transpose(0, 1, 3, 2, 4, 5).reshape(4, 4, ci, n * th * tw)
    u = _transform(weight.transpose(2, 3, 0, 1), "filter")
    tiles = _transform(u @ v, "output").reshape(2, 2, co, n, th, tw)
    y = tiles.transpose(3, 2, 4, 0, 5, 1).reshape(n, co, 2 * th, 2 * tw)
    return y[:, :, :oh, :ow], WinogradContext(x.shape, padding, u if need_dx else None, v if need_dw else None)


def backward(dy, ctx):
    n, ci, h, w = ctx.shape
    ph, pw = ctx.padding
    _, co, oh, ow = dy.shape
    th, tw = (oh + 1) // 2, (ow + 1) // 2
    padded = xp.pad(dy, ((0, 0), (0, 0), (0, oh % 2), (0, ow % 2)))
    tiles = padded.reshape(n, co, th, 2, tw, 2).transpose(3, 5, 1, 0, 2, 4).reshape(2, 2, co, n * th * tw)
    dm = _transform(tiles, "output_grad")
    dx = dw = None
    if ctx.inputs is not None:
        du = dm @ ctx.inputs.swapaxes(-1, -2)
        dw = _transform(du, "filter_grad").transpose(2, 3, 0, 1)
    if ctx.filters is not None:
        dv = ctx.filters.swapaxes(-1, -2) @ dm
        pieces = _transform(dv, "input_grad").reshape(4, 4, ci, n, th, tw).transpose(0, 1, 3, 2, 4, 5)
        summed = xp.zeros((n, ci, 2 * th + 2, 2 * tw + 2), dtype=dy.dtype)
        for i in range(4):
            for j in range(4):
                summed[:, :, i : i + 2 * th : 2, j : j + 2 * tw : 2] += pieces[i, j]
        dx = summed[:, :, ph : ph + h, pw : pw + w]
    return dx, dw
