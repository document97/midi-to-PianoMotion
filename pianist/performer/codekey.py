"""A cache key for a computation: the source of the code it can reach from its entry points, and the module constants
that code names. An edit elsewhere in the same files (a wrist path constant, another stage's function) leaves a cached
result valid; an edit to anything the computation can run invalidates it.

Reach is followed statically through names (globals, imports from the project's packages, classes as a whole); an
attribute call on an object (``solver.solve``) is covered by naming the object's class among the entry points."""
import dis
import hashlib
import importlib
import inspect
import sys
import types

import numpy as np

PACKAGES = ('performer', 'planner')


def _module(obj):
    """The real module name of a function or class, from its module's spec (``performer.pipeline`` run as __main__ too)."""
    g = getattr(obj, '__globals__', None)
    if g is None:
        g = vars(sys.modules[obj.__module__]) if obj.__module__ in sys.modules else {}
    spec = g.get('__spec__')
    return spec.name if spec is not None else (getattr(obj, '__module__', None) or '')


def _ours(obj):
    return _module(obj).split('.')[0] in PACKAGES


def _codes(code):
    yield code
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            yield from _codes(c)


def _constant(value):
    """Bytes of a module constant, or None when it is not data."""
    if isinstance(value, np.ndarray):
        return value.dtype.str.encode() + repr(value.shape).encode() + np.ascontiguousarray(value).tobytes()
    if isinstance(value, (bool, int, float, str, bytes, type(None))):
        return repr(value).encode()
    if isinstance(value, (tuple, list, frozenset, set)):
        parts = [_constant(v) for v in (sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value)]
        return None if any(p is None for p in parts) else b'(' + b','.join(parts) + b')'
    if isinstance(value, dict):
        parts = []
        for k in sorted(value, key=repr):
            a, b = _constant(k), _constant(value[k])
            if a is None or b is None:
                return None
            parts.append(a + b':' + b)
        return b'{' + b','.join(parts) + b'}'
    return None


def _local_imports(fn):
    """{name: object} imported inside fn (``from .pose import WIDE_SEEDS``)."""
    out = {}
    package = _module(fn).rpartition('.')[0]
    for code in _codes(fn.__code__):
        level, module = 0, None
        for ins in dis.get_instructions(code):
            if ins.opname == 'LOAD_CONST' and isinstance(ins.argval, int):
                level = ins.argval
            elif ins.opname == 'IMPORT_NAME':
                name = ins.argval
                try:
                    if level:
                        base = package.rsplit('.', level - 1)[0] if level > 1 else package
                        module = importlib.import_module(('.' + name) if name else '.', base) if name else importlib.import_module(base)
                    else:
                        module = importlib.import_module(name)
                except ImportError:
                    module = None
            elif ins.opname == 'IMPORT_FROM' and module is not None and hasattr(module, ins.argval):
                out[ins.argval] = getattr(module, ins.argval)
    return out


def code_key(*roots, extra=b''):
    """Hex digest over everything reachable from ``roots`` (functions, classes)."""
    digest = hashlib.sha256(extra)
    seen, todo = set(), list(roots)
    while todo:
        obj = todo.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, type):
            digest.update(f'class {_module(obj)}.{obj.__qualname__}:'.encode() + inspect.getsource(obj).encode())
            functions = [v for v in vars(obj).values() if isinstance(v, (types.FunctionType, staticmethod, classmethod, property))]
            functions = [getattr(f, '__func__', None) or getattr(f, 'fget', None) or f for f in functions]
            # dataclass defaults and class constants
            for name, v in sorted(vars(obj).items()):
                b = _constant(v)
                if b is not None:
                    digest.update(f'{_module(obj)}.{obj.__qualname__}.{name}='.encode() + b)
            for base in obj.__mro__[1:]:
                if _ours(base):
                    todo.append(base)
        else:
            functions = [obj]
        for fn in functions:
            if not isinstance(fn, types.FunctionType):
                continue
            try:
                source = inspect.getsource(fn).encode()
            except (OSError, TypeError):           # generated (a dataclass's __init__): its class's source covers it
                continue
            digest.update(f'{_module(fn)}.{fn.__qualname__}:'.encode() + source)
            names = {n for c in _codes(fn.__code__) for n in c.co_names}
            g = dict(fn.__globals__)
            g.update(_local_imports(fn))
            for n in sorted(names):
                if n not in g:
                    continue
                v = g[n]
                if isinstance(v, (types.FunctionType, type)) and _ours(v):
                    todo.append(v)
                elif not isinstance(v, types.ModuleType):
                    b = _constant(v)
                    if b is not None:
                        digest.update(f'{_module(fn)}.{n}='.encode() + b)
        if isinstance(obj, type):
            digest.update(f'class {_module(obj)}.{obj.__qualname__}'.encode())
    return digest.hexdigest()
