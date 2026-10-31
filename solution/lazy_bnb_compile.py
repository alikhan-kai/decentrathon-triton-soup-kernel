"""Opt-in import-only deferral for four version-pinned BNB 0.50.2 decorators.

The original BNB decorator still owns runtime fallback. No compiler flags,
function bodies, dispatch registrations, or torch installation files change.
"""
from contextlib import contextmanager
import functools
import importlib.machinery
import importlib.metadata
import importlib.util
import sys
import threading
import types

_MODULE = "bitsandbytes.backends.default.ops"
_OPTIONS = {
    "_dequantize_blockwise_compute": {"dynamic": False},
    "_dequantize_4bit_compute": {"dynamic": False},
    "_optimizer_precondition_32bit": {},
    "_optimizer_update_32bit": {},
}
_SCOPE_LOCK = threading.Lock()
_RECORDS = []
_STATUS = "disabled"


class _ConstructionFailed(Exception):
    """Let the unchanged BNB wrapper invoke its original body exactly once."""


class _Deferred:
    def __init__(self, compiler, fn, args, options):
        self.compiler, self.fn = compiler, fn
        self.args, self.options = args, dict(options)
        self.lock = threading.Lock()
        self.ready = False
        self.compiled = None
        self.calls = self.constructions = self.runtime_failures = 0
        self.constructor_failure = None
        self.last_runtime_failure = None

    def snapshot(self):
        with self.lock:
            return {"name": self.fn.__name__, "calls": self.calls,
                    "constructions": self.constructions,
                    "constructor_failure": self.constructor_failure,
                    "runtime_failures": self.runtime_failures,
                    "last_runtime_failure": self.last_runtime_failure,
                    "options": dict(self.options)}

    def __call__(self, *args, **kwargs):
        with self.lock:
            self.calls += 1
            if not self.ready:
                self.constructions += 1
                try:
                    self.compiled = self.compiler(self.fn, *self.args, **self.options)
                except Exception as exc:
                    # BNB's constructor failure is permanent. Raising this
                    # private signal on every call makes its existing wrapper
                    # call fn once, even if that eager body itself raises.
                    self.constructor_failure = type(exc).__name__
                self.ready = True
            compiled = self.compiled
            failed = self.constructor_failure is not None
        if failed:
            raise _ConstructionFailed() from None
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            # BNB retries compiled execution on the NEXT call after a runtime
            # failure; do not permanently replace compiled with the eager body.
            with self.lock:
                self.runtime_failures += 1
                self.last_runtime_failure = type(exc).__name__
            raise


def _defer(compiler, fn, args, options):
    record = _Deferred(compiler, fn, args, options)

    @functools.wraps(fn)
    def lazy(*a, **kw):
        return record(*a, **kw)

    lazy._torchdynamo_orig_callable = fn
    return lazy, record


def _code_children(code):
    return {c.co_name: c for c in code.co_consts if isinstance(c, types.CodeType)}


def _admit():
    """Admit by normal package metadata; never open installed package source."""
    if any(n == "bitsandbytes" or n.startswith("bitsandbytes.") for n in sys.modules):
        return None, "bitsandbytes already imported or importing"
    if threading.current_thread() is not threading.main_thread():
        return None, "import scope requires main thread"
    spec = importlib.util.find_spec("bitsandbytes")
    if (spec is None or type(spec.loader) is not importlib.machinery.SourceFileLoader
            or not spec.origin or not spec.submodule_search_locations
            or len(spec.submodule_search_locations) != 1):
        return None, "unsupported package loader"
    if importlib.metadata.version("bitsandbytes") != "0.50.2":
        return None, "distribution version mismatch"
    # Only names and loader metadata are inspected. The real loaded functions
    # and the real compiler are retained; no function body or compiler option changes.
    return (spec.origin.rsplit("/", 1)[0].rsplit("\\", 1)[0], dict(_OPTIONS)), "admitted"


def _matches(fn, args, options, admission, caller, owner, seen):
    if threading.get_ident() != owner or args or not isinstance(fn, types.FunctionType):
        return False
    name = fn.__name__
    if (name not in _OPTIONS or name in seen or options.keys() != _OPTIONS[name].keys()
            or any(options[k] is not v for k, v in _OPTIONS[name].items())
            or fn.__qualname__ != name or fn.__module__ != _MODULE):
        return False
    mod = sys.modules.get(_MODULE)
    spec = getattr(mod, "__spec__", None)
    if (spec is None or not getattr(spec, "_initializing", False)
            or type(spec.loader) is not importlib.machinery.SourceFileLoader
            or not spec.origin or fn.__globals__ is not vars(mod)
            or fn.__code__.co_filename != spec.origin):
        return False
    normalized = spec.origin.replace("\\", "/")
    package = admission[0].replace("\\", "/")
    if normalized != package + "/backends/default/ops.py":
        return False
    original_decorator = getattr(mod, "_try_torch_compile", None)
    # Check the actual active decorator frame against the loaded wrapper's
    # nested code object. This is an in-memory identity check, not a file read.
    decorator = (_code_children(original_decorator.__code__).get("decorator")
                 if isinstance(original_decorator, types.FunctionType) else None)
    return (decorator is not None and caller.f_globals is vars(mod)
            and caller.f_code is decorator and caller.f_locals.get("fn") is fn)


@contextmanager
def install(*, enabled=False):
    """Scope around apply's imports; disabled/preimported/unknown cases are no-ops.

    Only the importing thread's canonical decorator calls are intercepted.
    Calls from other threads and all unrelated calls forward verbatim. A
    concurrent importer owning BNB simply uses the original compiler. Nested
    scopes are no-ops. No background imports or global compiler options change.
    """
    global _STATUS
    if not enabled:
        yield
        return
    if not _SCOPE_LOCK.acquire(blocking=False):
        yield
        return
    try:
        try:
            admission, _STATUS = _admit()
        except Exception as exc:
            admission, _STATUS = None, "admission failed: " + type(exc).__name__
        if admission is None:
            yield
            return
        import torch
        original = torch.compile  # lint: ok - preserve dependency compiler; never credited as our kernel
        owner = threading.get_ident()
        seen = set()

        def selective(*args, **kwargs):
            if args and _matches(args[0], args[1:], kwargs, admission,
                                 sys._getframe(1), owner, seen):
                lazy, record = _defer(original, args[0], args[1:], kwargs)
                seen.add(args[0].__name__)
                _RECORDS.append(record)
                if len(seen) == len(_OPTIONS):
                    # The pinned source has exactly these four decorators.
                    # Restore before default.ops (and BNB) finishes importing.
                    if torch.compile is selective:  # lint: ok - preserve dependency compiler; never credited as our kernel
                        torch.compile = original  # lint: ok - preserve dependency compiler; never credited as our kernel
                return lazy
            return original(*args, **kwargs)

        torch.compile = selective  # lint: ok - preserve dependency compiler; never credited as our kernel
        try:
            yield
        finally:
            if torch.compile is selective:  # lint: ok - preserve dependency compiler; never credited as our kernel
                torch.compile = original  # lint: ok - preserve dependency compiler; never credited as our kernel
            _STATUS = "restored; deferred=" + str(len(seen))
    finally:
        _SCOPE_LOCK.release()


def stats():
    """Python counters only: no tensor inspection, device access, or compilation."""
    return {"status": _STATUS, "functions": [r.snapshot() for r in tuple(_RECORDS)]}
