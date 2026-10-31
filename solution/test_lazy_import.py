"""Fresh-process import guard checks against installed bitsandbytes 0.50.2.

Run this file separately from tests that import BNB before test collection.
"""
from pathlib import Path
import inspect
import sys
import types
from unittest.mock import patch

import pytest
import torch
from solution import lazy_bnb_compile as lazy


def test_real_import_delegation_without_package_source_reads():
    if 'bitsandbytes' in sys.modules:
        pytest.skip('Run this file separately to audit fresh import admission')
    original = torch.compile  # lint: ok - dependency compiler identity, not original kernels
    match = lazy._matches
    accepted_names = []

    def audited(fn, args, opts, admission, caller, owner, seen):
        accepted = match(fn, args, opts, admission, caller, owner, seen)
        if accepted:
            accepted_names.append(fn.__name__)
            assert not match(fn, (object(),), opts, admission, caller, owner, seen)
            assert not match(fn, args, dict(opts, disable=False), admission, caller, owner, seen)
            assert not match(fn, args, opts, admission, caller, owner + 1, seen)
            assert not match(fn, args, opts, admission, caller, owner, {fn.__name__})
            if opts:
                assert not match(fn, args, {'dynamic': 0}, admission, caller, owner, seen)
            wrong = types.SimpleNamespace(f_globals=caller.f_globals,
                                          f_code=audited.__code__, f_locals=caller.f_locals)
            assert not match(fn, args, opts, admission, wrong, owner, seen)
        return accepted

    read_bytes = Path.read_bytes

    def no_package_source(path, *args, **kwargs):
        assert not (path.suffix == '.py' and 'site-packages' in path.parts), str(path)
        return read_bytes(path, *args, **kwargs)

    with patch.object(lazy, '_matches', audited), patch.object(Path, 'read_bytes', no_package_source):
        with lazy.install(enabled=True):
            import bitsandbytes.backends.default.ops as ops
    assert set(accepted_names) == set(lazy._OPTIONS)
    assert torch.compile is original  # lint: ok - verifies restoration of dependency compiler
    records = lazy.stats()['functions']
    assert len(records) == 4
    assert all(r['calls'] == r['constructions'] == 0 for r in records)
    for name in lazy._OPTIONS:
        fn = getattr(ops, name)
        assert fn.__name__ == name and fn.__module__ == lazy._MODULE
        assert inspect.signature(fn) == inspect.signature(fn.__wrapped__)
    with lazy.install(enabled=True):
        assert torch.compile is original  # lint: ok - verifies pre-imported no-op path
    assert 'already imported' in lazy.stats()['status']
