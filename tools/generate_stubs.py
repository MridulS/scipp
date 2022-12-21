# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""
Generate stub files for the C++ components of scipp.

This script requires pybind11-stubgen to be installed and on the PATH
https://github.com/sizmailov/pybind11-stubgen
"""

import argparse
import ast
import inspect
import os
import re
import subprocess
from contextlib import contextmanager
from tempfile import TemporaryDirectory

import scipp as sc

PREAMBLE = '''# ATTENTION
# This file was generated by tools/generate_stubs.py
from __future__ import annotations
from pathlib import Path
import typing
import numpy
from ..typing import MetaDataMap, VariableLike
_Shape = typing.Tuple[int, ...]
'''


@contextmanager
def generate_pybind11_stubs():
    with TemporaryDirectory() as out_dir:
        subprocess.check_call([
            'pybind11-stubgen', 'scipp._scipp', '-o', out_dir, '--no-setup-py',
            '--ignore-invalid=all'
        ])
        with open(os.path.join(out_dir, 'scipp/_scipp-stubs/core/__init__.pyi'),
                  'r') as f:
            yield f


def format_all_dunder(names):
    return '__all__ = [\n    ' + ',\n    '.join('"' + name + '"'
                                                for name in names) + '\n]'


def format_stubs(stubs):
    return '\n'.join(stubs).replace('scipp._scipp.core.', '')


def signature_for_method(func):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        # Not a callable
        return None
    params = list(sig.parameters.values())
    if params[0].kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
        params[0] = params[0].replace(name='self', annotation=inspect.Parameter.empty)
    return sig.replace(parameters=params)


def replace_name_qualifications(s):
    typing_repl = ('Callable', 'Dict', 'Iterable', 'List', 'Optional', 'Sequence',
                   'Tuple', 'Union')
    for name in typing_repl:
        s = re.sub(rf'([^a-zA-Z0-9_.]){name}', repl=rf'\1typing.{name}', string=s)
    return s.replace('_cpp.', '').replace('scipp._scipp.core.',
                                          '').replace('NoneType', 'None')


def append_dynamic_functions(stub, class_node):
    try:
        cls = getattr(sc, class_node.name)
    except AttributeError:
        # Class is not exposed in scipp/__init__.py
        return stub

    statically_bound_methods = [
        node.name for node in ast.iter_child_nodes(class_node)
        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')
    ]
    for member in filter(
            lambda x: not x.startswith('_') and x not in statically_bound_methods,
            dir(cls)):
        if (sig := signature_for_method(getattr(cls, member))) is None:
            continue
        sig_str = replace_name_qualifications(str(sig))
        # Fixed indentation => this breaks for nested classes.
        stub += f'\n    def {member}{sig_str}: ...'

    return stub


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str, help='Output stub file')
    return parser.parse_args()


def main():
    args = command_line_args()
    with generate_pybind11_stubs() as stub_file:
        full_stub = stub_file.read()

    class_stubs = []
    class_names = []
    module_tree = ast.parse(full_stub)
    for class_node in filter(
            lambda node: not node.name.startswith('_'),
            filter(lambda node: isinstance(node, ast.ClassDef),
                   ast.iter_child_nodes(module_tree))):
        stub = ast.get_source_segment(full_stub, class_node)
        class_stubs.append(append_dynamic_functions(stub, class_node))
        class_names.append(class_node.name)

    with open(args.outfile, 'w') as f:
        f.write(PREAMBLE + '\n' + format_all_dunder(class_names) + '\n' +
                format_stubs(class_stubs))


if __name__ == '__main__':
    main()
