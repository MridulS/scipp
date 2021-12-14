# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock, Jan-Lukas Wynen

from fractions import Fraction
from typing import Dict, Iterable, List, Mapping, Set, Union

from ..core import DataArray, Dataset, NotFoundError, VariableError, bins
from ..logging import get_logger
from .coord_table import Coord, CoordTable, Destination
from .graph import GraphDict, RuleGraph, rule_sequence
from .options import Options
from .rule import ComputeRule, FetchRule, RenameRule, Rule, rule_output_names


def transform_coords(x: Union[DataArray, Dataset],
                     targets: Union[str, Iterable[str]],
                     graph: GraphDict,
                     *,
                     rename_dims: bool = True,
                     keep_aliases: bool = True,
                     keep_intermediate: bool = True,
                     keep_inputs: bool = True) -> Union[DataArray, Dataset]:
    """Compute new coords based on transformations of input coords.

    :param x: Input object with coords.
    :param targets: Name or list of names of desired output coords.
    :param graph: A graph defining how new coords can be computed from existing
                  coords. This may be done in multiple steps.
                  The graph is given by a ``dict`` where:

                  - Dict keys are ``str`` or ``tuple`` of ``str``, defining the
                    names of outputs generated by a dict value.
                  - Dict values are ``str`` or a callable (function). If ``str``,
                    this is a synonym for renaming a coord. If a callable,
                    it must either return a single variable or a dict of
                    variables. The argument names of callables must be coords
                    in ``x`` or be computable by other nodes in ``graph``.
    :param rename_dims: Rename dimensions if the corresponding dimension coords
                        are used as inputs and there is a single output coord
                        that can be associated with that dimension.
                        See the user guide for more details and examples.
                        Default is True.
    :param keep_aliases: If True, aliases for coords defined in graph are
                         included in the output. Default is True.
    :param keep_intermediate: Keep attributes created as intermediate results.
                              Default is True.
    :param keep_inputs: Keep consumed input coordinates or attributes.
                        Default is True.
    :return: New object with desired coords. Existing data and meta-data is
             shallow-copied.

    :seealso: The section in the user guide on
     `Coordinate transformations <../../user-guide/coordinate-transformations.rst>`_
    """
    options = Options(rename_dims=rename_dims,
                      keep_aliases=keep_aliases,
                      keep_intermediate=keep_intermediate,
                      keep_inputs=keep_inputs)
    targets = {targets} if isinstance(targets, str) else set(targets)
    if isinstance(x, DataArray):
        return _transform_data_array(x,
                                     targets=targets,
                                     graph=RuleGraph(graph),
                                     options=options)
    else:
        return _transform_dataset(x,
                                  targets=targets,
                                  graph=RuleGraph(graph),
                                  options=options)


def show_graph(graph: GraphDict, size: str = None, simplified: bool = False):
    """
    Show graphical representation of a graph as required by
    :py:func:`transform_coords`

    Requires `python-graphviz` package.

    :param graph: Transformation graph to show.
    :param size: Size forwarded to graphviz, must be a string, "width,height"
                 or "size". In the latter case, the same value is used for
                 both width and height.
    :param simplified: If ``True``, do not show the conversion functions,
                       only the potential input and output coordinates.
    """
    return RuleGraph(graph).show(size=size, simplified=simplified)


def _transform_data_array(original: DataArray, targets: Set[str], graph: RuleGraph,
                          options: Options) -> DataArray:
    graph = graph.graph_for(original, targets)
    rules = rule_sequence(graph)
    working_coords = CoordTable(rules, targets, options)
    # _log_plan(rules, targets, dim_name_changes, working_coords)
    dim_coords = set()
    for rule in rules:
        for name, coord in rule(working_coords).items():
            working_coords.add(name, coord)
            if coord.has_dim(name):
                dim_coords.add(name)

    res = _store_results(original, working_coords, targets)
    dim_name_changes = (_dim_name_changes(graph, dim_coords)
                        if options.rename_dims else {})
    return res.rename_dims(dim_name_changes)


def _transform_dataset(original: Dataset, targets: Set[str], graph: RuleGraph, *,
                       options: Options) -> Dataset:
    # Note the inefficiency here in datasets with multiple items: Coord
    # transform is repeated for every item rather than sharing what is
    # possible. Implementing this would be tricky and likely error-prone,
    # since different items may have different attributes. Unless we have
    # clear performance requirements we therefore go with the safe and
    # simple solution
    return Dataset(
        data={
            name: _transform_data_array(
                original[name], targets=targets, graph=graph, options=options)
            for name in original
        })


def _log_plan(rules: List[Rule], targets: Set[str], dim_name_changes: Mapping[str, str],
              coords: CoordTable) -> None:
    inputs = set(rule_output_names(rules, FetchRule))
    byproducts = {
        name
        for name in (set(rule_output_names(rules, RenameRule))
                     | set(rule_output_names(rules, ComputeRule))) - targets
        if coords.total_usages(name) < 0
    }

    message = f'Transforming coords ({", ".join(sorted(inputs))}) ' \
              f'-> ({", ".join(sorted(targets))})'
    if byproducts:
        message += f'\n  Byproducts:\n    {", ".join(sorted(byproducts))}'
    if dim_name_changes:
        dim_rename_steps = '\n'.join(f'    {t} <- {f}'
                                     for f, t in dim_name_changes.items())
        message += '\n  Rename dimensions:\n' + dim_rename_steps
    message += '\n  Steps:\n' + '\n'.join(
        f'    {rule}' for rule in rules if not isinstance(rule, FetchRule))

    get_logger().info(message)


def _store_coord(da: DataArray, name: str, coord: Coord) -> None:
    def out(x):
        return x.coords if coord.destination == Destination.coord else x.attrs

    def del_other(x):
        try:
            if coord.destination == Destination.coord:
                del x.attrs[name]
            else:
                del x.coords[name]
        except NotFoundError:
            pass

    if coord.has_dense:
        out(da)[name] = coord.dense
        del_other(da)
    if coord.has_event:
        try:
            out(da.bins)[name] = coord.event
        except VariableError:
            # Thrown on mismatching bin indices, e.g. slice
            da.data = da.data.copy()
            out(da.bins)[name] = coord.event
        del_other(da.bins)


def _store_results(da: DataArray, coords: CoordTable, targets: Set[str]) -> DataArray:
    da = da.copy(deep=False)
    if da.bins is not None:
        da.data = bins(**da.bins.constituents)
    for name, coord in coords.items():
        if name in targets:
            coord.destination = Destination.coord
        _store_coord(da, name, coord)
    return da


def _color_dims(graph: RuleGraph,
                dim_coords: Set[str]) -> Dict[str, Dict[str, Fraction]]:
    graph = graph.dependency_graph

    colors = {
        coord: {dim: Fraction(0, 1)
                for dim in dim_coords}
        for coord in graph.nodes()
    }
    for dim in dim_coords:
        colors[dim][dim] = Fraction(1, 1)
        pending = [dim]
        while pending:
            coord = pending.pop()
            children = tuple(graph.children_of(coord))
            for child in children:
                # test for produced dim coords
                if child not in dim_coords:
                    colors[child][dim] += colors[coord][dim] * Fraction(
                        1, len(children))
            pending.extend(children)

    return colors


def _dim_name_changes(rule_graph: RuleGraph, dim_coords: Set[str]) -> Dict[str, str]:
    colors = _color_dims(rule_graph, dim_coords)
    nodes = list(rule_graph.dependency_graph.nodes_topologically())[::-1]
    name_changes = {}
    for dim in dim_coords:
        for node in nodes:
            c = colors[node]
            if all(f == 1 if d == dim else f != 1 for d, f in c.items()):
                name_changes[dim] = node
                break
    return name_changes
