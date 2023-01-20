import ast
import enum
import inspect
from string import Template
from typing import Iterable, List, Optional, Type

from .config import DISABLE_TYPE_CHECK_OVERRIDE, HEADER, INCLUDE_DOCS, TEMPLATE_FILE, \
    class_is_excluded
from .parse import parse_method, parse_property
from .transformer import RemoveDocstring, fix_method, fix_property


def _build_method(cls: Type[type], method_name: str) -> [ast.FunctionDef]:
    meth = inspect.getattr_static(cls, method_name)
    return [
        fix_method(m, cls_name=cls.__name__) for m in parse_method(meth, method_name)
    ]


def _build_property(cls: Type[type], property_name: str) -> [ast.FunctionDef]:
    prop = inspect.getattr_static(cls, property_name)
    return [fix_property(p) for p in parse_property(prop, property_name)]


def _format_dunder_all(names):
    return '__all__ = [\n    ' + ',\n    '.join('"' + name + '"'
                                                for name in names) + '\n]'


def _add_suppression_comments(code: str) -> str:

    def _add_override(s: str) -> str:
        for name in DISABLE_TYPE_CHECK_OVERRIDE:
            if name in s:
                return s + '  # type: ignore[override]'
        return s

    return '\n'.join(_add_override(line) for line in code.splitlines())


class _Member(enum.Enum):
    function = enum.auto()
    instancemethod = enum.auto()
    prop = enum.auto()
    skip = enum.auto()


def _classify(obj: object) -> _Member:
    if inspect.isbuiltin(obj):
        return _Member.skip
    if inspect.isdatadescriptor(obj):
        return _Member.prop
    if inspect.isfunction(obj):
        return _Member.function
    if inspect.isroutine(obj) and 'instancemethod' in repr(obj):
        return _Member.instancemethod
    return _Member.skip


def _get_bases(cls: Type[type]) -> List[ast.Name]:
    base_classes = [base for base in cls.__bases__ if 'pybind11' not in repr(base)]
    return [ast.Name(id=cls.__name__) for cls in base_classes]


def _build_class(cls: Type[type]) -> Optional[ast.ClassDef]:
    print(f'Generating class {cls.__name__}')
    body = []
    if cls.__doc__ and INCLUDE_DOCS:
        body.append(ast.Expr(value=ast.Constant(value=cls.__doc__)))

    for member_name, member in inspect.getmembers(cls):
        member_class = _classify(member)
        if member_class == _Member.skip:
            continue
        if member_class == _Member.prop:
            code = _build_property(cls, member_name)
        else:
            code = _build_method(cls, member_name)
        body.extend(code)

    if not body:
        body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))

    cls = ast.ClassDef(
        name=cls.__name__,
        bases=_get_bases(cls),
        keywords=[],
        decorator_list=[],
        body=body,
    )

    if not INCLUDE_DOCS:
        cls = ast.fix_missing_locations(RemoveDocstring().visit(cls))

    return ast.fix_missing_locations(cls)


def _cpp_classes() -> Iterable[Type[type]]:
    from scipp._scipp import core
    for name, cls in inspect.getmembers(core, inspect.isclass):
        if not class_is_excluded(name):
            yield cls


def generate_stub() -> str:
    classes = [cls for cls in map(_build_class, _cpp_classes()) if cls is not None]
    classes_code = '\n\n'.join(map(ast.unparse, classes))
    classes_code = _add_suppression_comments(classes_code)

    with TEMPLATE_FILE.open('r') as f:
        templ = Template(f.read())

    return templ.substitute(header=HEADER,
                            classes=classes_code,
                            dunder_all=_format_dunder_all(cls.name for cls in classes))
