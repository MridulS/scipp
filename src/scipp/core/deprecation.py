import warnings

from .util import VisibleDeprecationWarning


def _warn_attr_removal():
    warnings.warn(
        "sc.DataArray.attrs has been deprecated and will be removed in Scipp v24.12.0. "
        "The deprecation includes sc.DataArray.meta and sc.DataArray.drop_attrs. "
        "For unaligned coords, use sc.DataArray.coords and unset the alignment flag. "
        "For other attributes, use a higher-level data structure.",
        VisibleDeprecationWarning,
        stacklevel=3,
    )


def _deprecated_attrs(cls):
    """
    Dict of attrs.

    .. deprecated:: 23.9.0
       Use :py:attr:`coords` with unset alignment flag instead, or
       store attributes in higher-level data structures.
    """
    _warn_attr_removal()
    return cls.deprecated_attrs


def _deprecated_meta(cls):
    """
    Dict of coords and attrs.

    .. deprecated:: 23.9.0
       Use :py:attr:`coords` with unset alignment flag instead, or
       store attributes in higher-level data structures.
    """
    _warn_attr_removal()
    return cls.deprecated_meta


def _deprecated_drop_attrs(cls, *args, **kwargs):
    """
    Drop attrs.

    .. deprecated:: 23.9.0
       Use :py:attr:`coords` with unset alignment flag instead, or
       store attributes in higher-level data structures.
    """
    _warn_attr_removal()
    return cls.deprecated_drop_attrs(*args, **kwargs)
