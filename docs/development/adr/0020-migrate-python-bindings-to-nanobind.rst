ADR 0020: Migrate Python bindings from pybind11 to nanobind
===========================================================

- Status: accepted
- Deciders: Simon, Jan-Lukas, Neil, Mridul
- Date: 2026-06-10

Context
-------

Background
~~~~~~~~~~

Scipp's Python layer is a single ``_scipp`` extension module of roughly 6,300 lines of binding code, historically built with `pybind11 <https://pybind11.readthedocs.io/>`_.
The binding layer is on the hot path for a large class of user code: slicing, coordinate lookups, attribute access, dict-like iteration over datasets, and per-bin Python loops are all dispatch-bound rather than kernel-bound.
For these workloads the binding framework's per-call overhead, not the C++ compute kernels, dominates the runtime.

Three further pressures motivated re-evaluating the framework:

- **Compile time and binary size.** The binding translation units are slow to compile and produce a large shared object, which inflates both developer iteration time and wheel size.
- **Dispatch overhead.** pybind11's overload resolution and argument casting add measurable per-call cost that is paid by every small operation.
- **Free-threading maturity.** Scipp opts out of the GIL at module scope. nanobind provides more mature free-threading support with localized locking, rather than relying on a single central lock.

`nanobind <https://nanobind.readthedocs.io/>`_ is a successor framework by the original author of pybind11 that targets these exact axes, at the cost of a stricter and smaller API surface.

Measured results
~~~~~~~~~~~~~~~~~

A dedicated binding microbenchmark suite was run on the same machine, compiler, and Python for both frameworks (the kernel-bound asv suite cannot detect binding-layer changes).
The headline figures:

.. list-table::
   :header-rows: 1
   :widths: 40 18 18 24

   * - Metric
     - pybind11
     - nanobind
     - Factor
   * - Binding microbenchmarks (geomean)
     - —
     - —
     - ~2.0x faster (median 1.65x)
   * - ``_scipp`` extension size
     - 11.6 MB
     - 2.5 MB
     - 4.6x smaller
   * - Bindings compile time (CPU, 113 TUs, no ccache)
     - 467 s
     - 205 s
     - 2.3x faster
   * - ``import scipp`` (total)
     - 175 ms
     - 112 ms
     - 1.6x faster
   * - Slicing (``var['x', 0]``, ellipsis, 2-D)
     - 1.2–1.5 µs
     - 0.2–0.3 µs
     - 5–7x faster
   * - Dict-like iteration (``ds.keys()``, ``items()``)
     - ~240 µs
     - ~16–20 µs
     - 12–15x faster
   * - Attribute access (``.unit``, ``.dims``, ``.ndim``)
     - 150–380 ns
     - 60–170 ns
     - 1.6–2.6x faster

The gains are concentrated in dispatch-bound code.
**Large-array and kernel-bound operations are unchanged**, as expected for compute-bound, framework-independent work: small binary ops (``a + b``, 4 elements) remain ~1.2x (C++ kernel bound), and the large 1e6-element values setter is at parity.

There is one notable regression: the ``.values`` getter on small arrays is **~1.5x slower** (0.6 µs to 0.9 µs, i.e. ~0.66x), and the analogous ``attr_values_large`` case is likewise ~0.66x.
Both stem from nanobind's ``ndarray``-to-numpy export overhead, not from a memcpy control; the zero-copy semantics themselves are unchanged.
We therefore do **not** claim blanket large-array parity: bulk math is unchanged, but the numpy *export* path carries a fixed nanobind overhead.

Constraints
~~~~~~~~~~~

- **Atomic cutover.** pybind11 and nanobind cannot share overload chains or exchange bound types. Because scipp is a single ``_scipp`` module, the core port must land as one change rather than incrementally.
- **No mixed-framework type exchange.** Any downstream C++ extension that passes scipp bound types through pybind11 casters will break and must be migrated in lockstep.
- ``nb::ndarray`` is DLPack-based and supports only int/uint/float/complex/bool (plus float16/bfloat16). It has no ``datetime64``, object, or string dtype, no ``py::dtype`` equivalent, no buffer protocol, and expresses strides in *elements*, not bytes. Every affected scipp path needed an explicit workaround.

Decision
--------

Adopt nanobind as the Python binding framework for scipp, replacing pybind11.

The core of the migration is a construct-by-construct mapping of the riskiest pybind11 usages to verified nanobind equivalents.
The following are the constructs that required design (rather than mechanical renaming), with the approach adopted for each.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - pybind11 construct
     - nanobind approach
   * - ``py::dtype`` static APIs (``of<T>()``, ``from_args``, ``normalized_num``, ``kind``)
     - Thin helper layer over ``np.dtype`` via the Python C API; the ``scipp_dtype`` switch is rewritten as a kind + itemsize dispatch.
   * - ``py::array_t<T>`` including converting casts
     - ``nb::ndarray<const T, nb::numpy>``; lists/sequences require an explicit ``np.asarray`` front door (nanobind refuses non-array-likes by design); ``unchecked<N>`` becomes ``.view<nb::ndim<N>>()``.
   * - ``datetime64`` round-trips
     - int64 ``nb::ndarray`` plus a zero-copy ``.view("datetime64[unit]")`` on the Python side; verified zero-copy, stride- and readonly-preserving.
   * - ``py::dtype("datetime64[...]")`` + raw ``py::array`` ctor for ``.values``
     - Build an int64 ndarray with an owner handle and reinterpret via ``.view()``; readonly is expressed via ``const T`` instead of private writeable-flag manipulation.
   * - ``py::make_scalar`` (numpy scalar returns)
     - Numpy scalar constructors via import (restoring scipp's own prior code plus a bool branch).
   * - ``py::options`` + handwritten ``__init__`` docstring signatures
     - Per-function ``nb::sig(...)`` plus placement-new ``__init__``; the rendered ``__doc__`` first line stays byte-compatible so sphinx autodoc continues to work.
   * - ``py::object`` as a Variable element (``scipp::python::PyObject``)
     - Near drop-in. **Trap:** ``nb::cast_error`` is a ``std::bad_cast``, not a ``std::runtime_error``, so casts surrounded by ``catch`` clauses must be audited or dtype errors fail silently.
   * - Eigen casters (``Vector3d``/``Matrix3d``)
     - ``nanobind/eigen/dense.h``; the 3-arg ``nb::cast(value, rv_policy, parent)`` supports the ``.value`` view-with-owner pattern; ``const`` to readonly is expressed with an explicit ``nb::ndarray<const double>`` view; lists no longer convert to Eigen arguments.
   * - Shared binder helpers + ``make_iterator``
     - ``nb::make_iterator(scope, name, ...)``. **Trap:** the element return-value policy must be the *template* argument; a runtime ``rv_policy::move`` extra is silently ignored.
   * - ``py::dict`` converting ctor ``py::dict(obj)``
     - A small ``to_pydict()`` helper calling the Python dict type (``nb::dict`` deliberately has no converting ctor); everything else is rename-level.
   * - ``py::class_<T, Ignored...>&`` binder templates
     - Deduction works identically, **except** ``py::class_<VariableConcept, VariableConceptHandle>`` must drop the holder argument and use ``<nanobind/stl/shared_ptr.h>``.
   * - ``py::slice.compute()``
     - ``nb::slice::compute(size)`` returning a tuple; the signed ``Py_ssize_t`` out-types are better suited to scipp's negative-step handling.
   * - ``std::tuple<std::string, T>`` indexing args
     - Granular ``nanobind/stl/tuple.h`` etc.; two-pass overload resolution makes the historical registration-order workarounds less fragile (gated by the slicing test matrix).
   * - Umbrella ``pybind11.h`` header
     - Rebuilt as a nanobind umbrella header with an audited caster set: string, vector, tuple, **pair**, optional, variant, map, plus eigen/dense, operators, ndarray, make_iterator, and typing.

Cross-cutting semantic differences that each required a test or audit:

- **None arguments are rejected by default.** Every argument that may legitimately receive ``None`` needs ``nb::arg("x").none()`` (or ``= nb::none()``); scipp has many (``unit=None``, ``dtype=None``, ``coords=None``, ``out=None``, …). Omission is a silent API break.
- **Strides are in elements, not bytes.** The ndarray path must not multiply by ``sizeof(T)``; getting this wrong is a silent data-corruption hazard in the overlap checks.
- **Read-only numpy inputs are rejected** unless the parameter element type is ``const T`` (or ``nb::ro``); broadcast results are frequently read-only.
- **Two-pass overload resolution** changes dispatch for numpy scalars (the first pass is exact-type-strict), so the indexing and label-slicing matrices must be retested.

Consequences
------------

Positive:
~~~~~~~~~

- Dispatch-bound user code (slicing, coordinate lookups, dataset iteration, per-bin loops) is roughly 2x faster on the microbenchmark suite, with some operations 5–15x faster.
- The ``_scipp`` extension shrinks 4.6x (11.6 MB to 2.5 MB), reducing wheel size.
- Bindings compile ~2.3x faster (CPU time), improving developer iteration time.
- ``import scipp`` is ~1.6x faster.
- More mature free-threading support, with localized locking rather than a single central lock.
- Per-instance wrapper memory overhead is reduced.

Negative:
~~~~~~~~~

- **Behavior changes visible to users** (documented as migration notes):

  - ``np.bool_`` values (e.g. ``np.True_``) are no longer accepted where a Python ``bool`` argument is expected; nanobind's bool caster requires an exact ``bool``.
  - ``bytes`` objects are no longer auto-decoded to ``str`` (e.g. as dict keys, dimension names, or unit strings).
  - Arithmetic operand dtype rules follow numpy: ``int64_var + np.int64(x)`` correctly stays int64 (parity with the pre-migration behavior), and ``np.True_`` as an operand resolves to float64 on numpy ≥ 2 (numpy removed ``np.bool_.__index__``).

- The ``.values`` numpy-export path carries a fixed nanobind overhead, making the getter on small arrays ~1.5x slower; bulk array math is unaffected.
- Internal mechanics required care: ``nb::cast_error`` derives from ``std::bad_cast`` (not ``std::runtime_error``); ``__hash__`` must be set to ``None`` on ``__eq__``-defining classes (pybind11 did this automatically); weakref support and rendered ``update()`` signatures had to be restored explicitly.
- int64 ``.values`` arrays carry the ``NPY_LONGLONG`` descriptor (type num 9 rather than 7); it compares equal to int64 but differs in identity.

Remaining work:
~~~~~~~~~~~~~~~

- Free-threaded wheels (e.g. ``cp314t``) are not yet restored.
- Cross-platform CI and wheel/conda packaging need to be exercised against the nanobind CMake configuration.
- Downstream packages with their own C++ extensions (scippneutron, scippnexus, ess*, Mantid converters) must be audited and migrated, since frameworks cannot exchange bound types.
