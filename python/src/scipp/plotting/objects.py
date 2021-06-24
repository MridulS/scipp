# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

from .. import config
from .formatters import make_formatter
from .tools import parse_params
from .._scipp.core import DimensionError
from .controller1d import PlotController1d
from .controller2d import PlotController2d
from .view1d import PlotView1d
from .view2d import PlotView2d
from .widgets import PlotWidgets


def make_params(*,
                cmap=None,
                norm=None,
                vmin=None,
                vmax=None,
                masks=None,
                color=None):
    # Scan the input data and collect information
    params = {"values": {}, "masks": {}}
    globs = {
        "cmap": cmap,
        "norm": norm,
        "vmin": vmin,
        "vmax": vmax,
        "color": color
    }
    masks_globs = {"norm": norm, "vmin": vmin, "vmax": vmax}
    # Get the colormap and normalization
    params["values"] = parse_params(globs=globs)
    params["masks"] = parse_params(params=masks,
                                   defaults={
                                       "cmap": "gray",
                                       "cbar": False,
                                       "under_color": None,
                                       "over_color": None
                                   },
                                   globs=masks_globs)
    # Set cmap extend state: if we have sliders then we need to extend.
    # We also need to extend if vmin or vmax are set.
    extend_cmap = "neither"
    if (vmin is not None) and (vmax is not None):
        extend_cmap = "both"
    elif vmin is not None:
        extend_cmap = "min"
    elif vmax is not None:
        extend_cmap = "max"

    params['extend_cmap'] = extend_cmap
    return params


def make_errorbar_params(arrays, errorbars):
    """
    Determine whether error bars should be plotted or not.
    """
    if errorbars is None:
        params = {}
    else:
        if isinstance(errorbars, bool):
            params = {name: errorbars for name in arrays}
        elif isinstance(errorbars, dict):
            params = errorbars
        else:
            raise TypeError("Unsupported type for argument "
                            "'errorbars': {}".format(type(errorbars)))
    for name, array in arrays.items():
        has_variances = array.variances is not None
        if name in params:
            params[name] &= has_variances
        else:
            params[name] = has_variances
    return params


def make_formatters(arrays, labels):
    array = next(iter(arrays.values()))
    labs = {dim: dim for dim in array.dims}
    if labels is not None:
        labs.update(labels)
    formatters = {dim: make_formatter(array, labs[dim]) for dim in array.dims}
    return labs, formatters


def make_profile(ax, mask_color):
    from .profile import PlotProfile
    pad = config.plot.padding.copy()
    pad[2] = 0.77
    return PlotProfile(ax=ax,
                       mask_color=mask_color,
                       figsize=(1.3 * config.plot.width / config.plot.dpi,
                                0.6 * config.plot.height / config.plot.dpi),
                       padding=pad,
                       legend={
                           "show": True,
                           "loc": (1.02, 0.0)
                       })


class DataArrayDict(dict):
    """
    Dict of data arrays with matching dimension labels and units. Shape and
    coordinates may mismatch.
    """
    @property
    def dims(self):
        return next(iter(self.values())).dims

    @property
    def sizes(self):
        return next(iter(self.values())).sizes

    @property
    def unit(self):
        return next(iter(self.values())).unit

    @property
    def meta(self):
        return next(iter(self.values())).meta


class PlotDict():
    """
    The Plot object is used as output for the plot command.
    It is a small wrapper around python dict, with an `_ipython_display_`
    representation.
    The dict will contain one entry for each entry in the input supplied to
    the plot function.
    More functionalities can be added in the future.
    """
    def __init__(self, *args, **kwargs):
        self._items = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._items[key]

    def __len__(self):
        return len(self._items)

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def items(self):
        return self._items.items()

    def _ipython_display_(self):
        """
        IPython display representation for Jupyter notebooks.
        """
        return self._to_widget()._ipython_display_()

    def _to_widget(self):
        """
        Return plot contents into a single VBocx container
        """
        import ipywidgets as ipw
        contents = []
        for item in self.values():
            if item is not None:
                contents.append(item._to_widget())
        return ipw.VBox(contents)

    def show(self):
        """
        """
        for item in self.values():
            item.show()

    def hide_widgets(self):
        for item in self.values():
            item.hide_widgets()

    def close(self):
        """
        Close all plots in dict, making them static.
        """
        for item in self.values():
            item.close()

    def redraw(self):
        """
        Redraw/update  all plots in dict.
        """
        for item in self.values():
            item.redraw()

    def set_draw_no_delay(self, value):
        """
        When set to True, try to update plots as soon as possible.
        This is useful in the case where one wishes to update the plot inside
        a loop (e.g. when listening to a data stream).
        The plot update is then slightly more expensive than when it is set to
        False.
        """
        for item in self.values():
            item.set_draw_no_delay(value)


class Plot:
    """
    Base class for plot objects. It uses the Model-View-Controller pattern to
    separate displayed figures and user-interaction via widgets from the
    operations performed on the data.

    It contains:
      - a `PlotModel`: contains the input data and performs all the heavy
          calculations.
      - a `PlotView`: contains a `PlotFigure` which is displayed and handles
          communications between `PlotController` and `PlotFigure`, as well as
          updating the `PlotProfile` depending on signals captured by the
          `PlotFigure`.
      - some `PlotWidgets`: a base collection of sliders and buttons which
          provide interactivity to the user.
      - a `PlotPanel` (optional): an extra set of widgets which is not part of
          the base `PlotWidgets`.
      - a `PlotProfile` (optional): used to display a profile plot under the
          `PlotFigure` to show one of the slider dimensions as a 1 dimensional
          line plot.
      - a `PlotController`: handles all the communication between all the
          pieces above.
    """
    def __init__(self,
                 scipp_obj_dict,
                 figure,
                 profile_figure=None,
                 errorbars=None,
                 panel=None,
                 labels=None,
                 resolution=None,
                 params=None,
                 axes=None,
                 norm=False,
                 scale=None,
                 positions=None,
                 view_ndims=None):

        self._scipp_obj_dict = scipp_obj_dict
        self.controller = None
        self.model = None
        self.panel = panel
        self.profile = None
        self.view = None
        self.widgets = None

        self.show_widgets = True
        self.view_ndims = view_ndims

        # Shortcut access to the underlying figure for easier modification
        self.fig = None
        self.ax = None

        # TODO use option to provide keys here
        array = next(iter(scipp_obj_dict.values()))

        self.name = list(scipp_obj_dict.keys())[0]
        self.dims = scipp_obj_dict[self.name].dims
        for dim in self.dims[:-view_ndims]:
            if dim in array.meta and len(array.meta[dim].dims) > 1:
                raise DimensionError("A ragged coordinate cannot lie along "
                                     "a slider dimension, it must be one of "
                                     "the displayed dimensions.")
        if positions:
            if not array.meta[positions].dims:
                raise ValueError(f"{positions} cannot be 0 dimensional"
                                 f" on input object\n\n{array}")
            else:
                self.position_dims = array.meta[positions].dims
        else:
            self.position_dims = None

        self._tool_button_states = {}
        if norm:
            self._tool_button_states['toggle_norm'] = True
        for dim in {} if scale is None else scale:
            self._tool_button_states[f'log_{dim}'] = scale[dim] == 'log'

        errorbars = make_errorbar_params(scipp_obj_dict, errorbars)
        figure.errorbars = errorbars
        if profile_figure is not None:
            profile_figure.errorbars = errorbars
        labels, formatters = make_formatters(scipp_obj_dict, labels)
        View = {1: PlotView1d, 2: PlotView2d}[view_ndims]
        self.view = View(figure=figure, formatters=formatters)
        self.profile = profile_figure

        self.widgets = PlotWidgets(dims=self.dims,
                                   formatters=formatters,
                                   ndim=self.view_ndims,
                                   dim_label_map=labels,
                                   masks=self._scipp_obj_dict)

        self.controller = self._make_controller(norm=norm,
                                                scale=scale,
                                                resolution=resolution,
                                                params=params)
        self._render()

    def _ipython_display_(self):
        """
        IPython display representation for Jupyter notebooks.
        """
        return self._to_widget()._ipython_display_()

    def _to_widget(self):
        """
        Get the SciPlot object as an `ipywidget`.
        """
        import ipywidgets as ipw
        widget_list = [self.view._to_widget()]
        if self.profile is not None:
            widget_list.append(self.profile._to_widget())
        if self.show_widgets:
            widget_list.append(self.widgets._to_widget())
        if self.panel is not None and self.show_widgets:
            widget_list.append(self.panel._to_widget())

        return ipw.VBox(widget_list)

    def hide_widgets(self):
        """
        Hide widgets for 1d and 2d (matplotlib) figures
        """
        self.show_widgets = False if self.view_ndims < 3 else True

    def close(self):
        """
        Send close signal to the view.
        """
        self.view.close()

    def show(self):
        """
        Call the show() method of a matplotlib figure.
        """
        self.view.show()

    def _render(self):
        """
        Perform some initial calls to render the figure once all components
        have been created.
        """
        self.view.figure.initialize_toolbar(
            log_axis_buttons=self.dims, button_states=self._tool_button_states)
        if self.profile is not None:
            self.profile.initialize_toolbar(
                log_axis_buttons=self.dims,
                button_states=self._tool_button_states)
        self.controller.render()
        if hasattr(self.view.figure, "fig"):
            self.fig = self.view.figure.fig
        if hasattr(self.view.figure, "ax"):
            self.ax = self.view.figure.ax

    def savefig(self, filename=None):
        """
        Save plot to file.
        Possible file extensions are `.jpg`, `.png` and `.pdf`.
        The default directory for writing the file is the same as the
        directory where the script or notebook is running.
        """
        self.view.savefig(filename=filename)

    def redraw(self):
        """
        Redraw the plot. Use this to update a figure when the underlying data
        has been modified.
        """
        self.controller.redraw()

    def set_draw_no_delay(self, value):
        """
        When set to True, try to update plots as soon as possible.
        This is useful in the case where one wishes to update the plot inside
        a loop (e.g. when listening to a data stream).
        The plot update is then slightly more expensive than when it is set to
        False.
        """
        self.view.set_draw_no_delay(value)
        if self.profile is not None:
            self.profile.set_draw_no_delay(value)

    def _make_controller(self, norm, scale, resolution, params):
        from .model1d import PlotModel1d
        from .model2d import PlotModel2d
        Model = {1: PlotModel1d, 2: PlotModel2d}[self.view_ndims]
        model = Model(scipp_obj_dict=self._scipp_obj_dict,
                      name=self.name,
                      resolution=resolution)
        profile_model = PlotModel1d(scipp_obj_dict=self._scipp_obj_dict,
                                    name=self.name)
        Controller = {
            1: PlotController1d,
            2: PlotController2d
        }[self.view_ndims]
        return Controller(dims=self.dims,
                          name=self.name,
                          vmin=params["values"]["vmin"],
                          vmax=params["values"]["vmax"],
                          norm=norm,
                          scale=scale,
                          widgets=self.widgets,
                          model=model,
                          profile_model=profile_model,
                          view=self.view,
                          panel=self.panel,
                          profile=self.profile)
