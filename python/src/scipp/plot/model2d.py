
from .. import config
from .model import PlotModel
from .tools import parse_params, make_fake_coord, to_bin_edges, to_bin_centers, mask_to_float
from .._utils import name_with_unit, value_to_string
from .._scipp import core as sc

# Other imports
import numpy as np
import warnings


class PlotModel2d(PlotModel):

    def __init__(self,
                 controller=None,
                 scipp_obj_dict=None,
                 # axes=None,
                 # masks=None,
                 # cmap=None,
                 # log=None,
                 # vmin=None,
                 # vmax=None,
                 # color=None,
                 resolution=None):

        super().__init__(controller=controller,
            scipp_obj_dict=scipp_obj_dict)

        self.axparams = {"x": {}, "y": {}}
        self.button_dims = [None, None]
        self.dim_to_xy = {}
        self.xyrebin = {}
        self.xywidth = {}
        self.image_pixel_size = {}

        if resolution is not None:
            if isinstance(resolution, int):
                self.image_resolution = {"x": resolution, "y": resolution}
            else:
                self.image_resolution = resolution
        else:
            self.image_resolution = {
                "x": config.plot.width,
                "y": config.plot.height
            }

        return


    # def update_buttons(self, owner, event, dummy):
    #     toggle_slider = False
    #     if not self.controller.widgets.slider[owner.dim].disabled:
    #         toggle_slider = True
    #         self.controller.widgets.slider[owner.dim].disabled = True
    #         self.controller.widgets.thickness_slider[owner.dim].disabled = True
    #     for dim, button in self.controller.widgets.buttons.items():
    #         if (button.value == owner.value) and (dim != owner.dim):
    #             if self.controller.widgets.slider[dim].disabled:
    #                 button.value = owner.old_value
    #             else:
    #                 button.value = None
    #             button.old_value = button.value
    #             if toggle_slider:
    #                 self.controller.widgets.slider[dim].disabled = False
    #                 self.controller.widgets.thickness_slider[dim].disabled = False
    #     owner.old_value = owner.value
    #     self.update_axes()
    #     return

    def update_axes(self, limits):
        # Go through the buttons and select the right coordinates for the axes
        # extents = {}
        # for dim, button in self.controller.widgets.buttons.items():
        #     if self.controller.widgets.slider[dim].disabled:

        for dim in limits:

            but_val = limits[dim]["button"]
            # but_val = button.value.lower()
            # self.controller.extent[but_val] = self.slider_xlims[self.name][dim].values
            # self.axparams[but_val]["lims"] = self.controller.extent[but_val].copy()

            # extents[but_val] = self.slider_xlims[self.name][dim].values
            # self.axparams[but_val]["lims"] = self.slider_xlims[self.name][dim].values
            self.axparams[but_val]["lims"] = limits[dim]["xlims"]

            if getattr(self.controller,
                       "log" + but_val) and (self.axparams[but_val]["lims"][0] <= 0):
                self.axparams[but_val]["lims"][
                    0] = 1.0e-03 * self.axparams[but_val]["lims"][1]
            # self.axparams[but_val]["labels"] = name_with_unit(
            #     self.slider_label[self.name][dim]["coord"],
            #     name=self.slider_label[self.name][dim]["name"])
            self.axparams[but_val]["labels"] = name_with_unit(
                self.data_arrays[self.name].coords[dim])
            self.axparams[but_val]["dim"] = dim
            # Get the dimensions corresponding to the x/y buttons
            # self.button_dims[but_val == "x"] = button.dim
            # TODO: is using dim here ok?
            self.button_dims[but_val == "x"] = dim
            self.dim_to_xy[dim] = but_val

        # extent_array = np.array(list(self.controller.extent.values())).flatten()
        # self.controller.current_lims['x'] = extent_array[:2]
        # self.controller.current_lims['y'] = extent_array[2:]

        # TODO: if labels are used on a 2D coordinates, we need to update
        # the axes tick formatter to use xyrebin coords
        for xy, param in self.axparams.items():
            # Create coordinate axes for resampled array to be used as image
            # offset = 2 * (xy == "y")
            self.xyrebin[xy] = sc.Variable(
                dims=[param["dim"]],
                values=np.linspace(param["lims"][0],
                                   param["lims"][1],
                                   self.image_resolution[xy] + 1),
                unit=self.data_arrays[self.name].coords[param["dim"]].unit)

        # # Set axes labels
        # self.controller.ax.set_xlabel(self.axparams["x"]["labels"])
        # self.controller.ax.set_ylabel(self.axparams["y"]["labels"])
        # for xy, param in self.axparams.items():
        #     axis = getattr(self.controller.ax, "{}axis".format(xy))
        #     is_log = getattr(self.controller, "log{}".format(xy))
        #     axis.set_major_formatter(
        #         self.axformatter[self.name][param["dim"]][is_log])
        #     axis.set_major_locator(
        #         self.axlocator[self.name][param["dim"]][is_log])

        # # Set axes limits and ticks
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=UserWarning)
        #     self.controller.image.set_extent(extent_array)
        #     # if len(self.masks[self.name]) > 0:
        #     for m, im in self.controller.mask_image.items():
        #         im.set_extent(extent_array)
        #     self.controller.ax.set_xlim(self.axparams["x"]["lims"])
        #     self.controller.ax.set_ylim(self.axparams["y"]["lims"])

        # # If there are no multi-d coords, we update the edges and widths only
        # # once here.
        # if not self.contains_multid_coord[self.name]:
        #     self.slice_coords()
        # Update the image using resampling




        # ==================================
        # self.update_slice()
        # ==================================




        # # Some annoying house-keeping when using X/Y buttons: we need to update
        # # the deeply embedded limits set by the Home button in the matplotlib
        # # toolbar. The home button actually brings the first element in the
        # # navigation stack to the top, so we need to modify the first element
        # # in the navigation stack in-place.
        # if self.fig is not None:
        #     if self.fig.canvas.toolbar is not None:
        #         if len(self.fig.canvas.toolbar._nav_stack._elements) > 0:
        #             # Get the first key in the navigation stack
        #             key = list(self.fig.canvas.toolbar._nav_stack._elements[0].
        #                        keys())[0]
        #             # Construct a new tuple for replacement
        #             alist = []
        #             for x in self.fig.canvas.toolbar._nav_stack._elements[0][
        #                     key]:
        #                 alist.append(x)
        #             alist[0] = (*self.slider_xlims[self.name][
        #                 self.button_dims[1]].values, *self.slider_xlims[
        #                     self.name][self.button_dims[0]].values)
        #             # Insert the new tuple
        #             self.fig.canvas.toolbar._nav_stack._elements[0][
        #                 key] = tuple(alist)
        # self.controller.reset_home_button()

        # self.controller.rescale_to_data()


        if self.controller.profile is not None:
            self.update_profile_axes()

        return self.axparams



    def slice_data(self, slices):
        """
        Recursively slice the data along the dimensions of active sliders.
        """
        data_slice = self.data_arrays[self.name]

        # Slice along dimensions with active sliders
        # for dim, val in self.controller.widgets.slider.items():
        for dim in slices:
            # if not val.disabled:
                # self.lab[dim].value = self.make_slider_label(
                #     self.slider_label[self.engine.name][dim]["coord"], val.value)
                # print(self.slider_axformatter)
                # self.lab[dim].value = self.make_slider_label(
                #     val.value, self.slider_axformatter[self.engine.name][dim][False])
                # self.lab[dim].value = self.slider_axformatter[self.engine.name][dim][False].format_data_short(val.value)

            # deltax = self.controller.widgets.thickness_slider[dim].value
            deltax = slices[dim]["thickness"]
            loc = slices[dim]["location"]

            # print(data_slice)
            # print(sc.Variable([dim], values=[val.value - 0.5 * deltax,
            #                                                      val.value + 0.5 * deltax],
            #                                             unit=data_slice.coords[dim].unit))

            # TODO: see if we can call resample_image only once with
            # rebin_edges dict containing all dims to be sliced.
            data_slice = self.resample_image(data_slice,
                    # coord_edges={dim: self.slider_coord[self.engine.name][dim]},
                    rebin_edges={dim: sc.Variable([dim], values=[loc - 0.5 * deltax,
                                                                 loc + 0.5 * deltax],
                                                        unit=data_slice.coords[dim].unit)})[dim, 0]
                # depth = self.slider_xlims[self.engine.name][dim][dim, 1] - self.slider_xlims[self.engine.name][dim][dim, 0]
                # depth.unit = sc.units.one
            data_slice *= (deltax * sc.units.one)

            # data_slice = data_slice[val.dim, val.value]


        # Update the xyedges and xywidth
        for xy, param in self.axparams.items():
            # # Create bin-edge coordinates in the case of non bin-edges, since
            # # rebin only accepts bin edges.
            # if not self.histograms[self.engine.name][param["dim"]][param["dim"]]:
            #     self.xyedges[xy] = to_bin_edges(self.cslice[param["dim"]],
            #                                     param["dim"])
            # else:
            #     self.xyedges[xy] = self.cslice[param["dim"]].astype(
            #         sc.dtype.float64)
            # # Pixel widths used for scaling before rebin step
            # self.compute_bin_widths(xy, param["dim"])
            self.xywidth[xy] = (
                data_slice.coords[param["dim"]][param["dim"], 1:] -
                data_slice.coords[param["dim"]][param["dim"], :-1])
            self.xywidth[xy].unit = sc.units.one


        self.vslice = data_slice
        # Scale by bin width and then rebin in both directions
        # Note that this has to be written as 2 inplace operations to avoid
        # creation of large 2D temporary from broadcast
        self.vslice *= self.xywidth["x"]
        self.vslice *= self.xywidth["y"]

    # def update_slice(self, change=None):
    def update_slice(self, slices, mask_names):
        """
        Slice data according to new slider value and update the image.
        """
        # # If there are multi-d coords in the data we also need to slice the
        # # coords and update the xyedges and xywidth
        # if self.contains_multid_coord[self.engine.name]:
        #     self.slice_coords()
        self.slice_data(slices)
        # Update image with resampling
        new_values = self.update_image(mask_names=mask_names)
        return new_values


    def update_image(self, extent=None, mask_names=None):
        # The order of the dimensions that are rebinned matters if 2D coords
        # are present. We must rebin the base dimension of the 2D coord first.
        xy = "yx"
        if len(self.vslice.coords[self.button_dims[1]].dims) > 1:
            xy = "xy"

        dimy = self.xyrebin[xy[0]].dims[0]
        dimx = self.xyrebin[xy[1]].dims[0]

        rebin_edges = {
            dimy: self.xyrebin[xy[0]],
            dimx: self.xyrebin[xy[1]]
        }

        resampled_image = self.resample_image(self.vslice,
                                              # coord_edges={
                                              #     self.xyrebin[xy[0]].dims[0]:
                                              #     self.xyedges[xy[0]],
                                              #     self.xyrebin[xy[1]].dims[0]:
                                              #     self.xyedges[xy[1]]
                                              # },
                                              rebin_edges=rebin_edges)

        # Use Scipp's automatic transpose to match the image x/y axes
        # TODO: once transpose is available for DataArrays,
        # use sc.transpose(dslice, self.button_dims) instead.
        shape = [
            self.xyrebin["y"].shape[0] - 1, self.xyrebin["x"].shape[0] - 1
        ]
        self.dslice = sc.DataArray(coords=rebin_edges,
                                   data=sc.Variable(dims=self.button_dims,
                                                    values=np.ones(shape),
                                                    variances=np.zeros(shape),
                                                    dtype=self.vslice.dtype,
                                                    unit=sc.units.one))

        self.dslice *= resampled_image


        # return self.dslice.values

        # Update the matplotlib image data
        new_values = {"values": self.dslice.values, "masks": {}}


        arr = self.dslice.values
        # self.controller.image.set_data(arr)
        # if extent is not None:
        #     self.controller.image.set_extent(extent)

        # Handle masks
        if len(mask_names[self.name]) > 0:
            # Use scipp's automatic broadcast functionality to broadcast
            # lower dimension masks to higher dimensions.
            # TODO: creating a Variable here could become expensive when
            # sliders are being used. We could consider performing the
            # automatic broadcasting once and store it in the Slicer class,
            # but this could create a large memory overhead if the data is
            # large.
            # Here, the data is at most 2D, so having the Variable creation
            # and broadcasting should remain cheap.
            base_mask = sc.Variable(dims=self.dslice.dims,
                                    values=np.ones(self.dslice.shape,
                                                   dtype=np.int32))
            for m in mask_names[self.name]:
                if m in self.dslice.masks:
                    msk = base_mask * sc.Variable(
                        dims=self.dslice.masks[m].dims,
                        values=self.dslice.masks[m].values.astype(np.int32))
                    # self.controller.mask_image[m].set_data(
                    #     mask_to_float(msk.values, arr))
                    new_values["masks"][m] = mask_to_float(msk.values, arr)
                    # if extent is not None:
                    #     self.controller.mask_image[m].set_extent(extent)
                else:
                    # self.controller.mask_image[m].set_visible(False)
                    # self.controller.mask_image[m].set_url("hide")
                    new_values["masks"][m] = None

        # if self.autoscale_cbar:
        #     cbar_params = parse_params(globs=self.vminmax,
        #                                array=arr,
        #                                min_val=self.global_vmin,
        #                                max_val=self.global_vmax)
        #     self.global_vmin = cbar_params["vmin"]
        #     self.global_vmax = cbar_params["vmax"]
        #     self.params["values"][self.engine.name]["norm"] = cbar_params["norm"]
        #     self.image.set_norm(self.params["values"][self.engine.name]["norm"])
        #     if len(self.masks[self.engine.name]) > 0:
        #         for m in self.masks[self.engine.name]:
        #             self.members["masks"][m].set_norm(
        #                 self.params["values"][self.engine.name]["norm"])

        # self.controller.fig.canvas.draw_idle()
        return new_values



    def update_viewport_image(self, xylims):

        for xy, param in self.axparams.items():
            # Create coordinate axes for resampled image array
            self.xyrebin[xy] = sc.Variable(
                dims=[param["dim"]],
                values=np.linspace(xylims[xy][0], xylims[xy][1],
                                   self.image_resolution[xy] + 1),
                unit=self.data_arrays[self.name].coords[param["dim"]].unit)
        self.update_image(extent=np.array(list(xylims.values())).flatten())
