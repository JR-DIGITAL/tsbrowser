#!/usr/bin/env PythonInterpreter

import argparse
import asyncio
import importlib
import math
import os
import pickle
import sys
import time
from datetime import datetime
from itertools import compress
from pathlib import Path

sLibPath = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
)
if sLibPath not in sys.path:
    sys.path.append(sLibPath)
    sys.path.append(os.path.abspath(os.path.join(sLibPath, "..", "individual", "vij")))

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import shapely
import xarray as xr
from pyproj import Transformer

from vhr import get_vhr

prompt = "--> "
flag_labels = set(map(str, range(10)))

config = dict()


def load_config(sConfigFile):
    if os.path.exists(sConfigFile):
        sDirname, sBasename = os.path.split(sConfigFile)
        sys.path.append(sDirname)
        sModule = sBasename.replace(".py", "")
        config["vars"] = importlib.import_module(sModule)
    else:
        raise RuntimeError("The given configuration file does not exist")


class UiEventHandler:
    def __init__(self, args, ts, vhr_layers, ax, handles):
        self.args = args
        self.geoarray = ts
        self.t = mdates.date2num(ts.time)
        self.vhr_layers = vhr_layers
        self.vhr_ann_top = None
        self.vhr_ann_bottom = None
        self.ax = ax
        self.handles = handles
        self.i = 0
        self.i_vhr = len(vhr_layers) - 1
        self.flags = dict()
        self.flag_val = dict()
        for key, band_name in iter(config["vars"].timeseries.items()):
            layer_index = config["vars"].layermap[band_name]
            y = ts.sel(band=layer_index).isel(x=len(ts.x)//2, y=len(ts.x)//2)
            setattr(self, key, y.data)

    def on_pick(self, event):
        try:
            i = event.ind.item()
            if event.mouseevent.button == 3:
                self.toggle_flag_state(i, config["vars"].default_flag_label)
            elif event.mouseevent.button == 1:
                self.update(i)
        except ValueError:
            print(f"Error: more than one entity picked\n{prompt}", end="")

    def on_key(self, event):
        try:
            modifier, key = event.key.split("+")
        except ValueError:
            modifier = None
        if modifier == "alt":
            if key in ("right", "left"):
                if key == "right":
                    i = self.i + 1
                elif key == "left":
                    i = self.i - 1
                self.update(self.limit_i(i))
            elif key in ("down", "up"):
                if key == "up":
                    i = self.i_vhr - 1
                elif key == "down":
                    i = self.i_vhr + 1
                self.update_vhr(self.limit_i_vhr(i))
            # elif key == 'down':
            # self.toggle_flag_state(self.i, config['vars'].default_flag_label)
            elif key in flag_labels:
                self.toggle_flag_state(self.i, key)

    def on_scroll(self, event):
        i = self.limit_i_vhr(self.i_vhr - int(event.step))
        # self.update(i)
        self.update_vhr(i)
        self.i_vhr = i

    def update(self, i=0):
        for key, val in iter(self.handles.items()):
            if key.startswith("ts"):
                val.set_data([self.t[i]], [getattr(self, key).data[i]])
            elif key[4] in ("L", "R"):
                img = prepare_rgb(
                    self.geoarray.isel(time=i), 
                    config["vars"].images[key],
                    self.args.vis
                )
                val.set_data(img)
        self.i = i

    def update_vhr(self, i=0):
        self.handles["img_VHR"].set_data(self.vhr_layers[i]["image"])
        label_top = "Approx. acquisition: {}".format(
            self.vhr_layers[i]["approximate_acquisition_date"]
        )
        label_bottom = "Publication: {}".format(self.vhr_layers[i]["publish_date"])
        if self.vhr_ann_top is not None:
            self.vhr_ann_top.remove()
        self.vhr_ann_top = self.ax["img_VHR"].annotate(
            label_top,
            (0.04, 0.96),
            color="w",
            backgroundcolor="k",
            va="top",
            xycoords="axes fraction",
            fontsize="small",
        )
        if self.vhr_ann_bottom is not None:
            self.vhr_ann_bottom.remove()
        self.vhr_ann_bottom = self.ax["img_VHR"].annotate(
            label_bottom,
            (0.04, 0.04),
            color="w",
            backgroundcolor="k",
            va="bottom",
            xycoords="axes fraction",
            fontsize="small",
        )
        self.i_vhr = i

    def toggle_flag_state(self, i, label):
        if i in self.flags.keys():
            for line2d, ann in self.flags[i]:
                line2d.remove()
                ann.remove()
            del self.flags[i]
            del self.flag_val[i]
        else:
            self.flag_val[i] = label
            self.flags[i] = []
            for key in iter(config["vars"].timeseries.keys()):
                y = getattr(self, key)
                line2d = self.ax[key].plot(self.t[i], y[i], "x", color="r")[0]
                ann = self.ax[key].annotate(
                    label,
                    (self.t[i], y[i]),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )
                self.flags[i].append((line2d, ann))

    def limit_i(self, i):
        if i < 0:
            return self.t.size - 1
        if i >= self.t.size:
            return 0
        return i

    def limit_i_vhr(self, i):
        if i < 0:
            return len(self.vhr_layers) - 1
        if i >= len(self.vhr_layers):
            return 0
        return i

    def set_flags(self, flag_val):
        for i, label in iter(flag_val.items()):
            self.toggle_flag_state(i, label)


def setup_figure(args):
    fig = plt.figure(figsize=(10 * args.scalewindow[0], 7.5 * args.scalewindow[1]))
    fig.canvas.manager.set_window_title(args.pid)
    gs = fig.add_gridspec(4, 3, height_ratios=[2 / 5, 1 / 5, 1 / 5, 1 / 5])
    ax = dict()
    ax["img_L"] = fig.add_subplot(gs[0, 0])
    ax["img_R"] = fig.add_subplot(gs[0, 1], sharex=ax["img_L"], sharey=ax["img_L"])
    ax["img_VHR"] = fig.add_subplot(gs[0, 2])
    ax["ts_B1"] = fig.add_subplot(gs[3, :])
    ax["ts_B2"] = fig.add_subplot(gs[2, :], sharex=ax["ts_B1"])
    ax["ts_B3"] = fig.add_subplot(gs[1, :], sharex=ax["ts_B1"])
    ax["ts_B2"].label_outer()
    ax["ts_B3"].label_outer()
    #    ax['ts_B1'].xaxis.set_major_locator(monthLoc)
    #    ax['ts_B1'].xaxis.set_major_formatter(monthFmt)
    for key, val in iter(config["vars"].timeseries.items()):
        ax[key].grid(True, "major", "x")
        ax[key].grid(True, "major", "y")
        ax[key].set_ylabel(val)
    plt.tight_layout(pad=1.4, h_pad=-1.0)
    return fig, ax


def init_plots(args, ts, vhr_layer, ax):
    handles = dict()
    for ax_name, band_name in iter(config["vars"].timeseries.items()):
        # point timeseries is the point in the middle of the image (floored)
        layer_index = config["vars"].layermap[band_name]
        y = ts.sel(band=layer_index).isel(x=len(ts.x)//2, y=len(ts.x)//2)
        ax[ax_name].plot(ts.time, y, "o", picker=5)
        handles[ax_name] = ax[ax_name].plot(
            ts.time[0], y[0], "o", fillstyle="none", markeredgewidth=2
        )[0]
        if args.semilogy:
            ax[ax_name].set_yscale("log")
        # TODO: move helper function into codebase to enable this
        # else:
        #     if config["vars"].ylim[0] == "auto":
        #         ymin = y.min().item()
        #     else:
        #         ymin = config["vars"].ylim[0]
        #     if config["vars"].ylim[1] == "auto":
        #         ymax = y.max().item()
        #     else:
        #         ymin = config["vars"].ylim[1]
        #     lim = helpers.compute_axis_limits(
        #         ymin,
        #         ymax,
        #         6,
        #         upper_padding=0,
        #         round_to_multiples_of=config["vars"].ytick_multiples,
        #     )
        #     ax[ax_name].set_ylim(lim[:2])
        #     ax[ax_name].set_yticks(lim[2])
    # get first acquisition in the timeseries
    first_acquisition = ts.isel(time=0)
    for axis_name, band_combination in iter(config["vars"].images.items()):
        img = prepare_rgb(
            first_acquisition, 
            band_combination,
            args.vis
        )
        handles[axis_name] = ax[axis_name].imshow(img)
        ax[axis_name].xaxis.tick_top()
    handles["img_VHR"] = ax["img_VHR"].imshow(vhr_layer["image"])
    ax["img_VHR"].xaxis.tick_top()
    ax["img_VHR"].yaxis.tick_right()
    ax["ts_B1"].xaxis.set_major_locator(config["vars"].monthLoc)
    ax["ts_B1"].xaxis.set_major_formatter(config["vars"].monthFmt)
    return handles


def add_patches(ax, row, col, oMapping60m=None):
    #  I think mapping == affine transform?
    # TODO handle 60m
    # if oMapping60m is not None:
        # Xpix60, Ypix60 = map(round, oMapping60m.Backward(Xgeo10, Ygeo10))
        # Xgeo60, Ygeo60 = oMapping60m.Forward(Xpix60, Ypix60)
        # Xpix60, Ypix60 = map(round, oMapping10m.Backward(Xgeo60 - 25, Ygeo60 + 25))
    for key, val in iter(config["vars"].images.items()):
        if oMapping60m is None:
            patch10 = patches.Rectangle(
                (row - 1.5, col - 1.5), 3, 3, fill=False, ec="y"
            )
            ax[key].add_patch(patch10)
        # else:
        #     patch10 = patches.Rectangle(
        #         (row - 0.5, col - 0.5), 1, 1, fill=False, ec="y"
        #     )
        #     patch60 = patches.Rectangle(
        #         (Xpix60 - 0.5, Ypix60 - 0.5), 6, 6, fill=False, ec="y"
        #     )
        #     ax[key].add_patch(patch10)
        #     ax[key].add_patch(patch60)


def add_patch_vhr(ax, offset_line, offset_col):
    patch = patches.Rectangle(
        (offset_line - 6, offset_col - 6), 12, 12, fill=False, ec="y"
    )
    ax["img_VHR"].add_patch(patch)


def load_stack(files: list[Path], times: list[datetime], point, config):
    with rasterio.open(files[0], "r") as tif:
        transform = tif.transform
        pixel_size_x = transform.a
        pixel_size_y = -transform.e  # negative because of raster orientation

    # Convert projected coordinates to pixel indices
    rows, cols = rasterio.transform.rowcol(transform, [point.x], [point.y])
    row = rows[0]
    col = cols[0]

    half_width_px = config["vars"].chip_width / (2 * pixel_size_x)
    half_height_px = config["vars"].chip_height / (2 * pixel_size_y)

    chips = []
    for tif_path in files:
        da = rioxarray.open_rasterio(tif_path, mask_and_scale=True)  # chunks="auto")
        # Slice the chip
        if not config["vars"].legacy_mode:
            da = da.isel(
                y=slice(
                    math.ceil(row - half_height_px), math.floor(row + half_height_px)
                ),
                x=slice(
                    math.ceil(col - half_width_px), math.floor(col + half_width_px)
                ),
            )
        chips.append(da)
    # Combine along new "band" or "variable" dimension
    chip_stack = xr.concat(chips, dim=pd.Index(times, name="time"))
    # Set up temporal filter
    start = pd.Timestamp(args.startdate) if args.startdate else None
    end = pd.Timestamp(args.stopdate) if args.stopdate else None
    chip_stack.sel(time=slice(start, end))
    return chip_stack

def prepare_rgb(ds, bands, vis):
    band_indices = [config["vars"].layermap[band] for band in bands]
    lower_arr = np.array([vis[band][0] for band in bands])
    upper_arr = np.array([vis[band][1] for band in bands])
    img = (ds.sel(band=band_indices).transpose('y','x','band') - lower_arr) / upper_arr
    out_img = img.clip(0, 1).values
    return out_img

def main(args):
    # Load config variables
    load_config(args.config)
    flag_labels.update(config["vars"].add_flag_labels)

    if args.pause:
        if not os.path.exists(args.pause_file):
            with open(args.pause_file, "w") as f:
                f.write("pause")

    # Set up acquisition timestamp read function
    if config["vars"].t_mode == "metadata":
        # TODO fix this path
        RetrieveTimestamp = helpers.RetrieveAcquisitionDateTime
    elif config["vars"].t_mode == "filename":
        RetrieveTimestamp = lambda filename: datetime.strptime(
            filename.name[config["vars"].t_slice], config["vars"].t_format
        )

    # Possibly overwrite with command line arguments
    if args.pattern:
        config["vars"].i_pattern = args.pattern

    # Load vector file
    geom_df = gpd.read_file(config["vars"].path)
    attr_names_to_check = set([config["vars"].attr_id, config["vars"].attr_i_loc])
    if not config["vars"].legacy_mode:
        attr_names_to_check.add(config["vars"].attr_q_loc)

    # subset to only the pid, if pid is passed
    if args.pid:
        if pd.api.types.is_numeric_dtype(geom_df[config["vars"].attr_id].dtype):
            geom_df = geom_df.query(f"{config['vars'].attr_id} == {args.pid}")
        else:
            geom_df = geom_df.query(f"{config['vars'].attr_id} == '{args.pid}'")

    # TODO Error handling if attributes not available
    # if not oTab.ValidateAttributeSet(attr_names_to_check):
    #     raise RuntimeError('Attribute name error')

    # From here everything should be a function which works on a row by row basis

    def get_image_files(config: dict, sample_series: gpd.GeoSeries):
        tif_lists = dict()
        if not config["vars"].legacy_mode:
            # get quality files
            data_dir_q = Path(sample_series[config["vars"].attr_q_loc])
            if not data_dir_q.exists():
                raise RuntimeError("Raster quality directory does not exist")
            glob_files = data_dir_q.glob(
                f"{'**/' if config['vars'].q_recursive else ''}{config['vars'].q_pattern}"
            )
            tif_lists["q"] = sorted(list(glob_files))

        # Get image input directory
        data_dir = Path(sample_series[config["vars"].attr_i_loc])
        if not data_dir.exists():
            raise RuntimeError("Raster data directory does not exist")

        for res in ("10m", "20m", "60m"):
            res_path = data_dir / res
            if res_path.exists():
                glob_files = res_path.glob(
                    f"{'**/' if config['vars'].i_recursive else ''}{config['vars'].i_pattern}"
                )
                tif_lists[res] = sorted(list(glob_files))

            else:
                # if there are no resolution subdirs in the directory, assume that 10m data is directly there
                if res == "10m":
                    glob_files = data_dir.glob(
                        f"{'**/' if config['vars'].i_recursive else ''}{config['vars'].i_pattern}"
                    )
                    tif_lists[res] = sorted(list(glob_files))
                else:
                    tif_lists[res] = []

        # Create temporal consistency
        if config["vars"].legacy_mode:
            t_common = list(map(RetrieveTimestamp, tif_lists["10m"]))
        else:
            t_qa = list(map(RetrieveTimestamp, tif_lists["q"]))
            t_im = list(map(RetrieveTimestamp, tif_lists["10m"]))

            # Find common timestamps between q and 10m files
            t_qa_set = set(t_qa)
            t_im_set = set(t_im)
            t_common_set = t_qa_set.intersection(t_im_set)
            t_common = sorted(list(t_common_set))

            # Track discarded timestamps
            discarded_qa = t_qa_set - t_common_set
            discarded_im = t_im_set - t_common_set

            # Filter tif_lists to keep only files with common timestamps
            filtered_qa = []
            filtered_im = []

            for i, timestamp in enumerate(t_qa):
                if timestamp in t_common_set:
                    filtered_qa.append(tif_lists["q"][i])

            for i, timestamp in enumerate(t_im):
                if timestamp in t_common_set:
                    filtered_im.append(tif_lists["10m"][i])

            tif_lists["q"] = filtered_qa
            tif_lists["10m"] = filtered_im

            # Print discarded files
            if discarded_qa:
                print(
                    "Discarded quality files: {}".format(
                        ", ".join(
                            map(lambda x: x.strftime("%Y-%m-%d"), sorted(discarded_qa))
                        )
                    )
                )
            if discarded_im:
                print(
                    "Discarded image files: {}".format(
                        ", ".join(
                            map(lambda x: x.strftime("%Y-%m-%d"), sorted(discarded_im))
                        )
                    )
                )

        # Get target point geometry in raster data projection
        # TODO: Do that once outside the loop
        with rasterio.open(tif_lists["q"][0], "r") as tif:
            target_crs = tif.crs
            bounds = tif.bounds

        transformer = Transformer.from_crs(geom_df.crs, target_crs, always_xy=True)
        point_reproj = shapely.transform(
            sample_series.geometry, transformer.transform, interleaved=False
        )
        is_inside_bounds = (bounds.left <= point_reproj.x <= bounds.right) and (
            bounds.bottom <= point_reproj.y <= bounds.top
        )
        if not is_inside_bounds:
            raise RuntimeError("Sample point outside of raster extent")

        q_stack = load_stack(tif_lists["q"], t_common, point_reproj, config)
        # TODO: Do band subset in load_stack
        selected_band = q_stack.sel(band=config["vars"].q_band)
        if config["vars"].q_mode == "oqb":
            ts_q_bin = selected_band < config["vars"].threshold
        elif config["vars"].q_mode == "score":
            ts_q_bin = selected_band > config["vars"].threshold
        elif config["vars"].q_mode == "scl":
            ts_q_bin = ~selected_band.isin(config["vars"].masking_classes)
        overall_assessment = ts_q_bin.mean(dim=["x", "y"])
        row_slice = slice(
            len(q_stack.y) // 2 - config["vars"].specific_radius,
            len(q_stack.y) // 2 + config["vars"].specific_radius + 1,
        )
        col_slice = slice(
            len(q_stack.x) // 2 - config["vars"].specific_radius,
            len(q_stack.x) // 2 + config["vars"].specific_radius + 1,
        )
        specific_assessment = (
            ts_q_bin.isel(y=row_slice, x=col_slice).mean(dim=["x", "y"])
            >= config["vars"].specific_valid_ratio
        )
        selector = np.logical_and(overall_assessment, specific_assessment).values

        ts = load_stack(
            list(compress(tif_lists["10m"], selector)),
            list(compress(t_common, selector)),
            point_reproj,
            config,
        )

        # select appropriate bands
        selected_bands = list(config["vars"].layermap.values())
        ts = ts.sel(band=selected_bands)

        # Get 60m geotransform
        oMapping60m = None
        if tif_lists["60m"]:
            oMapping60m = load_stack(
                list(compress(tif_lists["10m"], selector)),
                list(compress(t_common, selector)),
                point_reproj,
                config,
            )

        # Fetch VHR data
        vhr_layers = asyncio.run(
            get_vhr(
                sample_series.geometry.y,
                sample_series.geometry.x,
                config["vars"].vhr_zoom,
                remove_duplicates=config["vars"].remove_duplicates,
            )
        )

        return (
            sample_series[config["vars"].attr_id], 
            {10: ts, 60: oMapping60m, "vhr": vhr_layers}
        )

    data = []
    for index, row in geom_df.iterrows():
        data.append(get_image_files(config, row))

    # Calculate/set vis bounds
    args.vis = {}
    for band_name, val in iter(config['vars'].contrast.items()):
        band_index = config["vars"].layermap[band_name]
        if isinstance(val, tuple):
            lower, upper = val
        elif val == 'mean_stddev':
            mean = ts.sel(band=band_index).mean()
            std = ts.sel(band=band_index).std()
            N = config['vars'].std_factor
            lower = mean - N*std
            upper = mean + N*std
        elif val == 'median_mad':
            med = np.ma.median(ts.sel(band=band_index))
            mad = np.ma.median(np.ma.fabs(ts.sel(band=band_index) - med))
            s = mad/.6745
            N = config['vars'].std_factor
            lower = med - N*s
            upper = med + N*s
        elif val == 'pct_clip':
            lower = np.nanpercentile(ts.sel(band=band_index).filled(np.nan), config['vars'].pct_min,
                method='lower')
            upper = np.nanpercentile(ts.sel(band=band_index).filled(np.nan), config['vars'].pct_max,
                method='higher')
        else:
            raise RuntimeError('invalid contrast stretch parameter "{}"'.format(val))
        print('{} contrast min {:.2f} max {:.2f}'.format(band_name, lower, upper))
        args.vis[band_name] = (lower, upper)

    # Load possibly existing flag values
    if config["vars"].flag_dir is None:
        flag_dir = os.path.dirname(args.path)
    else:
        if os.path.exists(config["vars"].flag_dir):
            flag_dir = config["vars"].flag_dir
        else:
            raise RuntimeError("Output directory for flag files does not exist")
    flags_file_path = os.path.join(flag_dir, f"flags_{args.pid}.pickle")
    if os.path.exists(flags_file_path):
        with open(flags_file_path, "rb") as flags_file:
            flag_val_datetime = pickle.load(flags_file)
        flag_val = dict()
        flags_not_shown = []
        for date_time, val in flag_val_datetime.items():
            try:
                # TODO figure out what is going on here
                flag_index = ts.lSensingTimes.index(date_time)
                flag_val[flag_index] = val
            except ValueError:
                flags_not_shown.append(f"{date_time:%Y-%m-%d %H:%M:%S}")
    else:
        flag_val = None
        flag_val_datetime = dict()
        flags_not_shown = []
    if flags_not_shown:
        print("\nFlags not shown: {}\n".format(", ".join(flags_not_shown)))

    # Now set up figure
    # TODO: Proper data handling
    # TODO pass object to next functions
    sample_id, img_data = data[0]
    ts = img_data[10]
    oMapping60m = img_data[60]
    vhr_layers = img_data["vhr"]

    plt.ion()
    # row, col of sample in image (might need to be done better)
    # right now it is just assumed to be in the middle of the image
    row = len(ts.y) // 2
    col = len(ts.x) // 2
    fig, ax = setup_figure(args)
    handles = init_plots(args, ts, vhr_layers[0], ax)
    add_patches(ax, row, col)
    add_patch_vhr(ax, *vhr_layers[0]["point_pixel_offset_xy"])
    EventHandler = UiEventHandler(args, ts, vhr_layers, ax, handles)
    if flag_val is not None:
        EventHandler.set_flags(flag_val)
    # breakpoint()
    EventHandler.update_vhr(len(vhr_layers) - 1)
    fig.canvas.mpl_connect("pick_event", EventHandler.on_pick)
    fig.canvas.mpl_connect("key_press_event", EventHandler.on_key)
    fig.canvas.mpl_connect("scroll_event", EventHandler.on_scroll)

    # Wait for the user to issue quit command
    _quit = False
    while not _quit:
        command = input(prompt)
        #        breakpoint()
        if command.startswith("q"):
            # Convert flag indices to datetime
            for flag_index, val in EventHandler.flag_val.items():
                flag_val_datetime[ts.lSensingTimes[flag_index]] = val
            # Remove deleted flags also from flag_val_datetime
            keys_to_delete = []
            for flag_date_time, val in flag_val_datetime.items():
                if flag_date_time in ts.lSensingTimes:
                    index_to_check = ts.lSensingTimes.index(flag_date_time)
                    if index_to_check not in EventHandler.flag_val:
                        keys_to_delete.append(flag_date_time)
            if keys_to_delete:
                for flag_date_time in keys_to_delete:
                    del flag_val_datetime[flag_date_time]
            with open(flags_file_path, "wb") as flags_file:
                pickle.dump(flag_val_datetime, flags_file)
            _quit = True

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Browse image time series")
    parser.add_argument("config", help="Configuration file", metavar="PATH")
    parser.add_argument("pid", help="ID of point to display", metavar="STR or INT")
    parser.add_argument("--pattern", help="Raster file search pattern", metavar="STR")
    parser.add_argument(
        "--semilogy",
        help="use logarithmic scaling of y-axis in time series plots",
        action="store_true",
    )
    parser.add_argument(
        "--scalewindow",
        help="Apply scale factors to width and height of the default window size",
        type=float,
        nargs=2,
        default=(1, 1),
        metavar="FLOAT",
    )
    parser.add_argument(
        "--startdate",
        help="Start date for time series display",
        default="",
        metavar="YYYYMMDD",
    )
    parser.add_argument(
        "--stopdate",
        help="Stop date for time series display",
        default="",
        metavar="YYYYMMDD",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Pause interactive session until externally released",
    )
    parser.add_argument(
        "--pause-file",
        type=str,
        default="pause.tmp",
        help="Path to the pause file used to delay the interactive session",
    )
    args = parser.parse_args()
    main(args)
