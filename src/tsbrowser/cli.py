import argparse
import asyncio
import importlib
import math
import gc
import json
import re
import sys
import queue
import threading
import logging
from datetime import datetime
from itertools import compress
from pathlib import Path
import warnings
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import shapely
import string
import xarray as xr
from pyproj import Transformer

from .vhr import get_vhr

prompt = "--> "
flag_labels = set(map(str, range(10)))

config = dict()
# disable garbabge collection to avoid issues with tkinter and thread safety
# we manually collect garbage after each figure is closed
gc.set_threshold(0)

# --- Logging Configuration ---
# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the minimum logging level

# Create handlers
# Console handler (for user-facing messages, possibly warnings/errors)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)  # Only show WARNING and above on console

# File handler (for all detailed pre-loading output)
# We will set the log file path dynamically later in main()
file_handler = logging.FileHandler("tsbrowser_preload.log")  # Default log file name
file_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
console_formatter = logging.Formatter("%(levelname)s: %(message)s")
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
# --- End Logging Configuration ---


def load_config(sConfigFile):
    config_path = Path(sConfigFile)
    if config_path.exists():
        config_dir = config_path.parent
        sys.path.append(str(config_dir))
        sModule = config_path.stem
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
        # used to display flags not tied to the current observation set
        self.t_extra = dict()  
        for key, band_name in iter(config["vars"].timeseries.items()):
            y = ts[band_name].isel(x=len(ts.x) // 2, y=len(ts.y) // 2)
            setattr(self, key, y.data)

    def on_pick(self, event):
        if event.artist.get_label().startswith("extra"):
            i = event.artist.get_label()[-1]
        elif event.artist.get_label().startswith("ts"):
            i = int(event.artist.get_label().split("_")[1])
        else:
            i = event.ind.item(0)
        if event.mouseevent.button == 3:
            self.toggle_flag_state(i, config["vars"].default_flag_label)
        elif event.mouseevent.button == 1 and not isinstance(i, str):
            self.update(i)

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
                    self.args.vis,
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
            if isinstance(i, str):
                del self.t_extra[i]
        else:
            self.flag_val[i] = label
            self.flags[i] = []
            for key in iter(config["vars"].timeseries.keys()):
                if isinstance(i, str):
                    line2d = self.ax[key].axvline(
                        self.t_extra[i], 0.0, 0.85, color="tab:olive", zorder=1
                    )
                    ann_label = "extra_{}".format(i)
                else:
                    line2d = self.ax[key].axvline(
                        self.t[i], 0.0, 0.85, color="tab:green", zorder=1
                    )
                    ann_label = "ts_{}".format(i)
                ann = self.ax[key].annotate(
                    label,
                    (0.0, 1.0),
                    xytext=(0, 0),
                    xycoords=line2d,
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    picker=5,
                    label=ann_label,
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


def setup_figure(args, pid, interpreter):
    fig = plt.figure(figsize=(10 * args.scalewindow[0], 7.5 * args.scalewindow[1]))
    fig.canvas.manager.set_window_title(f"PID: {pid} | Interpreter: {interpreter}")
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
        y = ts[band_name].isel(x=len(ts.x) // 2, y=len(ts.y) // 2)
        ax[ax_name].plot(ts.time, y, "o", picker=5)
        handles[ax_name] = ax[ax_name].plot(
            ts.time[0], y[0], "o", fillstyle="none", markeredgewidth=2
        )[0]
        if args.semilogy:
            ax[ax_name].set_yscale("log")
    # get first acquisition in the timeseries
    first_acquisition = ts.isel(time=0)
    for axis_name, band_combination in iter(config["vars"].images.items()):
        img = prepare_rgb(first_acquisition, band_combination, args.vis)
        handles[axis_name] = ax[axis_name].imshow(img)
        ax[axis_name].xaxis.tick_top()
    handles["img_VHR"] = ax["img_VHR"].imshow(vhr_layer["image"])
    ax["img_VHR"].xaxis.tick_top()
    ax["img_VHR"].yaxis.tick_right()
    ax["ts_B1"].xaxis.set_major_locator(config["vars"].monthLoc)
    ax["ts_B1"].xaxis.set_major_formatter(config["vars"].monthFmt)
    return handles


def add_patches(ax, row, col):
    for key, val in iter(config["vars"].images.items()):
        patch10 = patches.Rectangle((row - 1.5, col - 1.5), 3, 3, fill=False, ec="y")
        ax[key].add_patch(patch10)


def add_patch_vhr(ax, offset_line, offset_col):
    patch = patches.Rectangle(
        (offset_line - 6, offset_col - 6), 12, 12, fill=False, ec="y"
    )
    ax["img_VHR"].add_patch(patch)


def load_stack(files: list[Path], times: list[datetime], point, config, apply_mask=True):
    with rasterio.open(files[0], "r") as tif:
        transform = tif.transform
        pixel_size_x = transform.a
        pixel_size_y = -transform.e

    # Convert projected coordinates to pixel indices
    rows, cols = rasterio.transform.rowcol(transform, [point.x], [point.y])
    row = rows[0]
    col = cols[0]

    half_width_px = config["vars"].chip_width / (2 * pixel_size_x)
    half_height_px = config["vars"].chip_height / (2 * pixel_size_y)

    chips = []
    with warnings.catch_warnings():
        # rioxarray is warning about different scales per band, did not find a way to handle this warning
        # so we just ignore it
        warnings.filterwarnings("ignore", category=UserWarning, module="rioxarray")
        for tif_path in files:
            da = rioxarray.open_rasterio(tif_path, masked=apply_mask)
            # Slice the chip
            if not config["vars"].legacy_mode:
                da = da.isel(
                    y=slice(
                        math.ceil(row - half_height_px),
                        math.floor(row + half_height_px),
                    ),
                    x=slice(
                        math.ceil(col - half_width_px), math.floor(col + half_width_px)
                    ),
                )
            chips.append(da)
        # Combine along new "band" or "variable" dimension
        chip_stack = xr.concat(chips, dim=pd.Index(times, name="time"))
        # Set up temporal filter
        start = (
            pd.Timestamp(config["args"].startdate) if config["args"].startdate else None
        )
        end = pd.Timestamp(config["args"].stopdate) if config["args"].stopdate else None
        chip_stack.sel(time=slice(start, end))
    return chip_stack


def prepare_rgb(ds, bands, vis):
    lower_arr = np.array([vis[band][0] for band in bands])
    upper_arr = np.array([vis[band][1] for band in bands])
    img = (
        ds[[*bands]].to_dataarray("band").transpose("y", "x", "band") - lower_arr
    ) / upper_arr
    out_img = img.clip(0, 1).values
    return out_img


# New function to load data for a single PID and put it into the queue
def data_loader(pid_queue, preloaded_data_queue, args, original_geom_df, failed_pids):
    while not pid_queue.empty():
        current_pid = pid_queue.get(timeout=1)

        logger.info(
            f"Preloading data for PID: {current_pid}"
        )  # Changed print to logger.info

        # Subset to only the current pid
        if pd.api.types.is_numeric_dtype(
            original_geom_df[config["vars"].attr_id].dtype
        ):
            geom_df_subset = original_geom_df.query(
                f"{config['vars'].attr_id} == {current_pid}"
            )
        else:
            geom_df_subset = original_geom_df.query(
                f"{config['vars'].attr_id} == '{current_pid}'"
            )

        if geom_df_subset.empty:
            logger.warning(
                f"No data found for PID: {current_pid}"
            )  # Changed print to logger.warning
            pid_queue.task_done()
            failed_pids.append(current_pid)
            continue

        sample_series = geom_df_subset.iloc[0]

        # Set up acquisition timestamp read function
        if config["vars"].t_mode == "metadata":
            # TODO fix this path
            RetrieveTimestamp = None  # helpers.RetrieveAcquisitionDateTime
        elif config["vars"].t_mode == "filename":
            RetrieveTimestamp = lambda filename: datetime.strptime(
                filename.name[config["vars"].t_slice], config["vars"].t_format
            )

        # Possibly overwrite with command line arguments
        if args.pattern:
            config["vars"].i_pattern = args.pattern

        # Get image files
        tif_lists = dict()
        if not config["vars"].legacy_mode:
            # get quality files
            data_dir_q = Path(sample_series[config["vars"].attr_q_loc])
            if not data_dir_q.is_absolute():
                data_dir_q = args.geom_path.parent / data_dir_q
            if not data_dir_q.exists():
                logger.warning(
                    f"Raster quality directory does not exist: {data_dir_q} for PID {current_pid}"
                )  # Changed print to logger.warning
                pid_queue.task_done()
                failed_pids.append(current_pid)
                continue
            glob_files = data_dir_q.glob(
                f"{'**/' if config['vars'].q_recursive else ''}{config['vars'].q_pattern}"
            )
            tif_lists["q"] = sorted(list(glob_files))

        # Get image input directory
        data_dir = Path(sample_series[config["vars"].attr_i_loc])
        if not data_dir.is_absolute():
            data_dir = args.geom_path.parent / data_dir
        if not data_dir.exists():
            logger.warning(
                f"Raster data directory does not exist: {data_dir} for PID {current_pid}"
            )  # Changed print to logger.warning
            pid_queue.task_done()
            failed_pids.append(current_pid)
            continue

        for res in ("10m", "20m"):
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
                logger.info(  # Changed print to logger.info
                    "Discarded quality files: {}".format(
                        ", ".join(
                            map(lambda x: x.strftime("%Y-%m-%d"), sorted(discarded_qa))
                        )
                    )
                )
            if discarded_im:
                logger.info(  # Changed print to logger.info
                    "Discarded image files: {}".format(
                        ", ".join(
                            map(lambda x: x.strftime("%Y-%m-%d"), sorted(discarded_im))
                        )
                    )
                )

            if not tif_lists["q"] or not tif_lists["10m"]:
                logger.warning(
                    f"No common valid files found for PID {current_pid}. Skipping."
                )  # Changed print to logger.warning
                pid_queue.task_done()
                failed_pids.append(current_pid)
                continue

        # Get target point geometry in raster data projection
        with rasterio.open(tif_lists["10m"][0], "r") as tif:
            target_crs = tif.crs
            bounds = tif.bounds

        transformer = Transformer.from_crs(
            original_geom_df.crs, target_crs, always_xy=True
        )
        point_reproj = shapely.transform(
            sample_series.geometry, transformer.transform, interleaved=False
        )
        is_inside_bounds = (bounds.left <= point_reproj.x <= bounds.right) and (
            bounds.bottom <= point_reproj.y <= bounds.top
        )
        if not is_inside_bounds:
            logger.warning(
                f"Sample point outside of raster extent for PID {current_pid}. Skipping."
            )  # Changed print to logger.warning
            pid_queue.task_done()
            failed_pids.append(current_pid)
            continue

        if config["vars"].legacy_mode:
            selector = [True]*len(tif_lists["10m"])
        else:
            # this stack is loaded without applying the nd-mask to preserve integer data type
            # otherwise xarray converts everything to float because nd is represented as NaN
            q_stack = load_stack(tif_lists["q"], t_common, point_reproj, config, False)
            selected_band = q_stack.sel(band=config["vars"].q_band)
            if config["vars"].q_mode == "threshold_lt":
                ts_q_bin = selected_band < config["vars"].threshold
            elif config["vars"].q_mode == "threshold_gt":
                ts_q_bin = selected_band > config["vars"].threshold
            elif config["vars"].q_mode == "classes":
                if (
                    config["vars"].masking_classes is not None
                    and config["vars"].valid_classes is not None
                ):
                    raise ValueError(
                        "Cannot specify both masking_classes and valid_classes"
                    )
                if config["vars"].masking_classes is not None:
                    if config["vars"].eval_bitwise:
                        ts_q_bin = xr.ones_like(selected_band, dtype=bool) # start: all valid
                        for val in config["vars"].masking_classes:
                            not_masked = (selected_band & val) == 0
                            ts_q_bin = ts_q_bin & not_masked
                    else:
                        ts_q_bin = ~selected_band.isin(config["vars"].masking_classes)
                elif config["vars"].valid_classes is not None:
                    if config["vars"].eval_bitwise:
                        ts_q_bin = xr.zeros_like(selected_band, dtype=bool) # start: all invalid
                        for val in config["vars"].valid_classes:
                            not_masked = (selected_band & val) > 0
                            ts_q_bin = ts_q_bin | not_masked
                    else:
                        ts_q_bin = selected_band.isin(config["vars"].valid_classes)
                else:
                    raise ValueError(
                        "config error: either masking_classes or valid_classes must be specified"
                    )
            else:
                raise ValueError(
                    "config error: invalid parameter value for variable q_mode"
                )
            # Ensure that no-data pixels are set to False
            is_not_nd = ~selected_band.isin(selected_band._FillValue)
            ts_q_bin = ts_q_bin & is_not_nd
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

            if not any(selector):
                logger.warning(
                    f"No valid time steps after quality assessment for PID {current_pid}. Skipping."
                )  # Changed print to logger.warning
                pid_queue.task_done()
                failed_pids.append(current_pid)
                continue

        ts = load_stack(
            list(compress(tif_lists["10m"], selector)),
            list(compress(t_common, selector)),
            point_reproj,
            config,
        )

        # select appropriate bands and convert to dataset
        selected_bands = list(config["vars"].layermap.values())
        rename_dict = {v: k for k, v in config["vars"].layermap.items()}
        ts = ts.sel(band=selected_bands).to_dataset("band").rename_vars(rename_dict)

        # add any on the fly defined indices
        for index_name, index_formula in config["vars"].indices.items():
            # Replace each match with ts["match"]
            pattern = (
                r"\b(" + "|".join(re.escape(s) for s in rename_dict.values()) + r")\b"
            )
            result = re.sub(pattern, r'ts["\1"]', index_formula)
            ts[index_name] = eval(result)

        # Fetch VHR data
        vhr_layers = asyncio.run(
            get_vhr(
                sample_series.geometry.y,
                sample_series.geometry.x,
                config["vars"].vhr_zoom,
                remove_duplicates=config["vars"].remove_duplicates,
            )
        )
        if not vhr_layers:
            logger.warning(
                f"No VHR data found for PID {current_pid}. Skipping."
            )  # Changed print to logger.warning
            pid_queue.task_done()
            failed_pids.append(current_pid)
            continue

        # Calculate/set vis bounds
        vis = {}
        for band_name, val in iter(config["vars"].contrast.items()):
            if isinstance(val, tuple):
                lower, upper = val
            elif val == "mean_stddev":
                mean = ts[band_name].mean()
                std = ts[band_name].std()
                N = config["vars"].std_factor
                lower = mean - N * std
                upper = mean + N * std
            elif val == "median_mad":
                med = np.ma.median(ts[band_name])
                mad = np.ma.median(np.ma.fabs(ts[band_name] - med))
                s = mad / 0.6745
                N = config["vars"].std_factor
                lower = med - N * s
                upper = med + N * s
            elif val == "pct_clip":
                lower = np.nanpercentile(
                    ts[band_name].filled(np.nan),
                    config["vars"].pct_min,
                    method="lower",
                )
                upper = np.nanpercentile(
                    ts[band_name].filled(np.nan),
                    config["vars"].pct_max,
                    method="higher",
                )
            else:
                raise RuntimeError(
                    'invalid contrast stretch parameter "{}"'.format(val)
                )
            logger.info(
                f"{band_name} contrast min {lower:.2f} max {upper:.2f}"
            )  # Changed print to logger.info
            vis[band_name] = (lower, upper)

        # row, col of sample in image (might need to be done better)
        # right now it is just assumed to be in the middle of the image
        row = len(ts.y) // 2
        col = len(ts.x) // 2

        # Load possibly existing flag values
        if config["vars"].flag_dir is None:
            flag_dir = args.geom_path.parent
        else:
            flag_dir = Path(config["vars"].flag_dir)
            if not flag_dir.is_absolute():
                flag_dir = Path(args.config).parent / flag_dir
        flags_file_path = flag_dir / f"flags_{current_pid}.json"

        flag_val_datetime = {}
        if flags_file_path.exists():
            with open(flags_file_path, "r") as flags_file:
                flag_val_datetime = json.load(flags_file)

        preloaded_data_queue.put(
            {
                "pid": current_pid,
                "ts": ts,
                "vhr_layers": vhr_layers,
                "vis": vis,
                "row": row,
                "col": col,
                "flag_val_datetime": flag_val_datetime,
                "flags_file_path": flags_file_path,
            }
        )
        pid_queue.task_done()


def process_pid(args, preloaded_data):
    current_pid = preloaded_data["pid"]
    ts = preloaded_data["ts"]
    vhr_layers = preloaded_data["vhr_layers"]
    args.vis = preloaded_data["vis"]
    row = preloaded_data["row"]
    col = preloaded_data["col"]
    flag_val_datetime = preloaded_data["flag_val_datetime"]
    flags_file_path = preloaded_data["flags_file_path"]

    plt.ion()

    # Now set up figure
    fig, ax = setup_figure(args, current_pid, config["vars"].interpreter)
    handles = init_plots(args, ts, vhr_layers[0], ax)

    add_patches(ax, row, col)
    add_patch_vhr(ax, *vhr_layers[0]["point_pixel_offset_xy"])
    EventHandler = UiEventHandler(args, ts, vhr_layers, ax, handles)

    # Manage flags
    flag_val = None
    letters = iter(string.ascii_lowercase)
    ts_str_times = list(ts.time.dt.strftime("%Y-%m-%d %H:%M:%S.%f").values)
    if flag_val_datetime:
        flag_val = {}
        for dt_str, val in flag_val_datetime["flags"].items():
            try:
                flag_index = ts_str_times.index(dt_str)
                flag_val[flag_index] = val
            except ValueError:
                extra_letter = next(letters)
                flag_val[extra_letter] = (
                    val  # assign id for flag not tied to current obs set
                )
                EventHandler.t_extra[extra_letter] = datetime.strptime(
                    dt_str, "%Y-%m-%d %H:%M:%S.%f"
                )

    if flag_val is not None:
        EventHandler.set_flags(flag_val)

    # Show oldest VHR image in figure
    EventHandler.update_vhr(len(vhr_layers) - 1)

    # Register event callbacks
    fig.canvas.mpl_connect("pick_event", EventHandler.on_pick)
    fig.canvas.mpl_connect("key_press_event", EventHandler.on_key)
    fig.canvas.mpl_connect("scroll_event", EventHandler.on_scroll)

    # Prompt user immediately for interpretation confidence and comment
    previous_confidence = flag_val_datetime.get("confidence", None)
    confidence_str = (
        f" [ENTER to confirm previous value: {previous_confidence}]"
        if previous_confidence is not None
        else ""
    )
    previous_comment = flag_val_datetime.get("comment", None)
    comment_str = (
        f" [ENTER to confirm previous value: {previous_comment}]"
        if previous_comment is not None
        else ""
    )

    print(f"Interpretation for point {current_pid}:")
    try:
        confidence_input = (
            input(
                f"Enter interpretation confidence (high/h, medium/m, low/l){confidence_str}: "
            )
            .strip()
            .lower()
        )
        confidence = confidence_input or previous_confidence
        while confidence not in {"high", "medium", "low", "h", "m", "l", None}:
            confidence = input("Please enter 'high', 'medium', or 'low': ").strip().lower()
        if confidence in {"h", "m", "l"}:
            confidence = {"h": "high", "m": "medium", "l": "low"}[confidence]

        comment_input = input(
            f"Enter any comment about the interpretation{comment_str}: "
        ).strip()
        comment = comment_input or previous_comment
    except EOFError:
        return

    # Save current flag status from EventHandler
    flags = dict()
    for flag_index, flag_value in EventHandler.flag_val.items():
        if isinstance(flag_index, str):
            flags[EventHandler.t_extra[flag_index].strftime("%Y-%m-%d %H:%M:%S.%f")] = (
                flag_value
            )
        else:
            flags[ts_str_times[flag_index]] = flag_value

    # Add confidence and comment to the saved data
    output_data = {
        "flags": flags,
        "confidence": confidence,
        "comment": comment,
        "interpreter": config["vars"].interpreter,
    }

    with open(flags_file_path, "w") as flags_file:
        json.dump(output_data, flags_file, indent=4)

    plt.close(fig)
    plt.ioff()
    gc.collect()


def run_tsbrowser(args):
    # Load config variables
    load_config(args.config)
    flag_labels.update(config["vars"].add_flag_labels)
    config["args"] = args

    # Set up the log file path after config is loaded
    # Get the directory of the config file
    config_dir = Path(args.config).parent
    log_file_path = config_dir / "tsbrowser_preload.log"

    # Remove existing handlers to avoid adding multiple file handlers if main is called again (e.g. for testing)
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()  # Close the old file handler to release the file

    # Create a new file handler with the desired path
    new_file_handler = logging.FileHandler(log_file_path)
    new_file_handler.setLevel(logging.INFO)
    new_file_handler.setFormatter(file_formatter)
    logger.addHandler(new_file_handler)

    # Load vector file once
    geom_path = Path(config["vars"].path)
    if geom_path.is_absolute():
        geom_df = gpd.read_file(geom_path)
    else:
        # If the path is relative, construct it relative to the config directory
        geom_path = config_dir / geom_path
        geom_df = gpd.read_file(geom_path)
    args.geom_path = geom_path

    all_pids_to_process = []
    if not args.pid:
        print("No PIDs provided. Checking for existing flag files...")
        all_pids = geom_df[config["vars"].attr_id].astype(str).tolist()

        if config["vars"].flag_dir is None:
            flag_dir = args.geom_path.parent
        else:
            flag_dir = Path(config["vars"].flag_dir)
            if not flag_dir.is_absolute():
                flag_dir = Path(args.config).parent / flag_dir

        existing_flags = {
            f.stem[len("flags_") : ]
            for f in flag_dir.glob("flags_*.json")
        }

        all_pids_to_process = [
            pid for pid in all_pids if str(pid) not in existing_flags
        ]
        print(f"{len(all_pids_to_process)} samples to interpret.")

        if not all_pids_to_process:
            print("All PIDs already have flag files. Exiting.")
            return 0
    else:
        all_pids_to_process = args.pid

    attr_names_to_check = set([config["vars"].attr_id, config["vars"].attr_i_loc])
    if not config["vars"].legacy_mode:
        attr_names_to_check.add(config["vars"].attr_q_loc)

    # TODO Error handling if attributes not available
    # if not oTab.ValidateAttributeSet(attr_names_to_check):
    #     raise RuntimeError('Attribute name error')

    pid_queue = queue.Queue()
    preloaded_data_queue = queue.Queue(
        maxsize=5
    )  # Limit the queue size to avoid memory issues
    failed_pids = []
    num_preload_threads = args.preload_threads

    # Populate the PID queue
    for pid in all_pids_to_process:
        pid_queue.put(pid)

    # Start the background data loading threads
    loader_threads = []
    for _ in range(num_preload_threads):
        thread = threading.Thread(
            target=data_loader, args=(pid_queue, preloaded_data_queue, args, geom_df, failed_pids)
        )
        thread.daemon = True
        loader_threads.append(thread)
        thread.start()

    processed_pids_count = 0
    print(f"Waiting for next PID data to be preloaded...")
    while (len(failed_pids)+processed_pids_count) < len(all_pids_to_process):
        try:
            preloaded_data = preloaded_data_queue.get(timeout=1)
        except queue.Empty:
            continue
        processed_pids_count += 1
        print(
            f"Processing PID: {preloaded_data['pid']} ({processed_pids_count}/{len(all_pids_to_process)})"
        )
        try:
            process_pid(args, preloaded_data)
        except KeyboardInterrupt:
            break

        if processed_pids_count < len(all_pids_to_process):
            cont = input(
                f"Finished with PID {preloaded_data['pid']}. Press Enter to continue to next PID, or 'q' to quit: "
            )
            if cont.lower() == "q":
                break
            print(f"Waiting for next PID data to be preloaded...")

    print("All PIDs processed or skipped.")
    return 0


def main():
    """Entry point for the tsbrowser console script."""
    parser = argparse.ArgumentParser(description="Browse image time series")
    parser.add_argument("config", help="Configuration file", metavar="PATH")
    parser.add_argument(
        "--pid",
        help="ID(s) of point(s) to display",
        nargs="*",
        metavar="STR or INT",
    )
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
        "--preload-threads",
        help="Number of background threads to preload data (default: 5)",
        type=int,
        default=1,
        metavar="INT",
    )
    args = parser.parse_args()
    return run_tsbrowser(args)
