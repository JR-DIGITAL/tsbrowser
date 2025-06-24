#!/usr/bin/env PythonInterpreter
# -*- coding: utf-8 -*-

import argparse
import asyncio
from datetime import datetime, timedelta
from functools import partial
import importlib
from itertools import compress
import os
import pickle
import sys
import time

sLibPath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if sLibPath not in sys.path:
    sys.path.append(sLibPath)
    sys.path.append(os.path.abspath(os.path.join(sLibPath, '..', 'individual', 'vij')))

from impylib import gui, helpers, arrayio, vector
from vhr import get_vhr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import numpy as np


prompt = '--> '
flag_labels = set(map(str, range(10)))

config = dict()

def load_config(sConfigFile):
    if os.path.exists(sConfigFile):
        sDirname, sBasename = os.path.split(sConfigFile)
        sys.path.append(sDirname)
        sModule = sBasename.replace('.py', '')
        config['vars'] = importlib.import_module(sModule)
    else:
        raise RuntimeError('The given configuration file does not exist')


class UiEventHandler():

    def __init__(self, args, ts, vhr_layers, ax, handles):
        self.args = args
        self.geoarray = ts
        self.t = mdates.date2num(ts.lSensingTimes)
        self.vhr_layers = vhr_layers
        self.vhr_ann_top = None
        self.vhr_ann_bottom = None
        self.ax = ax
        self.handles = handles
        self.i = 0
        self.i_vhr = len(vhr_layers) - 1
        self.flags = dict()
        self.flag_val = dict()
        for key, val in iter(config['vars'].timeseries.items()):
            y = ts.aArray[:, ts.lBandNames.index(val), args.line, args.pixel]
            setattr(self, key, y.data)

    def on_pick(self, event):
        try:
            i = event.ind.item()
            if event.mouseevent.button == 3:
                self.toggle_flag_state(i, config['vars'].default_flag_label)
            elif event.mouseevent.button == 1:
                self.update(i)
        except ValueError:
            print('Error: more than one entity picked\n{}'.format(prompt), end='')

    def on_key(self, event):
        try:
            modifier, key = event.key.split('+')
        except ValueError:
            modifier = None
        if modifier == 'alt':
            if key in ('right', 'left'):
                if key == 'right':
                    i = self.i + 1
                elif key == 'left':
                    i = self.i - 1
                self.update(self.limit_i(i))
            elif key in ('down', 'up'):
                if key == 'up':
                    i = self.i_vhr - 1
                elif key == 'down':
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
            if key.startswith('ts'):
                val.set_data([self.t[i]], [getattr(self, key).data[i]])
            elif key[4] in ('L', 'R'):
                val.set_data(self.geoarray.GetRgbImage(i, config['vars'].images[key]))
        self.i = i
        
    def update_vhr(self, i=0):
        self.handles['img_VHR'].set_data(self.vhr_layers[i]['image'])
        label_top = 'Approx. acquisition: {}'.format(
            self.vhr_layers[i]['approximate_acquisition_date'])
        label_bottom = 'Publication: {}'.format(self.vhr_layers[i]['publish_date'])
        if self.vhr_ann_top is not None:
            self.vhr_ann_top.remove()
        self.vhr_ann_top = self.ax['img_VHR'].annotate(label_top, (.04, .96), color='w',
            backgroundcolor='k', va='top', xycoords='axes fraction', fontsize='small')
        if self.vhr_ann_bottom is not None:
            self.vhr_ann_bottom.remove()
        self.vhr_ann_bottom = self.ax['img_VHR'].annotate(label_bottom, (.04, .04), color='w',
            backgroundcolor='k', va='bottom', xycoords='axes fraction', fontsize='small')
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
            for key in iter(config['vars'].timeseries.keys()):
                y = getattr(self, key)
                line2d = self.ax[key].plot(self.t[i], y[i], 'x', color='r')[0]
                ann = self.ax[key].annotate(label, (self.t[i], y[i]), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')
                self.flags[i].append((line2d, ann))

    def limit_i(self, i):
        if i < 0:
            return self.t.size - 1
        elif i >= self.t.size:
            return 0
        else:
            return i

    def limit_i_vhr(self, i):
        if i < 0:
            return len(self.vhr_layers) - 1
        elif i >= len(self.vhr_layers):
            return 0
        else:
            return i

    def set_flags(self, flag_val):
        for i, label in iter(flag_val.items()):
            self.toggle_flag_state(i, label)


def setup_figure(args):
    fig = plt.figure(figsize=(10*args.scalewindow[0], 7.5*args.scalewindow[1]))
    fig.canvas.manager.set_window_title(args.pid)
    gs = fig.add_gridspec(4, 3, height_ratios=[2/5,1/5,1/5,1/5])
    ax = dict()
    ax['img_L'] = fig.add_subplot(gs[0, 0])
    ax['img_R'] = fig.add_subplot(gs[0, 1], sharex=ax['img_L'], sharey=ax['img_L'])
    ax['img_VHR'] = fig.add_subplot(gs[0, 2])
    ax['ts_B1'] = fig.add_subplot(gs[3, :])
    ax['ts_B2'] = fig.add_subplot(gs[2, :], sharex=ax['ts_B1'])
    ax['ts_B3'] = fig.add_subplot(gs[1, :], sharex=ax['ts_B1'])
    ax['ts_B2'].label_outer()
    ax['ts_B3'].label_outer()
#    ax['ts_B1'].xaxis.set_major_locator(monthLoc)
#    ax['ts_B1'].xaxis.set_major_formatter(monthFmt)
    for key, val in iter(config['vars'].timeseries.items()):
        ax[key].grid(True, 'major', 'x')
        ax[key].grid(True, 'major', 'y')
        ax[key].set_ylabel(val)
    plt.tight_layout(pad=1.4, h_pad=-1.0)
    return fig, ax


def init_plots(args, ts, vhr_layer, ax):
    handles = dict()
    for key, val in iter(config['vars'].timeseries.items()):
        y = ts.aArray[:, ts.lBandNames.index(val), args.line, args.pixel]
        ax[key].plot(ts.lSensingTimes, y, 'o', picker=5)
        handles[key] = ax[key].plot(ts.lSensingTimes[0], y[0], 'o',
               fillstyle='none', markeredgewidth=2)[0]
        if args.semilogy:
            ax[key].set_yscale('log')
        else:
            if config['vars'].ylim[0] == 'auto':
                ymin = y.min().item()
            else:
                ymin = config['vars'].ylim[0]
            if config['vars'].ylim[1] == 'auto':
                ymax = y.max().item()
            else:
                ymin = config['vars'].ylim[1]
            lim = helpers.compute_axis_limits(ymin, ymax, 6, upper_padding=0,
                round_to_multiples_of=config['vars'].ytick_multiples)
            ax[key].set_ylim(lim[:2])
            ax[key].set_yticks(lim[2])
    for key, val in iter(config['vars'].images.items()):
        img = ts.GetRgbImage(0, val)
        handles[key] = ax[key].imshow(img)
        ax[key].xaxis.tick_top()
    handles['img_VHR'] = ax['img_VHR'].imshow(vhr_layer['image'])
    ax['img_VHR'].xaxis.tick_top()
    ax['img_VHR'].yaxis.tick_right()
    ax['ts_B1'].xaxis.set_major_locator(config['vars'].monthLoc)
    ax['ts_B1'].xaxis.set_major_formatter(config['vars'].monthFmt)
    return handles


def add_patches(ax, Xpix10, Ypix10, oMapping10m, oMapping60m=None):
#    breakpoint()
    Xgeo10, Ygeo10 = oMapping10m.Forward(Xpix10, Ypix10)
    if oMapping60m is not None:
        Xpix60, Ypix60 = map(round, oMapping60m.Backward(Xgeo10, Ygeo10))
        Xgeo60, Ygeo60 = oMapping60m.Forward(Xpix60, Ypix60)
        Xpix60, Ypix60 = map(round, oMapping10m.Backward(Xgeo60 - 25, Ygeo60 + 25))
    for key, val in iter(config['vars'].images.items()):
        if oMapping60m is None:
            patch10 = patches.Rectangle((Xpix10 - 1.5, Ypix10 - 1.5), 3, 3, fill=False, ec='y')
            ax[key].add_patch(patch10)
        else:
            patch10 = patches.Rectangle((Xpix10 - .5, Ypix10 - .5), 1, 1, fill=False, ec='y')
            patch60 = patches.Rectangle((Xpix60 - .5, Ypix60 - .5), 6, 6, fill=False, ec='y')
            ax[key].add_patch(patch10)
            ax[key].add_patch(patch60)


def add_patch_vhr(ax, offset_line, offset_col):
    patch = patches.Rectangle((offset_line - 6, offset_col - 6), 12, 12, fill=False, ec='y')
    ax['img_VHR'].add_patch(patch)


def main(args):
    # Load config variables
    load_config(args.config)
    flag_labels.update(config['vars'].add_flag_labels)

    if args.pause:
        if not os.path.exists(args.pause_file):
            with open(args.pause_file, 'w') as f:
                f.write("pause")
    
    # Set up acquisition timestamp read function
    if config['vars'].t_mode == 'metadata':
        RetrieveTimestamp = helpers.RetrieveAcquisitionDateTime
    elif config['vars'].t_mode == 'filename':
        RetrieveTimestamp = partial(gui.RetrieveDateTimeFromString, 
            config['vars'].t_slice.start, config['vars'].t_slice.stop, config['vars'].t_format)
    
    # Possibly overwrite with command line arguments
    if args.pattern:
        config['vars'].i_pattern = args.pattern
    
    # Load vector file
    lGeom, oTab = vector.CAttributeTable().FromShapefile(config['vars'].path)
    attr_names_to_check = set([config['vars'].attr_id, config['vars'].attr_i_loc])
    if not config['vars'].legacy_mode:
        attr_names_to_check.add(config['vars'].attr_q_loc)
    if not oTab.ValidateAttributeSet(attr_names_to_check):
        raise RuntimeError('Attribute name error')
        
    # Find the correct table row
    if oTab.AttrTypes[oTab.AttrNames.index(config['vars'].attr_id)] is int:
        iRow = oTab[config['vars'].attr_id].index(int(args.pid))
    else:
        iRow = oTab[config['vars'].attr_id].index(args.pid) 
    tif_lists = dict()
        
    # Get quality raster input directory
    if not config['vars'].legacy_mode:
        data_dir_q = oTab[config['vars'].attr_q_loc][iRow]
        if not os.path.exists(data_dir_q):
            raise RuntimeError('Raster data directory does not exist')
        
    # Search for quality files   
        if config['vars'].q_recursive:
            tif_lists['q'] = sorted(helpers.SearchDirRecursively(data_dir_q,
                config['vars'].q_pattern, nMaxDepth=config['vars'].q_depth))
        else:
            tif_lists['q'] = sorted(helpers.SearchDir(data_dir_q, config['vars'].q_pattern))
        
    # Get image input directory
    data_dir = oTab[config['vars'].attr_i_loc][iRow]
    if not os.path.exists(data_dir):
        raise RuntimeError('Raster data directory does not exist')

    # Get lists of input files for each resolution
    data_dirs = {res: os.path.join(data_dir, res) for res in ('10m', '20m', '60m')}
    for key in iter(data_dirs.keys()):
        try:
            if config['vars'].i_recursive:
                tif_lists[key] = sorted(helpers.SearchDirRecursively(data_dirs[key],
                    config['vars'].i_pattern, nMaxDepth=config['vars'].i_depth))
            else:
                tif_lists[key] = sorted(helpers.SearchDir(data_dirs[key], config['vars'].i_pattern))
        except RuntimeError:
        # if there are no resolution subdirs in the directory, assume that 10m data is directly there
            if key == '10m':
                data_dirs[key] = os.path.realpath(data_dir)
                if config['vars'].i_recursive:
                    tif_lists[key] = sorted(helpers.SearchDirRecursively(data_dirs[key],
                        config['vars'].i_pattern, nMaxDepth=config['vars'].i_depth))
                else:
                    tif_lists[key] = sorted(helpers.SearchDir(data_dirs[key], config['vars'].i_pattern))
            else:
                data_dirs[key] = None
                tif_lists[key] = []
                
    # Create temporal consistency
    if config['vars'].legacy_mode:
        t_common = list(map(RetrieveTimestamp, tif_lists['10m']))
    else:
        t_qa = list(map(RetrieveTimestamp, tif_lists['q']))
        t_im = list(map(RetrieveTimestamp, tif_lists['10m']))
        discarded, t_common, i_qa, i_im = gui.IntersectInputs(
            tif_lists['q'], t_qa, tif_lists['10m'], t_im)
        tif_lists['q'] = i_qa
        tif_lists['10m'] = i_im
        if 0 in discarded:
            print('Discarded quality files: {}'.format(
                ', '.join(map(lambda x: x.strftime('%Y-%m-%d'), discarded[0]))))
        if 1 in discarded:
            print('Discarded image files: {}'.format(
                ', '.join(map(lambda x: x.strftime('%Y-%m-%d'), discarded[1]))))
        
    # Get target point geometry in raster data projection
    oSampleReader = arrayio.RasterIo.CFileReader(tif_lists['10m'][0])
    nRasterEpsg = oSampleReader.GetProjection().GetEPSGProjection()
    oMultiBand = oSampleReader.ReadMultiBandTile(0)
    oExtent = oMultiBand.GetBand(0).GetMappedTargetRect()
    oMultiBand= None
    oSampleReader = None
    if oTab.Epsg != nRasterEpsg:
        lGeomReprojected = vector.ReprojectGeom2d([lGeom[iRow]], 'point', oTab.Epsg, nRasterEpsg)
        oTargetPoint = lGeomReprojected[0]
    else:
        oTargetPoint = lGeom[iRow]
        
    # Fetch VHR data
    vhr_layers = asyncio.run(get_vhr(lGeom[iRow].GetY(), lGeom[iRow].GetX(), config['vars'].vhr_zoom, remove_duplicates=config['vars'].remove_duplicates))
       
    # Determine the subset for data extraction
    if config['vars'].legacy_mode:
        oSubset = None
    else:
        if not oExtent.IsPointInside(oTargetPoint):
            raise RuntimeError('Target point not on raster extent')
        oSubsetRect = vector.GraphObject2d.CRectDbl()
        oSubsetRect.SetXYWH(
            oTargetPoint.GetX() - config['vars'].chip_width/2,
            oTargetPoint.GetY() - config['vars'].chip_height/2,
            config['vars'].chip_width,
            config['vars'].chip_height
            )
            
        # if the target point is close to the raster border, move the chip a bit
        if not oExtent.IsPointInside(oSubsetRect.GetStart()):
            dShiftX = max(0, oExtent.GetStart().GetX() - oSubsetRect.GetStart().GetX())
            dShiftY = max(0, oExtent.GetStart().GetY() - oSubsetRect.GetStart().GetY())
            oSubsetRect.MoveX(dShiftX)
            oSubsetRect.MoveY(dShiftY)
        elif not oExtent.IsPointInside(oSubsetRect.GetEnd()):
            dShiftX = min(0, oExtent.GetEnd().GetX() - oSubsetRect.GetEnd().GetX())
            dShiftY = min(0, oExtent.GetEnd().GetY() - oSubsetRect.GetEnd().GetY())
            oSubsetRect.MoveX(dShiftX)
            oSubsetRect.MoveY(dShiftY)
        
        # Convert to subset tuple format
        oSubset = (
            oSubsetRect.GetStart().GetX(),
            oSubsetRect.GetEnd().GetY(),
            config['vars'].chip_width,
            config['vars'].chip_height
            )
        
    # Set up temporal filter    
    selector = [True]*len(tif_lists['10m'])
    if args.startdate or args.stopdate:
        try:
            start_date = datetime.strptime(args.startdate, '%Y%m%d')
        except ValueError:
            start_date = t_common[0]
        try:
            stop_date = datetime.strptime(args.stopdate, '%Y%m%d')
        except ValueError:
            stop_date = t_common[-1] + timedelta(1)
        for k, target_date in enumerate(t_common):
            selector[k] = target_date >= start_date and target_date < stop_date
        tmp = list(compress(tif_lists['10m'], selector))
        tif_lists['10m'] = tmp

    # Load quality data
    if config['vars'].legacy_mode:
        selector = [True]*len(tif_lists['10m'])
    else:
        q_layermap = {'B1': config['vars'].q_band}
        ts_q = arrayio.Read(list(compress(tif_lists['q'], selector)), q_layermap, oSubset=oSubset, 
            bProjected=True, oDateTimeRetriever=RetrieveTimestamp)
    
    # Set pixel coordinates
        float_pixel, float_line = ts_q.oMapping.Backward(oTargetPoint.GetX(), oTargetPoint.GetY())
        args.pixel = int(round(float_pixel))
        args.line = int(round(float_line))
        print('\n--> Evaluate {} at pixel/line time series {}/{}'.format(config['vars'].q_mode.upper(),
            args.pixel, args.line))
    
    # Evaluate quality
        if config['vars'].q_mode == 'oqb':
            ts_q_bin = ts_q.B1 < config['vars'].threshold
        elif config['vars'].q_mode == 'score':
            ts_q_bin = (ts_q.B1 > config['vars'].threshold).filled(False)
        elif config['vars'].q_mode == 'scl':
            ts_q_bin = np.ones_like(ts_q.B1, bool)
            for n_class in config['vars'].masking_classes:
                ts_q_bin[(ts_q.B1 == n_class).filled(False)] = False
        # breakpoint()
        overall_assessment = ts_q_bin.mean(2).mean(1) >= config['vars'].overall_valid_ratio
        row_slice = slice(args.line - config['vars'].specific_radius,
            args.line + config['vars'].specific_radius + 1)
        col_slice = slice(args.pixel - config['vars'].specific_radius,
            args.pixel + config['vars'].specific_radius + 1)
        specific_assessment = ts_q_bin[:, row_slice, col_slice].mean(2).mean(1) >= config['vars'].specific_valid_ratio
        selector = np.logical_and(overall_assessment, specific_assessment)

    # Load 10m data
    # breakpoint()
    ts = arrayio.Read(list(compress(tif_lists['10m'], selector)), config['vars'].layermap,
        sDataType='float', oSubset=oSubset, bProjected=True, oDateTimeRetriever=RetrieveTimestamp,
        bApplyZScales=config['vars'].apply_metadata_zscale)
    ts.Scale(config['vars'].zscale)
    # if config['vars'].legacy_mode:
    float_pixel, float_line = ts.oMapping.Backward(oTargetPoint.GetX(), oTargetPoint.GetY())
    args.pixel = int(round(float_pixel))
    args.line = int(round(float_line))
    print('\n--> Displaying pixel/line time series {}/{}'.format(args.pixel, args.line))
    print('')
    for key, val in iter(config['vars'].contrast.items()):
        if isinstance(val, tuple):
            lower, upper = val
        elif val == 'mean_stddev':
            mean = getattr(ts, key).mean()
            std = getattr(ts, key).std()
            N = config['vars'].std_factor
            lower = mean - N*std
            upper = mean + N*std
        elif val == 'median_mad':
            med = np.ma.median(getattr(ts, key))
            mad = np.ma.median(np.ma.fabs(getattr(ts, key) - med))
            s = mad/.6745
            N = config['vars'].std_factor
            lower = med - N*s
            upper = med + N*s
        elif val == 'pct_clip':
            lower = np.nanpercentile(getattr(ts, key).filled(np.nan), config['vars'].pct_min,
                method='lower')
            upper = np.nanpercentile(getattr(ts, key).filled(np.nan), config['vars'].pct_max,
                method='higher')
        else:
            raise RuntimeError('invalid contrast stretch parameter "{}"'.format(val))
        print('{} contrast min {:.2f} max {:.2f}'.format(key, lower, upper))
        ts.SetContrastSrcRange(key, lower, upper)

    # Get 60m geotransform
    oMapping60m = None
    if tif_lists['60m']:
        oRasterInfo = helpers.GetRasterInfo(tif_lists['60m'][0])
        oMapping60m = oRasterInfo.mapping

    # Load possibly existing flag values
    if config['vars'].flag_dir is None:
        flag_dir = os.path.dirname(args.path)
    else:
        if os.path.exists(config['vars'].flag_dir):
            flag_dir = config['vars'].flag_dir
        else:
            raise RuntimeError('Output directory for flag files does not exist')
    flags_file_path = os.path.join(flag_dir, 'flags_{}.pickle'.format(args.pid))
    if os.path.exists(flags_file_path):
        with open(flags_file_path, 'rb') as flags_file:
            flag_val_datetime = pickle.load(flags_file)
        flag_val = dict()
        flags_not_shown = []
        for date_time, val in flag_val_datetime.items():
            try:
                flag_index = ts.lSensingTimes.index(date_time)
                flag_val[flag_index] = val
            except ValueError:
                flags_not_shown.append('{:%Y-%m-%d %H:%M:%S}'.format(date_time))
    else:
        flag_val = None
        flag_val_datetime = dict()
        flags_not_shown = []
    if flags_not_shown:
        print('\nFlags not shown: {}\n'.format(', '.join(flags_not_shown)))

    # ****** PAUSE BLOCK START ******
    if args.pause:
        print(f"Pausing before interactive session. Waiting until file '{args.pause_file}' is removed...")
        while os.path.exists(args.pause_file):
            time.sleep(0.5)
        print("Release signal received. Starting interactive session.")
    # ****** PAUSE BLOCK END ******

    # Now set up figure
    plt.ion()
    fig, ax = setup_figure(args)
    handles = init_plots(args, ts, vhr_layers[0], ax)
    add_patches(ax, args.pixel, args.line, ts.oMapping, oMapping60m)
    add_patch_vhr(ax, *vhr_layers[0]['point_pixel_offset_xy'])
    EventHandler = UiEventHandler(args, ts, vhr_layers, ax, handles)
    if flag_val is not None:
        EventHandler.set_flags(flag_val)
    # breakpoint()
    EventHandler.update_vhr(len(vhr_layers) - 1)
    fig.canvas.mpl_connect('pick_event', EventHandler.on_pick)
    fig.canvas.mpl_connect('key_press_event', EventHandler.on_key)
    fig.canvas.mpl_connect('scroll_event', EventHandler.on_scroll)

    # Wait for the user to issue quit command
    _quit = False
    while not _quit:
        command = input(prompt)
#        breakpoint()
        if command.startswith('q'):
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
            with open(flags_file_path, 'wb') as flags_file:
                pickle.dump(flag_val_datetime, flags_file)
            _quit = True

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Browse image time series')
    parser.add_argument('config', help='Configuration file', metavar='PATH')
    parser.add_argument('pid', help='ID of point to display', metavar='STR or INT')
    parser.add_argument('--pattern', help='Raster file search pattern', metavar='STR')
    parser.add_argument('--semilogy', help='use logarithmic scaling of y-axis in time series plots',
        action='store_true')
    parser.add_argument('--scalewindow', help='Apply scale factors to width '
        'and height of the default window size', type=float,
        nargs=2, default=(1, 1), metavar='FLOAT')
    parser.add_argument('--startdate', help='Start date for time series display',
        default='', metavar='YYYYMMDD')
    parser.add_argument('--stopdate', help='Stop date for time series display',
        default='', metavar='YYYYMMDD')
    parser.add_argument('--pause', action='store_true',
                    help='Pause interactive session until externally released')
    parser.add_argument('--pause-file', type=str, default='pause.tmp',
                        help='Path to the pause file used to delay the interactive session')
    args = parser.parse_args()
    main(args)
