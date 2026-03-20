#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from utils import imgio as iio
import os
import shutil
import glob
import json
import datetime

import npy2bdv
import time
import sys
import re
import wx
import subprocess
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector


def get_dir(root_dir, pref, doc, pos=None):
    if pos:
        return os.path.join(root_dir, pref+doc+'__pos_'+pos, '')
    else:
        return os.path.join(root_dir, pref+doc, '')

ch_names = [
    "[2BLUE]",
    "[5GREEN]",
    "[6RED]",
    "[7FarRED]",
    "[8FarFarRED]",
]

scripts_root = os.path.dirname(os.path.abspath(__file__))


def get_channels_times(path):
    list_of_files_path = glob.glob(path+r'\*.tif')
    list_of_files = [os.path.basename(fp).split('.')[0] for fp in list_of_files_path]

    if len(list_of_files) == 0:
        return None
    idx_time = 0
    idx_ch = 0
    idx_chI = 0
    all_parts = list_of_files[0].split(' ')
    for p_idx, part in enumerate(all_parts):
        if '[' in part:
            idx_ch = p_idx
        if 'Time' in part:
            idx_time = p_idx
        if '_C' in part:
            idx_chI = p_idx

    channels = [f.split(' ')[idx_ch] for f in list_of_files]

    channelsI = [f.split(' ')[idx_chI] for f in list_of_files]
    channelsI = [f.split('_')[1] for f in channelsI]
    channelsI = [int(f.split('C')[1]) for f in channelsI]

    ch_id_map = {ch_idx: ch_n for ch_n, ch_idx in zip(channels, channelsI)}

    times = [f.split(' ')[idx_time] for f in list_of_files]
    times = set(times)
    times = list(times)
    times = [int(t.split('Time')[1]) for t in times]

    n_t = np.max(times)+1
    n_ch = np.max(channelsI)+1

    return ch_id_map, n_ch, n_t


def get_im_name(ds_dir, doc, ch_idx, t_idx, ch_map):
    ch_name = ch_map[ch_idx]
    return os.path.join(ds_dir,
                        doc+'_Doc1_PMT - PMT '+ ch_name + ' _C' + '%02d'%ch_idx
                        + '_Time Time' + '%04d'%t_idx + '.ome.tif')


def split_multiposition(root_dir, pref, doc, n_pos):
    try:
        mpos_path = [get_dir(root_dir, pref, doc, '%d'%pos) for pos in range(n_pos)]
        for pos_path in mpos_path:
            os.makedirs(pos_path, exist_ok=True)

        path = get_dir(root_dir, pref, doc)
        ch_map, n_chs, n_times = get_channels_times(path)
        print(ch_map, n_chs, n_times)

        for t_idx in range(n_times):
            pos_idx = t_idx % n_pos
            pos_t = t_idx // n_pos

            for ch_idx in range(n_chs):
                im_f = get_im_name(path, doc, ch_idx, t_idx, ch_map)
                im_fp = get_im_name(mpos_path[pos_idx], doc, ch_idx, pos_t, ch_map)
                if not os.path.exists(im_f):
                    print(im_f, 'not found')
                    break
                try:
                    os.link(im_f, im_fp)
                except OSError:
                    shutil.copy(im_f, im_fp)
    except Exception as e:
        print(e)


def recombine_multiposition(root_dir, pref, doc, n_pos, move=True):
    mpos_path = [get_dir(root_dir, pref, doc, '%d'%pos) for pos in range(n_pos)]
    for pos_path in mpos_path:
        os.makedirs(pos_path, exist_ok=True)

    path = get_dir(root_dir, pref, doc)
    ch_map, n_chs, n_times = get_channels_times(mpos_path[0])
    n_times = n_times * n_pos
    print(ch_map, n_chs, n_times)

    for t_idx in range(n_times):
        pos_idx = t_idx % n_pos
        pos_t = t_idx // n_pos

        for ch_idx in range(n_chs):
            im_f  = get_im_name(path, doc, ch_idx, t_idx, ch_map)
            im_fp = get_im_name(mpos_path[pos_idx], doc, ch_idx, pos_t, ch_map)
            if not os.path.exists(im_fp):
                print(im_fp, 'not found')
                break
            if move:
                shutil.move(im_fp, im_f)
            else:
                shutil.copy(im_fp, im_f)


def convert_to_h5(ds_dir, doc, res=(1,1,1), comp=False):
    ch_map, n_chs, n_times = get_channels_times(ds_dir)

    o_fn = os.path.dirname(get_im_name(ds_dir, doc, 0, 0, ch_map)) + '.h5'
    bdv_writer = npy2bdv.BdvWriter(o_fn, nchannels=n_chs, subsamp=((1, 1, 1),),
                                   blockdim=((32, 128, 128),),
                                   compression='gzip' if comp else None)

    t0 = time.perf_counter()
    for t_idx in range(n_times):
        for ch_idx in range(n_chs):
            stack = iio.read_mp_tiff(get_im_name(ds_dir, doc, ch_idx, t_idx, ch_map))
            bdv_writer.append_view(stack, time=t_idx, channel=ch_idx, voxel_size_xyz=res, voxel_units='um')
        print(f'{t_idx/n_times*100:.2f} % ', end='\r')

    bdv_writer.write_xml()
    bdv_writer.close()
    print(f'\ndone in {time.perf_counter() - t0:.1f}s')


def convert_to_h5_tmpl(ds_dir_wc, doc, res=(1,1,1), comp=False):
    ds_dir_arr = glob.glob(ds_dir_wc)
    if len(ds_dir_arr) == 0:
        raise ValueError('no directory found, idk how to proceed'+str(ds_dir_arr))
    elif len(ds_dir_arr) > 1:
        print('several dirs found, using last one:', ds_dir_arr)

    ds_dir = ds_dir_arr[-1]

    try:
        convert_to_h5(ds_dir, doc, comp=comp, res=res)
    except Exception as e:
        print(e)


def gen_dataset_cfg(workdir, dsname, runname,
                    ref_channel_idx, ref_ch_name, ch_id_map,
                    crop_ofs, crop_size, out_dir, output_ts=None):
    """Generate a complete per-run DistCorr cfg and save to out_dir."""
    template_path = os.path.join(scripts_root, 'cfg', 'DistCorr_align_template.cfg')
    with open(template_path, 'rt') as f:
        template = json.load(f)

    timestamp = datetime.datetime.now().strftime('%Y.%m.%d_%H%M%S')
    ref_ch_short = ref_ch_name.strip('[]')   # "[5GREEN]" -> "5GREEN"
    ch_letter    = ref_ch_name[2].upper()     # "[5GREEN]" -> 'G'
    crop_applied = any(v != -1 for v in crop_ofs) or any(v != -1 for v in crop_size)
    proc_sfx     = f'ALGN_CROP_{ch_letter}' if crop_applied else f'ALGN_{ch_letter}'

    distcorr = {
        "info":                    f"BatchSAC {timestamp} | ds: {dsname} | ref: {ref_ch_short}",
        "InputPath":               f"{workdir}\\{dsname}\\",
        "OutputPath":              f"{workdir}\\{dsname}_{proc_sfx}_{output_ts}\\" if output_ts else f"{workdir}\\{dsname}_{proc_sfx}\\",
        "NChannel":                str(len(ch_id_map)),
        "RefChannel":              str(ref_channel_idx),
        "ImageAlign":              "Y",
        "ImageRepositionToScanPos":"Y",
        "Resolution":              "1 1 1",
        "MaxOfs":                  "80 80 15",
        "ScanPosFile":             "$(ALIGNOUT)\\tile_000\\scanPos.txt",
    }

    for i in sorted(ch_id_map.keys()):
        distcorr[f"Channel_{i}"] = (
            f"{runname}_PMT - PMT {ch_id_map[i]} _C{i:02d}_Time Time%04d.ome.tif"
        )

    if any(v != -1 for v in crop_ofs):
        distcorr["AlignCropOfs"]  = " ".join(str(v) for v in crop_ofs)
    if any(v != -1 for v in crop_size):
        distcorr["AlignCropSize"] = " ".join(str(v) for v in crop_size)

    template["DistCorr"] = distcorr

    fname    = f'DistCorr_align_{dsname}_{ref_ch_short}_{timestamp}.cfg'
    cfg_path = os.path.join(out_dir, fname)
    with open(cfg_path, 'wt') as f:
        json.dump(template, f, indent='\t')

    return cfg_path


def gen_run_align_bat(run_root, run, date, doc, n_tiles,
                      ref_channel_idx, ref_ch_name, ch_id_map,
                      crop_ofs, crop_size, ts=None):
    try:
        workdir   = os.path.join(run_root, run)
        timestamp = doc[:8]
        runname   = f'{timestamp}_Doc1'
        ref_ch_short = ref_ch_name.strip('[]')

        bat_stem = f'proc_align_{run}_{date}_{doc}_{n_tiles}tl_{ref_ch_short}'
        run_dir  = os.path.join(workdir, 'applied_cfg_archive', bat_stem)
        os.makedirs(run_dir, exist_ok=True)

        bat  = '@echo off\n'
        bat += 'pushd %~dp0\n'
        bat += f'set proc_align={scripts_root}\\proc_align.bat\n'

        if n_tiles > 1:
            for pos in range(n_tiles):
                dsname   = f'{date}_Doc1_{doc}__pos_{pos}'
                cfg_path = gen_dataset_cfg(workdir, dsname, runname,
                                           ref_channel_idx, ref_ch_name, ch_id_map,
                                           crop_ofs, crop_size, out_dir=run_dir, output_ts=ts)
                bat += f'call %proc_align% "{cfg_path}"\n'
        else:
            dsname   = f'{date}_Doc1_{doc}'
            cfg_path = gen_dataset_cfg(workdir, dsname, runname,
                                       ref_channel_idx, ref_ch_name, ch_id_map,
                                       crop_ofs, crop_size, out_dir=run_dir, output_ts=ts)
            bat += f'call %proc_align% "{cfg_path}"\n'

        bat += 'popd\n'

        print(bat)

        fname = os.path.join(run_dir, f'{bat_stem}.bat')
        with open(fname, 'wt') as f:
            f.write(bat)

        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.call(fname, startupinfo=si)

    except Exception as e:
        print(e)


class ModalWin(wx.Dialog):
    def __init__(self, parent, dlg_class, pars=None):
        super().__init__(parent=parent, title='Split/Convert/Align 2pm data')
        self.SetEscapeId(12345)
        self.a = dlg_class(self, pars)
        self.buttonOk = wx.Button(self, wx.ID_OK, label='Next')
        self.sizerB = wx.StdDialogButtonSizer()
        self.sizerB.AddButton(self.buttonOk)
        self.sizerB.Realize()

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.a, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(self.sizerB, border=10, flag=wx.EXPAND|wx.ALL)

        self.SetSizerAndFit(self.sizer)
        self.SetPosition(pt=(550, 200))


class PathPanel(wx.Panel):
    def __init__(self, parent, pars):
        super().__init__(parent=parent)
        self.parent = parent
        self.text = wx.StaticText(self, label="1. Select Dir with datasets", pos=(10, 10))
        self.path = wx.TextCtrl(self, value='c:\\', pos=(10, 30))
        self.browse_btn = wx.Button(self, -1, "Browse", pos=(160, 30))
        self.Bind(wx.EVT_BUTTON, self.Browse, self.browse_btn)

    def Browse(self, event=None):
        try:
            dlg = wx.DirDialog(None, "Choose dataset directory", "",
                               wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
            if dlg.ShowModal() == wx.ID_OK:
                self.path.SetValue(dlg.GetPath())
            dlg.Destroy()
        except:
            pass


class CropSelectDialog(wx.Dialog):
    """Modal dialog showing 3 MIP projections of the first timeframe with
    linked rectangular crop selection across all three orthogonal views."""

    def __init__(self, parent, ds_path, ds_t, ch_idx, ch_map, crop_ofs, crop_size):
        super().__init__(parent, title=f'Select crop — {ds_t}',
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
                         size=(1100, 850))
        self._updating = False

        runname = f'{ds_t[:8]}_Doc1'
        fname   = f"{runname}_PMT - PMT {ch_map[ch_idx]} _C{ch_idx:02d}_Time Time0000.ome.tif"
        fpath   = os.path.join(ds_path, fname)

        try:
            stack = iio.read_mp_tiff(fpath)  # (Z, Y, X)
        except Exception as e:
            wx.MessageBox(f'Could not load image:\n{fpath}\n\n{e}',
                          'Error', wx.OK | wx.ICON_ERROR)
            self.EndModal(wx.ID_CANCEL)
            return

        nz, ny, nx = stack.shape
        self._shape = (nz, ny, nx)

        def norm(arr):
            lo, hi = np.percentile(arr, [2, 98])
            return np.clip((arr.astype(float) - lo) / max(hi - lo, 1), 0, 1)

        mip_xy = norm(np.max(stack, axis=0))  # (Y, X)
        mip_xz = norm(np.max(stack, axis=1))  # (Z, X)
        mip_yz = norm(np.max(stack, axis=2))  # (Z, Y)

        # Crop box: [x0, x1, y0, y1, z0, z1]
        x0 = float(crop_ofs[0]  if crop_ofs[0]  != -1 else 0)
        y0 = float(crop_ofs[1]  if crop_ofs[1]  != -1 else 0)
        z0 = float(crop_ofs[2]  if crop_ofs[2]  != -1 else 0)
        x1 = float(x0 + crop_size[0] if crop_size[0] != -1 else nx)
        y1 = float(y0 + crop_size[1] if crop_size[1] != -1 else ny)
        z1 = float(z0 + crop_size[2] if crop_size[2] != -1 else nz)
        self.box = [x0, x1, y0, y1, z0, z1]

        # Figure: orthogonal layout — XY top-left, YZ top-right, XZ bottom-left
        fig = Figure(figsize=(11, 8.5))
        gs  = fig.add_gridspec(2, 2,
                               width_ratios=[nx, nz],
                               height_ratios=[ny, nz],
                               hspace=0.08, wspace=0.08)

        ax_xy = fig.add_subplot(gs[0, 0])
        ax_yz = fig.add_subplot(gs[0, 1])
        ax_xz = fig.add_subplot(gs[1, 0])
        fig.add_subplot(gs[1, 1]).set_visible(False)

        ax_xy.imshow(mip_xy, cmap='gray', origin='upper', aspect='auto')
        ax_xy.set_title('XY  (Z-projection)'); ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y')

        ax_xz.imshow(mip_xz, cmap='gray', origin='upper', aspect='auto')
        ax_xz.set_title('XZ  (Y-projection)'); ax_xz.set_xlabel('X'); ax_xz.set_ylabel('Z')

        ax_yz.imshow(mip_yz.T, cmap='gray', origin='upper', aspect='auto')
        ax_yz.set_title('YZ  (X-projection)'); ax_yz.set_xlabel('Z'); ax_yz.set_ylabel('Y')

        sel_kw = dict(useblit=False, button=[1], interactive=True,
                      props=dict(facecolor='none', edgecolor='red',
                                 linewidth=1.5, alpha=0.8))

        self._sel_xy = RectangleSelector(ax_xy, self._on_sel_xy, **sel_kw)
        self._sel_xz = RectangleSelector(ax_xz, self._on_sel_xz, **sel_kw)
        self._sel_yz = RectangleSelector(ax_yz, self._on_sel_yz, **sel_kw)

        self._canvas = FigureCanvasWxAgg(self, -1, fig)
        self._update_selectors()

        btn_sizer = wx.StdDialogButtonSizer()
        btn_ok     = wx.Button(self, wx.ID_OK)
        btn_cancel = wx.Button(self, wx.ID_CANCEL)
        btn_sizer.AddButton(btn_ok)
        btn_sizer.AddButton(btn_cancel)
        btn_sizer.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self._canvas, 1, wx.EXPAND)
        sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, border=5)
        self.SetSizer(sizer)

    # ---- selector callbacks ------------------------------------------------

    def _on_sel_xy(self, eclick, erelease):
        if self._updating: return
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        self.box[0], self.box[1] = x0, x1
        self.box[2], self.box[3] = y0, y1
        self._update_selectors(skip='xy')

    def _on_sel_xz(self, eclick, erelease):
        if self._updating: return
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        z0, z1 = sorted([eclick.ydata, erelease.ydata])
        self.box[0], self.box[1] = x0, x1
        self.box[4], self.box[5] = z0, z1
        self._update_selectors(skip='xz')

    def _on_sel_yz(self, eclick, erelease):
        if self._updating: return
        z0, z1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        self.box[2], self.box[3] = y0, y1
        self.box[4], self.box[5] = z0, z1
        self._update_selectors(skip='yz')

    def _update_selectors(self, skip=None):
        self._updating = True
        x0, x1, y0, y1, z0, z1 = self.box
        if skip != 'xy': self._sel_xy.extents = (x0, x1, y0, y1)
        if skip != 'xz': self._sel_xz.extents = (x0, x1, z0, z1)
        if skip != 'yz': self._sel_yz.extents = (z0, z1, y0, y1)
        self._canvas.draw_idle()
        self._updating = False

    # ---- result ------------------------------------------------------------

    def get_crop(self):
        nz, ny, nx = self._shape
        x0, x1, y0, y1, z0, z1 = [round(v) for v in self.box]
        ofs_x = x0       if x0 > 0  else -1
        ofs_y = y0       if y0 > 0  else -1
        ofs_z = z0       if z0 > 0  else -1
        sz_x  = (x1-x0) if x1 < nx - 1 else -1
        sz_y  = (y1-y0) if y1 < ny - 1 else -1
        sz_z  = (z1-z0) if z1 < nz - 1 else -1
        return [ofs_x, ofs_y, ofs_z], [sz_x, sz_y, sz_z]


class DatasetProcConfigurator(wx.Panel):
    N_COLS = 17  # 10 data columns + 6 crop columns + 1 select button

    def __init__(self, parent, pars):
        super().__init__(parent=parent)
        self.parent = parent
        self.root_dir, self.runs_info = pars
        self.rows = []
        self.crop_visible = False
        self.crop_header_widgets = []

        self.outer_sizer = wx.BoxSizer(wx.VERTICAL)

        self.crop_btn = wx.Button(self, label='▶ Show crop params')
        self.Bind(wx.EVT_BUTTON, self.OnToggleCrop, self.crop_btn)
        self.outer_sizer.Add(self.crop_btn, flag=wx.LEFT|wx.TOP|wx.BOTTOM, border=5)

        self.sizer = wx.FlexGridSizer(self.N_COLS)
        self.gen_titel()

        for ds_dir, els in self.runs_info.items():
            ds_date, ds_times, ds_ress, ds_nposs = els
            for ds_t, ds_res, ds_npos in zip(ds_times, ds_ress, ds_nposs):
                path = get_dir(f'{self.root_dir}/{ds_dir}', f'{ds_date}_Doc1_', ds_t)
                ch_map, n_chs, n_times = get_channels_times(path)
                self.gen_ds_row(ds_dir, ds_date, ds_t, ds_res, ds_npos, ch_map)

        self.outer_sizer.Add(self.sizer)
        self.SetSizerAndFit(self.outer_sizer)

    def OnToggleCrop(self, event):
        self.crop_visible = not self.crop_visible
        self.crop_btn.SetLabel('▼ Hide crop params' if self.crop_visible else '▶ Show crop params')

        for w in self.crop_header_widgets:
            w.Show(self.crop_visible)
        for row in self.rows:
            for k in ['crop_ofs_x', 'crop_ofs_y', 'crop_ofs_z',
                      'crop_sz_x',  'crop_sz_y',  'crop_sz_z']:
                row[k].Show(self.crop_visible)
            row['btn_select'].Show(self.crop_visible)

        self.outer_sizer.Layout()
        self.SetSizerAndFit(self.outer_sizer)
        p = self.GetParent()
        if hasattr(p, 'sizer'):
            p.SetSizerAndFit(p.sizer)

    def OnSelectAll(self, event, col_key):
        val = event.GetEventObject().GetValue()
        for row in self.rows:
            row[col_key].SetValue(val)

    def _open_crop_select(self, row):
        ds_dir  = row['ds_dir_lbl'].Label
        ds_date = row['ds_date_lbl'].Label
        ds_t    = row['ds_t_lbl'].Label
        ds_path = get_dir(f'{self.root_dir}/{ds_dir}', f'{ds_date}_Doc1_', ds_t)

        ch_map  = row['ch_map']
        ch_name = row['ds_align_cb'].GetValue()
        ch_idx  = {v: k for k, v in ch_map.items()}.get(ch_name, min(ch_map.keys()))

        crop_ofs  = [int(row['crop_ofs_x'].Value), int(row['crop_ofs_y'].Value),
                     int(row['crop_ofs_z'].Value)]
        crop_size = [int(row['crop_sz_x'].Value),  int(row['crop_sz_y'].Value),
                     int(row['crop_sz_z'].Value)]

        dlg = CropSelectDialog(self, ds_path, ds_t, ch_idx, ch_map, crop_ofs, crop_size)
        if dlg.ShowModal() == wx.ID_OK:
            ofs, sz = dlg.get_crop()
            row['crop_ofs_x'].SetValue(str(ofs[0])); row['crop_ofs_y'].SetValue(str(ofs[1]))
            row['crop_ofs_z'].SetValue(str(ofs[2]))
            row['crop_sz_x'].SetValue(str(sz[0]));   row['crop_sz_y'].SetValue(str(sz[1]))
            row['crop_sz_z'].SetValue(str(sz[2]))
        dlg.Destroy()

    def gen_titel(self):
        hdr_process = wx.CheckBox(self, label='process')
        hdr_process.SetValue(True)
        self.Bind(wx.EVT_CHECKBOX,
                  lambda e: self.OnSelectAll(e, 'ds_process'), hdr_process)

        hdr_save_h5 = wx.CheckBox(self, label='save tile h5')
        hdr_save_h5.SetValue(True)
        self.Bind(wx.EVT_CHECKBOX,
                  lambda e: self.OnSelectAll(e, 'ds_save_tile_h5'), hdr_save_h5)

        hdr_align = wx.CheckBox(self, label='align frames')
        hdr_align.SetValue(True)
        self.Bind(wx.EVT_CHECKBOX,
                  lambda e: self.OnSelectAll(e, 'ds_align'), hdr_align)

        crop_hdrs = [wx.StaticText(self, label=lbl)
                     for lbl in ['ofs x', 'ofs y', 'ofs z', 'sz x', 'sz y', 'sz z']]
        for w in crop_hdrs:
            w.Show(False)
        hdr_select = wx.StaticText(self, label='')
        hdr_select.Show(False)
        self.crop_header_widgets = crop_hdrs + [hdr_select]

        title_widgets = [
            hdr_process,
            wx.StaticText(self, label='directory'),
            wx.StaticText(self, label='date'),
            wx.StaticText(self, label='dataset'),
            wx.StaticText(self, label='pixel xy, um'),
            wx.StaticText(self, label='pixel z, um'),
            wx.StaticText(self, label='num tiles'),
            hdr_save_h5,
            hdr_align,
            wx.StaticText(self, label='align channel'),
        ] + crop_hdrs + [hdr_select]

        for w in title_widgets:
            self.sizer.Add(w, border=5, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)

    def gen_ds_row(self, ds_dir, ds_date, ds_t, ds_res, ds_npos, ch_map):
        ds_process      = wx.CheckBox(self)
        ds_process.SetValue(True)
        ds_dir_lbl      = wx.StaticText(self, label=ds_dir)
        ds_date_lbl     = wx.StaticText(self, label=ds_date)
        ds_t_lbl        = wx.StaticText(self, label=ds_t)
        ds_res_xy_lbl   = wx.TextCtrl(self, value=f'{ds_res[0]:.3f}')
        ds_res_z_lbl    = wx.TextCtrl(self, value=f'{ds_res[2]:.3f}')
        n_tiles_inp     = wx.TextCtrl(self, value=f'{ds_npos}')
        ds_save_tile_h5 = wx.CheckBox(self)
        ds_save_tile_h5.SetValue(False)
        ds_align        = wx.CheckBox(self)
        ds_align.SetValue(True)

        ds_align_cb = wx.ComboBox(self, -1, style=wx.CB_READONLY)
        channels = sorted(ch_map.values())
        for ch in channels:
            ds_align_cb.Append(ch)
        green_idx = next((i for i, ch in enumerate(channels) if ch == '[5GREEN]'), 0)
        ds_align_cb.SetSelection(green_idx)

        crop_ofs_x = wx.TextCtrl(self, value='-1', size=(45, -1))
        crop_ofs_y = wx.TextCtrl(self, value='-1', size=(45, -1))
        crop_ofs_z = wx.TextCtrl(self, value='-1', size=(45, -1))
        crop_sz_x  = wx.TextCtrl(self, value='-1', size=(45, -1))
        crop_sz_y  = wx.TextCtrl(self, value='-1', size=(45, -1))
        crop_sz_z  = wx.TextCtrl(self, value='-1', size=(45, -1))
        for w in [crop_ofs_x, crop_ofs_y, crop_ofs_z, crop_sz_x, crop_sz_y, crop_sz_z]:
            w.Show(False)

        btn_select = wx.Button(self, label='Select', size=(55, -1))
        btn_select.Show(False)

        els = {
            'ds_process':      ds_process,
            'ds_dir_lbl':      ds_dir_lbl,
            'ds_date_lbl':     ds_date_lbl,
            'ds_t_lbl':        ds_t_lbl,
            'ds_res_xy_lbl':   ds_res_xy_lbl,
            'ds_res_z_lbl':    ds_res_z_lbl,
            'n_tiles_inp':     n_tiles_inp,
            'ds_save_tile_h5': ds_save_tile_h5,
            'ds_align':        ds_align,
            'ds_align_cb':     ds_align_cb,
            'crop_ofs_x':      crop_ofs_x,
            'crop_ofs_y':      crop_ofs_y,
            'crop_ofs_z':      crop_ofs_z,
            'crop_sz_x':       crop_sz_x,
            'crop_sz_y':       crop_sz_y,
            'crop_sz_z':       crop_sz_z,
            'btn_select':      btn_select,
            'ch_map':          ch_map,
        }
        self.rows.append(els)
        self.Bind(wx.EVT_BUTTON, lambda e, r=els: self._open_crop_select(r), btn_select)

        for w in [ds_process, ds_dir_lbl, ds_date_lbl, ds_t_lbl, ds_res_xy_lbl,
                  ds_res_z_lbl, n_tiles_inp, ds_save_tile_h5, ds_align, ds_align_cb,
                  crop_ofs_x, crop_ofs_y, crop_ofs_z, crop_sz_x, crop_sz_y, crop_sz_z,
                  btn_select]:
            self.sizer.Add(w, border=5, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)


def get_runs_info(run_root, dirs):
    res = [0.7731239092495636, 0.7731239092495636, 4.0]
    runs_info = {}

    for d in dirs:
        p = os.path.join(run_root, d, '*')
        dss = glob.glob(p)

        dates  = []
        times  = []
        ress   = []
        n_poss = []
        for ds_i in dss:
            bn = os.path.basename(ds_i)
            if '_bk' in bn or '_ALGN' in bn:
                print('skipping IGNORED dir', bn)
                continue
            if len(glob.glob(os.path.join(ds_i, '*tif'))) == 0:
                print('skipping EMPTY dir', bn)
                continue

            n_pos = 1 if '__pos' in bn else 4

            findres = re.findall('(.*)_Doc1_(.*)', bn)
            if len(findres) == 0:
                continue

            date, t = findres[0]
            dates.append(date)
            times.append(t)
            ress.append(res)
            n_poss.append(n_pos)

        dates = list(set(dates))
        assert len(dates) == 1

        runs_info[d] = (dates[0], times, ress, n_poss)

    return runs_info


def do_process(run_root, run, date, doc, res, n_tiles,
               do_tile_h5, do_align, align_channel, ch_id_map,
               crop_ofs, crop_size):
    print('\n\n Processing:')
    print(run_root, run, date, doc, res, n_tiles, do_tile_h5, do_align)

    timestamp = doc[:8]
    split_ts = datetime.datetime.now().strftime('%Y.%m.%d_%H%M%S')

    if n_tiles > 1:
        split_multiposition(f'{run_root}/{run}', f'{date}_Doc1_', doc, n_tiles)

    if do_tile_h5:
        if n_tiles > 1:
            for pos in range(n_tiles):
                ds_dir_wc = f'{run_root}/{run}/{date}_Doc1_{doc}__pos_{pos}/'
                convert_to_h5_tmpl(ds_dir_wc, timestamp, comp=True, res=res)
        else:
            ds_dir_wc = f'{run_root}/{run}/{date}_Doc1_{doc}/'
            convert_to_h5_tmpl(ds_dir_wc, timestamp, comp=True, res=res)

    if do_align:
        ref_channel_idx = {v: k for k, v in ch_id_map.items()}[align_channel]
        ch_letter = align_channel[2].upper()

        gen_run_align_bat(run_root, run, date, doc, n_tiles,
                          ref_channel_idx, align_channel, ch_id_map,
                          crop_ofs, crop_size, ts=split_ts)

        crop_applied = any(v != -1 for v in crop_ofs) or any(v != -1 for v in crop_size)
        algn_sfx = f'ALGN_CROP_{ch_letter}' if crop_applied else f'ALGN_{ch_letter}'
        if n_tiles > 1:
            for pos in range(n_tiles):
                ds_dir_wc = f'{run_root}/{run}/{date}_Doc1_{doc}__pos_{pos}_{algn_sfx}_{split_ts}/'
                convert_to_h5_tmpl(ds_dir_wc, timestamp, comp=True, res=res)
        else:
            ds_dir_wc = f'{run_root}/{run}/{date}_Doc1_{doc}_{algn_sfx}_{split_ts}/'
            convert_to_h5_tmpl(ds_dir_wc, timestamp, comp=True, res=res)


def process_all(ds_list):
    for item in ds_list:
        (run_root, run, date, doc, res, process, n_tiles,
         do_tile_h5, do_align, align_channel, ch_id_map,
         crop_ofs, crop_size) = item
        if process:
            try:
                do_process(run_root, run, date, doc, res, n_tiles,
                           do_tile_h5, do_align, align_channel, ch_id_map,
                           crop_ofs, crop_size)
            except Exception as e:
                print(e)


def run():
    app = wx.App()
    cont = True

    if cont:
        frameM = ModalWin(None, PathPanel)
        if frameM.ShowModal() != wx.ID_OK:
            cont = False
        path = frameM.a.path.Value
        frameM.Destroy()

    if cont:
        run_root  = os.path.dirname(path)
        dirs      = [os.path.basename(path)]
        runs_info = get_runs_info(run_root, dirs)

        frameM = ModalWin(None, DatasetProcConfigurator, (run_root, runs_info))
        if frameM.ShowModal() != wx.ID_OK:
            cont = False

        ds_list = []
        for row in frameM.a.rows:
            process       = row['ds_process'].Value
            run_          = row['ds_dir_lbl'].Label
            date          = row['ds_date_lbl'].Label
            doc           = row['ds_t_lbl'].Label
            rs_xy         = float(row['ds_res_xy_lbl'].Value)
            rs_z          = float(row['ds_res_z_lbl'].Value)
            n_tiles       = int(row['n_tiles_inp'].Value)
            do_tile_h5    = row['ds_save_tile_h5'].Value
            do_align      = row['ds_align'].Value
            align_channel = row['ds_align_cb'].Value
            ch_id_map     = row['ch_map']
            crop_ofs  = [int(row['crop_ofs_x'].Value),
                         int(row['crop_ofs_y'].Value),
                         int(row['crop_ofs_z'].Value)]
            crop_size = [int(row['crop_sz_x'].Value),
                         int(row['crop_sz_y'].Value),
                         int(row['crop_sz_z'].Value)]

            ds_list.append([run_root, run_, date, doc, [rs_xy, rs_xy, rs_z],
                            process, n_tiles, do_tile_h5, do_align, align_channel,
                            ch_id_map, crop_ofs, crop_size])
        frameM.Destroy()

    app.MainLoop()

    print(ds_list)
    if cont:
        process_all(ds_list)

    print('\n\n\n DONE!')


if __name__ == '__main__':
    run()
