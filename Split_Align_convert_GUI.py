#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from utils import imgio as iio
import os
import shutil
import glob

import npy2bdv
import sys 
import re
import wx
import subprocess

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

scripts_root = 'd:\\Trafficking_proc\\BatchSAT\\'

def get_channels_times(path):
    list_of_files_path = glob.glob(path+'\*.tif')
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
            
    #print(idx_ch, idx_time)
    
    channels = [f.split(' ')[idx_ch] for f in list_of_files]
    
    channelsI = [f.split(' ')[idx_chI] for f in list_of_files]
    channelsI = [f.split('_')[1] for f in channelsI]
    channelsI = [int(f.split('C')[1]) for f in channelsI]
    
    ch_id_map = {ch_idx:ch_n for ch_n, ch_idx in zip(channels, channelsI)}
    #print(ch_id_map)
    
    #channels = set(channels)
    times = [f.split(' ')[idx_time] for f in list_of_files]
    times = set(times)
    
    #print(times, channels)
    
    #ch_idxs = []
    #for ch_idx, ch in enumerate(ch_names):
    #    if ch in channels:
    #        ch_idxs.append(ch_idx)
            
    times = list(times)
    times = [int(t.split('Time')[1]) for t in times]
    
    n_t = np.max(times)+1
    n_ch =  np.max(channelsI)+1
    #print(times, ch_idxs, n_t)
    
    return ch_id_map, n_ch, n_t
    

def get_im_name(ds_dir, doc, ch_idx, t_idx, ch_map):
    ch_name = ch_map[ch_idx]
    
    return os.path.join(ds_dir, 
                        doc+'_Doc1_PMT - PMT '+ ch_name + ' _C' + '%02d'%ch_idx+ '_Time Time' + '%04d'%t_idx + '.ome.tif'
                       )


def split_multiposition(root_dir, pref, doc, n_pos, move=True):
    try:
        mpos_path = [get_dir(root_dir, pref, doc, '%d'%pos) for pos in range(n_pos)]
        for pos_path in mpos_path:
            os.makedirs(pos_path, exist_ok=True)


        path = get_dir(root_dir, pref, doc)
        ch_map, n_chs, n_times = get_channels_times(path)
        print( ch_map, n_chs, n_times)

        for t_idx in range(n_times):
            pos_idx = t_idx % n_pos
            pos_t = t_idx // n_pos

            for ch_idx in range(n_chs):
                im_f = get_im_name(path, doc, ch_idx, t_idx, ch_map)

                im_fp = get_im_name(mpos_path[pos_idx], doc, ch_idx, pos_t, ch_map)
                #print(im_f, '->', im_fp)
                if not os.path.exists(im_f):
                    print(im_f, 'not found')
                    break
                if move:
                    shutil.move(im_f, im_fp)
                else:
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
    print( ch_map, n_chs, n_times)
    
    for t_idx in range(n_times):
        pos_idx = t_idx % n_pos
        pos_t = t_idx // n_pos
        
        for ch_idx in range(n_chs):
            im_f = get_im_name(path, doc, ch_idx, t_idx, ch_map)
            
            im_fp = get_im_name(mpos_path[pos_idx], doc, ch_idx, pos_t, ch_map)
            #print(im_f, '->', im_fp)
            if not os.path.exists(im_fp):
                print(im_fp, 'not found')
                break
            if move:
                shutil.move(im_fp, im_f)
            else:
                shutil.copy(im_fp, im_f)


def convert_to_h5(ds_dir, doc, res=(1,1,1), comp=False):
    ch_map, n_chs, n_times = get_channels_times(ds_dir)

    
    o_tif_fn = os.path.dirname(get_im_name(ds_dir, doc, 0, 0, ch_map)) + '.h5'
    
    bdv_writer = npy2bdv.BdvWriter(o_tif_fn, nchannels=n_chs, subsamp=((1, 1, 1),)
                                  , compression='gzip' if comp else None
                                  )

    for t_idx in range(n_times):
        for ch_idx in range(n_chs):
            stack = iio.read_mp_tiff(get_im_name(ds_dir, doc, ch_idx, t_idx, ch_map))
            bdv_writer.append_view(stack, time=t_idx, channel=ch_idx, voxel_size_xyz=res, voxel_units='um')
            
        print(f'{t_idx/n_times*100:.2f} % ', end='\r')

    bdv_writer.write_xml()
    bdv_writer.close()


def convert_to_h5_tmpl(ds_dir_wc, doc, res=(1,1,1), comp=False):
    ds_dir_arr = glob.glob(ds_dir_wc)
    if len(ds_dir_arr) == 0:
        raise ValueError('no directory found, idk how to proceed'+str(ds_dir_arr))
    elif len(ds_dir_arr) > 1:
        print('several dirs found, using last one:', ds_dir_arr)

    ds_dir = ds_dir_arr[-1]

    #print(ds_dir, res)

    try:
        convert_to_h5(ds_dir, doc, comp=True, res=res)
    except Exception as e:
        print(e)



class ModalWin(wx.Dialog):
    def __init__(self, parent, dlg_class, pars=None):
        super().__init__(parent=parent, title='Split/Conver/Align 2pm data')
        ####---- Variables
        self.SetEscapeId(12345)
        ####---- Widgets
        self.a = dlg_class(self, pars)
        self.buttonOk = wx.Button(self, wx.ID_OK, label='Next')
        ####---- Sizers
        self.sizerB = wx.StdDialogButtonSizer()
        self.sizerB.AddButton(self.buttonOk)
        
        self.sizerB.Realize()

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.a, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(self.sizerB, border=10, 
            flag=wx.EXPAND|wx.ALIGN_RIGHT|wx.ALL)

        self.SetSizerAndFit(self.sizer)

        self.SetPosition(pt=(550, 200))


class PathPanel(wx.Panel):
    def __init__(self, parent, pars):
        super().__init__(parent=parent)
        ####---- Variables
        self.parent = parent
        ####---- Widgets
        label = ("1. Select Dir with datasets")
        self.text = wx.StaticText(self, label=label, pos=(10, 10))
        self.path = wx.TextCtrl(self, value='c:\\', pos=(10, 30))
        
        self.browse_btn = wx.Button(self, -1, "Browse", pos=(160, 30))
        self.Bind(wx.EVT_BUTTON, self.Browse, self.browse_btn)
        
    def Browse(self, event=None):
        try:
            dlg = wx.DirDialog (None, "Choose dataset diretory", "", wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
            if dlg.ShowModal() == wx.ID_OK:
                RootPath = dlg.GetPath()
                self.path.SetValue(RootPath)

            dlg.Destroy()
        
        except:
            pass
        
        
class DatasetProcConfigurator(wx.Panel):
    def __init__(self, parent, pars):
        super().__init__(parent=parent)
        ####---- Variables
        self.parent = parent
        self.root_dir, self.runs_info = pars
        self.rows = []
        ####---- Widgets
        
        self.sizer = wx.FlexGridSizer(10)
        self.gen_titel()
        
        for ds_dir, els in self.runs_info.items():
            ds_date, ds_times, ds_ress, ds_nposs = els
            
            for ds_t, ds_res, ds_npos in zip(ds_times, ds_ress, ds_nposs):
                path = get_dir(f'{self.root_dir}/{ds_dir}', f'{ds_date}_Doc1_', ds_t)
                ch_map, n_chs, n_times = get_channels_times(path)
                self.gen_ds_row(ds_dir, ds_date, ds_t, ds_res, ds_npos, ch_map)
                
        
        self.SetSizerAndFit(self.sizer)
        
    def gen_ds_row(self, ds_dir, ds_date, ds_t, ds_res, ds_npos, ch_map):
        ds_process = wx.CheckBox(self)
        ds_process.SetValue(True)
        ds_dir_lbl = wx.StaticText(self, label=ds_dir)
        ds_date_lbl = wx.StaticText(self, label=ds_date)
        ds_t_lbl = wx.StaticText(self, label=ds_t)
        ds_res_xy_lbl = wx.TextCtrl(self, value=f'{ds_res[0]:.3f}')
        ds_res_z_lbl = wx.TextCtrl(self, value=f'{ds_res[2]:.3f}')
        n_tiles_inp = wx.TextCtrl(self, value=f'{ds_npos}')
        ds_save_tile_h5 = wx.CheckBox(self)
        ds_save_tile_h5.SetValue(True)

        ds_align = wx.CheckBox(self, pos=(10, 220))
        ds_align.SetValue(True)
        
        ds_align_cb = wx.ComboBox(self, -1, style=wx.CB_READONLY)
        
        channels = sorted(list(ch_map.values()))
        for ch in channels:
            ds_align_cb.Append(ch)
        ds_align_cb.SetSelection(1)  # wx.NOT_FOUND
        
        els = {
            'ds_process': ds_process,
            'ds_dir_lbl' : ds_dir_lbl,
            'ds_date_lbl' : ds_date_lbl,
            'ds_t_lbl' : ds_t_lbl,
            'ds_res_xy_lbl' : ds_res_xy_lbl,
            'ds_res_z_lbl' : ds_res_z_lbl,
            'n_tiles_inp': n_tiles_inp,
            'ds_save_tile_h5' : ds_save_tile_h5,
            'ds_align' : ds_align,
            'ds_align_cb' : ds_align_cb
        }
        
        self.rows.append(els)
        
        self.sizer.Add(ds_process, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_dir_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_date_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_t_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_res_xy_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_res_z_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(n_tiles_inp, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_save_tile_h5, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_align , border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_align_cb , border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)


    def gen_titel(self):
        ds_process = wx.StaticText(self, label='process')
        ds_dir_lbl = wx.StaticText(self, label='directory')
        ds_date_lbl = wx.StaticText(self, label='date')
        ds_t_lbl = wx.StaticText(self, label='dataset')
        ds_res_xy_lbl = wx.StaticText(self, label='pixel xy, um')
        ds_res_z_lbl = wx.StaticText(self, label='pixel z, um')
        n_tiles_inp = wx.StaticText(self, label='num tiles')
        ds_save_tile_h5 = wx.StaticText(self, label='save tile h5')
        ds_align = wx.StaticText(self, label='align frames')
        ds_align_cb = wx.StaticText(self, label='align channel')

        self.sizer.Add(ds_process, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_dir_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_date_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_t_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_res_xy_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_res_z_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(n_tiles_inp, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_save_tile_h5, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_align , border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_align_cb , border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)


def get_runs_info(run_root, dirs):
    res = [0.7731239092495636, 0.7731239092495636, 4.0]
    runs_info = {}

    for d in dirs:
        p = os.path.join(run_root, d, '*' )
        dss = glob.glob(p)

        dates = []
        times = []
        ress = []
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
            if len(findres)==0:
                continue
                
            date, t = findres[0]
            
            
            dates.append(date)
            times.append(t)
            ress.append(res)
            n_poss.append(n_pos)

        dates = list(set(dates))
        assert len(dates)==1

        runs_info[d] = (dates[0], times, ress, n_poss)

    return runs_info

def gen_run_align_bat(run_root, run, date, doc, n_tiles, align_channel_code):
    try:
        bat = '@echo off\n'
        bat += f'set root={run_root}\n'
        bat += f'pushd %~dp0\n'
        bat += f'set proc_align={scripts_root}proc_align.bat\n'

        align_channel_code_u = align_channel_code.upper()
        timestamp = doc[:8]
        
        if n_tiles>1:
            for pos in range(n_tiles):
                s = 'call %proc_align% %root%'+f'\\{run} {date}_Doc1_{doc}__pos_{pos}  {timestamp}_Doc1     ALGN_{align_channel_code_u} {align_channel_code}'
                bat += s+'\n'
        else:
            s = 'call %proc_align% %root%'+f'\\{run} {date}_Doc1_{doc}  {timestamp}_Doc1     ALGN_{align_channel_code_u} {align_channel_code}'
            bat += s+'\n'


        bat += f'popd\n'

        #print(bat)

        fname = os.path.join(run_root, f'proc_align_{run}_{date}_{doc}_{n_tiles}tl_{align_channel_code}.bat')
        with open(fname, 'wt') as f:
            f.write(bat)

        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.call(fname, startupinfo=si)
        
    except Exception as e:
        print(e)


def do_process(run_root, run, date, doc, res, n_tiles, do_tile_h5, do_align, align_channel):
    print('\n\n Proocessing:')
    print(run_root, run, date, doc, res, n_tiles, do_tile_h5, do_align)
    
    timestamp = doc[:8]
    #split
    if n_tiles >1:
        split_multiposition(f'{run_root}/{run}', f'{date}_Doc1_', doc, n_tiles, move=True)
        pass
    
    # convert tiled data to h5
    if do_tile_h5:
        if n_tiles >1:
            for pos in range(n_tiles):
                ds_dir_wc = f'{run_root}/{run}/{date}_Doc1_{doc}__pos_{pos}/'
                convert_to_h5_tmpl(ds_dir_wc, timestamp, comp=True, res=res)
        else:
            ds_dir_wc = f'{run_root}/{run}/{date}_Doc1_{doc}/'
            convert_to_h5_tmpl(ds_dir_wc, timestamp, comp=True, res=res)
            
    
    if do_align:
        align_channel_code = align_channel[2].lower()
        align_channel_code_u = align_channel_code.upper()
        gen_run_align_bat(run_root, run, date, doc, n_tiles, align_channel_code)

        if n_tiles >1:
            for pos in range(n_tiles):
                ds_dir_wc = f'{run_root}/{run}/{date}_Doc1_{doc}__pos_{pos}_ALGN_{align_channel_code_u}/'
                convert_to_h5_tmpl(ds_dir_wc, timestamp, comp=True, res=res)
        else:
            ds_dir_wc = f'{run_root}/{run}/{date}_Doc1_{doc}_ALGN_{align_channel_code_u}/'
            convert_to_h5_tmpl(ds_dir_wc, timestamp, comp=True, res=res)
    
def process_all(ds_list):
    for run_root, run, date, doc, res, process, n_tiles, do_tile_h5, do_align, align_channel in ds_list:
        if process:
            try:
                do_process(run_root, run, date, doc, res, n_tiles, do_tile_h5, do_align, align_channel)
            except Exception as e:
                print(e)
            
def run():
    app = wx.App()
    cont = True

    if cont:
        frameM = ModalWin(None, PathPanel)
        if frameM.ShowModal() == wx.ID_OK:
            #print("Exited by Ok button")
            pass
        else:
            #print("Exited by X button")
            cont = False
            
        path = frameM.a.path.Value
        frameM.Destroy()

    if cont:
        run_root = os.path.dirname(path)
        dirs = [os.path.basename(path)]
        runs_info = get_runs_info(run_root, dirs)
        
        #print(runs_info)
        
        frameM = ModalWin(None, DatasetProcConfigurator, (run_root, runs_info))
        if frameM.ShowModal() == wx.ID_OK:
            #print("Exited by Ok button")
            pass
        else:
            #print("Exited by X button")
            cont = False
            
        ds_list = []
        for row in frameM.a.rows:
            #for k, el in row.items():
            #    try:
            #        v = el.Value
            #    except:
            #        v = el.Label
            #    
            #    print(k, v)
                
            #run, (date, docs, rs)
            process = row['ds_process'].Value
            run = row['ds_dir_lbl'].Label
            date = row['ds_date_lbl'].Label
            doc  = row['ds_t_lbl'].Label
            
            rs_xy = float(row['ds_res_xy_lbl'].Value)
            rs_z = float(row['ds_res_z_lbl'].Value)
            
            n_tiles = int(row['n_tiles_inp'].Value)
            do_tile_h5 = row['ds_save_tile_h5'].Value
            do_align = row['ds_align'].Value
            align_channel = row['ds_align_cb'].Value
            
            ds_list.append([run_root, run, date, doc, [rs_xy, rs_xy, rs_z], process, n_tiles, do_tile_h5, do_align, align_channel])
        frameM.Destroy()
    app.MainLoop()
    
    
    print(ds_list)
    if cont:
        process_all(ds_list)
        
    print('\n\n\n DONE!')

if __name__=='__main__':
    run()

