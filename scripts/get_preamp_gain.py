#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E. Tamura 2024.06.11
# 
# 全面的に改修
# 高速化のため、library化
# read_gain_caldb を呼んで、caldb FITS を読み込む
# 次に、指定した温度の parameter set を calc_gain_table_at_T で読み込む
# 最後に get_gain_at_F で補完関数をつくって、複数周波数での gain を戻す
# 
# based on get_preamp2_gain.py
# read caldb file for FA/FM preamp gain and get it at any temp and freq
# 
# v0.1 Emi Tamura (2024.05.13)
# determine fits filename from device
# 
# ver 0.0 initial release

import os
import glob
import sys
import re
import numpy as np
import argparse
import matplotlib.pyplot as plt
import padasip as pa
from astropy.io import fits
from scipy import interpolate

#########################################################################
# class for struct of params
#########################################################################
class LUSEE_GAIN:
    def __init__(self):
        self.pro='get_preamp_gain.py'
        self.version='v0.2'
        self.dev_name='fmpre0'
        self.caldb_dir='/home/lusee/etamura/LuSEE/caldb/data/'
        self.gains=np.empty(0)
        self.debug=False
        self.freqs=np.empty(0)
        self.paras=np.empty(0)
        self.min_freq=1.e4
        self.max_freq=1.e8

    #########################################################################
    # calc_gain_table_at_T
    #   calculate gain table from polynomial parameters at specified temperature
    #########################################################################
    def calc_gain_table_at_T (self, temp):
        self.gains=np.empty(0)
        for ifreq in range (len(self.freqs)):
            val=0
            for ipara in range (self.paras.shape[1]):
                val+=temp**(self.paras.shape[1]-ipara-1)*self.paras[ifreq,ipara]
            self.gains=np.append(self.gains, val)
            if self.debug:
                print (self.freqs[ifreq]*1e6, val)
        #return gains

    #########################################################################
    # get_gain_at_F
    #   calculate gain at specified frequency
    #########################################################################
    def get_gain_at_F (self, get_freqs):
        # limit freq
        #index=np.where((self.min_freq<=get_freqs) & (get_freqs<=self.max_freq))[0]
        # interpolate
        f1=interpolate.interp1d(self.freqs, self.gains, kind="cubic")
        return f1(get_freqs/1.e6)

    #########################################################################
    # read_gain_caldb
    #   read gain caldb
    #########################################################################
    def read_gain_caldb (self):
        self.fitsfile=f"{self.caldb_dir}{self.dev_name}_gain_temp_freq_dep.fits"

        # FITS
        hdul = fits.open(self.fitsfile)
        # initialize for repeated call
        self.freqs=np.empty(0)
        self.paras=np.empty(0)
        # extension の数は len(hdul) でもってこれる 便利すぎ
        for iext in range (1, len(hdul)):
            para_num=hdul[iext].data.shape[0]
            for idata in range (len(hdul[iext].data[0])):
                d=hdul[iext].data.names[idata].split()
                self.freqs=np.append(self.freqs, float(d[0]))
                for ipara in range (para_num):
                    self.paras=np.append(self.paras, hdul[iext].data[ipara][idata])

        self.paras=self.paras.reshape(int(len(self.paras)/para_num),para_num)
        #self.interp_gain=interpolate.interp1d(freqs, self.gains, kind="cubic")
        #self.calc_gain_table_at_T(freqs, paras)
        #gain=params.get_gain_at_F(freqs, gains)

#########################################################################
# main
#########################################################################
def get_preamp_gain (*_args):

    #print ('!', *_args)
    #########################################################################
    #引数処理
    #########################################################################
    #global params
    params=LUSEE_GAIN() # just define
    params.dev_name=_args[0]
    params.temp=float(_args[1])
    params.get_freqs=float(_args[2])/1.e6
    params.fitsfile=f"{params.caldb_dir}{params.dev_name}_gain_temp_freq_dep.fits"
    #print ('pop', params.fitsfile, params.temp, params.freq)

    if params.debug:
        print (params.fitsfile, params.temp, params.get_freqs)

    # FITS
    hdul = fits.open(params.fitsfile)
    # extension の数は len(hdul) でもってこれる 便利すぎ
    freqs=np.empty(0)
    paras=np.empty(0)
    for iext in range (1, len(hdul)):
        para_num=hdul[iext].data.shape[0]
        for idata in range (len(hdul[iext].data[0])):
            d=hdul[iext].data.names[idata].split()
            freqs=np.append(freqs, float(d[0]))
            for ipara in range (para_num):
                paras=np.append(paras, hdul[iext].data[ipara][idata])

    paras=paras.reshape(int(len(paras)/para_num),para_num)
    gains=params.calc_gain_table_at_T(freqs, paras)
    gain=params.get_gain_at_F(freqs, gains)
    #print ('!', *_args, gain)
    #print (gain)
    #return gain

if __name__ == '__main__':
    get_preamp_gain(sys.argv)