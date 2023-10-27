#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:19:58 2023

@author: tomas
"""

import pandas
import warnings

import cine

# mm between each marking on the tube
TENTH_DISTANCE = 5.5

# mL between each marking on the tube
TENTH_VOL = 0.1

def get_table():
    table = pandas.read_csv("unit_convert_table.csv")
    del table['ratio']
    table['lin_scale'] = TENTH_DISTANCE / table['distance(pixels)']
    table['vol_scale'] = TENTH_VOL / table['distance(pixels)']
    return table

class UnitConversion:
    
    def __init__(self):
        self.table = get_table()
        framerates = []
        for name in self.table.filename:
            c = cine.Cine(name)
            framerate = c.framerate
            c.close()
            framerates.append(framerate)
        self.table['framerate'] = framerates
        
    def get_attrib(self, name, attrib):
        for i, filename in enumerate(self.table['filename']):
            if filename.endswith(name):
                row = self.table.iloc[i]
                if pandas.isnull(row[attrib]):
                    warnings.warn("Video-specific scale not available, using average scale", Warning)
                    return self.table[attrib].mean()
                else:
                    return row[attrib]
        
        raise LookupError(
            "could not find file {} in list of known recordings".format(
                name)
            )
    
    def get_vol_factor(self, name):
        return self.get_attrib(name, 'vol_scale')
    
    def get_lin_factor(self, name):
        return self.get_attrib(name, 'lin_scale')
    
    def get_framerate(self, name):
        return self.get_attrib(name, 'framerate')
    
