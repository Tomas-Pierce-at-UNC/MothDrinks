#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:41:51 2023

@author: tomas
"""

import glob
import pathlib

csvnames = glob.glob("framedata/**/*.csv", recursive=True)
for name in csvnames:
    csvpath = pathlib.Path(name)
    txtpath = csvpath.with_suffix(".txt")
    with open(csvpath) as csvhandle:
        csvtxt = csvhandle.read()
    with open(txtpath, "w") as txthandle:
        txthandle.write(csvtxt)
