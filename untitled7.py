# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:38:28 2021

@author: kubaf
"""
import os
import os.path
import re
from datetime import datetime
import math
import glob
import pandas as pd
import numpy as np
from tika import parser
os.chdir('C:/Users/kubaf/Documents/Skoki')

competitions = pd.read_csv(os.getcwd()+'\\all_comps.csv')
qual_competitions = competitions[(competitions['id'].str.contains('RLQ')) & (competitions['season']<2018)]
comp=qual_competitions.iloc[45]
parsed = parser.from_file(os.getcwd()+'//PDFs//'+comp['id']+'.pdf')
tekst = parsed["content"]
tekst = tekst.lower()
tekst_lin = tekst.splitlines()
tekst_lin = [i for i in tekst_lin if i]
word1 = 'prequalified'
word2 = 'weather information'
try:
    start = min([i for i, x in enumerate(tekst_lin) if x.count(word1)])
    end = max([i for i, x in enumerate(tekst_lin) if x.count(word2)])
except ValueError:
    start=0
    end=len(tekst_lin)
