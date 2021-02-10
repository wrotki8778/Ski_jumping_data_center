# -*- coding: utf-8 -*-
"""
Test module
"""
import os
import pandas as pd
import numpy as np
import re
from tika import parser
os.chdir('C:/Users/kubaf/Documents/Skoki')
actual_comps = pd.read_csv(os.getcwd()+'\\all_comps.csv')
actual_comps = actual_comps.sort_values(['id'], ascending=[True])
os.chdir('C:/Users/kubaf/Documents/Skoki/PDFs')

def parse_weather(comp):
    if not os.path.isfile(os.getcwd()+'//'+comp['id']+'.pdf'):
        return [[comp['id'], '']]
    print(comp.name)
    parsed = parser.from_file(comp['id']+'.pdf')
    tekst=parsed["content"]
    tekst=tekst.lower()
    tekst_lin=tekst.splitlines()
    tekst_lin = [i for i in tekst_lin if i] 
    word1 = 'weather information'
    word2 = 'statistics'
    try:
        start = min([i for i,x in enumerate(tekst_lin) if x.count(word1)])
        end = max([i for i,x in enumerate(tekst_lin) if x.count(word2)])
    except ValueError:
        return [[comp['id'], '']]
    tekst_lin = tekst_lin[start:end]
    word_acc = ['1st round', 'trial round', '2nd round',
                'training', 'qualification', 'final round']
    tekst_lin = [[comp['id'], x] for x in tekst_lin 
                 if sum(c.isdigit() for c in x) > 4
                 and sum([x.count(word) for word in word_acc])]
    return tekst_lin

def process_weather(data,comps):
    fis_code = data[0]
    line = data[1]
    comp = comps[comps['id'] == fis_code].iloc[0]
    print(comp)
    month = int(comp['date'][5:7])
    if comp['type'] in (0,4,5) and int(comp['season']>2012):
        tmp = line.split(' ')
        humid = float(tmp[-3])
        snow = float(tmp[-6])
        air = float(tmp[-9])
        weather_type = ''
        round_type = ''
        i = -12
        while not sum(c.isdigit() for c in tmp[i]):
            weather_type = tmp[i] + ' ' + weather_type
            i = i-1
        i = 0
        while not sum(c.isdigit() for c in tmp[i]):
            round_type = tmp[i] + ' ' + round_type
            i = i+1    
        return [fis_code,humid,snow,air,weather_type,round_type]
    return [fis_code,np.nan,np.nan,np.nan,np.nan,np.nan]
    
weather_data = [parse_weather(comp) for i,comp in actual_comps.iterrows()]
all_data = [item for sublist in weather_data for item in sublist]
processed_data = [process_weather(x, actual_comps) for x in all_data]
comp=actual_comps.iloc[10] 






