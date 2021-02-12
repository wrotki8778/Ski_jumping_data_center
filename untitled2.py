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
    if not line:
        return [fis_code,np.nan,np.nan,np.nan,np.nan,np.nan]
    comp = comps[comps['id'] == fis_code].iloc[0]
    print(comp)
    month = int(comp['date'][5:7])
    if comp['type'] in (0,2,4,5):
        tmp = line.split(' ')
        max_wind, avg_wind, min_wind, humid = \
            [float(tmp[-2]), float(tmp[-1]), float(tmp[-3]),float(tmp[-4])]
        if comp['type'] != 2:
            if int(comp['season']<2012):
                i = -7
                snow, air = \
                [float(tmp[-5]), float(tmp[-6])]
            else:
                i= -13
                snow, air = \
                [float(tmp[-7]), float(tmp[-10])]
        else:
            if int(comp['season']<2012):
                i = -6
                snow, air = \
                [np.nan, float(tmp[-5])]
            else:
                i= -10
                snow, air = \
                [np.nan, float(tmp[-7])]
        weather_type = ''
        round_type = tmp[0]
        while not sum(c.isdigit() for c in tmp[i]):
            weather_type = tmp[i] + ' ' + weather_type
            i = i-1
        i = 1
        while not sum(c.isdigit() for c in tmp[i]):
            round_type = round_type + ' ' + tmp[i]
            i = i+1
        return [fis_code, humid, snow, air, weather_type,
                round_type, max_wind, avg_wind, min_wind]
    else:
        round_type=''
        round_types = ['1st round', '2nd round', 'training 1',
                       'training 2', 'trial round', 'final round',
                       'training 3']
        for tag in round_types:
            if line.count(tag):
                round_type=tag
                line = line.replace(tag, '')
                break
        tmp = line.split(' ')
        max_wind, avg_wind, min_wind, humid = \
            [float(tmp[-2]), float(tmp[-1]), float(tmp[-3]), float(tmp[-4])]
        if month > 4 and month < 11:
            snow, air = \
                [np.nan, float(tmp[-5])]
        else:
            if len(tmp[-5]) == 2:
                snow, air = \
                    [float(tmp[-5][0]), float(tmp[-5][1])]
            else:
                last_minus = max([i for i,x in enumerate(tmp[-5]) if x == '-'])
                snow, air = \
                    [float(tmp[-5][:last_minus]), float(tmp[-5][last_minus:])]
        weather_type = tmp[0]
        i = 1
        while not sum(c.isdigit() for c in tmp[i]):
            weather_type = weather_type + ' ' + tmp[i]
            i = i+1
        return [fis_code,np.nan,np.nan,np.nan,weather_type,round_type,
                np.nan, np.nan, np.nan]
    return [fis_code,np.nan,np.nan,np.nan,np.nan,np.nan]

weather_data = [parse_weather(comp) for i,comp in actual_comps.iterrows()]
all_data = [item for sublist in weather_data for item in sublist]
processed_data = [process_weather(x, actual_comps) for x in all_data]
comp=actual_comps.iloc[10] 






