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

def rozdziel(string):
    """Process a string to split a multi-'.'-substrings."""
    new_string = string
    if string.count('-'):
        new_string = string.split('-')[0]+' ' \
            + ' '.join(['-'+e for e in string.split('-')[1:] if e])
    tmp = new_string.split(' ')
    if not([i for i in tmp if i.count('.') > 1]):
        return new_string
    if new_string.count('.') > 1:
        index = min([i for i, x in enumerate(new_string) if x == '.'])+2
        return new_string[:index]+' ' + new_string[index:]
    return new_string


def parse_weather(comp):
    if not os.path.isfile(os.getcwd()+'//'+comp['id']+'.pdf'):
        return [[[comp['id'], '']], [[comp['id'], '']]]
    print(comp.name)
    parsed = parser.from_file(comp['id']+'.pdf')
    tekst = parsed["content"]
    tekst = tekst.lower()
    tekst_lin = tekst.splitlines()
    tekst_lin = [i for i in tekst_lin if i]
    word1 = 'weather information'
    word2 = 'statistics'
    try:
        start = min([i for i, x in enumerate(tekst_lin) if x.count(word1)])
        end = max([i for i, x in enumerate(tekst_lin) if x.count(word2)])
    except ValueError:
        return [[[comp['id'], '']], [[comp['id'], '']]]
    tekst_lin_1 = tekst_lin[start:end]
    tekst_lin_2 = tekst_lin[end:]
    word_acc = ['1st round', '2nd round', '3rd round', '4th round',
                'training 1', 'training 2', 'trial round',
                'final round', 'training 3']
    tekst_lin_1 = [[comp['id'], x] for x in tekst_lin_1
                   if sum(c.isdigit() for c in x) > 4
                   and sum([x.count(word) for word in word_acc])]
    tekst_lin_2 = [[comp['id'], x] for x in tekst_lin_2
                   if sum(c.isdigit() for c in x) > 4
                   and sum([x.count(word) for word in word_acc])]
    return tekst_lin_1, tekst_lin_2


def process_weather_init(data,comps):
    fis_code = data[0]
    line = data[1] + ' '
    if not data[1]:
        return [fis_code, np.nan, np.nan, np.nan, np.nan, np.nan]
    comp = comps[comps['id'] == fis_code].iloc[0]
    print(comp)
    month = int(comp['date'][5:7])
    round_type = ''
    round_types = ['1st round ', '2nd round ', '3rd round ', '4th round ',
                   'training 1 ', 'training 2 ', 'trial round ',
                   'final round ', 'training 3 ']
    if comp['type'] in (0, 2, 4, 5):
        tmp = line.split(' ')
        tmp = [x for x in tmp if x]
        max_wind, avg_wind, min_wind, humid = \
            [float(tmp[-2]), float(tmp[-1]), float(tmp[-3]), float(tmp[-4])]
        if comp['type'] != 2:
            if int(comp['season']) < 2012:
                i = -7
                snow, air = [float(tmp[-5]), float(tmp[-6])]
            else:
                i = -13
                snow, air = [float(tmp[-7]), float(tmp[-10])]
        else:
            if int(comp['season']) < 2012:
                i = -6
                snow, air = [np.nan, float(tmp[-5])]
            else:
                i = -10
                snow, air = [np.nan, float(tmp[-7])]
        weather_type = ''
        round_type = tmp[0]
        while not sum(c.isdigit() for c in tmp[i]):
            weather_type = tmp[i] + ' ' + weather_type
            i = i-1
        for tag in round_types:
            if line.count(tag):
                round_type = tag
                break
        return [fis_code, humid, snow, air, weather_type,
                round_type, max_wind, avg_wind, min_wind]
    else:
        for tag in round_types:
            if line.count(tag):
                round_type = tag
                line = line.replace(tag, '')
                break
        tmp = line.split(' ')
        tmp = [x for x in tmp if x]
        max_wind, avg_wind, min_wind, humid = \
            [float(tmp[-1]), float(tmp[-2]), float(tmp[-3]), float(tmp[-4])]
        if month > 4 and month < 11:
            snow, air = \
                [np.nan, float(tmp[-5])]
        else:
            if len(tmp[-5]) == 2:
                snow, air = \
                    [float(tmp[-5][0]), float(tmp[-5][1])]
            else:
                last_minus = max([i for i, x in enumerate(tmp[-5]) if x == '-'])
                air, snow = \
                    [float(tmp[-5][:last_minus]), float(tmp[-5][last_minus:])]
        weather_type = tmp[0]
        i = 1
        while not sum(c.isdigit() for c in tmp[i]):
            weather_type = weather_type + ' ' + tmp[i]
            i = i+1
        return [fis_code, humid, snow, air, weather_type,
                round_type, max_wind, avg_wind, min_wind]
    return [fis_code,np.nan,np.nan,np.nan,np.nan,np.nan]


def process_stats_init(data,comps):
    fis_code = data[0]
    line = data[1]
    line = line.replace('/',' ')
    if not line:
        return [fis_code,np.nan,np.nan,np.nan,np.nan,np.nan]
    comp = comps[comps['id'] == fis_code].iloc[0]
    gate = np.nan
    all_jumpers = np.nan
    counted_jumpers = np.nan
    all_countries = np.nan
    round_type = np.nan
    round_types = ['1st round ', '2nd round ', '3rd round ', '4th round ',
                   'training 1 ', 'training 2 ', 'trial round ',
                   'final round ', 'training 3 ']
    for tag in round_types:
        if line.count(tag):
            round_type = tag
            line = line.replace(tag, '')
            break
    tmp = line.split(' ')
    tmp = [rozdziel(x) for x in tmp if x]
    if comp['type'] in (0, 2, 4, 5):
        if comp['team']:
            gate = float(tmp[2])
        else:
            gate = float(tmp[0])
        counted_jumpers = float(tmp[-2])
        all_jumpers = float(tmp[-4])
        all_countries = float(tmp[-3])
        return [fis_code, round_type, gate, counted_jumpers,
                all_jumpers, all_countries]
    else:
        gate = float(tmp[0])
        counted_jumpers = float(tmp[-2])
        all_jumpers = float(tmp[-4])
        all_countries = float(tmp[-3])
        return [fis_code, round_type, gate, counted_jumpers,
                all_jumpers, all_countries]


def process_weather(data, comps):
    try:
        return process_weather_init(data, comps)
    except ValueError:
        return [data[0], np.nan, np.nan,
                np.nan, np.nan, 'error',
                np.nan, np.nan, np.nan]


def process_stats(data, comps):
    try:
        return process_stats_init(data, comps)
    except ValueError:
        return [data[0], 'error', np.nan, np.nan, np.nan, np.nan]


data = [parse_weather(comp) for i, comp in actual_comps.iterrows()]
weather_data = [x[0] for x in data]
stats_data = [x[1] for x in data]
all_weather_data = [item for sublist in weather_data for item in sublist]
all_stats_data = [item for sublist in stats_data for item in sublist]
processed_weather_data = [process_weather(x, actual_comps)
                          for x in all_weather_data]
weather_names = ['fis_code', 'humid', 'snow', 'air', 'weather_type',
                 'round_type', 'max_wind', 'avg_wind', 'min_wind']
weather_dataframe = pd.DataFrame(processed_weather_data,
                                 columns = weather_names)
processed_stats_data = [process_stats(x, actual_comps)
                        for x in all_stats_data]
stats_names = ['fis_code', 'round_type', 'gate', 'counted_jumpers',
               'all_jumpers', 'all_countries']
stats_dataframe = pd.DataFrame(processed_stats_data,
                                 columns = stats_names)
complete_dataframe = pd.merge(weather_dataframe, stats_dataframe,
                              on = ['fis_code','round_type'], how = 'left')
w = process_weather_init(all_weather_data[0], actual_comps)
# s = process_stats_init(all_stats_data[6004], actual_comps)
all_error_weather_data = [all_weather_data[i]
                          for i, x in enumerate(processed_weather_data)
                          if x[5] == 'error']
all_error_stats_data = [all_stats_data[i]
                        for i, x in enumerate(processed_stats_data)
                        if x[1] == 'error']
os.chdir('C:/Users/kubaf/Documents/Skoki')
complete_dataframe.to_csv('all_stats.csv',index=False,na_rep='NA')

