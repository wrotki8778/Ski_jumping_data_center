# -*- coding: utf-8 -*-
"""
Module to parse stats, weather infos
and to include round information in a results .csv file.

@author: wrotki8778
"""
import os
import os.path
import glob
import pandas as pd
import numpy as np
from tika import parser
os.chdir('C:/Users/kubaf/Documents/Skoki')


def disperse_text(string):
    """Process a string to split a multi-'.'-substrings."""
    new_string = string
    if string.count('-'):
        new_string = string.split('-')[0]+' '\
            + ' '.join(['-'+e for e in string.split('-')[1:] if e])
    tmp = new_string.split(' ')
    if not([i for i in tmp if i.count('.') > 1]):
        return new_string
    if new_string.count('.') > 1:
        index = min([i for i, x in enumerate(new_string) if x == '.'])+2
        return new_string[:index]+' ' + new_string[index:]
    return new_string


def parse_weather(comp):
    """
    Return lines from a PDF with stats and weather information.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py file (check "database" output for details)..

    Returns
    -------
    list
        List containing two list of strings - one for weather information,
        second for statistics.

    """
    if not os.path.isfile(os.getcwd()+'//PDFs//'+comp['id']+'.pdf'):
        return ['', '']
    parsed = parser.from_file(os.getcwd()+'//PDFs//'+comp['id']+'.pdf')
    text = parsed["content"]
    text = text.lower()
    line_text = text.splitlines()
    line_text = [i for i in line_text if i]
    word1 = 'weather information'
    word2 = 'statistics'
    try:
        start = min([i for i, x in enumerate(line_text) if x.count(word1)])
        end = max([i for i, x in enumerate(line_text) if x.count(word2)])
    except ValueError:
        return ['', '']
    line_text_1 = line_text[start:end]
    line_text_2 = line_text[end:]
    line_text_2 = [x.replace('tiral', 'trial') for x in line_text_2]
    word_acc = ['1st round', '2nd round', '3rd round', '4th round',
                'training 1', 'training 2', 'trial round',
                'final round', 'training 3', 'qualification', 'prologue']
    line_text_1 = [x for x in line_text_1
                   if sum(c.isdigit() for c in x) > 4
                   and sum([x.count(word) for word in word_acc])]
    line_text_2 = [x for x in line_text_2
                   if sum(c.isdigit() for c in x) > 4
                   and sum([x.count(word) for word in word_acc])]
    return line_text_1, line_text_2


def process_weather_init(data, comp):
    """
    Return a list with extracted values from data about weather.

    Parameters
    ----------
    data : string
        Initial string containing weather data.
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py file (check "database" output for details).

    Returns
    -------
    list
        List with following information from the day of the competition:
            - humid - humidity in percent (0%-100%),
            - snow - snow temperature in Celsius degrees,
            - air - air temperature in Celsius degrees,
            - weather_type - string with description of weather
            (sunny, cloudy etc.)
            - round_type - string with the name of round
            (1st round, 2nd round, trial round etc.)
            - max_wind/avg_wind/min_wind - three numbers which describes the
            variability of wind in a given round. Negative values indicate
            tail wind and vice versa - positive values suggest head wind.

    """
    fis_code = comp['id']
    if not data:
        return [fis_code, np.nan, np.nan, np.nan, np.nan, np.nan]
    line = data + ' '
    month = int(comp['date'][5:7])
    round_type = ''
    round_types = ['1st round ', '2nd round ', '3rd round ', '4th round ',
                   'training 1 ', 'training 2 ', 'trial round ',
                   'final round ', 'training 3 ',
                   'qualification ', 'prologue ']
    if comp['type'] in (0, 2, 4, 5):
        tmp = line.split(' ')
        tmp = [x for x in tmp if x]
        humid = float(tmp[-4])
        wind = sorted([float(tmp[-1]), float(tmp[-2]), float(tmp[-3])])
        max_wind = wind[2]
        avg_wind = wind[1]
        min_wind = wind[0]
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
    for tag in round_types:
        if line.count(tag):
            round_type = tag
            line = line.replace(tag, '')
            break
    tmp = line.split(' ')
    tmp = [x for x in tmp if x]
    humid = float(tmp[-4])
    wind = sorted([float(tmp[-1]), float(tmp[-2]), float(tmp[-3])])
    max_wind = wind[2]
    avg_wind = wind[1]
    min_wind = wind[0]
    if 4 < month < 11:
        snow, air = [np.nan, float(tmp[-5])]
    else:
        if len(tmp[-5]) == 2:
            snow, air = [float(tmp[-5][0]), float(tmp[-5][1])]
        else:
            last_minus = max([i for i, x in enumerate(tmp[-5])
                              if x == '-'])
            air, snow = \
                [float(tmp[-5][:last_minus]), float(tmp[-5][last_minus:])]
    weather_type = tmp[0]
    i = 1
    while not sum(c.isdigit() for c in tmp[i]):
        weather_type = weather_type + ' ' + tmp[i]
        i = i+1
    return [fis_code, humid, snow, air, weather_type,
            round_type, max_wind, avg_wind, min_wind]


def process_stats_init(data, comp):
    """
    Return a list with extracted values from data about stats.

    Parameters
    ----------
    data : string
        Initial string containing weather data.
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py file (check "database" output for details).

    Returns
    -------
    list
        List with following information from the day of the competition:
            - fis_code - FIS code of a competition in a ,
            - snow - snow temperature in Celsius degrees,
            - air - air temperature in Celsius degrees,
            - weather_type - string with description of weather
            (sunny, cloudy etc.)
            - round_type - string with the name of round
            (1st round, 2nd round, trial round etc.)
            - max_wind/avg_wind/min_wind - three numbers which describes the
            variability of wind in a given round. Negative values indicate
            tail wind and vice versa - positive values suggest head wind.

    """
    fis_code = comp['id']
    line = data
    line = line.replace('/', ' ')
    if not line:
        return [fis_code, np.nan, np.nan, np.nan, np.nan, np.nan]
    gate = np.nan
    all_jumpers = np.nan
    counted_jumpers = np.nan
    all_countries = np.nan
    round_type = np.nan
    round_types = ['1st round ', '2nd round ', '3rd round ', '4th round ',
                   'training 1 ', 'training 2 ', 'trial round ',
                   'final round ', 'training 3 ',
                   'qualification ', 'prologue ']
    for tag in round_types:
        if line.count(tag):
            round_type = tag
            line = line.replace(tag, '')
            break
    tmp = line.split(' ')
    tmp = [disperse_text(x) for x in tmp if x]
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
    gate = float(tmp[0])
    counted_jumpers = float(tmp[-2])
    all_jumpers = float(tmp[-4])
    all_countries = float(tmp[-3])
    return [fis_code, round_type, gate, counted_jumpers,
            all_jumpers, all_countries]


def process_stats(comp):
    """
    Merge infos from process_weather_init and process_stats_init
    into a single Pandas series.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).

    Returns
    -------
    complete_series : Pandas dataframe
        Merged infos from process_weather_init and process_stats_init
        for all competition rounds in comp series. For the variables inside
        you can read the output of the two mentioned procedures.

    """
    data = parse_weather(comp)
    try:
        weather_data = [process_weather_init(x, comp) for x in data[0]]
    except ValueError:
        weather_data = [[comp['id'], np.nan, np.nan,
                        np.nan, np.nan, 'error',
                        np.nan, np.nan, np.nan] for x in data[0]]
    try:
        stats_data = [process_stats_init(x, comp) for x in data[1]]
    except ValueError:
        stats_data = [[comp['id'], 'error', np.nan, np.nan, np.nan, np.nan]
                      for x in data[1]]
    weather_names = ['fis_code', 'humid', 'snow', 'air', 'weather_type',
                     'round_type', 'max_wind', 'avg_wind', 'min_wind']
    stats_names = ['fis_code', 'round_type', 'gate', 'counted_jumpers',
                   'all_jumpers', 'all_countries']
    weather_series = pd.DataFrame(weather_data, columns=weather_names)
    stats_series = pd.DataFrame(stats_data, columns=stats_names)
    complete_series = pd.merge(weather_series, stats_series,
                               on=['fis_code', 'round_type'], how='outer')
    return complete_series


def get_round_names(comp):
    """Return names of all counted rounds in a competition from comp variable."""
    types = ['WC', 'COC', 'GP', 'FC', 'SFWC', 'WSC', 'WJC']
    list_of_files = glob.glob(os.getcwd()+'/stats/*'+str(comp['season'])
                              + '*'+types[comp['type']]+'*')
    if not list_of_files:
        return []
    names = []
    for item in list_of_files:
        database = pd.read_csv(item)
        names = names + [x['round_type'] for i, x in database.iterrows()
                         if x['fis_code'] == comp['id']]
    names = ['NA']+[str(x) for x in list(np.unique(names))]
    return names


def cummulative(vector, comp):
    """
    Return a vector of integers containing information about round.

    Parameters
    ----------
    vector : list of strings
        List of subsequent names in a given competition round
        (have to be sorted by the achieved points)
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    Returns
    -------
    output : list of integers
        List of subsequent rounds in a given competition round
        (have to be sorted by the achieved points). 0 means
        "first (in order) round", 1 means "second (in order) round" etc.
    """
    i = 1
    counter = 1
    no_comps = len(get_round_names(comp))
    output = [counter]
    while i < len(vector):
        if vector[i] == vector[i-1]:
            counter = counter + 1
        else:
            counter = 1
        output = output + [counter]
        i = i+1
    if comp['id'].count('RLT') and (comp['type'] in (1, 3, 6)
                                    or comp['season'] < 2016
                                    or (comp['type'] == 2
                                        and comp['season'] == 2016)):
        for i in range(len(vector)):
            print(output[i]+1, no_comps)
            if output[i+1:]:
                if (output[i] + 1 < no_comps and output[i+1] == 1):
                    print('replace')
                    output[i-output[i]+1:i+1] = np.repeat(0, output[i], axis=0)
            elif output[i] + 1 < no_comps:
                output[i-output[i]+1:i+1] = np.repeat(0, output[i], axis=0)
    print(output)
    return output


def get_round(comp):
    """
    Return a improved results containing information about round.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).

    Returns
    -------
    results: Pandas dataframe
        File read from the appropriate .csv file
        (see the untitled0.py script for reference) with attached info about
        the round (1st round, 2nd round etc.)

    """
    round_names = get_round_names(comp)
    print(round_names)
    if round_names in (['NA'], ['NA', 'error']):
        return []
    directory = os.getcwd()+'/results/'+comp['id']+'.csv'
    if not os.path.isfile(directory):
        return []
    results = pd.read_csv(os.getcwd()+'/results/'+comp['id']+'.csv')
    print(results)
    tmp = [round_names[i] for i in cummulative(results['name'], comp)]
    print(tmp)
    results['round'] = tmp
    return results


list_of_files = glob.glob(os.getcwd()+'/comps/*')
directory = max(list_of_files, key=os.path.getctime)

comps = pd.read_csv(directory)
comps = comps[comps['k-point'].notnull()]
# comps['type'] = 0
all_stats_names = ['fis_code', 'humid', 'snow', 'air', 'weather_type',
                   'round_type', 'max_wind', 'avg_wind', 'min_wind',
                   'gate', 'counted_jumpers', 'all_jumpers', 'all_countries']
stats_dataframe = pd.DataFrame([], columns=all_stats_names)

directory_stats = directory.replace('comps', 'stats')
directory_stats_2 = directory.replace('comps', 'elastic_stats')

for k, comp_to_process in comps.iterrows():
    if os.path.isfile(directory_stats):
        continue
    content = process_stats(comp_to_process)
    stats_dataframe = stats_dataframe.append(content, ignore_index=True)

stats_dataframe = stats_dataframe[stats_dataframe['round_type'] != 'error']
if not os.path.isfile(directory_stats):
    stats_dataframe.to_csv(directory_stats, index=False)
stats_dataframe.to_csv(directory_stats_2, index=False)

comps = pd.read_csv(directory)
for k, comp_to_process in comps.iterrows():
    corrected_results = pd.DataFrame(get_round(comp_to_process))
    if not corrected_results.empty:
        stats_dataframe =\
            corrected_results.to_csv(os.getcwd()+'\\elastic_results\\'
                                     + comp_to_process['id']+'.csv',
                                     index=False)
