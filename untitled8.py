"""
Script to analyze results from untitled0.py and untitled6.py.

@author: wrotki8778
"""
import os
import pandas as pd
import numpy as np
os.chdir('C:/Users/kubaf/Documents/Skoki')


def merge_names(comps, directory):
    names = pd.DataFrame([], columns=['bib', 'name'])
    names_fis = pd.DataFrame([], columns=['bib', 'codex', 'name'])
    for i, comp in comps.iterrows():
        print(i)
        file_name = directory+str(comp['season'])+'JP'+str(comp['codex'])
        try:
            tmp_naz = pd.read_csv(file_name+'naz.csv', sep=';', header=None)
            tmp_naz.columns = ['bib', 'name']
            names = names.append(tmp_naz)
        except FileNotFoundError:
            pass
        except pd.errors.EmptyDataError:
            pass
        try:
            tmp_nazfis = pd.read_csv(file_name+'nazfis.csv',
                                     sep=';', header=None)
            if not comp['team']:
                tmp_nazfis.columns = ['bib', 'codex', 'name']
            else:
                tmp_nazfis.columns = ['codex', 'name']
            names_fis = names_fis.append(tmp_nazfis)
        except FileNotFoundError:
            pass
        except pd.errors.EmptyDataError:
            pass
    names_fis['name'] = names_fis['name'].str.lower()
    names_fis = pd.merge(names_fis, names, how='right', on=['name'])
    names_fis = names_fis.drop_duplicates(['name', 'codex'])
    names_fis = names_fis.drop(['bib_x', 'bib_y'], axis=1)
    names = names.drop(['bib'], axis=1)
    names = names.drop_duplicates()
    names_fis = pd.merge(names_fis, names, how='left', on=['name'])
    return names_fis


def merge_comps(names, comps, directory):
    columns_names = ['name', 'wind', 'wind_comp',
                     'dist', 'speed', 'dist_points',
                     'note_1', 'note_2', 'note_3',
                     'note_4', 'note_5', 'note_points',
                     'points', 'loc', 'gate',
                     'gate_points', 'id']
    results = pd.DataFrame([], columns=columns_names)
    for i, comp in comps.iterrows():
        print(i)
        try:
            tmp = pd.read_csv(directory+str(comp['id'])+'.csv',
                              sep=',', na_values=['', 'NA'])
            tmp['id'] = comp['id']
            results = results.append(tmp)
        except FileNotFoundError:
            pass
        except pd.errors.EmptyDataError:
            pass
    results = pd.merge(results, names, how='left', on=['name'])
    results = results.drop(['name', 'Unnamed: 0'], axis=1)
    return results

def merge_stats(directory):
    columns_names = ['fis_code', 'humid', 'snow',
                     'air', 'weather_type', 'round_type', 'max_wind',
                     'avg_wind', 'min_wind', 'gate', 'counted_jumpers',
                     'all_jumpers', 'all_countries']
    stats = pd.DataFrame([], columns=columns_names)
    list_of_files = os.listdir(directory)
    for item in list_of_files:
        tmp = pd.read_csv(directory+'\\'+item, sep=',')
        stats = stats.append(tmp)
    stats = stats.drop_duplicates(['fis_code', 'round_type'])
    stats[['humid', 'snow', 'air', 'max_wind',
           'avg_wind', 'min_wind', 'gate', 'counted_jumpers',
           'all_jumpers', 'all_countries']] =\
        stats[['humid', 'snow', 'air', 'max_wind',
               'avg_wind', 'min_wind', 'gate', 'counted_jumpers',
               'all_jumpers', 'all_countries']].apply(pd.to_numeric)
    return stats


def merge_infos(directory):
    columns_names = ['codex', 'place', 'gender', 'hill_size_x',
                     'team', 'season', 'hill_size_y', 'k-point',
                     'meter value', 'gate factor', 'wind factor',
                     'type', 'date', 'id', 'training']
    comps = pd.DataFrame([], columns=columns_names)
    list_of_files = os.listdir(directory)
    for item in list_of_files:
        tmp = pd.read_csv(directory+'\\'+item, sep=',')
        country = [x.rsplit(' ', 1)[1][1:4] for x in tmp['place']]
        new_place = [x.rsplit(' ', 1)[0] for x in tmp['place']]
        tmp['place'] = pd.DataFrame(new_place)
        tmp['country'] = pd.DataFrame(country)
        comps = comps.append(tmp)
    comps = comps.drop_duplicates(['id'])
    comps[['hill_size_x', 'team', 'season', 'hill_size_y', 'k-point',
           'meter value', 'gate factor', 'wind factor', 'type', 'training']] =\
        comps[['hill_size_x', 'team',
               'season', 'hill_size_y', 'k-point',
               'meter value', 'gate factor',
               'wind factor', 'type', 'training']].apply(pd.to_numeric)
    return comps


def new_rating(ratingi, k):
    delty = []
    for i, ocena in enumerate(ratingi):
        if np.isnan(ocena):
            delty.append(0)
            continue
        delta = 0
        exp_score = 0
        fact_score = 0
        for j, inna_ocena in enumerate(ratingi):
            if np.isnan(inna_ocena):
                continue
            if i == j:
                continue
            if i < j:
                exp_score = exp_score+1/(10**((inna_ocena-ocena)/400)+1)
                fact_score = fact_score+1
            else:
                exp_score = exp_score+1/(10**((inna_ocena-ocena)/400)+1)
        delta = k*(fact_score-exp_score)/10
        delty.append(delta)
    return delty


def append_rating(results, i, comp, rating_act, rating_db, k, round_name):
    ratingi = results['cumm_rating']
    results['delty'] = new_rating(ratingi, k)
    results['id'] = comp['id']
    results['round'] = round_name
    results['number'] = i
    new_rating_act = rating_act.append(results, ignore_index=True)
    new_rating_db = pd.merge(rating_db, results[['codex', 'delty']],
                             on='codex', how='left')
    new_rating_db['cumm_rating'] = new_rating_db['cumm_rating']\
        + new_rating_db.fillna(0)['delty']
    new_rating_db = new_rating_db.drop(['delty'], axis=1)
    return [new_rating_act, new_rating_db]


def build_rating(comps, results, names):
    rating_db = pd.DataFrame(names['codex'])
    rating_db = rating_db.drop_duplicates()
    rating_db.dropna()
    rating_db['cumm_rating'] = 1000
    rating_act = pd.DataFrame()
    for i, comp in comps.iterrows():
        k = 8
        omit_sort=0
        print(k)
        all_results = results[(results['id'] == comp['id'])
                              & (results['codex'].notna())]
        if all_results.empty:
            omit_sort=1
            try:
                file_name = os.getcwd()+'\\nazwy\\'+comp['id'][:10]+'nazfis.csv'
                all_results = pd.read_csv(file_name, sep=';', header=None)
                if comp['team']:
                    continue
                all_results.columns = ['bib', 'codex', 'name']
                all_results['round'] = 'whole competition '
                all_results['points'] = 1
                print('imported from nazfis.csv file')
                round_names = ['whole competition ']
            except pd.errors.EmptyDataError:
                continue
        else:
            round_names = np.unique([x['round']
                                     for i, x in all_results.iterrows()])
        all_results = pd.DataFrame(all_results[['codex', 'round', 'points']])
        for round_name in round_names:
            result = all_results[all_results['round'] == round_name][['codex','points']]
            if not omit_sort:
                result = result.sort_values(['points'], ascending=[False]).reset_index()['codex']
            else:
                result = result['codex']
            print(comp)
            print(round_name)
            if result.empty:
                print('omitted')
                continue
            result.columns = ['codex', 'rating']
            result = pd.merge(result, rating_db, how='left', on='codex')
            rating_act, rating_db = append_rating(result, i, comp, rating_act,
                                                  rating_db, k, round_name)
    return rating_act, rating_db


def show_rating(comps, names, rating_act, take_all=True, index=False):
    names = names.drop_duplicates(subset=['codex'])
    if not index:
        index = len(comps) - 1
    if take_all:
        results = rating_act[(rating_act['number'] <= index)
                             & (rating_act['number'] > index - 150)]
        results = results.sort_values(['number'], ascending=False)
        results = results.drop_duplicates(['codex'])
        results = results.drop(['delty', 'id', 'round'], axis = 1)
    else:
        results = rating_act[rating_act['number'] == index]
    results = pd.merge(results, names, how='left')
    return results


actual_comps = merge_infos(os.getcwd()+'\\comps\\')
actual_comps.to_csv(os.getcwd()+'\\all_comps.csv', index=False, na_rep='NA')
actual_stats = merge_stats(os.getcwd()+'\\stats\\')
actual_stats.to_csv(os.getcwd()+'\\all_stats.csv', index=False, na_rep='NA')
actual_comps = actual_comps[actual_comps['training'] == 0]
actual_comps = actual_comps.sort_values(['date', 'id'],
                                        ascending=[True, False])
actual_comps = actual_comps.reset_index()
actual_names = pd.read_csv(os.getcwd()+'\\all_names.csv')
actual_results = pd.read_csv(os.getcwd()+'\\all_results.csv')
comps_to_process = actual_comps
actual_rating = build_rating(comps_to_process,
                             actual_results, actual_names)
actual_standings = show_rating(comps_to_process, actual_names,
                               actual_rating[0], False, 1587)
ryoyu = actual_rating[0][actual_rating[0]['codex'] == 5585]
actual_rating[0].to_csv(os.getcwd()+'\\all_ratings.csv',
                        index=False, na_rep='NA')
