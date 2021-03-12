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
            tmp_nazfis = pd.read_csv(file_name+'nazfis.csv', sep=';', header=None)
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
            tmp = pd.read_csv(directory+str(comp['id'])+'.csv', sep=',', na_values=['','NA'])
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
    stats = stats.drop_duplicates(['fis_code','round_type'])
    stats[['humid','snow', 'air', 'max_wind',
           'avg_wind', 'min_wind', 'gate', 'counted_jumpers',
           'all_jumpers', 'all_countries']] =\
    stats[['humid','snow', 'air', 'max_wind',
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
        country = [x.rsplit(' ',1)[1][1:4] for x in tmp['place']]
        new_place = [x.rsplit(' ',1)[0] for x in tmp['place']]
        tmp['place']=pd.DataFrame(new_place)
        tmp['country']=pd.DataFrame(country)
        comps = comps.append(tmp)
    comps = comps.drop_duplicates(['id'])
    comps[['hill_size_x','team', 'season', 'hill_size_y', 'k-point',
           'meter value', 'gate factor', 'wind factor', 'type', 'training']] =\
    comps[['hill_size_x','team', 'season', 'hill_size_y', 'k-point',
           'meter value', 'gate factor', 'wind factor', 'type', 'training']].apply(pd.to_numeric)

    return comps


def new_rating(ratingi, k):
    delty = []
    for i, ocena in enumerate(ratingi):
        if np.isnan(ocena):
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


def doklej_rating(results, i, comp, rating_db, k):
    ratingi = results['rating']
    delty = pd.DataFrame(new_rating(ratingi, k))
    delty.columns = ['rating']
    codeksy = results['codex'].reset_index()
    delty['codex'] = codeksy['codex']
    delty['id'] = comp['id']
    delty['number'] = i
    new_rating_db = rating_db.append(delty, ignore_index=True)
    return new_rating_db


def build_rating(comps, results, names):
    rating_db = pd.DataFrame(names['codex'])
    rating_db = rating_db.drop_duplicates()
    rating_db['id'] = '2000JP0000RL'
    rating_db['rating'] = 1000
    rating_db['number'] = 0
    rating_act = rating_db[['codex', 'rating']]
    for i, comp in comps.iterrows():
        k = 8
        omit_sort=0
        print(k)
        all_results = results[results['id'] == comp['id']]
        if all_results.empty:
            omit_sort=1
            try:
                file_name = os.getcwd()+'\\nazwy\\'+comp['id'][:10]+'nazfis.csv'
                all_results = pd.read_csv(file_name, sep=';', header=None)
                print(all_results)
                if comp['team']:
                    continue
                else:
                    all_results.columns = ['bib', 'codex', 'name']
                all_results['round'] = 'whole competition '
                all_results['points'] = 1
                print('imported from nazfis.csv file')
                round_names = ['whole competition ']
            except pd.errors.EmptyDataError:
                continue
        else:
            round_names = np.unique([x['round'] for i,x in all_results.iterrows()])
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
            result = pd.merge(result, rating_act, how='left')
            result.columns = ['codex', 'rating']
            rating_db = doklej_rating(result, i, comp, rating_db, k)
            rating_act = rating_db.groupby('codex')['rating'].sum().reset_index()[['codex', 'rating']]
            rating_act.columns = ['codex', 'next_rating']
            result = pd.merge(result, rating_act, how='left')
            result['delta'] = result['next_rating']-result['rating']
    return rating_db


def show_rating(comps, names, rating_db, take_all=True, index = False):
    if not index:
        index = len(comps) - 1
    names = names.drop_duplicates(subset=['codex'])
    pre_comps = comps.iloc[:index]['id'].values.tolist()+['2000JP0000RL']
    comp = comps.iloc[index]
    if take_all:
        rating_cut = rating_db[rating_db['id'].isin(pre_comps)]
    else:
        comp_codex = rating_db[rating_db['id'] == comp['id']]['codex']
        rating_cut = rating_db[rating_db['id'].isin(pre_comps) & (rating_db['codex'].isin(comp_codex))]
    rating_cut = rating_cut.groupby('codex')['rating'].sum().reset_index()
    rating_cut = pd.merge(names, rating_cut, how='inner')
    rating_prev = rating_cut.groupby('codex')['rating'].sum().reset_index()
    new_results = pd.merge(names, rating_prev, how='inner', on='codex')
    if not take_all:
        rating_after = rating_db[(rating_db['id'] == comp['id']) & (rating_db['codex'].isin(comp_codex))]
        rating_after['position'] = rating_after.index
        new_results = pd.merge(new_results, rating_after, on='codex', how='inner')
        print(new_results)
    # new_results = new_results.drop_duplicates(subset=['codex'])
    return new_results


actual_comps = merge_infos(os.getcwd()+'\\comps\\')
actual_comps.to_csv(os.getcwd()+'\\all_comps.csv',index=False,na_rep='NA')
actual_stats = merge_stats(os.getcwd()+'\\stats\\')
actual_stats.to_csv(os.getcwd()+'\\all_stats.csv',index=False,na_rep='NA')
actual_comps = actual_comps[actual_comps['training']==0]
actual_comps = actual_comps.sort_values(['date', 'id'], ascending=[True, False])
actual_comps = actual_comps.reset_index()
# actual_names = merge_names(actual_comps, os.getcwd()+'\\nazwy\\')
# actual_names.to_csv(os.getcwd()+'\\nazwy\\all_names.csv',index=False)
# actual_results = merge_comps(actual_names, actual_comps, os.getcwd()+'\\results\\')
# actual_results.to_csv(os.getcwd()+'\\results\\all_results.csv',index=False)
actual_names = pd.read_csv(os.getcwd()+'\\all_names.csv')
actual_results = pd.read_csv(os.getcwd()+'\\all_results.csv')
actual_rating = build_rating(actual_comps, actual_results, actual_names)
actual_standings = show_rating(actual_comps, actual_names, actual_rating, False,1252)
ryoyu = actual_rating[actual_rating['codex'] == 5585]
ryoyu['progress'] = np.cumsum(ryoyu['rating'])
