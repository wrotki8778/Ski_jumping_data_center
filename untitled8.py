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
        try:
            tmp = pd.read_csv(directory+str(comp['id'])+'.csv', sep=',')
            tmp['id'] = comp['id']
            results = results.append(tmp)
        except FileNotFoundError:
            pass
        except pd.errors.EmptyDataError:
            pass
    results = pd.merge(results, names, how='left', on=['name'])
    results = results.drop(['name', 'Unnamed: 0'], axis=1)
    return results


def merge_infos(directory):
    columns_names = ['codex', 'place', 'gender', 'hill_size_x',
                     'team', 'season', 'hill_size_y', 'k-point',
                     'meter value', 'gate factor', 'wind factor',
                     'type', 'date', 'id', 'training']
    comps = pd.DataFrame([], columns=columns_names)
    list_of_files = os.listdir(directory)
    list_of_files.remove('all_comps.csv')
    for item in list_of_files:
        tmp = pd.read_csv(directory+'\\'+item, sep=',')
        comps = comps.append(tmp)
    comps = comps.drop_duplicates(['id'])
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
    ratingi = results.loc[:, 'rating']
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
        k = 16
        if comp['id'].count("Q"):
            k = k/2
        print(k)
        if not comp['team']:
            part_results = results[results['id'] == comp['id']]
            if part_results.empty:
                try:
                    file_name = os.getcwd()+'\\nazwy\\'+comp['id'][:10]+'nazfis.csv'
                    part_results = pd.read_csv(file_name, sep=';', header=None)
                    part_results.columns = ['bib', 'codex', 'name']
                    print('imported from nazfis.csv file')
                except pd.errors.EmptyDataError:
                    continue
            part_results = pd.DataFrame(part_results['codex'])
            part_results = [part_results.drop_duplicates()]
        else:
            k = k/2
            print(comp)
            team_results = actual_results[actual_results['id'] == comp['id']]
            second_round = [i for i, x in team_results.iloc[1:].iterrows() 
                            if team_results.loc[i-1]['codex'] == x['codex']]
            first_round = [i for i, x in team_results.iterrows() 
                           if i not in second_round]
            first_round_results = team_results.loc[first_round]
            second_round_results = team_results.loc[second_round]
            first_round_results = first_round_results.sort_values(['points'], ascending=[False])
            second_round_results = second_round_results.sort_values(['points'], ascending=[False])
            part_results = [pd.DataFrame(first_round_results['codex'])] + \
                           [pd.DataFrame(second_round_results['codex'])]
            if second_round_results.empty:
                part_results = [pd.DataFrame(first_round_results['codex'])]
        for result in part_results:
            if result.empty:
                print('omitted')
                continue
            result = result.drop_duplicates()
            result = pd.merge(result, rating_act, how='left')
            result.columns = ['codex', 'rating']
            rating_db = doklej_rating(result, i, comp, rating_db, k)
            rating_act = rating_db.groupby('codex')['rating'].sum().reset_index()[['codex', 'rating']]
            rating_act.columns = ['codex', 'next_rating']
            result = pd.merge(result, rating_act, how='left')
            result['delta'] = result['next_rating']-result['rating']
            if comp['team']:
                print(comp)
                print(result)
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
actual_comps.to_csv(os.getcwd()+'\\comps\\all_comps.csv',index=False)
actual_comps = actual_comps[actual_comps['training'] != 1]
actual_comps = actual_comps.sort_values(['date', 'id'], ascending=[True, False])
actual_comps = actual_comps.reset_index()
actual_names = merge_names(actual_comps, os.getcwd()+'\\nazwy\\')
actual_names.to_csv(os.getcwd()+'\\nazwy\\all_names.csv')
actual_results = merge_comps(actual_names, actual_comps, os.getcwd()+'\\results\\')
actual_rating = build_rating(actual_comps, actual_results, actual_names)
actual_standings = show_rating(actual_comps, actual_names, actual_rating, True)
ryoyu = actual_rating[actual_rating['codex'] == 5262]
ryoyu['progress'] = np.cumsum(ryoyu['rating'])
