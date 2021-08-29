"""
Script to analyze results from untitled0/2/6.py and compute ratings.

@author: wrotki8778
"""
import os
import pandas as pd
import numpy as np
import numexpr as ne
import math
os.chdir('C:/Users/kubaf/Documents/Skoki')

def import_stuff(directory, duplicates = []):
    list_of_files = os.listdir(directory)
    import_data = {file[:-4]: pd.read_csv(directory+file, sep=',')
                   for file in list_of_files}
    if not duplicates:
        for file in list_of_files:
            import_data[file[:-4]]['id'] = file[:-4]
    all_data = pd.DataFrame().append(list(import_data.values()))
    if duplicates:
        all_data = all_data.drop_duplicates(duplicates)
    return all_data

def import_names(directory):
    list_of_files = set(os.listdir(directory))
    naz_files = {x for x in list_of_files if 'naz.csv' in x}
    nazfis_files = {x for x in list_of_files if 'nazfis.csv' in x}
    import_nazfis_data = {file[:10]: pd.read_csv(directory+file, sep=';')
                          for file in nazfis_files}
    import_naz_data = {file[:10]: pd.read_csv(directory+file, sep=';')
                       for file in naz_files}
    for file in nazfis_files:
        if import_nazfis_data[file[:10]].empty:
            continue
        elif len(import_nazfis_data[file[:10]].columns) == 2:
            import_nazfis_data[file[:10]].columns = ['codex', 'name']
        else:
            import_nazfis_data[file[:10]].columns = ['bib', 'codex', 'name']
    for file in naz_files:
        if import_naz_data[file[:10]].empty:
            continue
        else:
            import_naz_data[file[:10]].columns = ['bib', 'name']
    all_nazfis_data = pd.DataFrame().append(list(import_nazfis_data.values())).drop_duplicates(['name'])
    all_naz_data = pd.DataFrame().append(list(import_naz_data.values())).drop_duplicates(['name'])
    all_nazfis_data['name'] = all_nazfis_data['name'].str.lower()
    all_nazfis_data = all_nazfis_data.drop(['bib'],axis=1)
    all_naz_data = all_naz_data.drop(['bib'],axis=1)
    results = pd.merge(all_naz_data, all_nazfis_data, how='left',
                       on = ['name']).drop_duplicates(['name'])
    return results

def new_rating(ratings, k):
    """
    Compute deltas to update ratings.

    Parameters
    ----------
    ratings : vector of floats
        Ordered vector of initial ratings (before computing deltas).
    k : float
        Magnitude of rating change (bigger k <-> bigger changes).

    Returns
    -------
    delty : vector of floats
        Ordered vector of changes in initial ratings.

    """
    delty = np.array(ratings)
    n = len(delty)
    delty_matrix = np.tile(delty, (n, 1))
    delty_matrix = (delty_matrix - delty_matrix.T)/400
    results = ne.evaluate('1/(10**delty_matrix + 1)')
    exp_score = np.sum(results, axis=1, initial = -0.5)
    fact_score = np.flip(np.arange(n))
    return k*(fact_score - exp_score)/10

def append_rating(results, comp, rating_db, k, round_name):
    """
    Update ratings in a incremental way.

    Parameters
    ----------
    results : Pandas dataframe
        Dataframe with the results of the given competition round.
    i : integer
        Variable counting the number of competitions before a given round
    comps : Pandas dataframe
        Infos about all competitions gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    rating_act : Pandas dataframe
        Dataframe containing all deltas until a given competition round.
    rating_db : Pandas dataframe
        Dataframe containing aggregated deltas until a given competition round.
    k : float
        Magnitude of rating change (bigger k <-> bigger changes).
    round_name : string
        Name of round for which the rating is updated.

    Returns
    -------
    list
        new_rating_act : Pandas dataframe
            Dataframe containing all deltas after a given competition round.
        new_rating_db : Pandas dataframe
            Dataframe containing aggregated deltas
            after a given competition round.

    """
    ratings = results['cumm_rating']
    results['delty'] = new_rating(ratings, 8)
    results['id'] = comp.name
    results['round'] = round_name
    #results['short_rating'] = short_rating_compute(i, results, comps,
    #                                               rating_act)
    new_rating_db = pd.merge(rating_db, results[['codex', 'delty']],
                             on='codex', how='left')
    new_rating_db['cumm_rating'] = new_rating_db['cumm_rating']\
        + new_rating_db.fillna(0)['delty']*(1 - comp['training'])*(k/8)
    new_rating_db = new_rating_db.drop(['delty'], axis=1)
    return [results, new_rating_db]


def neighborhood_comps(comps,code):
    comp = comps[comps['id'] == code]
    comp_number = comp.index.item()
    codes = comps.loc[comp_number-12:comp_number]
    filtered_codes = codes[codes['place'] == comp.iloc[0]['place']]['id']
    return(list(filtered_codes))


def short_rating_compute(i, results, comps, rating_act):
    if comps.loc[i]['training']:
        return 0
    correct_ids = neighborhood_comps(comps, comps.loc[i]['id'])
    if not correct_ids:
        return 0
    rating_tmp = rating_act[rating_act['id'].isin(correct_ids)][['codex',
                                                                 'delty']]
    rating_tmp = rating_tmp.groupby('codex').mean(['delty'])
    results = pd.merge(results, rating_tmp, how='left', on='codex')
    return results['delty_y'].fillna(0)


def initialize_build_rating(names):
    rating_db = pd.DataFrame(names['codex'])
    rating_db = rating_db.drop_duplicates()
    rating_db.dropna()
    rating_db['cumm_rating'] = 1000
    return rating_db


def build_rating(comp, names, results=pd.DataFrame(),
                 rating_db=pd.DataFrame(), round_name=False):
    """
    Create a dataframe with ratings of all athletes.

    Ratings indicate their level over time.

    Parameters
    ----------
    comp : Pandas dataframe
        Series with the information about the competition.
    results : Pandas dataframe
        Dataframe with the results of given competition round.
    names : Pandas dataframe
        Dataframe with one-to-many map connecting every athlete with his/hers
        FIS code.

    Returns
    -------
    rating_act : Pandas dataframe
        Dataframe containing deltas (increments) of rating of every athlete
        after all competition rounds.
    rating_db : Pandas dataframe
        Dataframe containing aggregated deltas of every athlete
        after all competition rounds.

    """
    if rating_db.empty:
        rating_db = initialize_build_rating(names)
    k = 8 * (1 + max(2013 - comp['season'], 0))
    # print(k)
    # print(comp)
    # print(round_name)
    if results.empty:
        if comp['training'] == 1:
            # print('omitted')
            return pd.DataFrame(), rating_db
        file_name = os.getcwd()+'\\nazwy\\'+comp.name[:10]+'nazfis.csv'
        if os.path.exists(file_name):
            results = pd.read_csv(file_name, sep=';', header=None)
            if comp['team']:
                return pd.DataFrame(), rating_db
            results.columns = ['bib', 'codex', 'name']
            results = results.drop(['bib','name'], axis=1)
            # results['round'] = 'whole competition '
            # results['points'] = 1
            # results['dist_points'] = 1
            # results['dist'] = 1
            # print('imported from nazfis.csv file')
            round_name = 'whole competition '
            k = 2*k
        else:
            return pd.DataFrame(), rating_db
    new_results = pd.merge(results, rating_db, how='left', on='codex')
    new_rating_act, new_rating_db = append_rating(new_results, comp,
                                                  rating_db, k, round_name)
    return new_rating_act, new_rating_db

all_comps = import_stuff('C:/Users/kubaf/Documents/Skoki/comps/',['id'])
all_comps.to_csv(os.getcwd()+'\\all_comps.csv', index=False)
print('competitions loaded')
all_stats = import_stuff('C:/Users/kubaf/Documents/Skoki/stats/',
                         ['fis_code','round_type'])
all_stats.to_csv(os.getcwd()+'\\all_stats.csv', index=False)
print('stats loaded')
all_names = import_names('C:/Users/kubaf/Documents/Skoki/nazwy/')
all_names.to_csv(os.getcwd()+'\\all_names.csv', index=False)
print('names loaded')
all_results = import_stuff('C:/Users/kubaf/Documents/Skoki/elastic_results/')
all_results = pd.merge(all_results,all_names, how='left', on='name').drop(['name'], axis=1)
all_results.to_csv(os.getcwd()+'\\all_results.csv', index=False)
print('results loaded')
all_stats_rounds = {_: list(x['round_type'])
                       for _, x in all_stats.groupby(['fis_code'])}

all_comps = all_comps[all_comps['season'] > 2009]
all_comps = all_comps.sort_values(['date', 'id'],
                                        ascending=[True, False])
all_comps = all_comps.set_index(['id'])
all_results_split = {_: x.sort_values(by=['points', 'dist_points', 'loc'],
                                         ascending=[False, False, True])['codex'].dropna()
                        for _, x in all_results.groupby(['id', 'round'])}
print('rating evaluation started')
rating_db = pd.DataFrame()
rating_act = pd.DataFrame()
rating_results = {}
for primary_id in list(all_comps.index):
    if primary_id in all_stats_rounds.keys():
        round_names = all_stats_rounds[primary_id]
    else:
        continue
    comp = all_comps.loc[primary_id]
    for round_name in round_names:
        if (primary_id, round_name) in all_results_split.keys():
            results = all_results_split[(primary_id, round_name)]
        else:
            results = pd.DataFrame()
        rating_act, rating_db = build_rating(comp, all_names, results,
                                             rating_db=rating_db,
                                             round_name=round_name)
        rating_results[(primary_id, round_name)] = rating_act
rating_final = pd.DataFrame().append(list(rating_results.values()))
# Execution time 4x reduced (280s -> 70s)
rating_final.to_csv(os.getcwd()+'\\all_ratings.csv',
                        index=False)


