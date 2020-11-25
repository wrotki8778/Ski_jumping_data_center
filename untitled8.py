# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:05:01 2020

@author: kubaf
"""
import os
import pandas as pd
import numpy as np
os.chdir('C:/Users/kubaf/Documents/Skoki')
def merge_names(comps,directory):
    names=pd.DataFrame([],columns=['bib','name'])  
    names_fis=pd.DataFrame([],columns=['bib','codex','name'])  
    for i,comp in comps.iterrows():
        try:
            tmp_naz=pd.read_csv(directory+str(comp['season'])+'JP'+str(comp['codex'])+'naz.csv',sep=';',header=None)
            tmp_naz.columns=['bib','name']
            names=names.append(tmp_naz)
        except FileNotFoundError:
            pass
        except pd.io.common.EmptyDataError:
            pass
        try:
            tmp_nazfis=pd.read_csv(directory+str(comp['season'])+'JP'+str(comp['codex'])+'nazfis.csv',sep=';',header=None)
            if not comp['team']:
                tmp_nazfis.columns=['bib','codex','name']
            else:
                tmp_nazfis.columns=['codex','name']
            names_fis=names_fis.append(tmp_nazfis)
        except FileNotFoundError:
            pass
        except pd.io.common.EmptyDataError:
            pass
    names_fis['name']=names_fis['name'].str.lower()
    names_fis=pd.merge(names_fis,names,how='right',on=['name'])
    names_fis=names_fis.drop_duplicates(['name','codex'])
    names_fis=names_fis.drop(['bib_x','bib_y'],axis=1)
    names=names.drop(['bib'],axis=1)
    names=names.drop_duplicates()
    names_fis=pd.merge(names_fis,names,how='left',on=['name'])
    return(names_fis)
def merge_comps(names,comps,directory):
    columns_names=['name','wind','wind_comp','dist','speed','dist_points','note_1','note_2','note_3','note_4','note_5','note_points','points','loc','gate','gate_points','id']
    results=pd.DataFrame([],columns=columns_names)
    for i,comp in comps.iterrows():
        try:
            tmp=pd.read_csv(directory+str(comp['id'])+'.csv',sep=',')
            tmp['id']=comp['id']
            results=results.append(tmp)
        except FileNotFoundError:
            pass
        except pd.io.common.EmptyDataError:
            pass
    results=pd.merge(results,names,how='left',on=['name'])
    results=results.drop(['name','Unnamed: 0'],axis=1)
    return(results)
def merge_infos(directory):
    columns_names=['codex','place','gender','hill_size_x','team','season','hill_size_y','k-point','meter value','gate factor','wind factor','type','date','id']
    comps=pd.DataFrame([],columns=columns_names)
    list=os.listdir(directory)
    for i,item in enumerate(list):
        tmp=pd.read_csv(directory+'\\'+item,sep=',')
        comps=comps.append(tmp)
    comps=comps.drop_duplicates(['id'])
    return(comps)
comps=merge_infos(os.getcwd()+'\\comps\\')
comps=comps.sort_values(['date','id'],ascending=[True,False])
comps=comps.reset_index()
names=merge_names(comps,os.getcwd()+'\\nazwy\\')
results=merge_comps(names,comps,os.getcwd()+'\\results\\')
def new_rating(ratingi,k,print_exp):
    delty=[]
    for i,ocena in enumerate(ratingi):
        if np.isnan(ocena):
            continue
        delta=0
        exp_score=0
        fact_score=0
        for j,inna_ocena in enumerate(ratingi):
            """print(inna_ocena,ocena)"""
            if np.isnan(inna_ocena):
                continue
            if i==j:
                continue
            elif i<j:
                exp_score=exp_score+1/(10**((inna_ocena-ocena)/400)+1)
                fact_score=fact_score+1
            else:
                exp_score=exp_score+1/(10**((inna_ocena-ocena)/400)+1)
            if print_exp:
                """print(inna_ocena,ocena,1/(10**((inna_ocena-ocena)/400)+1),tmp)"""
        delta=k*(fact_score-exp_score)/10
        delty.append(delta)
        if print_exp:
            print(i,ocena,exp_score,fact_score,delta)
        """print(delta,exp_score,fact_score)"""
    return(delty)

def doklej_rating(results,i,comp,rating_db,k,print_exp):
    ratingi=results.loc[:,'rating']
    delty=pd.DataFrame(new_rating(ratingi,k,print_exp))
    delty.columns=['rating']
    codeksy=results['codex'].reset_index()
    delty['codex']=codeksy['codex']
    delty['id']=comp['id']
    delty['number']=i
    
    """print(pd.merge(rating_act,delty,how='left'))"""
    new_rating_db=rating_db.append(delty,ignore_index=True)
    return(new_rating_db)
def build_rating(comps,results):
    rating_db=pd.DataFrame(names['codex'])
    rating_db=rating_db.drop_duplicates()
    rating_db['id']='2010JP0000RL'
    rating_db['rating']=1000
    rating_db['number']=0
    rating_act=rating_db[['codex','rating']]
    for i, comp in comps.iterrows():
        k=16
        if comp['id'].count("Q"):
            k=k/2
        print(k)
        if not comp['team']:
            print(comp)
            part_results=results[results['id']==comp['id']]
            if part_results.empty:
                continue
            part_results=pd.DataFrame(part_results['codex'])
            part_results=part_results.drop_duplicates()
            part_results=pd.merge(part_results,rating_act,how='left')
            part_results.columns=['codex','rating']
            rating_db=doklej_rating(part_results,i,comp,rating_db,k,0)
            rating_act=rating_db.groupby('codex')['rating'].sum().reset_index()[['codex','rating']]
            rating_act.columns=['codex','next_rating']
            part_results=pd.merge(part_results,rating_act,how='left')
            part_results['delta']=part_results['next_rating']-part_results['rating']
    return(rating_db)
        
def show_rating(comps,index,names,rating_db,take_all=True):
    names=names.drop_duplicates(subset=['codex']) 
    pre_comps=list(comps.iloc[:index]['id'])+['2010JP0000RL']
    comp=comps.iloc[index]
    if take_all:
        rating_cut=rating_db[rating_db['id'].isin(pre_comps)]
    else:
        comp_codex=rating_db[rating_db['id']==comp['id']]['codex']
        rating_cut=rating_db[rating_db['id'].isin(pre_comps) & (rating_db['codex'].isin(comp_codex))]
        print(rating_cut)
    rating_cut=rating_cut.groupby('codex')['rating'].sum().reset_index()
    rating_cut=pd.merge(names,rating_cut,how='inner')
    rating_prev=rating_cut.groupby('codex')['rating'].sum().reset_index()
    new_results=pd.merge(names,rating_prev,how='inner',on='codex')
    if not take_all:
        rating_after=rating_db[(rating_db['id']==comp['id']) & (rating_db['codex'].isin(comp_codex))]
        rating_after['position']=rating_after.index
        new_results=pd.merge(new_results,rating_after,on='codex',how='inner')
    new_results=new_results.drop_duplicates(subset=['codex'])
    return(new_results)

rating_db=build_rating(comps,results)
results=show_rating(comps,800,names,rating_db,False)
ryoyu=rating_db[rating_db['codex']==5658]
ryoyu['progress']=np.cumsum(ryoyu['rating'])
"""quals_results.to_csv('new_qual_results_fix.csv')"""


