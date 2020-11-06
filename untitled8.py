# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:05:01 2020

@author: kubaf
"""
import os
import pandas as pd
import numpy as np
import re
os.chdir('C:/Users/kubaf/Documents/Skoki')

comps=pd.read_csv('comps_all_WC_CC_SGP.csv')
quals_results=pd.read_csv('results_all.csv')
names=pd.read_csv('nazwy_all.csv')
rating_db=pd.DataFrame(names['codex'])
rating_db=rating_db.drop_duplicates()
rating_db['date']=pd.to_datetime('2011-01-01')
rating_db['rating']=1000
rating_db['number']=0

rating_act=rating_db[['codex','rating']]
def new_rating(ratingi,k,print_exp):
    delty=[]
    for i,ocena in enumerate(ratingi):
        delta=0
        exp_score=0
        fact_score=0
        for j,inna_ocena in enumerate(ratingi):
            """print(inna_ocena,ocena)"""
            if i==j:
                continue
            elif i<j:
                exp_score=exp_score+1/(10**((inna_ocena-ocena)/400)+1)
                tmp=1
                fact_score=fact_score+1
            else:
                exp_score=exp_score+1/(10**((inna_ocena-ocena)/400)+1)
                tmp=0
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
    delty['date']=comp['date']
    delty['number']=i
    
    """print(pd.merge(rating_act,delty,how='left'))"""
    new_rating_db=rating_db.append(delty,ignore_index=True)
    return(new_rating_db)


for i, comp in comps.iterrows():
    year=comp['season']
    comp_codex=comp['codex']
    k=16
    if comp['qual']:
        k=k/2
    if comp['type']==1:
        k=k
    print(k)
    x=comp['Unnamed: 0']
    if 0:
        break
    if comp['qual']:
        print(comp)
        sort_qual=quals_results[quals_results['id']==x]
        sort_qual=pd.DataFrame(sort_qual['codex'])
        sort_qual=pd.merge(sort_qual,rating_act,how='left')
        sort_qual.columns=['codex','rating']
        input_qual=sort_qual
        rating_db=doklej_rating(input_qual,i,comp,rating_db,k,0)
        rating_act=rating_db.groupby('codex')['rating'].sum().reset_index()[['codex','rating']]
        rating_act.columns=['codex','next_rating']
        sort_qual=pd.merge(sort_qual,rating_act,how='left')
        sort_qual['delta']=sort_qual['next_rating']-sort_qual['rating']
        """print(sort_qual)"""
    elif not(comp['team']) and not(comp['qual']):
        print(comp)
        nazwa=str(year)+'JP'+str(comp_codex)+'naz.csv'
        sciezka=os.getcwd()+'\\WC\\nazwy\\'+nazwa
        sort_comp=pd.read_csv(sciezka,header=None,sep=';')
        sort_comp.columns=['bib','codex','name']
        sort_comp=pd.merge(sort_comp,rating_act,how='left')
        sort_comp.columns=['bib','codex','name','rating']
        input_comp=pd.DataFrame(sort_comp[['codex','rating']])
        if i==25:
            rating_db=doklej_rating(input_comp,i,comp,rating_db,k,1)
        else:
            rating_db=doklej_rating(input_comp,i,comp,rating_db,k,0)
        rating_act=rating_db.groupby('codex')['rating'].sum().reset_index()[['codex','rating']]
        rating_act.columns=['codex','next_rating']
        sort_comp=pd.merge(sort_comp,rating_act,how='left')
        sort_comp['delta']=sort_comp['next_rating']-sort_comp['rating']
        """print(sort_comp)"""
def show_rating(names,rating_db,index=1,take_all=True):
    names=names.drop_duplicates(subset=['codex'])
    comp_index=index
    if take_all:
        rating_cut=rating_db[(rating_db['number']<comp_index)]
    else:
        comp_codex=rating_db[rating_db['number']==comp_index]['codex']
        rating_cut=rating_db[(rating_db['number']<comp_index) & (rating_db['codex'].isin(comp_codex))]
        
    rating_cut=rating_cut.groupby('codex')['rating'].sum().reset_index()
    rating_cut=pd.merge(names,rating_cut,how='inner')
    rating_prev=rating_cut.groupby('codex')['rating'].sum().reset_index()
    new_results=pd.merge(names,rating_prev,how='inner',on='codex')
    if not take_all:
        rating_after=rating_db[(rating_db['number']==comp_index) & (rating_db['codex'].isin(comp_codex))]
        rating_after['position']=rating_after.index
        new_results=pd.merge(new_results,rating_after,on='codex',how='inner')
    new_results=new_results.drop_duplicates(subset=['codex'])
    return(new_results)

results=show_rating(names,rating_db,1437,True)
ryoyu=rating_db[rating_db['codex']==4141]
ryoyu['progress']=np.cumsum(ryoyu['rating'])
"""quals_results.to_csv('new_qual_results_fix.csv')"""


