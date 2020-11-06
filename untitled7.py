# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:21:09 2020

@author: kubaf
"""
import os
import pandas as pd
import re
os.chdir('C:/Users/kubaf/Documents/Skoki/WC/SGP/juz_przerobione')
from tika import parser
lista_pdf=os.listdir()
lista_pdf=[x for x in lista_pdf if x[-4:]=='.pdf']
comps=pd.read_csv('comps_all.csv')
info=['codex','season','hill_size','k-point','meter value','gate factor','wind factor']
comps_infos=pd.DataFrame([],columns=info)
for k,row in comps.iterrows():
    """index=282
    row=quals.iloc[index] """  
    year=str(row['season'])
    codex=str(row['codex'])
    parsed = parser.from_file(year+'JP'+codex+'RL.pdf')
    tekst=parsed["content"]
    tekst=tekst.lower()
    tekst_lin=tekst.splitlines()
    tekst_lin = [i for i in tekst_lin if i] 
    word=[]
    word.append('hill size')
    word.append('k-point')
    word.append('meter value')
    word.append('gate factor')
    word.append('wind factor')
    infos=[]
    param_infos=[]
    for words in word:
        add=[i for i in tekst_lin if words in i]
        """print(add)"""
        infos.append(add[0])
    """print(infos)"""
    new_info=pd.Series([year]+[codex]+infos, index = comps_infos.columns)
    """print(new_info)"""
    comps_infos=comps_infos.append(new_info,ignore_index=True)
    
    """lista=[]
    for i,line in enumerate(tekst_lin):
        if len(line)==sum(c.isdigit() for c in line) and sum(c.isalpha() for c in tekst_lin[i+1]):
            if row['season']<2016:
                const=1
            else:
                const=2
            next_line=tekst_lin[i+const]
            lista.append([line,next_line])
            if sum(c.isdigit() for c in next_line) or max(1-next_line.count(' '),0):
                print('Alert: w konkursie nr ' +str(k)+' zawodnik z nr '+line+' nazywa się '+next_line+'!')"""

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def take_number(string):
    string=string.replace('m','')
    tmp=string.split(' ')
    take=[x for x in tmp if is_number(x)]
    print(take)
    return float(take[0])

comps_infos['wind factor']=comps_infos['wind factor'].apply(take_number)
comps_infos.to_csv('comps_all_fix.csv')