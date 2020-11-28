# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:01:40 2020

@author: kubaf
"""
import os
import pandas as pd
import re
os.chdir('C:/Users/kubaf/Documents/Skoki/WC/to_do')
from tika import parser
lista=os.listdir()
lista=[x for x in lista if x[-4:]=='.pdf']
"""alt_lista=[x[:-4]+'.pdf' for x in lista if x[-4:]=='.csv']
alt_lista=alt_lista[6:]"""
lista.reverse()
"""lista=lista[-1:]"""
def training_initial_processing(comp):
    init_results=zwroc_skoki(comp)
    for i,skok in enumerate(init_results):
        skok=tekst_lin[numer[i]:numer[i+1]]
        skok=[i for i in skok if i]
        skok=skok+['']*(18-len(skok))
        print(skok)
        if 'TRIA' in nazwa:
            skok_final=[skok[0]]
            print(pre_2016)
            if not(pre_2016):
                const=4  
            else:
                const=5
            skok_final=skok_final+[skok[6-const]]
            dns=skok[const].count('dns')
            if skok[const].count('.')==1:
                skok[const]='0.0 '+skok[const]
            sklej=' '.join([skok[const]]+skok[const+2:])
            sklej=sklej.split(' ')
            sklej=[x for x in sklej if x]
            sklej=' '.join(sklej)
            print(sklej)
            sklej=''.join([i for i in sklej if not(i.isalpha())])
            print(sklej)
            if dns or not(sklej):
                sklej='dns'
            print(sklej)
            skok_final=skok_final+[sklej]
        if 'RLT' in nazwa:
            layout='rlt'
            skok_final=[skok[0]]
            print(pre_2016)
            if not(pre_2016):
                const=4  
            else:
                const=5
            skok_final=skok_final+[skok[6-const]]
            shift=[0,2,3]
            for i in range(3):
                indeksy=[j+const+4*i for j in shift]
                print(indeksy)
                dns=skok[indeksy[0]].count('dns')
                if skok[indeksy[0]].count('.')==1:
                    skok[indeksy[0]]='0.0 '+skok[indeksy[0]]
                sklej=' '.join([skok[i] for i in indeksy])
                sklej=sklej.split(' ')
                sklej=[x for x in sklej if x]
                sklej=' '.join(sklej)
                print(sklej)
                sklej=''.join([i for i in sklej if not(i.isalpha())])
                print(sklej)
                if dns or not(sklej):
                    sklej='dns'
                print(sklej)
                skok_final=skok_final+[sklej]
        print(skok_final)
        skoki.append(skok_final)
    return([skoki,pre_2016,layout])    

def przeksztalc(string,pre_2016,layout):
    string=string.replace('©', '')
    nowy_string=string.split()
    nowy_string=[x for x in nowy_string if x]
    """print(nowy_string)"""
    if nowy_string.count('dns'):
        if layout=='tria':
            return([0,0,0,0,0,0,0])
        else:
            return([0,0,0,0,0,0,0,0])
    if layout=='tria':
        nofy_string=nowy_string[-4:-1]+nowy_string[:-4]
    else:
        nofy_string=nowy_string[-2:]+nowy_string[:-2]
    print(nofy_string)
    return(nofy_string)
def znowu_przeksztalc(skok,pre_2016,layout):
    output = [idx for idx, line in enumerate(skok) if line.count('.')>2] 
    output = output+[idx for idx, line in enumerate(skok) if line.count('dns')] 
    info=['bib','name','wind','wind_comp','dist_points','speed','dist','gate','gate_points']
    if layout=='rlt':
        info=['bib','name','wind_comp','loc','speed','dist','gate','wind','dist_points','gate_points']
    new_jump=pd.DataFrame([],columns=info)
    for i in range(len(output)):
        bib=skok[0]
        name=skok[1]
        notes_pre=przeksztalc(skok[output[i]],pre_2016,layout)
        notes=[float(x) for x in notes_pre]
        if layout=='tria':
            passed_values=7
        else:
            passed_values=8
        if(len(notes)!=passed_values):
            notes.append(0)
        if len(notes)<3:
            notes=[0,0,0,0,0,0,0]
        data=pd.Series([bib]+[name]+notes, index = new_jump.columns)
        new_jump=new_jump.append(data,ignore_index=True)
    print(new_jump)
    return(new_jump)
def collect(jumps,pre_2016,layout):
    info=['bib','name','wind','wind_comp','dist_points','speed','dist','gate','gate_points']
    if layout=='rlt':
        info=['bib','name','wind_comp','loc','speed','dist','gate','wind','dist_points','gate_points']
    database=pd.DataFrame([],columns=info)
    for i in range(len(jumps)):
        new_jumps=znowu_przeksztalc(jumps[i],pre_2016,layout)
        database=database.append(new_jumps,ignore_index=True)
    return(database)

przyklad='2014JP3714RLT.pdf'
parsed = parser.from_file(przyklad)
tekst=parsed["content"]
tekst=tekst.lower()
tekst_lin=tekst.splitlines()
tekst_lin = [i for i in tekst_lin if i] 
wynik=zwroc_skoki(nazwa=przyklad,tekstlin=tekst_lin)
wynik_2=przeksztalc(wynik[0][4][1],wynik[1],wynik[2])
wynik_3=collect(wynik[0],wynik[1],wynik[2])
wynik_3.to_csv(przyklad[:-4]+'.csv')

for plik in lista:
    print(plik)
    content=zwroc_skoki(nazwa=plik)
    result=collect(content[0],content[1],content[2])
    result.to_csv(plik[:-4]+'.csv')
    






