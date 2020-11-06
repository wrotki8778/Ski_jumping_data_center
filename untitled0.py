# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 14:14:52 2020

@author: kubaf
"""
import os
import pandas as pd
import re
from tika import parser
os.chdir('C:/Users/kubaf/Documents/Skoki')
lista_pdf=os.listdir()
lista_pdf=[x for x in lista_pdf if x[-4:]=='.pdf']
lista_csv=os.listdir('C:/Users/kubaf/Documents/Skoki/WC/Kwalifikacje/csvki')
lista_csv=[x for x in lista_csv if x[-7:]=='RLQ.csv']
lista=[x for x in lista_pdf if x[:-7]+'RLQ.csv' not in lista_csv]

lista.reverse()
lista=[lista[1]]
def zwroc_skoki(nazwa=[],tekstlin=[],TCS=0):
    kwale=1
    team=0
    pre_2016=0
    if nazwa:
        if nazwa[-5]=='Q':
            kwale=1
        if int(nazwa[0:4])<2016:
            pre_2016=1
        else:
            pre_2016=0
        parsed = parser.from_file(nazwa)
        tekst=parsed["content"]
        tekst=tekst.lower()
        tekst_lin=tekst.splitlines()
        tekst_lin = [i for i in tekst_lin if i] 
    if tekstlin:
        tekst_lin=tekstlin
    for line in range(len(tekst_lin[0:8])):
        if tekst_lin[line].count('team')>0:
            team=1
    end=[]
    word2='competition / weather information'
    for i,line in enumerate(tekst_lin):
        if word2 in line: # or word in line.split() to search for full words
            end.append(i)
    tekst_lin=tekst_lin[:end[0]]
    totale=[]
    kw=[]
    pejdze=[]
    punkty=[]
    word='total'
    word_alt='points'
    word2='not qual'
    word3='page'
    word4='1st r'
    old_war=0
    war=0
    for i,line in enumerate(tekst_lin):
        inne=line.count('dsq')+line.count('dns')+line.count('man of')+line.count('the day')
        if inne>0: # or word in line.split() to search for full words
            """print(line)"""
            tekst_lin[i]=''
    for i,line in enumerate(tekst_lin):
        if word2 in line: # or word in line.split() to search for full words
            kw.append(i)
        if word4 in line and word2 not in line: # or word in line.split() to search for full words
            print(line)
            kwale=0
    """kwale=1"""
    for i,line in enumerate(tekst_lin):
        if word in line or line==word_alt: # or word in line.split() to search for full words
            totale.append(i)
        if word3 in line: # or word in line.split() to search for full words
            pejdze.append(i-1)
        old_war=war
        war=line.count('.')
        if i<len(tekst_lin)-1:
            empty=not(tekst_lin[i-1]) or not(tekst_lin[i+1])
        """print(i,old_war,war,empty)"""
        if (war>=8 and old_war>=8) or (war>=8 and kwale==1) or (war>=8 and kw and i>kw[0]) or (war>=8 and empty):
            punkty.append(i)

    if (kwale==0) and (team==0): 
        totale=totale[1::2]
        
    pejdze=pejdze[0:len(totale)]
    f3 = lambda x: x <= max(pejdze)
    punkty=[x for x in punkty if f3(x)]
    alles=sorted(totale+kw+punkty) 

    skoki=[]
    for i in range(len(alles)-1):
        if (alles[i] in pejdze) or (alles[i+1] in kw):
            continue
        else:
            skok=tekst_lin[alles[i]+1:alles[i+1]+1]
            skok=[i for i in skok if i] 
            if TCS==1:
                podziel=skok[-1].split()
            else:
                podziel=skok[0].split()
            print(podziel)
            """and podziel[0].count('.')>0"""
            if skok and podziel[0][0].isdigit():
                skoki.append(skok)
            if skok and podziel[0][0].isdigit() and podziel[0].count('-')>0 and team==1:
                skoki.append(skok)
    if team==1 and kwale==0:
        for i in range(len(skoki)):
            tmp=[]       
            for j in range(len(skoki[i])):
                line=skoki[i][j]
                div=line.split()
                if (line.count('.')>7) or (div[0][0].isdigit() and div[0].count('-')>0 and ('dns' not in div) and ('dsq' not in div)):
                    tmp=tmp+[line]
            skoki[i]=tmp
        for i in range(len(skoki)):
            div=skoki[i][0].split()
            if len(skoki[i])==1 or (len(skoki[i])==2 and div[0].count('-')==0):
                index=min([i for i, s in enumerate(div) if '.' in s])
                skoki[i]=[' '.join(div[0:index]),' '.join(div[index:])]
            if len(skoki[i])>=3 or (len(skoki[i])==2 and div[0].count('-')>0):
                div=skoki[i][0].split(' ',1)
                skoki[i][0]=div[1]
    if team==1 and kwale==1:
        for i in range(len(skoki)):
            tmp=[]
            skoki[i]=[skoki[i][-3],skoki[i][-1]]
    return([skoki,kwale,team,pre_2016,TCS])    

def przeksztalc(string,kwale=0,team=0,TCS=0):
    if TCS:
        string=string.replace('pq', '0.')
    pozycja=string.find('.')+2
    nowy_string=string[:pozycja]+' '+string[pozycja:]
    nowy_string=re.sub(r'[a-z]+', '', nowy_string, re.I)
    nowy_string=nowy_string.replace('©', '')
    znacznik=nowy_string.find('*')
    if znacznik:
        nowy_string=nowy_string[(znacznik+1):]
    wyrazy=nowy_string.rsplit(' ', 2)
    nowy_string=wyrazy[1] + ' ' + wyrazy[2] + ' ' + wyrazy[0]
    
    n=[12]
    offset=[1]
    if kwale==1:
        n=[]
        offset=[]
    if kwale==1 and team==1:
        n=[12]
        offset=[1]
    if TCS==1:
        n=[12]
        offset=[1]
    
    kropki=[i for i, a in enumerate(nowy_string) if a == '.']
    print(kropki,len(nowy_string),n)
    kropki=[kropki[i] for i in n]
    if n:
        nofy_string=[nowy_string[0:kropki[0]+offset[0]]]+[nowy_string[kropki[i]+offset[i]:kropki[i+1]+offset[i+1]] for i in range(len(kropki)-1)]+[nowy_string[kropki[-1]+offset[-1]:]]
        print(nofy_string)
        nofy_string=' '.join(nofy_string)
        wyrazy=nofy_string.split(' ')
    else:
        nofy_string=nowy_string
    if TCS and kwale:
        nofy_string=' '.join(wyrazy[0:3]) + ' ' + ' '.join(wyrazy[4:13]) + ' ' + ' '.join(wyrazy[14:])
    print(nofy_string)
    return(nofy_string)
def znowu_przeksztalc(skok,kwale=0,team=0,pre_2016=0,TCS=0):
    output = [idx for idx, line in enumerate(skok) if line.count('.')>7] 
    info=['name','wind','wind_comp','speed','dist','dist_points','note_1','note_2','note_3','note_4','note_5','note_points','points','loc','gate','gate_points']
    ind_name=2
    if pre_2016 or TCS:
        ind_name=1
    if kwale==1 and team==0:
        info=['name','wind','wind_comp','points','speed','dist','dist_points','note_1','note_2','note_3','note_4','note_5','note_points','gate','gate_points']
    if team==1:
        ind_name=0
    """ind_name=2"""
    """print(ind_name)"""
    new_jump=pd.DataFrame([],columns=info)
    for i in range(len(output)):
        """print(ind_name, pre_2016, skok[ind_name])"""
        name=skok[ind_name]
        notes_pre=przeksztalc(skok[output[i]],kwale,team,TCS)
        notes=[float(x) for x in notes_pre.split()]
        passed_values=14-kwale
        if min(kwale,team)==1:
            passed_values=14
        if(len(notes)==passed_values):
            notes.append(0)
        print([name]+notes)
        data=pd.Series([name]+notes, index = new_jump.columns)
        new_jump=new_jump.append(data,ignore_index=True)
        """print(new_jump)"""
    return(new_jump)
def collect(jumps,kwale=0,team=0,pre_2016=0,TCS=0):
    info=['name','wind','wind_comp','dist','speed','dist_points','note_1','note_2','note_3','note_4','note_5','note_points','points','loc','gate','gate_points']
    if kwale==1 and team==0:
        info=['name','wind','wind_comp','points','speed','dist','dist_points','note_1','note_2','note_3','note_4','note_5','note_points','gate','gate_points']
    database=pd.DataFrame([],columns=info)
    for i in range(len(jumps)):
        new_jumps=znowu_przeksztalc(jumps[i],kwale,team,pre_2016,TCS)
        database=database.append(new_jumps,ignore_index=True)
    return(database)

for plik in lista:
    print(plik)
    content=zwroc_skoki(nazwa=plik,TCS=0)
    result=collect(content[0],content[1],content[2],content[3],content[4])
    result.to_csv(plik[:-4]+'.csv')

przyklad='2014JP3796RLQ.pdf'
parsed = parser.from_file(przyklad)
tekst=parsed["content"]
tekst=tekst.lower()
tekst_lin=tekst.splitlines()
tekst_lin = [i for i in tekst_lin if i] 
content=zwroc_skoki(tekstlin=tekst_lin,TCS=0)
troche_dalej=znowu_przeksztalc(content[0][33],content[1],content[2],1,content[4])
dalej=collect(content[0],content[1],content[2],1,content[4])
dalej.to_csv(przyklad[:-4]+'.csv')






"""wyrazy=nowy_string.split(' ')
wyrazy=[x for x in wyrazy if x!='']
nowy_string=' '.join(wyrazy[:-1])+' '+wyrazy[-1][:-2]+' '+wyrazy[-1][-2:]
print(nowy_string)"""

    

    


    