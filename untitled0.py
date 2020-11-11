# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 14:14:52 2020

@author: kubaf
"""
import os
import pandas as pd
import numpy as np
import re
from tika import parser
import requests
import csv
from bs4 import BeautifulSoup
import time
os.chdir('C:/Users/kubaf/Documents/Skoki')

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
    return float(take[0])

def scraping_fis(linki):
    info=['codex','place','month','day','year','gender','hill_size','team','season']
    database=pd.DataFrame([],columns=info)
    names_all=[]
    for i,item in enumerate(linki):
            time.sleep(4)
            [link,year]=item
            r = requests.get(link, headers={'user-agent': 'abcd'})
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all('span', {'class': 'event-details__field'}):
                codex=a.text[-4:]
            for a in soup.find_all('h1',{'class': 'heading heading_l2 heading_white heading_off-sm-style'}):
                place=a.text
            for a in soup.find_all('span',{'class': 'date__full'}):
                date=a.text
                date=a.text.replace(',','').split()
            for a in soup.find_all('div',{'class': 'event-header__kind'}):
                tmp=a.text.replace('','').split()
                gender=tmp[0][:-2]
                hill_size=tmp[-1][2:]
                if len(tmp)>=3:
                    team=1
                else:
                    team=0
            new_comp=pd.Series([codex]+[place]+date+[gender]+[hill_size]+[team]+[year], index = database.columns)
            database=database.append(new_comp,ignore_index=True)
            r = requests.get(link, headers={'user-agent': 'abcd'})
            soup = BeautifulSoup(r.text, "lxml")
            names_list=[]
            if team:
                klas='table-row table-row_theme_additional'
            else:
                klas='table-row'
            for a in soup.find_all('a',{'class': klas}):
                tmp=a.text.replace('\n','<')
                tmp=tmp.split('<')
                tmp=[x for x in tmp if x]
                for i,line in enumerate(tmp):
                    nazwa=line.split(' ')
                    nazwa=[x for x in nazwa if x]
                    tmp[i]=' '.join(nazwa)
                    if(len(nazwa)>1):
                        tmp[i+1:]=[]
                        break
                if team:
                    names_list.append(tmp[-4:])
                else:
                    names_list.append(tmp[-3:])
            names_all=names_all+[names_list]
    return([database,names_all])

def import_links(years=[2021],genre='GP',to_download=['RL','RLQ','SLQ','SLR1','RLT','RTRIA'],import_data=[[],[],[],[],[]],scrap=True):
    [linki_tmp,linki,kody,database,names_list]=import_data
    if not linki_tmp:
        for i in range(len(years)):
            time.sleep(5)
            url = 'https://www.fis-ski.com/DB/?eventselection=&place=&sectorcode=JP&seasoncode='+str(years[i])+'&categorycode='+genre+'&disciplinecode=&gendercode=&racedate=&racecodex=&nationcode=&seasonmonth=X-'+str(years[i])+'&saveselection=-1&seasonselection='
            r = requests.get(url, headers={'user-agent': 'ejdzent'})
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all('a', {'class': 'g-sm justify-left hidden-xs hidden-md-up bold'},href=True):
                linki_tmp.append(a['href'])
        linki_tmp=list(dict.fromkeys(linki_tmp))
    if not linki:
        for url in linki_tmp:
            time.sleep(4)
            year=url[-4:]
            r = requests.get(url, headers={'user-agent': 'ejdzent'})
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all('a', {'class': 'px-1 g-lg-3 g-md-3 g-sm-4 g-xs-4 justify-left'},href=True):
                linki.append([a['href'],year])
    if not kody:
        for item in linki: 
            time.sleep(4)
            url=item[0]
            year=item[1]
            r = requests.get(url, headers={'user-agent': 'ejdzent'})
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all('span', {'class': 'event-details__field'}):
                codex=a.text[-4:]
            for suffix in to_download:
                tmp_file_name=year+'JP'+codex+suffix
                kody.append(tmp_file_name)
    for kod in kody:
        time.sleep(4)
        data=kod[0:4]
        cc=kod[6:10]
        url='http://data.fis-ski.com/pdf/'+data+'/JP/'+cc+'/'+kod+'.pdf'
        r = requests.get(url, allow_redirects=True)
        open(kod+'.pdf', 'wb').write(r.content)
        print('Pobrano konkurs: '+kod+'.pdf')
    if scrap:
        [database,names_list]=scraping_fis(linki)
    return([linki_tmp,linki,kody,database,names_list])
#new_data=import_links()

to_process=['SLQ','SLR1']
to_process=[x+'.pdf' for x in to_process]
lista=os.listdir()
lista=[x for x in lista if any(t for t in to_process if t in x)]
lista.reverse()

def import_start_list(nazwa): 
    qual=len([x for x in nazwa if x=='Q'])
    year=nazwa[:4]
    codex=nazwa[6:10]
    if qual:
        file_name=str(year)+'JP'+str(codex)+'SLQ.pdf'
    else:
        file_name=str(year)+'JP'+str(codex)+'SLR1.pdf'
    id=str(year)+'JP'+str(codex)+'RL'
    if qual:
        id=id+'Q'
    parsed = parser.from_file(file_name)
    tekst=parsed["content"]
    tekst=tekst.replace('* ','')
    tekst=tekst.lower()
    tekst_lin=tekst.splitlines()
    tekst_lin = [i for i in tekst_lin if i] 
    lista=[]
    for i,line in enumerate(tekst_lin):        
        if len(line)==sum(c.isdigit() for c in line)+line.count('-') and sum(c.isalpha() for c in tekst_lin[i+1]):
            if int(year)<2016:
                const=1
            else:
                const=2
            next_line=tekst_lin[i+const]
            lista.append([line,next_line])
            if sum(c.isdigit() for c in next_line) or max(1-next_line.count(' '),0):
                print('Alert: zawodnik z nr '+line+' nazywa się '+next_line+'!')
    info=['season','codex','hill_size','k-point','meter value','gate factor','wind factor','id']
    comps_infos=pd.DataFrame([],columns=info)
    word=['hill size','k-point','meter value','gate factor','wind factor']
    infos=[]
    for words in word:
        add=[i for i in tekst_lin if words in i]
        if add:
            infos.append(take_number(add[0]))
        else:
            infos.append(np.nan)
    new_info=pd.Series([year]+[codex]+infos+[id], index = comps_infos.columns)
    comps_infos=comps_infos.append(new_info,ignore_index=True)
    return([lista,comps_infos])  

start_lists=[]
comps_infos_all=pd.DataFrame([],index=['season','codex','hill_size','k-point','meter value','gate factor','wind factor','id'])
for nazwa in lista:
    [lista,comps_infos]=import_start_list(nazwa)
    comps_infos_all=comps_infos_all.append(comps_infos,ignore_index=True)
    with open(nazwa[:-4]+'.csv','w+') as result_file:
        for line in lista:
            mod_line=';'.join(line)
            result_file.write(mod_line)
            result_file.write('\n')
    result_file.close()
    start_lists=start_lists+[[lista]]

comps_infos_all=pd.merge(comps_infos_all,new_data[3],on=['season','codex'],how='inner')    
to_process=['RLQ','RL']
to_process=[x+'.pdf' for x in to_process]
lista=os.listdir()
lista=[x for x in lista if any(t for t in to_process if t in x)]
lista.reverse()
lista=lista[0:4]

def zwroc_skoki(nazwa,tekstlin=[],TCS=0):
    kwale=1
    team=0
    pre_2016=0
    names_list=[]
    if nazwa[-5]=='Q':
        kwale=1
        names_jumpers=nazwa[0:10]+'SLQ'
    else:
        names_jumpers=nazwa[0:10]+'SLR1' 
    names_list=pd.read_csv(names_jumpers+'.csv',sep=';',header=None)
    names_list.columns=['bib','name'] 
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
    word='round'
    word2='competition / weather information'
    for i,line in enumerate(tekst_lin):
        if word2 in line: # or word in line.split() to search for full words
            end.append(i)
        if word in line: # or word in line.split() to search for full words
            kwale=0
    tekst_lin=tekst_lin[:end[0]]
    lista=[i for i,x in enumerate(tekst_lin) if any(t for t in names_list['name'] if x.count(t))]+[len(tekst_lin)]
    indices=[(lista[i],lista[i+1]) for i in range(len(lista)-1)]
    skoki=[tekst_lin[s:e] for s,e in indices]
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
    kropki=[kropki[i] for i in n]
    if n:
        nofy_string=[nowy_string[0:kropki[0]+offset[0]]]+[nowy_string[kropki[i]+offset[i]:kropki[i+1]+offset[i+1]] for i in range(len(kropki)-1)]+[nowy_string[kropki[-1]+offset[-1]:]]
        nofy_string=' '.join(nofy_string)
        wyrazy=nofy_string.split(' ')
    else:
        nofy_string=nowy_string
    if TCS and kwale:
        nofy_string=' '.join(wyrazy[0:3]) + ' ' + ' '.join(wyrazy[4:13]) + ' ' + ' '.join(wyrazy[14:])
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
    new_jump=pd.DataFrame([],columns=info)
    for i in range(len(output)):
        name=skok[0]
        notes_pre=przeksztalc(skok[output[i]],kwale,team,TCS)
        notes=[float(x) for x in notes_pre.split()]
        passed_values=14-kwale
        if min(kwale,team)==1:
            passed_values=14
        if(len(notes)==passed_values):
            notes.append(0)
        data=pd.Series([name]+notes, index = new_jump.columns)
        new_jump=new_jump.append(data,ignore_index=True)
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
przyklad='2021JP3139RL.pdf'
parsed = parser.from_file(przyklad)
tekst=parsed["content"]
tekst=tekst.lower()
tekst_lin=tekst.splitlines()
tekst_lin = [i for i in tekst_lin if i] 
content=zwroc_skoki(przyklad)
troche_dalej=znowu_przeksztalc(content[0][17],content[1],content[2],content[3],content[4])
dalej=collect(content[0],content[1],content[2],content[3],content[4])
dalej.to_csv(przyklad[:-4]+'.csv')

for plik in lista:
    print(plik)
    content=zwroc_skoki(nazwa=plik)
    result=collect(content[0],content[1],content[2],content[3],content[4])
    result.to_csv(plik[:-4]+'.csv')

    

    


    