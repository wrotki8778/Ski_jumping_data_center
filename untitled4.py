# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 10:52:56 2020

@author: kubaf
"""
import requests
import csv
from bs4 import BeautifulSoup
import time
import os
import pandas as pd
os.chdir('C:/Users/kubaf/Documents/Skoki/WC/SGP/juz_przerobione/Kwalifikacje')
lista=os.listdir()
lista=[x for x in lista if (x[-6:]=='RL.pdf') | (x[-7:]=='RLQ.pdf')]
linki=[]
linki_tmp=[]
with open('linki_tmp.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        linki_tmp.append(row)
with open('linki_tmp_braki.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        linki_tmp.append(row)
with open('linki.csv', 'r') as file_2:
    reader = csv.reader(file_2)
    for row in reader:
        linki.append(row)
with open('linki_braki.csv', 'r') as file_2:
    reader = csv.reader(file_2)
    for row in reader:
        linki.append(row)
year=[]
for line in linki_tmp:
    year.append(line[0][-19:-15])
    
comps=pd.read_csv('comps_all.csv')
"""comps['season']=year
comps.to_csv('comps.csv')"""

braki=[]
for i,line in enumerate(lista):
    year=line[0:4]
    codex=int(line[6:10])
    print(year,codex)
    filtr=comps[(comps['season']==year)&(comps['codex']==codex)]
    """print(filtr)"""
    if filtr.empty:
        print("alert brak konkursu:"+lista[i])
        braki.append(lista[i])

info=['codex','place','month','day','year','gender','hill_size','team','season']
summer=pd.DataFrame([],columns=info)
for i,link in enumerate(linki):
    kod=linki_tmp[i]
    codex=kod[-4:]
    year=kod[-19:-15]
    time.sleep(4)
    url = link
    r = requests.get(url, headers={'user-agent': 'abcd'})
    soup = BeautifulSoup(r.text, "lxml")
    for a in soup.find_all('h1',{'class': 'heading heading_l2 heading_white heading_off-sm-style'}):
        print(a.text)
        place=a.text
    for a in soup.find_all('span',{'class': 'date__full'}):
        print(a.text)
        date=a.text
        date=a.text.replace(',','').split()
    for a in soup.find_all('div',{'class': 'event-header__kind'}):
        print(a.text)
        tmp=a.text.replace('','').split()
        gender=tmp[0][:-2]
        hill_size=tmp[-1][2:]
        if len(tmp)>=3:
            team=1
        else:
            team=0
    new_summer=pd.Series([codex]+[place]+date+[gender]+[hill_size]+[team]+[year], index = summer.columns)
    print(new_summer)
    summer=summer.append(new_summer,ignore_index=True)

"""comps_all=comps.append(comps_braki,ignore_index=True)
del comps_all['Unnamed: 0']"""
summer.to_csv('summer_quals.csv',index=False)
for k,url in enumerate(linki):
    link=url
    time.sleep(4)
    name=str(summer['season'][k])+'JP'+str(summer['codex'][k])+'naz.csv'
    print(name)
    r = requests.get(link, headers={'user-agent': 'abcd'})
    soup = BeautifulSoup(r.text, "lxml")
    lista=[]
    if summer['team'][k]:
        klas='table-row table-row_theme_additional'
    else:
        klas='table-row'
    for a in soup.find_all('a',{'class': klas}):
        """print(a.text)"""
        tmp=a.text.replace('\n','<')
        tmp=tmp.split('<')
        """print(tmp)"""
        tmp=[x for x in tmp if x]
        """print(tmp)"""
        for i,line in enumerate(tmp):
            nazwa=line.split(' ')
            nazwa=[x for x in nazwa if x]
            tmp[i]=' '.join(nazwa)
            if(len(nazwa)>1):
                tmp[i+1:]=[]
                break
        if summer['team'][k]:
            lista.append(tmp[-4:])
        else:
            lista.append(tmp[-3:])
    file = open(name, 'w+', newline ='') 
# writing the data into the file 
    with open(name,'w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        for line in lista:
            file.write(';'.join(line))
            file.write('\n')
    file.close()