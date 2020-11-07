# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 14:14:52 2020

@author: kubaf
"""
import os
import pandas as pd
import re
from tika import parser
import requests
import csv
from bs4 import BeautifulSoup
import time
os.chdir('C:/Users/kubaf/Documents/Skoki')
lista_pdf=os.listdir()
lista_pdf=[x for x in lista_pdf if x[-4:]=='.pdf']
lista_csv=os.listdir('C:/Users/kubaf/Documents/Skoki/WC/Kwalifikacje/csvki')
lista_csv=[x for x in lista_csv if x[-7:]=='RLQ.csv']
lista=[x for x in lista_pdf if x[:-7]+'RLQ.csv' not in lista_csv]
lista.reverse()
lista=[lista[1]]

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
            new_comp=pd.Series([codex]+[place]+date+[gender]+[hill_size]+[team]+[year], index = database.columns)
            print(new_comp)
            database=database.append(new_comp,ignore_index=True)
            name=str(year)+'JP'+str(codex)+'naz.csv'
            print(name)
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
                print(a['href'])
        linki_tmp=list(dict.fromkeys(linki_tmp))
    if not linki:
        for url in linki_tmp:
            time.sleep(4)
            year=url[-4:]
            r = requests.get(url, headers={'user-agent': 'ejdzent'})
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all('a', {'class': 'px-1 g-lg-3 g-md-3 g-sm-4 g-xs-4 justify-left'},href=True):
                linki.append([a['href'],year])
                print([a['href'],year])
        #linki=list(dict.fromkeys(linki))
        #kody=list(dict.fromkeys(kody))
        print(linki)
    if not kody:
        for item in linki: 
            time.sleep(4)
            print(item)
            url=item[0]
            year=item[1]
            r = requests.get(url, headers={'user-agent': 'ejdzent'})
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all('span', {'class': 'event-details__field'}):
                codex=a.text[-4:]
                print(codex)
            for suffix in to_download:
                tmp_file_name=year+'JP'+codex+suffix
                print(tmp_file_name)
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
new_data=import_links(years=[2010,2020],genre='WC')
def import_start_list(comp): 
    qual=comp['qual']
    year=comp['season']
    codex=comp['codex']
    if qual:
        file_name=str(year)+'JP'+str(codex)+'SLQ.pdf'
    else:
        file_name=str(year)+'JP'+str(codex)+'SLR1.pdf'
    parsed = parser.from_file(file_name)
    tekst=parsed["content"]
    tekst=tekst.lower()
    tekst_lin=tekst.splitlines()
    tekst_lin = [i for i in tekst_lin if i] 
    lista=[]
    for i,line in enumerate(tekst_lin):
        if len(line)==sum(c.isdigit() for c in line) and sum(c.isalpha() for c in tekst_lin[i+1]):
            if comp['season']<2016:
                const=1
            else:
                const=2
            next_line=tekst_lin[i+const]
            lista.append([line,next_line])
            if sum(c.isdigit() for c in next_line) or max(1-next_line.count(' '),0):
                print('Alert: w konkursie nr ' +str(k)+' zawodnik z nr '+line+' nazywa się '+next_line+'!')
     return(lista)  
 
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

    

    


    