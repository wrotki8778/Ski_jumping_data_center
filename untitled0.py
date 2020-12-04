# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 14:14:52 2020

@author: kubaf
"""
import os
import os.path
import pandas as pd
import numpy as np
import re
from tika import parser
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
os.chdir('C:/Users/kubaf/Documents/Skoki')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def decimal(list,list_k):
    output=[]
    for i,number in enumerate(list):
        output=output+[abs(list_k[i]*number-int(list_k[i]*number))>0]
    return output
def take_number(string):
    string=string.replace('m','')
    string=string.replace('/',' ')
    tmp=string.split(' ')
    take=[x for x in tmp if is_number(x)]
    return float(take[0])
def to_date(day,month,year):
    string=str(day)+' '+month+' '+str(year)
    return(datetime.strptime(string, '%d %B %Y'))
def validate(date_text):
    if len(date_text)<6:
        return False
    test=date_text[:3] + date_text[3].upper() + date_text[4:]
    try:
        if test != datetime.strptime(test, '%d %b %Y').strftime('%d %b %Y'):
            raise ValueError
        return True
    except ValueError:
        return False
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
            file_name=str(year)+'JP'+str(codex)+'nazfis.csv'
            with open(os.getcwd()+'\\nazwy\\'+file_name,'w+') as result_file:
                for i,line in enumerate(names_list):
                    mod_line=';'.join(line)
                    result_file.write(mod_line)
                    result_file.write('\n')
            result_file.close()    
            names_all=names_all+[names_list]
    return([database,names_all])
def import_links(years=[2021],genre='GP',to_download=['RL','RLQ','SLQ','SLR1','RLT','RTRIA'],import_data=[[],[],[],[],[]],import_num=0,scrap=True):
    [linki_tmp,linki,kody,database,names_list]=import_data
    if not linki_tmp:
        for i in range(len(years)):
            time.sleep(5)
            url = 'https://www.fis-ski.com/DB/?eventselection=results&place=&sectorcode=JP&seasoncode='+str(years[i])+'&categorycode='+genre+'&disciplinecode=&gendercode=&racedate=&racecodex=&nationcode=&seasonmonth=X-'+str(years[i])+'&saveselection=-1&seasonselection='
            r = requests.get(url, headers={'user-agent': 'ejdzent'})
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all('a', {'class': 'g-sm justify-left hidden-xs hidden-md-up bold'},href=True):
                linki_tmp.append(a['href'])  
        if import_num:
            linki_tmp=linki_tmp[:import_num]
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
        data=kod[0:4]
        cc=kod[6:10]
        url='http://medias3.fis-ski.com/pdf/'+data+'/JP/'+cc+'/'+kod+'.pdf'
        if not os.path.isfile(os.getcwd()+'\\PDFs\\'+kod+'.pdf'):
            time.sleep(1)
            r = requests.get(url, allow_redirects=True)
        else:
            print('Pominięto konkurs: '+kod+'.pdf')
            continue
        if r.status_code==404:
            continue
        else:
            time.sleep(4)
            open(os.getcwd()+'\\PDFs\\'+kod+'.pdf', 'wb').write(r.content)
            if os.path.getsize(os.getcwd()+'\\PDFs\\'+kod+'.pdf')<15:
                os.remove(os.getcwd()+'\\PDFs\\'+kod+'.pdf')
            else:
                print('Pobrano konkurs: '+kod+'.pdf')
    if scrap:
        [database,names_list]=scraping_fis(linki)
    return([linki_tmp,linki,kody,database,names_list])
def find_names(tekst_lin,year,tick):
    lista=[]
    if tick!=1:
        for i,line in enumerate(tekst_lin):        
            if len(line)==sum(c.isdigit() for c in line)+line.count('-') and sum(c.isalpha() for c in tekst_lin[i+1]):
                if int(year)<2016:
                    const=1
                else:
                    const=2
                next_line=tekst_lin[i+const]
                if sum(c.isdigit() for c in next_line) or max(1-next_line.count(' '),0):
                    print('Alert: w konkursie '+ nazwa +' zawodnik z nr '+line+' nazywa się '+next_line+'!')
                    const=1
                    next_line=tekst_lin[i+const]
                    print('Teraz w konkursie '+ nazwa +' zawodnik z nr '+line+' nazywa się '+next_line+'!')
                lista.append([line,next_line])
    else:
        indexes=[i for i,x in enumerate(tekst_lin) if validate(x)]
        lista=[[tekst_lin[i+1]]+[tekst_lin[i-3]] for i in indexes]
    return(lista)
def import_start_list(nazwa,tick=0,new_data=[],block=False): 
    year=nazwa[:4]
    codex=nazwa[6:10]
    try:
        file_name=nazwa[:10]+'SLQ.pdf'
        parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+file_name)
    except FileNotFoundError:
        file_name=nazwa[:10]+'SLR1.pdf'
        parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+file_name)
    except:
        return([])

    tekst=parsed["content"]
    tekst=tekst.replace('* ','')
    tekst=tekst.replace('*','')
    tekst=tekst.lower()
    tekst_lin=tekst.splitlines()
    tekst_lin = [i for i in tekst_lin if i] 
    lista=find_names(tekst_lin,year,tick)              
    info=['season','codex','hill_size','k-point','meter value','gate factor','wind factor','id','training']
    comps_infos=pd.DataFrame([],columns=info)
    word=['hill size','k-point','meter value','gate factor','wind factor']
    infos=[]
    for words in word:
        add=[i for i in tekst_lin if words in i]
        if add:
            infos.append(take_number(add[0]))
        else:
            infos.append(np.nan)
    ID=nazwa[:-4]
    training=nazwa.count('RLT') or nazwa.count('RTRIA')
    new_info=pd.Series([year]+[codex]+infos+[ID]+[training], index = comps_infos.columns)
    comps_infos=comps_infos.append(new_info,ignore_index=True)
    file_name=str(year)+'JP'+str(codex)+'naz.csv'
    if not os.path.isfile(os.getcwd()+'\\nazwy\\'+file_name): 
        with open(os.getcwd()+'\\nazwy\\'+file_name,'w+') as result_file:
            for i,line in enumerate(lista):
                mod_line=';'.join(line)
                result_file.write(mod_line)
                result_file.write('\n')
        result_file.close()
    if not block:
        return([lista,comps_infos])  
    else:
        return([[],comps_infos])
def zwroc_skoki(comp=[],names=[],nazwa=[],tekstlin=[]):
    if not comp.empty:
        nazwa=comp['id']+'.pdf'
    tick=comp['type']
    kwale=1
    team=0
    TCS=0
    names_list=[]
    if nazwa[-5]=='Q':
        kwale=2
    if not names:
        names_list=pd.DataFrame(import_start_list(nazwa,tick)[0],columns=['bib','name'])
    else:
        names_list=pd.DataFrame(names,columns=['bib','codex','name'])
        names_list['name']=names_list['name'].str.lower()
    parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+nazwa)
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
    word2='weather information'
    word3='large hill ko'
    for i,line in enumerate(tekst_lin):
        if word2 in line: # or word in line.split() to search for full words
            end.append(i)
        if word in line and i<=80: # or word in line.split() to search for full words
            kwale=0
        if word3 in line and i<=80: # or word in line.split() to search for full words
            TCS=1
    tekst_lin=tekst_lin[:end[0]]
    lista=[(i,[t for t in names_list['name'] if x.count(t)][0]) for i,x in enumerate(tekst_lin) if any(t for t in names_list['name'] if x.count(t))]+[(len(tekst_lin),'end')]
    indices=[(lista[i][0],lista[i+1][0],x[1]) for i,x in enumerate(lista[:-1])]
    skoki=[[x]+tekst_lin[s:e] for s,e,x in indices]
    if len(indices)<len(names_list):
        print('Warning: in '+comp['id']+' '+str(len(names_list) - len(indices))+' not found!')
    next_skoki=[conc_numbers(skok,comp) for i,skok in enumerate(skoki)]
    return([next_skoki,kwale,team,TCS])    
def conc_numbers(skok,comp):
    if comp['type']==1:
        return(conc_numbers_coc(skok,comp))
    if not comp['training']:
        return(skok)
    try:
        start=min([i for i,x in enumerate(skok) if x.count('.')])
        end=max([i for i,x in enumerate(skok) if min(x.count('.'),sum([t.isnumeric() for t in x if t.isnumeric()]))])
    except ValueError:
        return([skok[0]]+[10*'0.0 '])
    try:
        end_2=min([i for i,x in enumerate(skok) if x.count('page')])-1
    except ValueError:
        end_2=end
    if comp['id'].count('RTRIA'):
        line=' '.join([skok[start][4:]]+skok[start+1:min(end,end_2)+1])
        if skok[start].count('.')==1:
            line='0.0 '+line
        return([skok[0],line])
    elif comp['id'].count('RLT'):
        indexes=[(i,i+4) for i in range(start,min(end,end_2)) if not((i-start)%4)]
        lines=[' '.join(skok[i:j]) for i,j in indexes]
        new_lines=[skok[0]]
        for i,line in enumerate(lines):
            tmp=line.split(' ')
            tmp=[x for x in tmp if x=='dns' or not sum([t.isalpha() for t in x])]
            while(tmp[0]=='dns'):
                new_lines.append(10*'0.0 ')
                tmp=tmp[1:]
            new_lines.append(' '.join(tmp))
        return(new_lines)
    else:
        return(skok)
def conc_numbers_coc(skok,comp):
    if comp['training']:
        return(skok)
    try:
        start=min([i for i,x in enumerate(skok) if x.count('.') and sum([t.isnumeric() for t in x if t.isnumeric()])])
    except ValueError:
        return([skok[0]]+[10*'0.0 '])
    ciach_skok=skok[start:]
    try:
        end=start+min([i for i,x in enumerate(ciach_skok) if sum([t.isalpha() for t in x if t.isalpha()])])
    except ValueError:
        end=len(skok)
    print(start,end)
    if end-start-1:
        pierwszy=[i for i in range(start,end) if not((i-start)%2)]
        drugi=[i for i in range(start,end) if (i-start)%2 and i!=start+3]
        print(pierwszy,drugi)
        return([skok[0]]+[' '.join([skok[i] for i in pierwszy])]+[' '.join([skok[i] for i in drugi])])
    else:
        return([skok[0],skok[start]])
def przeksztalc(comp,string,kwale=0,team=0,TCS=0):
    nazwa=comp['id']
    if nazwa.count('RTRIA'):
        return(przeksztalc_rlt(string,kwale,team,TCS,'rtria'))
    elif nazwa.count('RLT'):
        return(przeksztalc_rlt(string,kwale,team,TCS,'rlt'))
    elif nazwa.count('RL'):
        return(przeksztalc_rl_rlq(string,kwale,team,TCS))
    else:
        return([])
def przeksztalc_rl_rlq(string,kwale,team,TCS):
    string=string.replace('pq', '0.')
    tmp=string.split(' ')
    if team and tmp[0].count('-'):
            del tmp[0]
    tmp=[x for x in tmp if not sum(i.isalpha() for i in x)]
    string=' '.join(tmp)
    pozycja=string.find('.')+2
    nowy_string=string[:pozycja]+' '+string[pozycja:]
    nowy_string=re.sub(r'[a-z]+', '', nowy_string, re.I)
    nowy_string=nowy_string.replace('©', '')
    nowy_string=nowy_string.replace('#', '')
    znacznik=nowy_string.find('*')
    if znacznik:
        nowy_string=nowy_string[(znacznik+1):]
    wyrazy=nowy_string.rsplit(' ', 2)
    nowy_string=wyrazy[1] + ' ' + wyrazy[2] + ' ' + wyrazy[0]  
    n=[12]
    offset=[1]
    if kwale:
        n=[]
        offset=[]
    if kwale and team:
        n=[12]
        offset=[1]
    if TCS and kwale==2:
        n=[11]
        offset=[1]  
    kropki=[i for i, a in enumerate(nowy_string) if a == '.']
    kropki=[kropki[i] for i in n]
    if n:
        nofy_string=[nowy_string[0:kropki[0]+offset[0]]]+[nowy_string[kropki[i]+offset[i]:kropki[i+1]+offset[i+1]] for i in range(len(kropki)-1)]+[nowy_string[kropki[-1]+offset[-1]:]]
        nofy_string=' '.join(nofy_string)
        wyrazy=nofy_string.split(' ')
    else:
        nofy_string=nowy_string
    if TCS and kwale==2:
        nofy_string=' '.join(wyrazy[0:3]) + ' ' + ' '.join(wyrazy[4:13]) + ' ' + ' '.join(wyrazy[14:])
    return(nofy_string)
def przeksztalc_rlt(string,kwale,team,TCS,layout):
    string=string.replace('©', '')
    nowy_string=string.split()
    nowy_string=[x for x in nowy_string if x]
    if nowy_string.count('dns') and layout=='rtria':
        return([0,0,0,0,0,0,0,0])
    elif nowy_string.count('dns') and layout=='rlt':
        return([0,0,0,0,0,0,0,0])
    if layout=='rtria':
        nofy_string=nowy_string[:2]+nowy_string[-4:]+nowy_string[4:-4]
    else:
        nofy_string=nowy_string[:2]+nowy_string[-2:]+nowy_string[4:-2]
    return(nofy_string+(8-len(nofy_string))*['0.0'])
def column_info(comp,kwale,team,TCS):
    info=['name','wind','wind_comp','speed','dist','dist_points','note_1','note_2','note_3','note_4','note_5','note_points','points','loc','gate','gate_points']
    if kwale and not(team):
        info=['name','wind','wind_comp','points','speed','dist','dist_points','note_1','note_2','note_3','note_4','note_5','note_points','gate','gate_points']
    nazwa=comp['id']
    if nazwa.count('RTRIA'):
        info=['name','speed','dist','wind','wind_comp','dist_points','loc','gate','gate_points']
    elif nazwa.count('RLT'):
        info=['name','speed','dist','wind_comp','loc','gate','wind','dist_points','gate_points']
    return(info)
def znowu_przeksztalc(comp,skok,kwale=0,team=0,TCS=0):
    exit_code=0
    output = [idx for idx, line in enumerate(skok) if line.count('.')>7] 
    if len(output)>2 and not comp['training']:
        print('Uwaga: zawodnik '+skok[0]+' oddał '+len(output)+" skoki!")
    if kwale and len(output)>1:
        print('Uwaga: zawodnik '+skok[0]+' oddał '+len(output)+" skoki w jednoseryjnym konkursie!")
    info=column_info(comp,kwale,team,TCS)
    new_jump=pd.DataFrame([],columns=info)
    for i in range(len(output)):
        name=skok[0]
        notes_pre=przeksztalc(comp,skok[output[i]],kwale,team,TCS)
        if not comp['training']:
            notes_pre=[x for x in notes_pre.split(' ') if x]
        notes=[float(x) for x in notes_pre]
        passed_values=14-bool(kwale)-5*comp['training']
        if min(kwale,team):
            passed_values=14
        if(len(notes)==passed_values):
            notes.append(0)
        data=pd.Series([name]+notes, index = new_jump.columns)
        if not comp['training']:
            conds=[abs(data['wind'])>3,abs(data['wind_comp'])>60,data['note_points']>60,data['note_5']>20]+decimal([data['wind_comp'],data['points'],data['dist_points'],data['gate_points'],data['speed'], data['note_1'], data['note_5'],data['note_points'],data['dist'],data['gate']],[10,10,10,10,10,2,2,2,2,1])
            condition=any(conds)
            if condition:
                exit_code=1
                print(conds)
                print(data)
        else:
            conds=[abs(data['wind'])>3,abs(data['wind_comp'])>60]+decimal([data['wind_comp'],data['dist_points'],data['gate_points'],data['speed'],data['dist'],data['gate']],[10,10,10,10,2,1])
            condition=any(conds)
            if condition:
                exit_code=1
                print(conds)
                print(data)
        new_jump=new_jump.append(data,ignore_index=True)
    return([new_jump,exit_code])
def collect(comp=[]):
    jumps,kwale,team,TCS=zwroc_skoki(comp)
    exit_code=0
    info=column_info(comp,kwale,team,TCS)
    database=pd.DataFrame([],columns=info)
    for i in range(len(jumps)):
        new_jumps,exit_code_tmp=znowu_przeksztalc(comp,jumps[i],kwale,team,TCS)
        exit_code=exit_code+exit_code_tmp
        database=database.append(new_jumps,ignore_index=True)
    return([database,exit_code])

years=[2021]
tick=1
types=['WC','COC','GP']
new_data=import_links(years=years,genre=types[tick],import_num=0)

comps_init=new_data[3]
comps_init_id=comps_init.apply(lambda x: x['season']+'JP'+x['codex'],axis=1).tolist()
to_process=['RLQ','RL','RLT','RTRIA']
to_process=[x+'.pdf' for x in to_process]
lista=os.listdir(os.getcwd()+'\\PDFs\\')
lista=[x for x in lista if any(t for t in to_process if t in x) and any(t for t in comps_init_id if t in x)]
lista.reverse()

start_lists=[]
comps=pd.DataFrame([],index=['season','codex','hill_size','k-point','meter value','gate factor','wind factor','id'])

for nazwa in lista:
    [list,comps_infos]=import_start_list(nazwa,tick)
    comps=comps.append(comps_infos,ignore_index=True)
    start_lists=start_lists+[[list]]
comps=pd.merge(comps,comps_init,on=['season','codex'],how='inner')  
comps['date']=comps.apply(lambda x: to_date(x['day'],x['month'],x['year']),axis=1)
comps=comps.drop(['month','day','year'],axis=1)
comps['type']=tick
name='_'.join([str(x) for x in years])+'_'+str(types[tick])+'.csv'
comps.to_csv(os.getcwd()+'\\comps\\'+name,index=False)


#comps=pd.read_csv(os.getcwd()+'\\comps\\2021_WC.csv')

n=1
comp=comps.iloc[n]
parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+comp['id']+'.pdf')
tekst=parsed["content"]
tekst=tekst.lower()
tekst_lin=tekst.splitlines()
tekst_lin = [i for i in tekst_lin if i] 
content=zwroc_skoki(comp,tekstlin=tekst_lin)
dalej,exit_code=collect(comp)
dalej.to_csv(comp['id']+'.csv',index=False)

for i,comp in comps.iterrows():
    content=zwroc_skoki(comp)
    [dalej,exit_code]=collect(comp)
    if exit_code:
        print(comp)
    if not os.path.isfile(os.getcwd()+'\\results\\'+comp['id']+'.csv'):
        dalej.to_csv(os.getcwd()+'\\results\\'+comp['id']+'.csv',index=False)









    

    


    