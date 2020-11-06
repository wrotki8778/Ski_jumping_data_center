# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 13:54:30 2020

@author: kubaf
"""
import requests
import csv
from bs4 import BeautifulSoup
import time
import os
os.chdir('C:/Users/kubaf/Documents/Skoki/WC/SGP/juz_przerobione/kwalifikacje')
lista=os.listdir()
lista=[x for x in lista if x[-4:]=='.pdf']
"""lista=lista[:10]"""
linki_tmp=[]
linki=[]
for kod in lista:
    time.sleep(3)
    url = 'http://www.fis-ski.com/DB/general/event-details.html?sectorcode=JP&seasoncode='+str(kod[0:4])+'&racecodex='+str(kod[6:10])
    r = requests.get(url, headers={'user-agent': 'abcd'})
    soup = BeautifulSoup(r.text, "lxml")
    # pierwszym argumentem konstruktora klasy BeautifulSoup jest zawartośc dokumentu HTML
    # drugim parametrem jest parser i nad nim się chwilę zatrzymamy
    for a in soup.find_all('a',{'class': 'px-xs-0 g-lg-2 g-md-2 g-sm-2 g-xs-4 justify-left'}, href=True):
        if('eventid' in a['href']):
            linki_tmp.append(a['href'])
            print(a['href'])
    linki_tmp=list(dict.fromkeys(linki_tmp))
for url in linki_tmp:
    time.sleep(2)
    """link=linki_tmp_braki[-1]"""
    kod=url[-4:]
    print(kod)
    r_2 = requests.get(url, headers={'user-agent': 'abcd'})
    soup_2 = BeautifulSoup(r_2.text, "lxml")
    for a in soup_2.find_all('a',{'class': 'g-lg-2 g-md-2 g-sm-3 g-xs-4 justify-center'}, href=True):
        if(a.string==kod):
            linki.append(a['href'])
            print(a['href'])
    
    
file = open('linki.csv', 'w+', newline ='') 

# writing the data into the file 
with open('linki.csv','w+') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    for line in linki:
        file.write(line)
        print(line)
        file.write('\n')        
        
file.close()
file = open('linki_tmp.csv', 'w+', newline ='') 

# writing the data into the file 
with open('linki_tmp.csv','w+') as result_file_2:
    wr_2 = csv.writer(result_file_2, dialect='excel')
    for line in linki_tmp:
        file.write(line)
        print(line)
        file.write('\n')  
file.close()