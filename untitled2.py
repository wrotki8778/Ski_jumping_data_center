# -*- coding: utf-8 -*-
"""
Test module
"""
import os
import pandas as pd
import re
from tika import parser
os.chdir('C:/Users/kubaf/Documents/Skoki')
actual_comps = pd.read_csv(os.getcwd()+'\\all_comps.csv')
os.chdir('C:/Users/kubaf/Documents/Skoki/PDFs')

def find_weather(comp):
    if not os.path.isfile(os.getcwd()+'//'+comp['id']+'.pdf'):
        return [[comp['id'], '']]
    print(comp.name)
    parsed = parser.from_file(comp['id']+'.pdf')
    tekst=parsed["content"]
    tekst=tekst.lower()
    tekst_lin=tekst.splitlines()
    tekst_lin = [i for i in tekst_lin if i] 
    word1 = 'weather information'
    word2 = 'statistics'
    try:
        start = min([i for i,x in enumerate(tekst_lin) if x.count(word1)])
        end = max([i for i,x in enumerate(tekst_lin) if x.count(word2)])
    except ValueError:
        return [[comp['id'], '']]
    tekst_lin = tekst_lin[start:end]
    word_acc = ['1st round', 'trial round', '2nd round',
                'training', 'qualification', 'final round']
    tekst_lin = [[comp['id'], x] for x in tekst_lin 
                 if sum(c.isdigit() for c in x) > 4
                 and sum([x.count(word) for word in word_acc])]
    return tekst_lin


weather_data = [find_weather(comp) for i,comp in actual_comps.iterrows()]
all_data = [item for sublist in weather_data for item in sublist]
    






