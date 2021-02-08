# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:01:40 2020

@author: kubaf
"""
import os
import pandas as pd
import re
from tika import parser
os.chdir('C:/Users/kubaf/Documents/Skoki')
actual_comps = pd.read_csv(os.getcwd()+'\\all_comps.csv')
os.chdir('C:/Users/kubaf/Documents/Skoki/PDFs')


comp = actual_comps.loc[357]
parsed = parser.from_file(comp['id']+'.pdf')
tekst=parsed["content"]
tekst=tekst.lower()
tekst_lin=tekst.splitlines()
tekst_lin = [i for i in tekst_lin if i] 

word1 = 'weather information'
word2 = 'statistics'
start = min([i for i,x in enumerate(tekst_lin) if x.count(word1)])
end = max([i for i,x in enumerate(tekst_lin) if x.count(word2)])
tekst_lin = tekst_lin[start:end]

    






