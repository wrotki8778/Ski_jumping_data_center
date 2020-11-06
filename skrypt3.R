library(tidyverse)
setwd('C://Users//kubaf//Documents//Skoki//')
wyniki_q<-read.csv('~/Skoki/WC/Kwalifikacje/wyniki_zbiorcze.csv',header=TRUE)
wyniki<-read.csv('~/Skoki/WC/Konkursy/comp_results.csv',header=TRUE)
nazwy<-read.csv('~/Skoki/WC/nazwy_all.csv',header=TRUE)
nazwy=nazwy[!duplicated(nazwy$name), ]
nazwy['X']=NULL
nejmy_q=wyniki_q['name']
wyniki_q['codex'] = left_join(wyniki_q['name'],nazwy)$codex
wyniki['codex'] = left_join(wyniki['name'],nazwy)$codex
wyniki['name']=NULL
wyniki_q['name']=NULL
write.csv(wyniki,'new_comp_results.csv')
write.csv(wyniki_q,'new_qual_results.csv')