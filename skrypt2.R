library(tidyverse)
setwd('C://Users//kubaf//Documents//Skoki//')
wyniki_q<-read.csv('~/Skoki/WC/Kwalifikacje/wyniki_zbiorcze.csv',header=TRUE)
nazwy_q<-wyniki_q['name']
setwd('C://Users//kubaf//Documents//Skoki//WC//Konkursy')
#comps_results <- read.csv("~/Skoki/WC/Konkursy/comps_results.txt")
#comps_all <- read.csv("~//Skoki//WC//Konkursy//comps_all.csv",header=TRUE)
#cc_comps_all <- read.csv("~/Skoki/WC/CC/cc_comps.csv",header=TRUE)
#summer_comps_all <- read.csv("~/Skoki/WC/CC/cc_comps.csv",header=TRUE)
nazwy_pre<-read.csv("~/Skoki/nazwy_all.csv",header=TRUE)

comps_all<-read.csv("~/Skoki/comps_all_WC_CC_SGP.csv",header=TRUE)
comps=comps_all %>%filter(qual==0)

#nazwy_konk<-comps_results['name']
nazwy=c('bib','codex','name')
comp <- data.frame(matrix(ncol = length(nazwy), nrow = 0))
pusty=data.frame(matrix(ncol = 3, nrow = 0))
colnames(pusty)=c('bib','codex','name')
"nazwy=c('year',colnames(comp))
colnames(pusty)=nazwy"
for (i in 1:nrow(comps)){
  codex=comps$codex[i]
  year=comps$season[i]
  team=comps$team[i]
  print(paste(i,codex,year,team))
  comp <- read.csv(paste("~/Skoki/WC/nazwy/",year,"JP",codex,"naz.csv",sep=''),sep=';',header=FALSE)
  #print(comp)
  if (comps$team[i]==1){
    print('team')
    colnames(comp)=c('codex','name')
    comp['bib']=NA
    print(comp)
  }
  else{
    colnames(comp)=c('bib','codex','name')
  }
  "comp['year']=year
  comp['codex']=codex
  comp[,1]=NULL
  if(!('loc' %in% colnames(comp))){
    comp['loc']=NA
  }"
  pusty=rbind(pusty,comp)
}
pusty_q=data.frame(matrix(ncol = 2, nrow = 0))
colnames(pusty_q)=c('bib','name')
quals = comps_all %>%filter(qual==1)
quals=quals[-310,]
for (i in 1:nrow(quals)){
  codex=quals$codex[i]
  year=quals$season[i]
  print(paste(i,codex,year))
  comp <- read.csv(paste("~//Skoki//WC//nazwy_slq//",year,"JP",codex,"SLQ.csv",sep=''),sep=';',header=FALSE)
  colnames(comp)=c('bib','name')
  "comp['year']=year
  comp['codex']=codex
  comp[,1]=NULL
  if(!('loc' %in% colnames(comp))){
    comp['loc']=NA
  }"
  
  pusty_q=rbind(pusty_q,comp)
}
pusty['bib']=NULL
pusty_q['bib']=NULL
pusty_q$name=gsub(' [*]','',pusty_q$name)
pusty=unique(pusty)
pusty_q=unique(pusty_q)
for (i in 1:nrow(pusty)){
  pusty$name[i]=tolower(pusty$name[[i]])
}
#nazwy_all=unique(rbind(nazwy_konk,nazwy_q))
nazwy_all= unique(full_join(nazwy_pre,pusty,pusty_q,by=c('name')))
write.csv(nazwy_all,'nazwy_all.csv')