library(tidyverse)
setwd('C://Users//kubaf//Documents//Skoki//')
pdf_comps_infos<-read.csv('~/Skoki/WC/SGP/juz_przerobione/summer_comps_infos.csv',header=TRUE)
pdf_quals_infos<-read.csv('~/Skoki/WC/SGP/juz_przerobione/summer_quals_infos.csv',header=TRUE)
comps_infos<-read.csv('~/Skoki/WC/SGP/juz_przerobione/summer_comps.csv',header=TRUE)
quals_infos<-read.csv('~/Skoki/WC/SGP/juz_przerobione/summer_quals.csv',header=TRUE)
comps= comps_infos %>% full_join(pdf_comps_infos,by=c("codex"="season","season"="codex"))
quals= quals_infos %>% full_join(pdf_quals_infos,by=c("codex"="season","season"="codex"))
comps['qual']=0
quals['qual']=1
comps['type']=2
quals['type']=2
comps_all=rbind(comps,quals)
lct <- Sys.setlocale("LC_TIME", "C")
date=comps_all[c('month','day','year')]
date=unite(date, value, sep=" ")
date=apply(date,1,as.Date,format='%B %d %Y')
date=as.Date(date,format='%d.%m.%Y', origin = "01.01.1970")
comps_all['date']=date
comps_all[c('month','day','year')]=NULL
wc_comps_all=read.csv('~/Skoki/comps_date_new.csv',header=TRUE)
comps_final=rbind(comps_all,wc_comps_all)
comps_final=comps_final[
  order( comps_final$date, 1-comps_final$qual ),
]
write.csv(comps_final,'comps_all_WC_CC_SGP.csv')