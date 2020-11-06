require(dplyr)
require(graphics)
setwd('C://Users//kubaf//Documents//Skoki')
#comps<- read.csv("~//Skoki//WC//Konkursy//comps_all.csv",header=TRUE)
#comps_infos<- read.csv("~//Skoki//WC//Konkursy//comps_infos_fix.csv",header=TRUE)
#comps_all= comps %>% full_join(comps_infos,by=c("codex"="season","season"="codex"))
#comps<- read.csv("~//Skoki//WC//Kwalifikacje//quals.csv",header=TRUE)
#comps_infos<- read.csv("~//Skoki//WC//Kwalifikacje//comps_infos_fix.csv",header=TRUE)
comps_all= read.csv("~/Skoki/comps_all_WC_CC_SGP.csv",header=TRUE)
subset=comps_all %>% filter(type==2 & qual==0)
subset= subset[order(subset$season,subset$codex),]
years=c(2012,2013,2014,2015,2016,2017,2018,2019,2020)
places=c('Innsbruck')
#subset<-comps_all %>% filter(season %in% years) # & grepl('Innsbruck', place)
comp <- read.csv("~//Skoki//new_comp_results.csv",header=TRUE)
nazwy=c('year',colnames(comp))
pusty=data.frame(matrix(ncol = length(nazwy), nrow = 0))
colnames(pusty)=nazwy
for (i in 1:nrow(subset)){
  codex=subset$codex[i]
  year=subset$season[i]
  print(paste(codex,year))
  comp <- read.csv(paste("~/Skoki/WC/SGP/csvki/Konkursy/",year,"JP",codex,"RL.csv",sep=''),header=TRUE)
  comp['id']=subset$X.1[i]
  comp[,1]=NULL
  if(!('loc' %in% colnames(comp))){
    comp['loc']=NA
  }
  pusty=rbind(pusty,comp)
}

write.csv(pusty,'SGP_comp_results.csv')
write.csv(comps_all,'info_zbiorcze.csv')
model=lm(dist~speed+wind+year,data=pusty)
pairs(pusty[c('speed','wind','dist','note_points','year')])
samp=pusty[c('name','dist','note_points')]
samp = samp %>% filter(note_points>45)
hs=130
k=120
#samp=as.data.frame(scale(samp))
samp['dist']=(samp['dist']-k)/(hs-k)
samp = samp %>% filter(-3*dist+note_points>48)
samp_2=samp[c('dist','note_points')]
plot(samp_2)
abline(a=48,b=3)
abline(lm(formula = note_points ~ dist, data = samp),col='red')
obrot=eigen(cov(samp_2))$vectors
skala=diag(eigen(cov(samp_2))$values,ncol=2)
skladowe=scale(as.matrix(samp_2)%*%obrot)
plot(skladowe)
lines(skladowe[samp$name=='kubacki dawid',],col='green',type='p')
