require(dplyr)
require(graphics)
setwd('C://Users//kubaf//Documents//Skoki')
sgp_comp_results<-read.csv('SGP_comp_results.csv',header=TRUE)
wc_comp_results<-read.csv('WC_comp_results.csv',header=TRUE)
sgp_qual_results<-read.csv('SGP_qual_results.csv',header=TRUE)
wc_qual_results<-read.csv('WC_qual_results.csv',header=TRUE)
nazwy_all<-read.csv('nazwy_all.csv',header=TRUE)
comps_all<-read.csv('comps_all_WC_CC_SGP.csv',header=TRUE)
nazwy_all$X=NULL
results_all=rbind(sgp_comp_results,sgp_qual_results,wc_comp_results,wc_qual_results)
id_athlete= results_all %>% left_join(nazwy_all,by='name')
id_athlete$name=NULL
write.csv(id_athlete,'results_all.csv')