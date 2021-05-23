setwd("~/GitHub/Ski_jumping_data_center")
all_ratings <- read.csv("all_ratings.csv")
all_comps_r <- read.csv("all_comps_r.csv")
all_results <- read.csv("all_results.csv")
all_results <- all_results[all_results['speed']>50 & all_results['speed']<115,]
all_results <- all_results[all_results['dist']>40,]
dataset <- merge(all_results,all_comps_r,by=c('id'),all.y=FALSE)
dataset['short_id'] <- apply(dataset['id'], 2, substr ,start=1, stop=10)
short_ratings <- all_ratings
short_ratings['short_id'] <- apply(short_ratings['id'], 2, substr ,start=1, stop=10)
short_ratings <- short_ratings[!duplicated(short_ratings[,c('short_id','codex')]),c('short_id','codex','cumm_rating')]
dataset['norm_dist'] = dataset['dist']/dataset['hill_size_x']
dataset <- merge(dataset,short_ratings,by.x=c('short_id','codex.x'),by.y=c('short_id','codex'),all.y=FALSE,all.x=TRUE)
dataset$gender = as.integer(as.factor(dataset$gender))
dataset$date_new = as.integer(as.Date.character(dataset$date))
simple_model <- lm(norm_dist~speed+wind+hill_size_x+cumm_rating+gender+date_new+training, data=dataset)
summary(simple_model)
library(mgcv) # contains our prime model
model<-gam(norm_dist~s(speed)+wind+hill_size_x+s(cumm_rating)+gender+training+s(date_new), data=dataset)
summary(model)
