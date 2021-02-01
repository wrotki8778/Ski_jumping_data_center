library(tidyverse)
setwd('C://Users//kubaf//Documents//Skoki//')

dir = paste(getwd(), '/all_names.csv', sep = '')
names <- read.csv2(dir, header = TRUE, sep = ',')

files_RL <-
  list.files(
    path = paste(getwd(), "/results", sep = ''),
    pattern = "*RL.csv",
    full.names = FALSE,
    recursive = FALSE
  )
files_RLQ <-
  list.files(
    path = paste(getwd(), "/results", sep = ''),
    pattern = "*RLQ.csv",
    full.names = FALSE,
    recursive = FALSE
  )
files_RLT <-
  list.files(
    path = paste(getwd(), "/results", sep = ''),
    pattern = "*RLT.csv",
    full.names = FALSE,
    recursive = FALSE
  )
files_RTRIA <-
  list.files(
    path = paste(getwd(), "/results", sep = ''),
    pattern = "*RTRIA.csv",
    full.names = FALSE,
    recursive = FALSE
  )

files_naz <-
  list.files(
    path = paste(getwd(), "/nazwy", sep = ''),
    pattern = "*naz.csv",
    full.names = FALSE,
    recursive = FALSE
  )

files_nazfis <-
  list.files(
    path = paste(getwd(), "/nazwy", sep = ''),
    pattern = "*nazfis.csv",
    full.names = FALSE,
    recursive = FALSE
  )

files = c(files_RL, files_RLQ, files_RLT, files_RTRIA)
files_names = c(files_naz, files_nazfis)
column_names = c(
  'name',
  'speed',
  'dist',
  'dist_points',
  'points',
  'note_1',
  'note_2',
  'note_3',
  'note_4',
  'note_5',
  'note_points',
  'gate',
  'gate_points',
  'wind',
  'wind_comp',
  'loc'
)
function_names = c('as.character', rep('as.double', 14), 'as.integer')
results_database = lapply(files, function(x) {
  dir = paste(getwd(), '/results/', x, sep = '')
  t <- read.csv2(dir, header = TRUE, sep = ',') # load file
  nowe_nazwy = setdiff(colnames(t), column_names)
  if (length(nowe_nazwy) > 0) {
    print(paste(x, nowe_nazwy))
  }
  for (i in 1:length(column_names)) {
    column_name = column_names[i]
    function_name = function_names[i]
    if (column_name %in% colnames(t)) {
      t[[column_name]] = apply(as.matrix(t[[column_name]]), 2, function_name)
    }
  }
  t[['id']] = rep(substr(x, 1, nchar(x) - 4), nrow(t))
  return(t)
})

names_naz_database = lapply(files_naz, function(x) {
  dir = paste(getwd(), '/nazwy/', x, sep = '')
  t <-
    tryCatch(
      read.csv2(dir, header = FALSE, sep = ';'),
      error = function(e)
        NULL
    ) # load file
  if (length(colnames(t))){
    colnames(t) = c('bib', 'name')
    t$bib = as.character(t$bib)
  }
  return(t)
})

names_nazfis_database = lapply(files_nazfis, function(x) {
  dir = paste(getwd(), '/nazwy/', x, sep = '')
  t <-
    tryCatch(
      read.csv2(dir, header = FALSE, sep = ';'),
      error = function(e)
        NULL
    ) # load file
  if (length(colnames(t)) == 2){
    colnames(t) = c('codex', 'name')
    t$name = tolower(t$name) 
  }
  else if (length(colnames(t)) == 3){
    colnames(t) = c('bib', 'codex', 'name')
    t$bib = as.character(t$bib)
    t$name = tolower(t$name)
  }
  return(t)
})
names = bind_rows(names_nazfis_database, .id = "column_label")
names = names %>% distinct(name,codex, .keep_all = TRUE)
names$bib = NULL
names$column_label = NULL

names_naz = bind_rows(names_naz_database, .id = "column_label")
names_naz = names_naz %>% distinct(name, .keep_all = TRUE)
names_naz$column_label = NULL
names_naz$bib = NULL

names = merge(names, names_naz, by='name', all.x = TRUE, all.y = TRUE)

results = bind_rows(results_database, .id = "column_label")
results$X <- 1:nrow(results)
results$Unnamed..0 = NULL
results$column_label = NULL

results = merge(results, names, by = c('name'))
results = results[order(results$X), ]

results$name = NULL
results$X = NULL

write.csv2(names, 'all_names.csv', row.names = FALSE)
write.csv2(results, 'all_results.csv', row.names = FALSE)

dir_c = paste(getwd(), '/all_comps.csv', sep = '')
competitions <- read.csv2(dir_c, header = TRUE, sep = ',')

area = 'Kuopio'
min_size = 115
max_size = 155
filtered_competition <-
  filter(
    competitions,
    str_detect(place, area) &
      as.double(hill_size_x) > min_size &
      as.double(hill_size_x) < max_size
  )[-c(56, 57, 58), ]
filtered_results <-
  filter(results, comp_name %in% filtered_competition$id)
filtered_results <- filter(filtered_results, gate != 0 & speed > 85)
boxplot(
  speed ~ gate,
  data = filtered_results ,
  main = "Correlation of speed with inrun length in Garmisch",
  xlab = "Gate",
  ylab = "Speed (km/h)",
  varwidth = TRUE
)
