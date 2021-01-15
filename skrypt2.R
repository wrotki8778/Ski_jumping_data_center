library(tidyverse)
setwd('C://Users//kubaf//Documents//Skoki//')
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
files = c(files_RL, files_RLQ, files_RLT, files_RTRIA)
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
  t[['comp_name']] = rep(substr(x, 1, nchar(x) - 4), nrow(t))
  return(t)
})
results = bind_rows(results_database, .id = "column_label")
results$X = NULL
results$Unnamed..0 = NULL
results$column_label = NULL

write.csv2(results,'all_results.csv',row.names = FALSE)

dir_c = paste(getwd(), '/comps/all_comps.csv', sep = '')
competitions <- read.csv2(dir_c, header = TRUE, sep = ',')

area = 'Oslo'
min_size = 115
max_size = 155
filtered_competition <-
  filter(competitions, str_detect(place, area) & as.double(hill_size_x) > min_size & as.double(hill_size_x) < max_size)[-c(42,43),]
filtered_results <-
  filter(results, comp_name %in% filtered_competition$id)
filtered_results <- filter(filtered_results, gate != 0 & speed != 0)
boxplot(
  speed ~ gate,
  data = filtered_results ,
  main = "Correlation of speed with inrun length in Garmisch",
  xlab = "Gate",
  ylab = "Speed (km/h)",
  varwidth = TRUE
)
