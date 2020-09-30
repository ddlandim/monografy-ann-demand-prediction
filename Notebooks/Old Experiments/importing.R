library(readr)
Inmet <- read_delim("experimentos_monografia/input/Inmet.csv", 
                    ";", escape_double = FALSE, col_types = cols(DATA = col_date(format = "%d/%m/%Y"), 
                                                                 HORA = col_character()), locale = locale(decimal_mark = ",", 
                                                                                                          grouping_mark = "."), trim_ws = TRUE)
View(Inmet)

Restaurante <- read_delim("experimentos_monografia/input/Restaurante.csv", 
                          ";", escape_double = FALSE, col_types = cols(DATA = col_date(format = "%d/%m/%Y")), 
                          locale = locale(decimal_mark = ",", grouping_mark = "."), 
                          trim_ws = TRUE)
View(Restaurante)
