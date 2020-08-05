library(dplyr)
Restaurante = Restaurante[order(nrow(Restaurante):1),]
dayslist = Restaurante$DATA
Inmet2 <- Inmet[Inmet$DATA %in% dayslist,]
Inmet3 <- Inmet2[Inmet2$HORA == "12:00",]
Restaurante2 = merge(Restaurante,Inmet3)
write.csv2(Restaurante2,"./experimentos_monografia/input/Todos.csv", row.names = FALSE)