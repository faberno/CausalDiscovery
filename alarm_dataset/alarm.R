library(bnlearn)

load("alarm_dataset/alarm.rda")

arc <- arcs(bn)

adj <- amat(bn)

write.table(adj,file="alarm_dataset/graph.csv", sep = ",")