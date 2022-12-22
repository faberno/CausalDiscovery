library(bnlearn)

sachs = read.table("sachs_dataset/sachs.data.txt", header = TRUE)
head(sachs)

dag.iamb = inter.iamb(sachs, test = "cor")

sachs.modelstring <- paste("[PKC][PKA|PKC][Raf|PKC:PKA][Mek|PKC:PKA:Raf][Erk|Mek:PKA][Akt|Erk:PKA][P38|PKC:PKA][Jnk|PKC:PKA][Plcg][PIP3|Plcg][PIP2|Plcg:PIP3]", sep = "")
dag.sachs <- model2network(sachs.modelstring)

graphviz.plot(dag.sachs, layout = "dot")