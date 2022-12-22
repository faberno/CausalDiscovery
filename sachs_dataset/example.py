import pandas as pd

a = pd.read_csv("adjmat.csv", index_col='source')
b = pd.read_csv("sachs.data.txt", sep="\t")