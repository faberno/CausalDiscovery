import sys
import bnlearn as bn
import os
import urllib
import gzip
import shutil

# ------config-------------------------------
name = 'insurance'
N = 1000
save_name = "insurance1"
# -------------------------------------------


name = name.lower()
cache = "network_files"
if not os.path.exists(cache):
    os.makedirs(cache)


save_path = os.path.join("artificial_datasets", save_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)


if os.path.exists(path := os.path.join(cache, f"{name}.bif")):
    model = bn.import_DAG(path)
else:
    model = bn.import_DAG(name, verbose=0)
    if len(model) == 0:
        BNREPO = "https://www.bnlearn.com/bnrepository/"
        ftpstream = urllib.request.urlopen(f"{BNREPO}{name}/{name}.bif.gz")
        with gzip.open(ftpstream) as f_in:
            with open(os.path.join(cache, f'{name}.bif'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        model = bn.import_DAG(path)

if model is None or len(model) == 0:
    sys.exit("Could not laod network.")

df = bn.sampling(model, n=1000, methodtype='bayes')
df_onehot = bn.df2onehot(df)[0]

model['adjmat'].to_csv(os.path.join(save_path, "graph.csv"))
df.to_csv(os.path.join(save_path, "data.csv"))
df_onehot.to_csv(os.path.join(save_path, "data_onehot.csv"))