from pathlib import Path

import sqlite3
import json

import pandas as pd
import numpy as np


targets = {}
p = Path.home() / "minipdb" / "programs"
targets = {d.name: 0.0 for d in p.iterdir() if d.is_dir()}

db = sqlite3.connect(Path.home() / ".minipdb" / "minipdb.sqlite")

for k in targets.keys():
    df = pd.read_sql_query(f"SELECT * FROM '{k}'", db)
    targets[k] = np.mean(df.iloc[:, 7:], axis = 0).values.copy()

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open("targets.json", 'w') as f:
    json.dump(json.dumps(targets, cls=NumpyEncoder), f)

with open('targets.json') as f:
    data = json.loads(json.load(f))
