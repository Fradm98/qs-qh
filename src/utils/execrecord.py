from utils.hexec import backends_objs_to_names
import pandas as pd
import numpy as np
import os

class ExecutionRecord:
    columns = ["J", "h", "lamb", "g", "final_time", "dt", "steps", "first_qb_or_all", "device"]
    def __init__(self, filepath):
        self.filepath = filepath
        if os.path.exists(filepath):
            self.df = pd.read_excel(filepath)
        else:
            self.df = pd.DataFrame(columns=self.columns)
            pd.to_excel(filepath, index=False)

    def save(self):
        self.df.to_excel(self.filepath, index=False)
    
    def add_record(self, J, h, lamb, g, final_time, dt, steps, first_qb_or_all, device):
        new_row = [np.array(J).tolist(), np.array(h).tolist(), np.array(lamb).tolist(), float(g), float(final_time), float(dt), int(steps), first_qb_or_all, backends_objs_to_names(device)]
        self.df.loc[-1] = new_row
        self.df.index = self.df.index + 1
        self.df = self.df.sort_index()
        self.save()

    def add_separator(self, lines=3):
        for i in range(lines):
            self.df.loc[-1] = [None]*self.columns
            self.df.index = self.df.index + 1
            self.df = self.df.sort_index()
            self.save()
        self.save()