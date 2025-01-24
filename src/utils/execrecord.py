from utils.hexec import backends_objs_to_names
from datetime import datetime
import pandas as pd
import numpy as np
import os

class ExecutionRecord:
    columns = ["J", "h", "lamb", "g", "final_time", "dt", "steps", "px", "py", "first_qb_or_all", "device", "date", "shots", "string_qubits", "odr"]
    def __init__(self, filepath):
        self.filepath = filepath
        if os.path.exists(filepath):
            self.df = pd.read_excel(filepath)
        else:
            self.df = pd.DataFrame(columns=self.columns)
            self.df.to_excel(filepath, index=False)

    def save(self):
        self.df.to_excel(self.filepath, index=False)
    
    def add_record(self, J, h, lamb, g, final_time, dt, steps, lattice, first_qb_or_all, device, shots, string_qubits, odr):
        new_row = [
            np.array(J).tolist(), 
            np.array(h).tolist(), 
            np.array(lamb).tolist(), 
            float(g), 
            float(final_time), 
            float(dt), 
            int(steps), 
            int(lattice.plaquettes_width), 
            int(lattice.plaquettes_height), 
            str(first_qb_or_all),
            backends_objs_to_names(device), 
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 
            shots,
            string_qubits,
            str(odr)
        ]
        self.df.loc[-1] = new_row
        self.df.index = self.df.index + 1
        self.df = self.df.sort_index()
        self.save()

    def add_separator(self, lines=3):
        for i in range(lines):
            self.df.loc[-1] = [None]*len(self.columns)
            self.df.index = self.df.index + 1
            self.df = self.df.sort_index()
        self.save()