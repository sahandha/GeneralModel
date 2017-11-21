#! python3
import pandas as pd
import numpy as np

class DataStore:
    def __init__(self, Model, datafolder='~/Research/GeneralModel/Data', **params):
        self._Model = Model
        self._DataFolder = datafolder

    def SaveState(self, filename, columns='Automatic'):
        time  = self._Model._Time
        state = self._Model._XX
        dim   = state.shape[1]
        if columns=='Automatic':
            columns = ['x_{}'.format(i+1) for i in range(dim)]
        columns = columns
        data = pd.DataFrame(state, index=time, columns=columns)
        data.to_csv(self._DataFolder+filename+'.csv',index_label="Time")

    def LoadData(self, filename):
        df = pd.read_csv(self._DataFolder + filename + '.csv')
        return df
