import pandas as pd
import numpy as np


def CalcIV(Xvar, Yvar):
    N_0 = np.sum(Yvar == 0)
    N_1 = np.sum(Yvar == 1)
    N_0_group = np.zeros(np.unique(Xvar).shape)
    N_1_group = np.zeros(np.unique(Xvar).shape)
    for i in range(len(np.unique(Xvar))):
        N_0_group[i] = Yvar[(Xvar == np.unique(Xvar)[i]) & (Yvar == 0)].count()
        N_1_group[i] = Yvar[(Xvar == np.unique(Xvar)[i]) & (Yvar == 1)].count()
    iv = np.sum((N_0_group / N_0 - N_1_group / N_1) *
                np.log((N_0_group / N_0) / (N_1_group / N_1)))
    return iv


def caliv_batch(df, Kvar, Yvar):
    df_Xvar = df.drop([Kvar, Yvar], axis=1)
    ivlist = []
    for col in df_Xvar.columns:
        iv = CalcIV(df[col], df[Yvar])
        ivlist.append(iv)
    names = list(df_Xvar.columns)
    iv_df = pd.DataFrame({'Var': names, 'Iv': ivlist}, columns=['Var', 'Iv'])

    return iv_df
