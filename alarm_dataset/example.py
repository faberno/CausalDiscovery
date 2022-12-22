import pandas as pd

a = pd.read_csv("graph.csv")

mapping = {
    'HISTORY': 'HIST',
    'CVP': 'CVP',
    'PCWP': 'PCWP',
    'HYPOVOLEMIA': 'HYP',
    'LVEDVOLUME': 'LVV',
    'LVFAILURE': 'LVF',
    'STROKEVOLUME': 'STKV',
    'ERRLOWOUTPUT': 'ERLO',
    'HRBP': 'HRBP',
    'HREKG': 'HREK',
    'ERRCAUTER': 'ERCA',
    'HRSAT': 'HRSA',
    'INSUFFANESTH': 'ANES',
    'ANAPHYLAXIS': 'APL',
    'TPR': 'TPR',
    'EXPCO2': 'ECO2',
    'KINKEDTUBE': 'KINK',
    'MINVOL': 'MINV',
    'FIO2': 'FIO2',
    'PVSAT': 'PVS',
    'SAO2': 'SAO2',
    'PAP': 'PAP',
    'PULMEMBOLUS': 'PMB',
    'SHUNT': 'SHNT',
    'INTUBATION': 'INT',
    'PRESS': 'PRSS',
    'DISCONNECT': 'DISC',
    'MINVOLSET': 'MVS',
    'VENTMACH': 'VMCH',
    'VENTTUBE': 'VTUB',
    'VENTLUNG': 'VLNG',
    'VENTALV': 'VALV',
    'ARTCO2': 'ACO2',
    'CATECHOL': 'CCHL',
    'HR': 'HR',
    'CO': 'CO',
    'BP': 'BP'
}

print("")
