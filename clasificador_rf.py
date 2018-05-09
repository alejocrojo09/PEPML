# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 02:05:40 2018

@author: acroj
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 23:37:21 2018

@author: acroj
"""

##############################################################################################
# Librerias 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
import scikitplot as scplt

from propy import PseudoAAC as PAAC
from propy import AAComposition as AC
from propy import CTD 
from propy import Autocorrelation as auto
from propy import QuasiSequenceOrder as qua
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.cross_validation import train_test_split as train
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, precision_score, matthews_corrcoef, accuracy_score, recall_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, SelectFpr, SelectPercentile

from propy import PseudoAAC as PAAC
from propy import AAComposition as AC
from propy import CTD 
from propy import Autocorrelation as auto
from propy import QuasiSequenceOrder as qua
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

from Bio import SeqIO

##############################################################################################
# Secuencias
sec_pos = []
sec_neg = []
for record in SeqIO.parse('AMP.fasta', 'fasta'):
    sec_pos.append(str(record.seq))
for record in SeqIO.parse('M_model_train_nonAMP_sequence.fasta', 'fasta'):
    sec_neg.append(str(record.seq))
seq = sec_pos + sec_neg

##############################################################################################
# Datos
var = propi()
X = var.iloc[:,:(len(var.columns))].values
Y = [1]*(len(sec_pos)) + [0]*(len(sec_neg))
Y = np.array(Y)
##############################################################################################
# Clasificación
X_train, X_test, Y_train, Y_test = train(X,Y, test_size = 0.3, random_state = 0)
clas_rf = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
rf = make_pipeline(StandardScaler(), SelectKBest(score_func = f_classif, k = 100), clas_rf)
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
##############################################################################################
# Reporte
print(classification_report(Y_test, rf_pred))
##############################################################################################
# Propiedades Fisicoquímicas
def propi():
    des_fis = GlobalDescriptor(seq)
    des_fis.calculate_all()
    prop_fis = des_fis.descriptor
    
    # Composición de aminoácidos
    amino_comp = map(AC.CalculateAAComposition,seq) # AA
    dipep_comp = map(AC.CalculateDipeptideComposition,seq) # Dipéptidos
    
    # Autocorrelación
    moreau_auto = map(auto.CalculateNormalizedMoreauBrotoAutoTotal,seq) # Moreau 
    moran_auto = map(auto.CalculateMoranAutoTotal,seq) # Moran
    geary_auto = map(auto.CalculateGearyAutoTotal,seq) # Geary
    
    # Composition - Distribution - Transition
    ctd = map(CTD.CalculateCTD,seq)
    
    # QuasiSequence
    sqa = map(lambda p: qua.GetQuasiSequenceOrder(p, maxlag = 5, weight = 0.1), seq)
    secq = map(lambda p: qua.GetSequenceOrderCouplingNumber(p, d = 1), seq)
    
    amino_comp =  pd.DataFrame.from_dict(amino_comp)
    amino_comp.reset_index(drop = True, inplace = True)
    dipep_comp =  pd.DataFrame.from_dict(dipep_comp)
    dipep_comp.reset_index(drop = True, inplace = True)
    
    moreau_auto = pd.DataFrame.from_dict(moreau_auto)
    moreau_auto.reset_index(drop = True, inplace = True)
    moran_auto = pd.DataFrame.from_dict(moran_auto)
    moran_auto.reset_index(drop = True, inplace = True)
    geary_auto = pd.DataFrame.from_dict(geary_auto)
    geary_auto.reset_index(drop = True, inplace = True)
    
    ctd = pd.DataFrame.from_dict(ctd)
    ctd.reset_index(drop = True, inplace = True)
    
    # PseudoAAC - Tipo I
    Hydrophobicity = PAAC._Hydrophobicity
    hydrophilicity = PAAC._hydrophilicity
    residuemass = PAAC._residuemass
    pK1 = PAAC._pK1
    pK2 = PAAC._pK2
    pI = PAAC._pI
    clasI_pse = map(lambda p: PAAC.GetPseudoAAC(p, lamda = 3, weight = 0.7, AAP = [Hydrophobicity, hydrophilicity,
                                                                                    residuemass, pK1, pK2, pI]),seq)
    clasI_pse = pd.DataFrame.from_dict(clasI_pse)
    clasI_pse.reset_index(drop = True, inplace = True)
    
    sqa = pd.DataFrame.from_dict(sqa)
    sqa.reset_index(drop = True, inplace = True)
    secq = pd.DataFrame.from_dict(secq)
    secq.reset_index(drop = True, inplace = True)
    
    prop_fis = pd.DataFrame(prop_fis)
    prop_fis.columns = ['Longitud','MW','Carga','DensCarga','pIso','InestInd','Aroma','Alifa','Boman','HidroRa']
    
    var = pd.concat([amino_comp,
               dipep_comp,
               moreau_auto,
               moran_auto,
               ctd,
               clasI_pse,
               sqa,
               secq,
               geary_auto,
               prop_fis], axis = 1)
    return var


