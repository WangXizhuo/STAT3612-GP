import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

aggr_dict = {'glucose': 'mean',
 'hematocrit': 'max',
 'sodium': 'mean',
 'creatinine': 'max',
 'potassium': 'mean',
 'blood urea nitrogen': 'max',
 'oxygen saturation': 'min',
 'hemoglobin': 'min',
 'platelets': 'max',
 'chloride': 'mean',
 'bicarbonate': 'mean',
 'white blood cell count': 'mean',
 'diastolic blood pressure': 'max',
 'heart rate': 'mean',
 'systolic blood pressure': 'max',
 'mean blood pressure': 'mean',
 'respiratory rate': 'min',
 'red blood cell count': 'min',
 'mean corpuscular hemoglobin concentration': 'min',
 'mean corpuscular hemoglobin': 'None',
 'mean corpuscular volume': 'min',
 'anion gap': 'max',
 'temperature': 'mean',
 'magnesium': 'mean',
 'prothrombin time inr': 'None',
 'prothrombin time pt': 'max',
 'partial thromboplastin time': 'max',
 'phosphate': 'mean',
 'calcium': 'mean',
 'phosphorous': 'mean',
 'ph': 'mean',
 'co2 (etco2, pco2, etc.)': 'mean',
 'partial pressure of carbon dioxide': 'max',
 'weight': 'mean',
 'lactate': 'mean',
 'glascow coma scale total': 'min',
 'co2': 'mean',
 'neutrophils': 'mean',
 'lymphocytes': 'mean',
 'monocytes': 'mean',
 'calcium ionized': 'mean',
 'positive end-expiratory pressure set': 'mean',
 'tidal volume observed': 'min',
 'ph urine': 'mean',
 'alanine aminotransferase': 'max',
 'asparate aminotransferase': 'max',
 'bilirubin': 'mean',
 'peak inspiratory pressure': 'max',
 'potassium serum': 'max',
 'lactic acid': 'min',
 'alkaline phosphate': 'max',
 'respiratory rate set': 'max',
 'tidal volume set': 'mean',
 'plateau pressure': 'max',
 'basophils': 'mean',
 'albumin': 'mean',
 'partial pressure of oxygen': 'mean',
 'tidal volume spontaneous': 'mean',
 'central venous pressure': 'mean',
 'fraction inspired oxygen set': 'mean',
 'troponin-t': 'mean',
 'lactate dehydrogenase': 'mean',
 'fibrinogen': 'mean',
 'positive end-expiratory pressure': 'mean',
 'fraction inspired oxygen': 'mean',
 'pulmonary artery pressure systolic': 'max',
 'height': 'mean',
 'creatinine urine': 'max',
 'cardiac index': 'min',
 'systemic vascular resistance': 'max',
 'cardiac output thermodilution': 'mean',
 'red blood cell count urine': 'max',
 'white blood cell count urine': 'max',
 'cholesterol': 'max',
 'cholesterol hdl': 'min',
 'cardiac output fick': 'min',
 'cholesterol ldl': 'max',
 'pulmonary artery pressure mean': 'mean',
 'chloride urine': 'mean',
 'lymphocytes atypical': 'max',
 'pulmonary capillary wedge pressure': 'max',
 'troponin-i': 'max',
 'total protein urine': 'max',
 'venous pvo2': 'mean',
 'post void residual': 'mean',
 'red blood cell count csf': 'max',
 'monocytes csl': 'max',
 'lymphocytes body fluid': 'mean',
 'lymphocytes ascites': 'mean',
 'red blood cell count ascites': 'max',
 'eosinophils': 'mean',
 'total protein': 'mean',
 'lactate dehydrogenase pleural': 'max',
 'lymphocytes pleural': 'mean',
 'red blood cell count pleural': 'max',
 'calcium urine': 'mean',
 'albumin urine': 'max',
 'albumin ascites': 'mean',
 'lymphocytes percent': 'mean',
 'albumin pleural': 'mean',
 'creatinine ascites': 'max',
 'creatinine pleural': 'mean',
 'lymphocytes atypical csl': 'mean',
 'creatinine body fluid': 'max'}



def aggregate_features(X_train,mask=False,time=False):
    df = pd.DataFrame()
    for feature in aggr_dict:
        if aggr_dict[feature] == 'mean':
            df[f'{feature}_mean'] = X_train[feature,'mean'].mean(axis=1)
        elif aggr_dict[feature] == 'max':
            df[f'{feature}_max'] = X_train[feature,'mean'].max(axis=1)
        elif aggr_dict[feature] == 'min':
            df[f'{feature}_min'] = X_train[feature,'mean'].min(axis=1)
        else:
            pass
        if mask == True:
           df[f'{feature}_mask'] = X_train[feature,'mask'].mean(axis=1) 
        if time == True:
            df[f'{feature}_time'] = X_train[feature,'time_since_measured'].mean(axis=1) 
    # df.replace(to_replace=0.0, value=np.nan, inplace=True)
    # df.fillna(df.mean(), inplace=True)
    return df

def auc_plot(model, X_test, Y_test):
    val_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = roc_curve(Y_test, val_pred)
    auc_train = auc(fpr, tpr)
    plt.figure(figsize=(10,8))
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr, tpr, "r", linewidth=3)
    plt.grid()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.text(0.15, 0.9, "AUC = " + str (round (auc_train, 4)))
    plt.show()

if __name__ == '__main__':
    X_train = pd.read_csv("X_valid.csv", index_col=[0], header=[0, 1, 2])
    df = aggregate_features(X_train)
    df.to_csv('aggregated_features_valid.csv', index=False)
    print(df.head())

