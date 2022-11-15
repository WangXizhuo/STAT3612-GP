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

# aggr_dict = {'glucose': 'None',
#  'hematocrit': 'None',
#  'sodium': 'None',
#  'creatinine': 'Max',
#  'potassium': 'None',
#  'blood urea nitrogen': 'Max',
#  'oxygen saturation': 'Min',
#  'hemoglobin': 'None',
#  'platelets': 'None',
#  'chloride': 'None',
#  'bicarbonate': 'Mean',
#  'white blood cell count': 'Mean',
#  'diastolic blood pressure': 'Max',
#  'heart rate': 'Mean',
#  'systolic blood pressure': 'Max',
#  'mean blood pressure': 'None',
#  'respiratory rate': 'None',
#  'red blood cell count': 'None',
#  'mean corpuscular hemoglobin concentration': 'Min',
#  'mean corpuscular hemoglobin': 'None',
#  'mean corpuscular volume': 'None',
#  'anion gap': 'Max',
#  'temperature': 'None',
#  'magnesium': 'Mean',
#  'prothrombin time inr': 'None',
#  'prothrombin time pt': 'Max',
#  'partial thromboplastin time': 'None',
#  'phosphate': 'None',
#  'calcium': 'None',
#  'phosphorous': 'None',
#  'ph': 'None',
#  'co2 (etco2, pco2, etc.)': 'None',
#  'partial pressure of carbon dioxide': 'None',
#  'weight': 'None',
#  'lactate': 'Mean',
#  'glascow coma scale total': 'Min',
#  'co2': 'None',
#  'neutrophils': 'None',
#  'lymphocytes': 'None',
#  'monocytes': 'Mean',
#  'calcium ionized': 'None',
#  'positive end-expiratory pressure set': 'None',
#  'tidal volume observed': 'None',
#  'ph urine': 'None',
#  'alanine aminotransferase': 'None',
#  'asparate aminotransferase': 'None',
#  'bilirubin': 'None',
#  'peak inspiratory pressure': 'Max',
#  'potassium serum': 'None',
#  'lactic acid': 'None',
#  'alkaline phosphate': 'None',
#  'respiratory rate set': 'None',
#  'tidal volume set': 'None',
#  'plateau pressure': 'None',
#  'basophils': 'None',
#  'albumin': 'Mean',
#  'partial pressure of oxygen': 'None',
#  'tidal volume spontaneous': 'None',
#  'central venous pressure': 'None',
#  'fraction inspired oxygen set': 'None',
#  'troponin-t': 'None',
#  'lactate dehydrogenase': 'None',
#  'fibrinogen': 'None',
#  'positive end-expiratory pressure': 'Mean',
#  'fraction inspired oxygen': 'None',
#  'pulmonary artery pressure systolic': 'None',
#  'height': 'None',
#  'creatinine urine': 'None',
#  'cardiac index': 'None',
#  'systemic vascular resistance': 'None',
#  'cardiac output thermodilution': 'None',
#  'red blood cell count urine': 'Max',
#  'white blood cell count urine': 'None',
#  'cholesterol': 'None',
#  'cholesterol hdl': 'None',
#  'cardiac output fick': 'None',
#  'cholesterol ldl': 'None',
#  'pulmonary artery pressure mean': 'None',
#  'chloride urine': 'None',
#  'lymphocytes atypical': 'None',
#  'pulmonary capillary wedge pressure': 'None',
#  'troponin-i': 'None',
#  'total protein urine': 'Max',
#  'venous pvo2': 'None',
#  'post void residual': 'Mean',
#  'red blood cell count csf': 'Max',
#  'monocytes csl': 'None',
#  'lymphocytes body fluid': 'None',
#  'lymphocytes ascites': 'None',
#  'red blood cell count ascites': 'None',
#  'eosinophils': 'None',
#  'total protein': 'None',
#  'lactate dehydrogenase pleural': 'None',
#  'lymphocytes pleural': 'None',
#  'red blood cell count pleural': 'None',
#  'calcium urine': 'None',
#  'albumin urine': 'None',
#  'albumin ascites': 'None',
#  'lymphocytes percent': 'None',
#  'albumin pleural': 'None',
#  'creatinine ascites': 'None',
#  'creatinine pleural': 'None',
#  'lymphocytes atypical csl': 'None',
#  'creatinine body fluid': 'None'}

# aggr_dict = {'glucose': 'Max',
#  'hematocrit': 'None',
#  'sodium': 'None',
#  'creatinine': 'Max', # try others
#  'potassium': 'Max',
#  'blood urea nitrogen': 'Max', # try others
#  'oxygen saturation': 'Min',
#  'hemoglobin': 'None',
#  'platelets': 'None',
#  'chloride': 'None',
#  'bicarbonate': 'Min',
#  'white blood cell count': 'Min', #need further check: min performs the best
#  'diastolic blood pressure': 'None',
#  'heart rate': 'Range', # max or range
#  'systolic blood pressure': 'Max', #try others or none
#  'mean blood pressure': 'None',
#  'respiratory rate': 'Range', # try others
#  'red blood cell count': 'None', # outliers looks the same after removing outliers
#  'mean corpuscular hemoglobin concentration': 'Min',
#  'mean corpuscular hemoglobin': 'None',
#  'mean corpuscular volume': 'None', # try others
#  'anion gap': 'Max',
#  'temperature': 'Range',
#  'magnesium': 'None',
#  'prothrombin time inr': 'None',
#  'prothrombin time pt': 'Max', # looks the same after removing outliers
#  'partial thromboplastin time': 'Max',
#  'phosphate': 'Mean', # Looks strange, try others
#  'calcium': 'None',
#  'phosphorous': 'None',
#  'ph': 'Range', # try max
#  'co2 (etco2, pco2, etc.)': 'Range', # try others
#  'partial pressure of carbon dioxide': 'Range',
#  'weight': 'None',
#  'lactate': 'None',
#  'glascow coma scale total': 'Max', #try all of them: using Max is the best
#  'co2': 'Min',
#  'neutrophils': 'Range', # all of them look similar try others
#  'lymphocytes': 'Min', # all of them look similar
#  'monocytes': 'Min',
#  'calcium ionized': 'None', # too many outliers
#  'positive end-expiratory pressure set': 'None',
#  'tidal volume observed': 'Range', # outlier in immoral dataset
#  'ph urine': 'Range',
#  'alanine aminotransferase': 'None', # too many outliers
#  'asparate aminotransferase': 'None',
#  'bilirubin': 'None',
#  'peak inspiratory pressure': 'max',
#  'potassium serum': 'None',
#  'lactic acid': 'Range',
#  'alkaline phosphate': 'None',
#  'respiratory rate set': 'Range', # may not be useful
#  'tidal volume set': 'Max',
#  'plateau pressure': 'Range',
#  'basophils': 'Range', # outlier in immoral dataset
#  'albumin': 'None',
#  'partial pressure of oxygen': 'Max',
#  'tidal volume spontaneous': 'Max', # too many outliers
#  'central venous pressure': 'None', 
#  'fraction inspired oxygen set': 'Max',
#  'troponin-t': 'Max',
#  'lactate dehydrogenase': 'None',
#  'fibrinogen': 'None',
#  'positive end-expiratory pressure': 'Max',
#  'fraction inspired oxygen': 'None',
#  'pulmonary artery pressure systolic': 'None',
#  'height': 'None',
#  'creatinine urine': 'None',
#  'cardiac index': 'None',
#  'systemic vascular resistance': 'None',
#  'cardiac output thermodilution': 'None',
#  'red blood cell count urine': 'max',
#  'white blood cell count urine': 'None',
#  'cholesterol': 'None',
#  'cholesterol hdl': 'None',
#  'cardiac output fick': 'None',
#  'cholesterol ldl': 'None',
#  'pulmonary artery pressure mean': 'None',
#  'chloride urine': 'None',
#  'lymphocytes atypical': 'None',
#  'pulmonary capillary wedge pressure': 'None',
#  'troponin-i': 'None',
#  'total protein urine': 'max',
#  'venous pvo2': 'None',
#  'post void residual': 'mean',
#  'red blood cell count csf': 'max',
#  'monocytes csl': 'None',
#  'lymphocytes body fluid': 'None',
#  'lymphocytes ascites': 'None',
#  'red blood cell count ascites': 'None',
#  'eosinophils': 'None',
#  'total protein': 'None',
#  'lactate dehydrogenase pleural': 'None',
#  'lymphocytes pleural': 'None',
#  'red blood cell count pleural': 'None',
#  'calcium urine': 'None',
#  'albumin urine': 'None',
#  'albumin ascites': 'None',
#  'lymphocytes percent': 'None',
#  'albumin pleural': 'None',
#  'creatinine ascites': 'None',
#  'creatinine pleural': 'None',
#  'lymphocytes atypical csl': 'None',
#  'creatinine body fluid': 'None'}

def aggregate_features(X_train):
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

