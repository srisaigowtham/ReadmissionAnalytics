import pandas as pd
import numpy as np

datain_dir = "~/Desktop/python/Data Mining Project/ReadmissionAnalytics/"
df_raw = pd.read_csv(datain_dir + "diabetic_data.csv")

df_raw = df_raw.replace('?', np.nan)

# Consider only one encounter per patient. Our goal is to avoid future visits, so we will avoid future encounters for patients in our data
df_raw.drop_duplicates(subset=['patient_nbr'], inplace=True)

df_raw.race.fillna('Missing', inplace=True)

# Remove unknown gender encounters; very rare
df_raw.drop(df_raw[df_raw.gender == 'Unknown/Invalid'].index, inplace=True)

# Modifying age variable to upper limit
df_raw['age'] = df_raw.age.str.extract('\d+-(\d+)')
df_raw['age'] = map(int, df_raw['age'])

# Remove encounters that resulted in hospice care or death to avoid skewing results
df_raw.drop(df_raw[df_raw.discharge_disposition_id == 11].index, inplace=True)
df_raw.drop(df_raw[df_raw.discharge_disposition_id == 13].index, inplace=True)
df_raw.drop(df_raw[df_raw.discharge_disposition_id == 14].index, inplace=True)
df_raw.drop(df_raw[df_raw.discharge_disposition_id == 19].index, inplace=True)
df_raw.drop(df_raw[df_raw.discharge_disposition_id == 20].index, inplace=True)
df_raw.drop(df_raw[df_raw.discharge_disposition_id == 21].index, inplace=True)

df_raw.medical_specialty.fillna('Missing', inplace=True)
# Medical speciality reducing grouping variable values with less than 3.5% values to Otherwise
df_raw.loc[((df_raw.medical_specialty != 'Missing') & (df_raw.medical_specialty != 'InternalMedicine') & (df_raw.medical_specialty != 'Family/GeneralPractice') & (df_raw.medical_specialty != 'Emergency/Trauma') & (df_raw.medical_specialty != 'Cardiology')),'medical_specialty'] = 'Otherwise'

# Discharged to home or discharged to other facilities with medical care
df_raw.loc[(df_raw.discharge_disposition_id == 1),'discharge_disposition_id'] = 'Discharged to home'
df_raw.loc[(df_raw.discharge_disposition_id != 'Discharged to home'),'discharge_disposition_id'] = 'Otherwise'

# Admission source: Emergency or Physician/Clinic referral or Otherwise
df_raw.loc[(df_raw.admission_source_id == 7),'admission_source_id'] = 'Admitted from emergency room'
df_raw.loc[(df_raw.admission_source_id == 1),'admission_source_id'] = 'Admitted because of physician/clinic referral'
df_raw.loc[(df_raw.admission_source_id == 2),'admission_source_id'] = 'Admitted because of physician/clinic referral'
df_raw.loc[((df_raw.admission_source_id != 'Admitted from emergency room') & (df_raw.admission_source_id != 'Admitted because of physician/clinic referral')),'admission_source_id'] = 'Otherwise'

# diag_1 represents primary diagnosis, this is the preliminary analysis of a patient. Initially the patient maybe identified with other diseases but later in diag_2 or diag_3, the disease may be determined as diabetes. Our goal is to predict the readmission after primary diagnosis, so diag_2 and diag_3 are not relevant to us.

df_raw.loc[(((df_raw.diag_1>=str(390)) & (df_raw.diag_1<=str(459))) | (df_raw.diag_1==str(785))), 'PrimaryDiagnosis'] = 'Circulatory'
df_raw.loc[(((df_raw.diag_1>=str(460)) & (df_raw.diag_1<=str(519))) | (df_raw.diag_1==str(786))), 'PrimaryDiagnosis'] = 'Respiratory'
df_raw.loc[(((df_raw.diag_1>=str(520)) & (df_raw.diag_1<=str(579))) | (df_raw.diag_1==str(787))), 'PrimaryDiagnosis'] = 'Digestive'
df_raw.loc[(df_raw.diag_1.str.contains('250', na=False)), 'PrimaryDiagnosis'] = 'Diabetes'
df_raw.loc[((df_raw.diag_1>=str(800)) & (df_raw.diag_1<=str(999))), 'PrimaryDiagnosis'] = 'Injury'
df_raw.loc[((df_raw.diag_1>=str(710)) & (df_raw.diag_1<=str(739))), 'PrimaryDiagnosis'] = 'Musculoskeletal'
df_raw.loc[(((df_raw.diag_1>=str(580)) & (df_raw.diag_1<=str(629))) | (df_raw.diag_1==str(788))), 'PrimaryDiagnosis'] = 'Genitourinary'
df_raw.loc[((df_raw.diag_1>=str(140)) & (df_raw.diag_1<=str(239))), 'PrimaryDiagnosis'] = 'Neoplasms'
df_raw.PrimaryDiagnosis.fillna('Other', inplace=True)

# In order to make the problem binary, Readmitted column is modified as Yes for <30 and No for >30
df_raw.loc[(df_raw.readmitted == '<30'),'readmitted'] = 1
df_raw.loc[(df_raw.readmitted == '>30'),'readmitted'] = 0
df_raw.loc[(df_raw.readmitted == 'NO'),'readmitted'] = 0

# Column details
col_data = df_raw.apply(lambda s: set(s.unique()), axis=0).to_frame('uni_val')
col_data['nan_rat'] = df_raw.isnull().sum(axis=0)/len(df_raw)
col_data['n_uni_vals'] = col_data.uni_val.apply(len)
col_data['uni_vals_str'] = col_data.uni_val.astype(str)
col_data = col_data.drop('uni_val', axis=1)

# Only 1 unique value present for all rows in column
col_data.loc[col_data['n_uni_vals']==1,'var_type'] = 'drop'
# Doesn't give any useful information
col_data.loc[['encounter_id'],['var_type']] = 'drop'
col_data.loc[['patient_nbr'],['var_type']] = 'drop'
col_data.loc[['payer_code'],['var_type']] = 'drop'
# Remove weight column as most of the values are empty
col_data.loc[['weight'],['var_type']] = 'drop'
# Remove diagnosis columns as we already classified each value appropriately
col_data.loc[['diag_1'],['var_type']] = 'drop'
col_data.loc[['diag_2'],['var_type']] = 'drop'
col_data.loc[['diag_3'],['var_type']] = 'drop'

col_data.loc[['num_procedures'],['var_type']] = 'cont'
col_data.loc[['age'],['var_type']] = 'cont'
col_data.loc[['time_in_hospital'],['var_type']] = 'cont'
col_data.loc[['number_emergency'],['var_type']] = 'cont'
col_data.loc[['number_inpatient'],['var_type']] = 'cont'
col_data.loc[['number_diagnoses'],['var_type']] = 'cont'
col_data.loc[['number_outpatient'],['var_type']] = 'cont'
col_data.loc[['num_medications'],['var_type']] = 'cont'
col_data.loc[['num_lab_procedures'],['var_type']] = 'cont'

col_data.loc[['readmitted'],['var_type']] = 'class'

col_data.loc[col_data.var_type.isnull(),'var_type'] = 'cat'

# Create a copy of data
df_raw2 = pd.DataFrame(df_raw, copy=True)

#Drop patient id and encounter id columns as they are not relevant to us
#df_raw.drop(['encounter_id','patient_nbr'], axis=1, inplace=True)

# Remove weight column as most of the values are empty
#df_raw.drop(['weight'], axis=1, inplace=True)

# Doesn't give any useful information
#df_raw.drop(['payer_code'], axis=1, inplace=True)

# Drop diag_1, diag_2, diag_3 columns as they are no longer needed
#df_raw.drop(['diag_1','diag_2','diag_3'], axis=1, inplace=True)

to_drop = col_data[col_data.var_type.str.contains('drop')].index
df_raw.drop(to_drop, axis=1, inplace=True)
df_raw2.drop(to_drop, axis=1, inplace=True)

# Create binarized columns of variables
cat_cols = col_data[col_data.var_type.str.contains('cat')].index
df_raw2 = pd.get_dummies(df_raw2, columns=cat_cols)

# Final dataframe
#df = pd.DataFrame(df_raw, copy=True) #Use this for categorical columns
#df = pd.DataFrame(df_raw2, copy=True) #Use this for all binarized values of categorical columns
df = df_raw.filter(['race','gender','age','discharge_disposition_id','admission_source_id','time_in_hospital','medical_specialty','PrimaryDiagnosis','A1Cresult','readmitted'],axis=1)

# Create binarized columns of variables
cat_cols = ['race','gender','discharge_disposition_id','admission_source_id','medical_specialty','PrimaryDiagnosis','A1Cresult']
df = pd.get_dummies(df, columns=cat_cols)

#Moving class label to the end
classLabel = df.pop('readmitted')
df['readmitted'] = classLabel

X = df.loc[:, df.columns != 'readmitted']
y = df.loc[:, df.columns == 'readmitted']

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns

os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['readmitted'])

data_final_vars=df.columns.values.tolist()
y=['readmitted']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())

final_column_list = []
for index in np.where(rfe.support_ == True)[0]:
    final_column_list.append(os_data_X.columns[index])

X=os_data_X[final_column_list]
y=os_data_y['readmitted']

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()