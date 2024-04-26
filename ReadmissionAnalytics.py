import pandas as pd
import numpy as np

datain_dir = "~/Desktop/python/Data Mining Project/ReadmissionAnalytics/"
df_raw = pd.read_csv(datain_dir + "diabetic_data.csv")

df_raw = df_raw.replace('?', np.nan)

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
# Remove Speciality column as we already converted each value to true false columns
col_data.loc[['medical_specialty'],['var_type']] = 'drop'
# Remove diagnosis columns as we already convertd each value to true false columns
col_data.loc[['diag_1'],['var_type']] = 'drop'
col_data.loc[['diag_2'],['var_type']] = 'drop'
col_data.loc[['diag_3'],['var_type']] = 'drop'
# Numeric continous variables
col_data.loc[['num_procedures'],['var_type']] = 'cont'
col_data.loc[['age'],['var_type']] = 'cont'
col_data.loc[['time_in_hospital'],['var_type']] = 'cont'
col_data.loc[['number_emergency'],['var_type']] = 'cont'
col_data.loc[['number_inpatient'],['var_type']] = 'cont'
col_data.loc[['number_diagnoses'],['var_type']] = 'cont'
col_data.loc[['number_outpatient'],['var_type']] = 'cont'
col_data.loc[['num_medications'],['var_type']] = 'cont'
col_data.loc[['num_lab_procedures'],['var_type']] = 'cont'
# Final column for result
col_data.loc[['readmitted'],['var_type']] = 'class'
# All others categorical variables, will be converted to binary
col_data.loc[col_data.var_type.isnull(),'var_type'] = 'cat'

# Speciality features converted to binary
spec_counts = df_raw.medical_specialty.value_counts()

spec_thresh = spec_counts.shape[0] # No of specialities (top N) to be modified based on results
for (spec, count) in spec_counts.head(spec_thresh).iteritems():
    new_col = 'spec_' + str(spec)
    df_raw[new_col] = (str(df_raw.medical_specialty) == str(spec))
    
# Diagnosis features converted to binary
diag_counts = df_raw.diag_1.value_counts().add(df_raw.diag_2.value_counts(),fill_value=0).add(df_raw.diag_3.value_counts(),fill_value=0)
diag_counts.sort_values(ascending=False)

diag_thresh = diag_counts.shape[0] # No of diagnoses (top N) to be modified based on results
for (icd9, count) in diag_counts.head(diag_thresh).iteritems():
    new_col = 'diag_' + str(icd9)
    df_raw[new_col] = (str(df_raw.diag_1) == str(icd9))|(str(df_raw.diag_2) == str(icd9))|(str(df_raw.diag_3) == str(icd9))
    
# Make a new copy of data
df_raw2 = pd.DataFrame(df_raw, copy=True)

df_raw2['age'] = df_raw2.age.str.extract('(\d+)-\d+')

to_drop = col_data[col_data.var_type.str.contains('drop')].index
df_raw2.drop(to_drop, axis=1, inplace=True)

#break out categorical variables into binaries
cat_cols = col_data[col_data.var_type.str.contains('cat')].index
df_raw2 = pd.get_dummies(df_raw2, columns=cat_cols)

df = pd.DataFrame(df_raw2)
df.shape
