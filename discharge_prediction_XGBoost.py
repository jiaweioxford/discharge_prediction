import pandas as pd
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import random

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score, average_precision_score
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import IsotonicRegression
from imblearn.pipeline import Pipeline
import xgboost as xgb
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval


def custom_classification_report(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    b_accuracy = balanced_accuracy_score(y_true, y_pred)
    acc = (tp+tn)/(tp+tn+fp+fn)
    sen = (tp)/(tp+fn)
    sp = (tn)/(tn+fp)
    ppv = (tp)/(tp+fp)
    npv = (tn)/(tn+fn)
    f1 = 2*(sen*ppv)/(sen+ppv)
    fpr = (fp)/(fp+tn)
    tpr = (tp)/(tp+fn)
    return (    '2X2 confusion matrix:', ['TP', tp, 'FP', fp, 'FN', fn, 'TN', tn],
                'Accuracy:', round(acc, 3),
                'Balanced accuracy:', round(b_accuracy, 3),
                'Sensitivity/Recall:', round(sen, 3),
                'Specificity:', round(sp, 3),
                'PPV/Precision:', round(ppv, 3),
                'NPV:', round(npv, 3),
                'F1-score:', round(f1, 3),
                'False positive rate:', round(fpr, 3),
                'True positive rate:', round(tpr, 3),
                'ROC AUC:', round(roc_auc_score(y_true, y_prob), 3),
                'AUPRC:', round(auc(recall, precision),3)
            )

def custom_classification_sum(y_true, y_pred, y_prob, X_test):
    y_true=y_true.reset_index(drop=True)
    n_bootstraps = 500
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = pd.DataFrame({'acc':[], 'bacc':[],'sen':[], 'sp':[], 'ppv':[], 'npv':[], 'f1':[], 'auroc':[], 'auprc':[], 'mae':[]})
    
    predicted_data=pd.DataFrame({'time':[], 'y':[], 'y_prob':[]})
    predicted_data['y'] = y_true.reset_index(drop=True)
    predicted_data['time'] = X_test.copy().reset_index(drop=True)['INDEX_DATE_index_dt_since_baseline'].apply(np.floor)
    predicted_data['y_prob'] = y_prob
    summary = predicted_data.groupby('time')[['y','y_prob']].sum()

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_prob), len(y_prob))
        tn, fp, fn, tp = confusion_matrix(y_true[indices], y_pred[indices]).ravel()
        precision, recall, _ = precision_recall_curve(y_true[indices], y_prob[indices])
        acc = (tp+tn)/(tp+tn+fp+fn)
        bacc = balanced_accuracy_score(y_true[indices], y_pred[indices])
        sen = (tp)/(tp+fn)
        sp = (tn)/(tn+fp)
        ppv = (tp)/(tp+fp)
        npv = (tn)/(tn+fn)
        f1 = 2*(sen*ppv)/(sen+ppv)

        auroc = round(roc_auc_score(y_true[indices], y_prob[indices]), 3)
        auprc = round(auc(recall, precision),3)

        sample = summary.sample(n=len(summary), replace=True)
        sample['diff']=abs(sample['y']-sample['y_prob'])
    
        mae = sample['diff'].median()/sample['y'].median()
    
        bootstrapped_scores.loc[i] = [acc, bacc, sen, sp, ppv, npv, f1, auroc, auprc, mae]
        


    metrics_df = pd.DataFrame({
    'Accuracy': bootstrapped_scores['acc'].mean(),
    'Accuracy-lci': bootstrapped_scores['acc'].quantile(0.025),
    'Accuracy-uci': bootstrapped_scores['acc'].quantile(0.975),
    'Balanced Accuracy': bootstrapped_scores['bacc'].mean(),
    'Balanced Accuracy-lci': bootstrapped_scores['bacc'].quantile(0.025),
    'Balanced Accuracy-uci': bootstrapped_scores['bacc'].quantile(0.975),
    'Sensitivity/Recall': bootstrapped_scores['sen'].mean(),
    'Sensitivity-lci': bootstrapped_scores['sen'].quantile(0.025),
    'Sensitivity-uci': bootstrapped_scores['sen'].quantile(0.975),
    'Specificity': bootstrapped_scores['sp'].mean(),
    'Specificity-lci': bootstrapped_scores['sp'].quantile(0.025),
    'Specificity-uci': bootstrapped_scores['sp'].quantile(0.975),
    'PPV/Precision': bootstrapped_scores['ppv'].mean(),
    'PPV/Precision-lci': bootstrapped_scores['ppv'].quantile(0.025),
    'PPV/Precision-uci': bootstrapped_scores['ppv'].quantile(0.975),
    'NPV': bootstrapped_scores['npv'].mean(),
    'NPV-lci': bootstrapped_scores['npv'].quantile(0.025),
    'NPV-uci': bootstrapped_scores['npv'].quantile(0.975),
    'F1-score': bootstrapped_scores['f1'].mean(),
    'F1-score-lci': bootstrapped_scores['f1'].quantile(0.025),
    'F1-score-uci': bootstrapped_scores['f1'].quantile(0.975),
    'ROC AUC': bootstrapped_scores['auroc'].mean(),
    'ROC AUC-lci': bootstrapped_scores['auroc'].quantile(0.025),
    'ROC AUC-uci': bootstrapped_scores['auroc'].quantile(0.975),
    'AUPRC': bootstrapped_scores['auprc'].mean(),
    'AUPRC-lci': bootstrapped_scores['auprc'].quantile(0.025),
    'AUPRC-uci': bootstrapped_scores['auprc'].quantile(0.975),
    'MAE': bootstrapped_scores['mae'].mean(),
    'MAE-lci': bootstrapped_scores['mae'].quantile(0.025),
    'MAE-uci': bootstrapped_scores['mae'].quantile(0.975)

}, index=[0])
    return metrics_df



### load data
e="elective"
t="12am"
df = pd.read_csv('~/data/extract_'+e+'_'+t+'.csv')

# generate outcome variable - 24 hr dishcarge
df['y'] = np.where(df['OUTCOME_future_los_this_adm_d'] < 1, 1, 0)

#train test split
df_train = df[(df['index_dt']<'2019-02-01') & (df['index_dt']>= "2017-02-01")].copy()
df_test = df[df['index_dt']>='2019-02-01'].copy()

id_train=df_train['id'].unique()
id_all=df['id'].unique()

id_train.shape[0];id_all.shape[0]
df_test = df[-df['id'].isin(id_train)]
df_train.shape[0];df_test.shape[0]

# remove columns present for info only, or as additional outcomes
info_columns = [c for c in df.columns if 'INFO' in c]
outcome_columns = [c for c in df.columns if 'OUTCOME' in c]
remove_other_columns = ['index_dt','pt','CURRENT_ADM_pt_classification','WARD_facility','MODEL_whole_days_since_adm','SHMI_LOS_shmi']
df_train.drop(columns=info_columns + outcome_columns + remove_other_columns, inplace=True)
df_test.drop(columns=info_columns + outcome_columns + remove_other_columns, inplace=True)

# set up data frames for modelling
X_train = df_train.drop(columns=['y'])
y_train = df_train['y']

X_test = df_test.drop(columns=['y'])
y_test = df_test['y']

# get categorical and continous columns
categorical_cols = ['INDEX_DATE_index_dt_weekday','INDEX_DATE_index_dt_month','DEMOGRAPHICS_sex',
 'DEMOGRAPHICS_ethnic_gp',
 'CURRENT_ADM_adm_weekday',
 'CURRENT_ADM_adm_month',
 'CURRENT_ADM_adm_source',
 'SPECIALTY_current_spec']

continuous_cols = [x for x in X_train.columns if x != categorical_cols]

cat_cols = X_train.select_dtypes(exclude="number").columns
num_cols = X_train.select_dtypes(include="number").columns

### preprocessing
categorical_pipeline = Pipeline(
    steps=[
        #("impute", SimpleImputer(strategy="most_frequent")),
        ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

numeric_pipeline = Pipeline(
    steps=[
        #("impute", SimpleImputer(strategy="median")),
        #("scale", StandardScaler()),
        ('preprocess', None)

    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, num_cols),
        ("categorical", categorical_pipeline, cat_cols),
    ]
)


X_train_processed = preprocessor.fit_transform(X_train)
feature_names = preprocessor.get_feature_names_out()
X_train_processed=pd.DataFrame(X_train_processed, columns=feature_names)
X_test_processed = preprocessor.transform(X_test)
X_test_processed=pd.DataFrame(X_test_processed, columns=feature_names)

#split into train and validation data
random.seed(42)
n = int(len(id_train) * 0.2)
id_val = random.sample(list(id_train), n)

X_val_processed=X_train_processed[X_train_processed['id'].isin(id_val)]
X_train_processed=X_train_processed[-X_train_processed['id'].isin(id_val)]
y_train=X_train_processed['y']
y_val=X_val_processed['y']
groups=X_train_processed['id']
X_train_processed = X_train_processed.drop(['id', 'y'], axis=1)
X_val_processed = X_val_processed.drop(['id','y'],axis=1)

X_train_processed.columns = X_train_processed.columns.str.replace('numeric__', '').str.replace('categorical__', '')
X_val_processed.columns = X_val_processed.columns.str.replace('numeric__', '').str.replace('categorical__', '')
X_test_processed.columns = X_test_processed.columns.str.replace('numeric__', '').str.replace('categorical__', '')



### LR baseline
X_train_LR = X_train_processed[['DEMOGRAPHICS_age', 'DEMOGRAPHICS_sex_F', 'DEMOGRAPHICS_sex_M', 'CURRENT_ADM_hours_since_adm'] + 
                               list(X_train_processed.columns[X_train_processed.columns.str.contains('INDEX_DATE_index_dt_weekday')])]
X_test_LR = X_test_processed[['DEMOGRAPHICS_age', 'DEMOGRAPHICS_sex_F', 'DEMOGRAPHICS_sex_M', 'CURRENT_ADM_hours_since_adm'] + 
                               list(X_test_processed.columns[X_test_processed.columns.str.contains('INDEX_DATE_index_dt_weekday')])]
X_val_LR = X_val_processed[['DEMOGRAPHICS_age', 'DEMOGRAPHICS_sex_F', 'DEMOGRAPHICS_sex_M', 'CURRENT_ADM_hours_since_adm'] + 
                               list(X_val_processed.columns[X_val_processed.columns.str.contains('INDEX_DATE_index_dt_weekday')])]

lr_baseline = LogisticRegression(max_iter=2000)
lr_baseline.fit(X_train_LR, y_train)

# performance on training data
y_pred = lr_baseline.predict(X_train_LR)
y_prob = lr_baseline.predict_proba(X_train_LR)[:,1]
print(custom_classification_report(y_train, y_pred, y_prob))

# performance on validation data
y_pred = lr_baseline.predict(X_val_LR)
y_prob = lr_baseline.predict_proba(X_val_LR)[:,1]
print(custom_classification_report(y_val, y_pred, y_prob))

# get the best threshold
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]
y_prob = lr_baseline.predict_proba(X_test_LR)[:,1]
y_pred = np.where(y_prob > best_thresh, 1, 0)

print(custom_classification_report(y_test, y_pred, y_prob))



### fit XGBoost
X_train_processed2=X_train_processed
X_val_processed2=X_val_processed
X_test_processed2=X_test_processed

X_train_processed=X_train_processed.drop(columns=['INDEX_DATE_index_dt_since_baseline'])
X_val_processed=X_val_processed.drop(columns=['INDEX_DATE_index_dt_since_baseline'])
X_test_processed=X_test_processed.drop(columns=['INDEX_DATE_index_dt_since_baseline'])


sp_weight=y_train.value_counts()[0]/y_train.value_counts()[1]

#tune hyperparameters
space = {
    'n_estimators': hp.choice('n_estimators', range(50, 1200, 25)),
    'learning_rate': hp.choice('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1]),
    'max_depth' : hp.choice('max_depth', range(3,12,1)),
    'gamma' : hp.choice('gamma', [i/10.0 for i in range(0,5)]),
    'colsample_bytree' : hp.choice('colsample_bytree', [i/10.0 for i in range(2,10)]),
    'min_child_weight': hp.choice('min_child_weight', range(1,6,1)),
    'subsample': hp.choice('subsample', [i/10.0 for i in range(3,10)]),
    'reg_alpha' : hp.choice('reg_alpha', [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]),
    'reg_lambda' : hp.choice('reg_lambda', [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100])
}

# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Objective function
def objective(params):
    xgboost = xgb.XGBClassifier(seed=0, **params, n_jobs=8, scale_pos_weight= sp_weight)
    score = cross_val_score(estimator=xgboost,
                            X=X_train_processed,
                            y=y_train,
                            cv=kfold,
                            scoring='roc_auc',
                            n_jobs=8).mean()
    # Loss is negative score
    loss = -score
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 100, trials = Trials())

# Print the values of the best parameters
print(space_eval(space, best))

# Train model using the best parameters
xgb_optim = xgb.XGBClassifier(seed=0,n_jobs=8, random_state=42, scale_pos_weight= sp_weight,
                           colsample_bytree=space_eval(space, best)['colsample_bytree'],
                           gamma=space_eval(space, best)['gamma'],
                           learning_rate=space_eval(space, best)['learning_rate'],
                           min_child_weight=space_eval(space, best)['min_child_weight'],
                           n_estimators=space_eval(space, best)['n_estimators'],
                           subsample=space_eval(space, best)['subsample'],
                           max_depth=space_eval(space, best)['max_depth'],
                           reg_alpha=space_eval(space, best)['reg_alpha'],
                           reg_lambda=space_eval(space, best)['reg_lambda']
                           ).fit(X_train_processed, y_train)


# calibration
labels_val = y_val.values
y_prob_val = xgb_optim.predict_proba(X_val_processed)[:,1]

isotonic = IsotonicRegression(out_of_bounds='clip',
                              y_min=y_prob_val.min(),
                              y_max=y_prob_val.max())
isotonic.fit(y_prob_val, y_val)
isotonic_probs = isotonic.predict(y_prob_val)

# get the best threshold
precision, recall, thresholds = precision_recall_curve(y_val, isotonic_probs)
F = 2*recall*precision/(recall+precision)
ix = np.argmax(F)
best_thresh = thresholds[ix]

# performance in test set
y_pred_optim_test = xgb_optim.predict(X_test_processed.values)
y_prob_optim_test = xgb_optim.predict_proba(X_test_processed.values)[:,1]
y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

print(custom_classification_report(y_test, y_pred_optim_test, y_prob_optim_test_calib))

y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

print(custom_classification_report(y_test, y_pred_optim_test_thresh, y_prob_optim_test_calib))
custom_classification_sum(y_test, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test)


# explain feature importance using SHAP
explainer = shap.TreeExplainer(xgb_optim)
shap_values = explainer.shap_values(X_train_processed)

shap.summary_plot(shap_values, X_train_processed, plot_type="bar")
fImportance=pd.DataFrame((zip(X_train_processed.columns[np.argsort(np.abs(shap_values).mean(0))][::-1], -np.sort(-np.abs(shap_values).mean(0)))),
columns=["Feature", "Importance"])



### subgroup analysis

# by sex
df_test_female = df_test[df_test['DEMOGRAPHICS_sex'] == 'F'].copy()
df_test_male = df_test[df_test['DEMOGRAPHICS_sex'] == 'M'].copy()
X_test_female = df_test_female.drop(columns=['y'])
y_test_female = df_test_female['y']
X_test_male = df_test_male.drop(columns=['y'])
y_test_male = df_test_male['y']

X_test_female_processed = preprocessor.transform(X_test_female)
X_test_female_processed = pd.DataFrame(X_test_female_processed, columns=feature_names)
X_test_male_processed = preprocessor.transform(X_test_male)
X_test_male_processed = pd.DataFrame(X_test_male_processed, columns=feature_names)

y_pred_optim_test = xgb_optim.predict(X_test_female_processed.values)
y_prob_optim_test = xgb_optim.predict_proba(X_test_female_processed.values)[:,1]
y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

custom_classification_sum(y_test_female, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test_female)

y_pred_optim_test = xgb_optim.predict(X_test_male_processed.values)
y_prob_optim_test = xgb_optim.predict_proba(X_test_male_processed.values)[:,1]
y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

custom_classification_sum(y_test_male, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test_male)


# by age
bins = [0, 40, 60, 80, float('inf')]
labels = ['<40', '40-60', '60-80', '>80']
df_test['DEMOGRAPHICS_age_cat'] = pd.cut(df_test['DEMOGRAPHICS_age'], bins=bins, labels=labels, right=False)

for i in ['<40', '40-60', '60-80', '>80']:
    df_test_age = df_test[df_test['DEMOGRAPHICS_age_cat']== i].copy()
    df_test_age.drop(columns=['DEMOGRAPHICS_age_cat'], inplace=True)

    X_test_age = df_test_age.drop(columns=['y'])
    y_test_age = df_test_age['y']
    X_test_age_processed = preprocessor.transform(X_test_age)
    X_test_age_processed = pd.DataFrame(X_test_age_processed, columns=feature_names)

    y_pred_optim_test = xgb_optim.predict(X_test_age_processed.values)
    y_prob_optim_test = xgb_optim.predict_proba(X_test_age_processed.values)[:,1]
    y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

    y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

    custom_classification_sum(y_test_age, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test_age)

# by ethnicity
for i in ['White',"Asian",'Black','Mixed','Other','Unknown']:
    df_test_ethnic = df_test[df_test['DEMOGRAPHICS_ethnic_gp']== i].copy()
    X_test_ethnic = df_test_ethnic.drop(columns=['y'])
    y_test_ethnic = df_test_ethnic['y']
    X_test_ethnic_processed = preprocessor.transform(X_test_ethnic)
    X_test_ethnic_processed = pd.DataFrame(X_test_ethnic_processed, columns=feature_names)


    y_pred_optim_test = xgb_optim.predict(X_test_ethnic_processed.values)
    y_prob_optim_test = xgb_optim.predict_proba(X_test_ethnic_processed.values)[:,1]
    y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

    y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

    custom_classification_sum(y_test_ethnic, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test_ethnic)

# by IMD
bins = [0, 6.5, 10.5, 16, float('inf')]
labels = ['<6.5', '6.5-10.5', '10.5-16', '>16']
df_test['DEMOGRAPHICS_imd_cat'] = pd.cut(df_test['DEMOGRAPHICS_imd'], bins=bins, labels=labels, right=False)

for i in ['<6.5', '6.5-10.5', '10.5-16', '>16']:
    df_test_imd = df_test[df_test['DEMOGRAPHICS_imd_cat']== i].copy()
    df_test_imd.drop(columns=['DEMOGRAPHICS_imd_cat'], inplace=True)

    X_test_imd = df_test_imd.drop(columns=['y'])
    y_test_imd = df_test_imd['y']
    X_test_imd_processed = preprocessor.transform(X_test_imd)
    X_test_imd_processed = pd.DataFrame(X_test_imd_processed, columns=feature_names)

    y_pred_optim_test = xgb_optim.predict(X_test_imd_processed.values)
    y_prob_optim_test = xgb_optim.predict_proba(X_test_imd_processed.values)[:,1]
    y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

    y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

    custom_classification_sum(y_test_imd, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test_imd)

#  by weekday
for i in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Holiday']:
    df_test_week = df_test[df_test['INDEX_DATE_index_dt_weekday']== i].copy()

    X_test_week = df_test_week.drop(columns=['y'])
    y_test_week = df_test_week['y']
    X_test_week_processed = preprocessor.transform(X_test_week)
    X_test_week_processed = pd.DataFrame(X_test_week_processed, columns=feature_names)

    y_pred_optim_test = xgb_optim.predict(X_test_week_processed.values)
    y_prob_optim_test = xgb_optim.predict_proba(X_test_week_processed.values)[:,1]
    y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

    y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

    custom_classification_sum(y_test_week, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test_week)

# by specialty
for i in ['Acute and general surgery','Acute, emergency and geriatric medicine','Surgical subspecialty','Trauma and orthopedics','Medical subspecialty','Other_spec']:
    df_test_spec = df_test[df_test['SPECIALTY_current_spec']== i].copy()
    X_test_spec = df_test_spec.drop(columns=['y'])
    y_test_spec = df_test_spec['y']
    X_test_spec_processed = preprocessor.transform(X_test_spec)
    X_test_spec_processed = pd.DataFrame(X_test_spec_processed, columns=feature_names)

    y_pred_optim_test = xgb_optim.predict(X_test_spec_processed.values)
    y_prob_optim_test = xgb_optim.predict_proba(X_test_spec_processed.values)[:,1]
    y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

    y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

    custom_classification_sum(y_test_spec, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test_spec)

# by comorbidity
df_test['COMORBIDITY_cat'] = pd.cut(df_test['COMORBIDITY_charlson_age_adj_comorbidity_score'], bins=[0, 1, 2, 4, float('inf')], labels=['0', '1-2', '3-4', '>=5'])
df_test['COMORBIDITY_cat'].value_counts()

for i in ['0', '1-2', '3-4', '>=5']:
    df_test_com = df_test[df_test['COMORBIDITY_cat']== i].copy()
    df_test_com.drop(columns=['COMORBIDITY_cat'], inplace=True)

    X_test_com = df_test_com.drop(columns=['y'])
    y_test_com = df_test_com['y']
    X_test_com_processed = preprocessor.transform(X_test_com)
    X_test_com_processed = pd.DataFrame(X_test_com_processed, columns=feature_names)

    y_pred_optim_test = xgb_optim.predict(X_test_com_processed.values)
    y_prob_optim_test = xgb_optim.predict_proba(X_test_com_processed.values)[:,1]
    y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

    y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

    custom_classification_sum(y_test_com, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test_com)

#by admission source
for i in df_test['CURRENT_ADM_adm_source'].unique():
    df_test_spec = df_test[df_test['CURRENT_ADM_adm_source']== i].copy()
    X_test_spec = df_test_spec.drop(columns=['y'])
    y_test_spec = df_test_spec['y']
    X_test_spec_processed = preprocessor.transform(X_test_spec)
    X_test_spec_processed = pd.DataFrame(X_test_spec_processed, columns=feature_names)

    y_pred_optim_test = xgb_optim.predict(X_test_spec_processed.values)
    y_prob_optim_test = xgb_optim.predict_proba(X_test_spec_processed.values)[:,1]
    y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

    y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

    custom_classification_sum(y_test_spec, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test_spec)

# by length of stay
def model_groups(d):
    if d < 7:
        return str(floor(d))
    if d < 10:
        return '7-10'
    if d < 14:
        return '10-14'
    if d < 21:
        return '14-21'
    if d < 28:
        return '21-28'
    if d >= 28:
        return '28+'

df_test['MODEL_group'] = df_test['MODEL_whole_days_since_adm'].apply(model_groups)
df_test['MODEL_group'].value_counts()

for i in df_test['MODEL_group'].unique():
    df_test_los = df_test[df_test['MODEL_group']== i].copy()
    X_test_los = df_test_los.drop(columns=['y'])
    y_test_los = df_test_los['y']
    X_test_los_processed = preprocessor.transform(X_test_los)
    X_test_los_processed = pd.DataFrame(X_test_los_processed, columns=feature_names)

    y_pred_optim_test = xgb_optim.predict(X_test_los_processed.values)
    y_prob_optim_test = xgb_optim.predict_proba(X_test_los_processed.values)[:,1]
    y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

    y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

    custom_classification_sum(y_test_los, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test_los)

# by outcome
for i in ['Alive','Died']:
    df_test_com = df_test[df_test['OUTCOME_died']== i].copy()
    df_test_com.drop(columns=['OUTCOME_died'], inplace=True)

    X_test_com = df_test_com.drop(columns=['y'])
    y_test_com = df_test_com['y']
    X_test_com_processed = preprocessor.transform(X_test_com)
    X_test_com_processed = pd.DataFrame(X_test_com_processed, columns=feature_names)

    y_pred_optim_test = xgb_optim.predict(X_test_com_processed.values)
    y_prob_optim_test = xgb_optim.predict_proba(X_test_com_processed.values)[:,1]
    y_prob_optim_test_calib = isotonic.predict(y_prob_optim_test)

    y_pred_optim_test_thresh = np.where(y_prob_optim_test_calib > best_thresh, 1, 0)

    custom_classification_sum(y_test_com, y_pred_optim_test_thresh, y_prob_optim_test_calib, X_test_com)
