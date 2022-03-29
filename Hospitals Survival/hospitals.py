import pandas as pd
import numpy as np
import statsmodels.imputation.mice as smi
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import matthews_corrcoef
import sys
import warnings
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
if len(sys.argv) != 3:
    print("Usage: python hospitals.py train_path test_path")
    # print("Usage: python3 hospitals.py total_path")
    exit(-1)

df_survival = pd.read_csv(sys.argv[1])
# df_survival = pd.read_csv('survival_train.csv')
try:
    # df_survival = pd.read_csv('Survival_dataset.csv')
    df_survival = df_survival.drop(columns = ['Length_of_stay', 'Survival'])
    df_survival.rename(columns = {'In-hospital_death':'Outcome', 'SAPS-I':'SAPSI'}, inplace=True)
except:
    pass

test_initial = pd.read_csv(sys.argv[2])
# test_initial = pd.read_csv('survival_test.csv')

try:
    test_initial = test_initial.drop(columns = ['Length_of_stay', 'Survival'])
    test_initial.rename(columns = {'In-hospital_death':'Outcome', 'SAPS-I':'SAPSI'}, inplace=True)
except:
    pass

def impute(data, outcomes):
    imp = smi.MICEData(pd.merge(data, outcomes, on='recordid'))
    imp.update_all(5)
    impute = imp.data
    return impute

def data_preprocessing(train:pd.DataFrame, test: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    ''' Take care of the missing values and class imbalance '''


    # try:
    #     test = test.drop(columns = ['Length_of_stay', 'Survival'])
    # except:
    #     pass

    train['index'] = train['recordid'].apply(lambda x: str(x)+'_train')
    test['index'] = test['recordid'].apply(lambda x: str(x)+'_test')
    # test.rename(columns = {'In-hospital_death':'Outcome', 'SAPS-I':'SAPSI'}, inplace=True)

    df_total = test.append(train)
    df_index = df_total[['index','recordid']]

    seq = ['first', 'last', 'lowest', 'highest', 'median']
    # id_columns = ['recordid','SAPSI','SOFA','Outcome','Age','Gender','Height','Weight','CCU','CSRU','SICU','MechVentStartTime','MechVentDuration','MechVentLast8Hour','UrineOutputSum']

    sub_reg_col = []
    for i in seq:
        sub_reg_col.append([x for x in df_total.columns if (i in x) | ('recordid' in x)])

    y_reg = df_total[['recordid','Outcome']]
    df_mice = impute(df_total[sub_reg_col[0]], y_reg)
    for col_reg in range(1,len(sub_reg_col)):
        df_mice = pd.merge(df_mice, impute(df_total[sub_reg_col[col_reg]], y_reg), right_on='recordid', left_on='recordid')

    # df_mice = pd.merge(df_mice, df_index, right_on='recordid', left_on='recordid')
    cols_to_drop = [x for x in df_mice.columns.to_list() if 'tcome' in x][:-1]
    df_mice = df_mice.drop(columns=cols_to_drop, axis=1)

    # replcae the filled NAs in the main total dataset
    columns_to_drop = list(df_mice.columns)[1:]
    df_total_filled = pd.merge(df_total.drop(columns_to_drop, axis=1), df_mice, right_on='recordid', left_on='recordid')

    # fill the total dataset now
    data_fill_all = impute(df_total_filled.loc[:,(df_total_filled.columns != 'Outcome') & (df_total_filled.columns != 'index')], y_reg)

    df_to_separate = pd.merge(data_fill_all, df_index, left_on='recordid', right_on='recordid')
    test_final = df_to_separate[df_to_separate['index'].apply(lambda x: x.split('_')[-1] == 'test')].drop('index', axis=1)
    data_fill_all = data_fill_all[~data_fill_all['recordid'].isin(test_final['recordid'].to_list())]

    oversample = SMOTE()
    X, y = oversample.fit_resample(data_fill_all.loc[:, data_fill_all.columns != 'Outcome'], data_fill_all['Outcome'])
    data_fill_all_balanced = pd.merge(X.reset_index(), y.reset_index(), left_on='index', right_on='index').drop(columns='index', axis=1)


    X_train = data_fill_all_balanced.loc[:,(data_fill_all_balanced.columns !='recordid') & (data_fill_all_balanced.columns !='Outcome')]
    y_train =  data_fill_all_balanced.loc[:,'Outcome']
    # print(test_final.loc[:,(test_final.columns !='recordid') & (test_final.columns !='Outcome')])
    X_test = test_final.loc[:,(test_final.columns !='recordid') & (test_final.columns !='Outcome')]
    y_test =  test_final.loc[:,'Outcome']

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    return X_train_scaled, y_train, X_test_scaled, y_test

def model_fit(X_train, y_train, X_test, y_test):
    ''' Fit the models with hyperparameter tuning, printing the scores and returning the probas with a threshold'''

    print('RF')
    RF = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(matthews_corrcoef(y_test, y_pred))

    print('SVM')
    SVM = make_pipeline(StandardScaler(), SVC(C=0.89, kernel='poly',gamma='auto')).fit(X_train, y_train)
    y_pred = SVM.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(matthews_corrcoef(y_test, y_pred))

    print('KNN')
    KNN=KNeighborsClassifier(n_neighbors=17,weights='uniform',algorithm='kd_tree').fit(X_train, y_train)
    y_pred = KNN.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(matthews_corrcoef(y_test, y_pred))

    print('LR')
    LR = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(matthews_corrcoef(y_test, y_pred))

    print('LDA')
    LDA = LinearDiscriminantAnalysis().fit(X_train, y_train)
    y_pred = LDA.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(matthews_corrcoef(y_test, y_pred))

    print('QDA')
    QDA = QuadraticDiscriminantAnalysis(reg_param=0.9).fit(X_train, y_train)
    y_pred = QDA.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(matthews_corrcoef(y_test, y_pred))

    print('AdaBoost')
    Ada=AdaBoostClassifier( n_estimators=70)
    Ada.fit(X_train, y_train)
    y_pred = Ada.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(matthews_corrcoef(y_test, y_pred))

    print('GradientBoostingClassifier')
    GBclf=GradientBoostingClassifier(n_estimators=30)
    GBclf.fit(X_train, y_train)
    y_pred = GBclf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(matthews_corrcoef(y_test, y_pred))


    estimators = [
    ('rf', RandomForestClassifier(random_state=0)),
    ('svm', make_pipeline(StandardScaler(),SVC(C=0.89, kernel='poly',gamma='auto'))),
    ('dt', DecisionTreeClassifier(random_state=0)),
    ('knn', KNeighborsClassifier(n_neighbors=17,weights='uniform',algorithm='kd_tree')),
    # ('lr',LogisticRegression()),
    ('nb', GaussianNB()),
    ('lda', LinearDiscriminantAnalysis()),
    ('qda', QuadraticDiscriminantAnalysis(reg_param=0.9)),
    ('ada',AdaBoostClassifier(n_estimators=30)),
    ('gb',GradientBoostingClassifier(n_estimators=30)),
    ('et',ExtraTreesClassifier(n_estimators=40)),
    ('rclf',RidgeClassifier()),
    ('bg',BaggingClassifier(base_estimator=QuadraticDiscriminantAnalysis(reg_param=0.9),n_estimators=10, random_state=0))]

    clf = StackingClassifier(
    estimators=estimators, final_estimator= QuadraticDiscriminantAnalysis(reg_param=0.9))
    
    print('Stacking')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(matthews_corrcoef(y_test, y_pred))


    BGclf = BaggingClassifier(base_estimator=QuadraticDiscriminantAnalysis(reg_param=0.9),n_estimators=10, random_state=0)

    print('Bagging')
    BGclf.fit(X_train, y_train)
    y_pred = BGclf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(matthews_corrcoef(y_test, y_pred))

    y_predict_proba = BGclf.predict_proba(X_test)[:,1]
    
    scores_threshold = []
    for thresholds in np.linspace(0, 1, num=100):
        y_pred_threshold = np.where(y_predict_proba>thresholds, 1, 0)
        scores_threshold.append(f1_score(y_test, y_pred_threshold))
    
    index = np.argmax(np.array(scores_threshold))

    threshold = np.linspace(0, 1, num=100)[index]
    
    
    
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict_proba)
    auc_precision_recall = auc(recall, precision)
    print("auc_precision_recall: ",auc_precision_recall)
    plt.plot(recall, precision)
    plt.show()
    average_precision = average_precision_score(y_test, y_predict_proba)
    disp = plot_precision_recall_curve(BGclf, X_test, y_test)
    disp.ax_.set_title('Binary class Precision-Recall curve: '
                    'AP={0:0.2f}'.format(average_precision))

    predict = dict({'predict_proba: ' : y_predict_proba, 'threshold: ' : threshold }) #threshold
    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle

    with open('predict_probas_model.p', 'wb') as fp:
        pickle.dump(predict, fp)







if __name__ == '__main__':

    train = df_survival
    test = test_initial

    print('Data preprocess')
    X_train, y_train, X_test, y_test = data_preprocessing(train, test)
    print('Model fit')
    model_fit(X_train, y_train, X_test, y_test)
