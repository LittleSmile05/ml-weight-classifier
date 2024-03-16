import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Məlumatları yükləyin və göstərin
məlumat = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
print(məlumat.describe())

def index_adi_ver(ind):
    if ind==0:
        return 'Aşırı Zayıf'
    elif ind==1:
        return 'Zayıf'
    elif ind==2:
        return 'Normal'
    elif ind==3:
        return 'Fazla Kilolu'
    elif ind==4:
        return 'Obezite'
    elif ind==5:
        return 'Aşırı Obez'

məlumat['Index'] = məlumat['Index'].apply(index_adi_ver)

sns.lmplot('Boy', 'Çəki', məlumat, hue='Index', size=7, aspect=1, fit_reg=False)

cins = məlumat['Cinsiyət'].value_counts()
kateqoriyalar = məlumat['Index'].value_counts()

# KİŞİLƏR ÜÇÜN STATİSTİKA
məlumat[məlumat['Cinsiyət']=='Kişi']['Index'].value_counts()

# QADINLAR ÜÇÜN STATİSTİKA
məlumat[məlumat['Cinsiyət']=='Qadın']['Index'].value_counts()

məlumat2 = pd.get_dummies(məlumat['Cinsiyət'])
məlumat.drop('Cinsiyət', axis=1, inplace=True)
məlumat = pd.concat([məlumat, məlumat2], axis=1)

y = məlumat['Index']
məlumat = məlumat.drop(['Index'], axis=1)

scaler = StandardScaler()
məlumat = scaler.fit_transform(məlumat)
məlumat = pd.DataFrame(məlumat)

X_train, X_test, y_train, y_test = train_test_split(məlumat, y, test_size=0.3, random_state=101)

param_grid = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 1000]}
grid_cv = GridSearchCV(RandomForestClassifier(random_state=101), param_grid, verbose=3)

grid_cv.fit(X_train, y_train)

print(grid_cv.best_params_)
# çəki kateqoriyası proqnozu
pred = grid_cv.predict(X_test)

print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print('Dəqiqlik --> ', accuracy_score(y_test, pred)*100)
print('\n')

def index_proqnozu(məlumatlar):
    cins = məlumatlar[0]
    boy = məlumatlar[1]
    çəki = məlumatlar[2]
    
    if cins == 'Kişi':
        məlumatlar = np.array([[np.float(boy), np.float(çəki), 0.0, 1.0]])
    elif cins == 'Qadın':
        məlumatlar = np.array([[np.float(boy), np.float(çəki), 1.0, 0.0]])
    
    y_pred = grid_cv.predict(scaler.transform(məlumatlar))
    return (y_pred[0])

# Canlı proqnoz
sizin_məlumatlarınız = ['Kişi', 175, 80]
print(index_proqnozu(sizin_məlumatları))
