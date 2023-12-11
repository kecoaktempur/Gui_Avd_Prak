import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import pingouin as pg
from scipy.stats import kendalltau
from scipy.stats import pointbiserialr

df = pd.read_csv(
    r"D:\Coolyeah\Mata Kuliah\SMT 5\Analisis Visualisasi Data Praktikum\6. UTS\AVD_UTS\AVD_UTS\Kuesioner Analisis Pola tidur terhadap BMI.csv")
df.head()

df.info()

df["label"].value_counts()

input = df.iloc[:, 0:15]
output = df.iloc[:, 15]
random_seed = 5
input_undersampling, output_undersampling = RandomUnderSampler(
    random_state=random_seed).fit_resample(input, output)
output_undersampling.value_counts()
df_undersampling = pd.concat(
    [input_undersampling, output_undersampling], axis=1)
df_undersampling.reset_index(drop=True, inplace=True)
df_undersampling.head()

df_undersampling.info()

df_undersampling['avg tidur siang'].value_counts()
df_undersampling['avg tidur siang'] = df_undersampling['avg tidur siang'].astype(
    int)

df_undersampling.isna().sum()
numerical_feature_1 = ["Umur", 'avg tidur malam',
                       'avg tidur siang', 'begadang', 'kualitas tidur', 'Indeks BMI']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=0)

from sklearn.naive_bayes import GaussianNB

model = GaussianNB(priors=[14/42, 14/42, 14/42])
model.fit(X_train, y_train)

pred_val = model.predict(X_test)
pred_val

from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred_val)*100

from sklearn import metrics

print(metrics.classification_report(y_test, pred_val))