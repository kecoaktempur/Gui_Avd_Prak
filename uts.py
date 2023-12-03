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

sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

sns.boxplot(data=df_undersampling[numerical_feature_1])
plt.title('Box Plots Visualization untuk deteksi outlier')
plt.xlabel('Features')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()

sns.boxplot(df_undersampling['avg tidur siang'])
plt.title("Boxplot avg tidur siang")

df_undersampling['avg tidur siang'] = df_undersampling['avg tidur siang'].replace([
                                                                                  3, 8], 1)
plt.show()
sns.boxplot(df_undersampling['avg tidur siang'])
plt.title("Boxplot avg tidur siang")

sns.boxplot(df_undersampling[["Tinggi Badan", "Berat Badan"]])
plt.title("Boxplot Visualization untuk deteksi outlier")
plt.xlabel("Features")

df_undersampling.info()

# Melakukan encoding untuk variable categorical
df_undersampling['set banyak alarm'] = df_undersampling['set banyak alarm'].replace({
                                                                                    "Ya": 1, "Tidak": 0})
df_undersampling['set banyak alarm'].value_counts()

df_undersampling['tidur setelah bangun'] = df_undersampling['tidur setelah bangun'].replace({
                                                                                            "Ya": 1, "Tidak": 0})
df_undersampling['tidur setelah bangun'].value_counts()

df_undersampling['label'].value_counts()

df_undersampling['label'] = df_undersampling['label'].replace(
    {"Underweight": 1, "Normal": 2, "Overweight": 3})
df_undersampling['label'].value_counts()

df_undersampling['jam tidur'] = df_undersampling['jam tidur'].replace(
    {"sekitar pukul 21.00 - 23.00 malam": 1, "diatas pukul 23.00 malam": 2})

df_undersampling['jam bangun'] = df_undersampling['jam bangun'].replace(
    {"dibawah pukul 05.00 pagi": 1, "sekitar pukul 05.00 - 06.30 pagi": 2, "diatas pukul 06.30 pagi": 3})
df_undersampling['jam bangun'].value_counts()

df_corr = df_undersampling.iloc[:, 4:16]
df_corr.head()

df_kontinu = df_undersampling[["Tinggi Badan", "Umur",
                               'Berat Badan', 'avg tidur malam', 'avg tidur siang', 'Indeks BMI']]
df_ordinal = df_undersampling[[
    'begadang', 'jam tidur', 'kualitas tidur', 'jam bangun', 'label']]
df_nominal = df_undersampling[['set banyak alarm', 'tidur setelah bangun']]

correlation_matrix = pd.DataFrame(
    index=df_ordinal.columns, columns=df_kontinu.columns)

for ordinal_feature in df_ordinal.columns:
    for continuous_feature in df_kontinu.columns:
        correlation_result = pg.corr(
            df_ordinal[ordinal_feature], df_kontinu[continuous_feature], method="spearman")
        correlation_matrix.at[ordinal_feature,
                              continuous_feature] = correlation_result['r'][0]
correlation_matrix = correlation_matrix.apply(pd.to_numeric)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True,
            cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Spearman Correlation Heatmap')
plt.show()

ordinal_features = ["begadang", "jam tidur",
                    "kualitas tidur", "jam bangun", "label"]

correlation_matrix = pd.DataFrame(
    index=ordinal_features, columns=ordinal_features)
for feature1 in ordinal_features:
    for feature2 in ordinal_features:
        correlation_coefficient, _ = kendalltau(
            df_undersampling[feature1], df_undersampling[feature2])
        correlation_matrix.at[feature1, feature2] = correlation_coefficient

correlation_matrix = correlation_matrix.apply(pd.to_numeric)


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True,
            cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Kendall Correlation Heatmap between Ordinal Features')
plt.show()

nominal_features = ['set banyak alarm', 'tidur setelah bangun']
ordinal_features = ['begadang', 'jam tidur',
                    'kualitas tidur', 'jam bangun', 'label']


correlation_matrix = pd.DataFrame(
    index=nominal_features, columns=ordinal_features)


for nominal_feature in nominal_features:
    for ordinal_feature in ordinal_features:
        correlation_result = pointbiserialr(
            df_undersampling[nominal_feature], df_undersampling[ordinal_feature])
        correlation_matrix.at[nominal_feature,
                              ordinal_feature] = correlation_result.correlation


correlation_matrix = correlation_matrix.apply(pd.to_numeric)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True,
            cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Point-Biserial Correlation Heatmap between Nominal and Ordinal Features')
plt.show()

df_bar = df_undersampling[['tidur setelah bangun',
                           'avg tidur siang', 'begadang', 'label']]

plt.figure(figsize=(12, 8))
plt.subplots_adjust(hspace=0.5)

for i, column in enumerate(df_bar.columns[:-1]):
    plt.subplot(3, 3, i+1)  # Create subplots
    sns.countplot(x=column, data=df_bar, palette='coolwarm')
    plt.title(f'Frequency of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

df_bar = df_undersampling[['tidur setelah bangun','avg tidur siang', 'begadang', 'label']]

plt.figure(figsize=(12, 8))
plt.subplots_adjust(hspace=0.5)

for i, column in enumerate(df_bar.columns[:-1]):
    plt.subplot(3, 3, i+1)  # Create subplots
    sns.countplot(x=column, data=df_bar, palette='coolwarm')
    plt.title(f'Frequency of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt


df_bar = df_undersampling[['Berat Badan',
                           'avg tidur siang', 'tidur setelah bangun', 'label']]


plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=0.5, wspace=0.5)


for i, column in enumerate(df_bar.columns[:-1]):
    ax = plt.subplot(3, 3, i+1)

    class_frequencies = df_bar.groupby(
        column)['label'].value_counts().unstack(fill_value=0)

    class_frequencies.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title(f'{column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

    # Add legend
    ax.legend(title='Class', loc='upper left', bbox_to_anchor=(1, 1))


plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

df_bar = df_undersampling[['Berat Badan',
                           'avg tidur siang', 'tidur setelah bangun', 'label']]

plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

for i, column in enumerate(df_bar.columns):
    ax = plt.subplot(3, 3, i+1)

    feature_frequencies = df_bar[column].value_counts()

    ax.pie(feature_frequencies, labels=feature_frequencies.index,
           autopct='%1.1f%%', startangle=90)
    ax.set_title(f'{column} Composition')
    ax.axis('equal')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt


df_bar = df_undersampling[['Berat Badan', 'avg tidur siang',
                           'tidur setelah bangun', 'label', 'avg tidur malam']]

plt.figure(figsize=(10, 6))

for label in df_bar['label'].unique():
    subset = df_bar[df_bar['label'] == label]
    plt.stackplot(subset.index, subset['avg tidur malam'], subset['avg tidur siang'], labels=[
                  f'avg tidur malam - Label {label}', f'avg tidur siang - Label {label}'], alpha=0.7)

# Customize the plot
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Stacked Area Chart')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
features = ['avg tidur siang', 'Berat Badan']
labels = df_bar['label']

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_bar['avg tidur siang'], df_bar['Berat Badan'], labels)
ax.set_xlabel('avg tidur siang')
ax.set_ylabel('Berat Badan')
ax.set_zlabel('Label')
plt.title('3D Scatter Plot')
plt.show()

import plotly.express as px


fig = px.scatter_3d(df_bar, x='tidur setelah bangun', y='jam bangun', z='label',
                    labels={'tidur setelah bangun': 'Tidur Setelah Bangun',
                            'jam bangun': 'Jam Bangun',
                            'label': 'Label'},
                    title='3D Scatter Plot with Categorical Axis')
fig.show()

import plotly.express as px

fig = px.line_3d(df_bar, x='avg tidur siang', y='Berat Badan', z='label', labels={'avg tidur siang': 'AVG Tidur Siang','Berat Badan': 'Berat Badan','label': 'Label'}, title='3D Line Plot')
fig.show()

import numpy as np

variables = ['tidur setelah bangun', 'jam bangun', 'label']

averages = df_undersampling[variables].mean()
sorted_variables = averages.sort_values()
sorted_variable_list = sorted_variables.index.tolist()


data_lists = [df_undersampling[var].tolist() for var in sorted_variable_list]
print(data_lists)
df_3d = pd.DataFrame(data_lists).transpose()
colors = ['red', 'green', 'blue', 'orange',
          'purple', 'pink', 'cyan', 'magenta', 'yellow']

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection="3d")
plt.yticks([i for i in range(len(sorted_variable_list))], sorted_variable_list)
z = list(df_3d)
for n, i in enumerate(df_3d):
    print('n', n)
    xs = np.arange(len(df_3d[i]))
    ys = [i for i in df_3d[i]]
    zs = z[n]

    cs = colors[n]
    print(' xs:', xs, 'ys:', ys, 'zs', zs, ' cs: ', cs)
    ax.bar(xs, ys, zs, zdir="y", color=cs, alpha=0.8)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

from sklearn.model_selection import train_test_split

X = df_undersampling[['avg tidur malam', 'avg tidur siang', 'jam bangun', 'Berat Badan']]
y = df_undersampling[['label']]

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