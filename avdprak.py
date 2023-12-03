import streamlit as st
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv(r"D:\Coolyeah\Mata Kuliah\SMT 5\Analisis Visualisasi Data Praktikum\6. UTS\AVD_UTS\AVD_UTS\Kuesioner Analisis Pola tidur terhadap BMI.csv")

# Undersampling
random_seed = 5
input_undersampling, output_undersampling = RandomUnderSampler(random_state=random_seed).fit_resample(df[['avg tidur malam', 'avg tidur siang', 'jam bangun', 'Berat Badan']], df['label'])
df_undersampling = pd.concat([input_undersampling, output_undersampling], axis=1)
df_undersampling.reset_index(drop=True, inplace=True)

# Convert 'jam bangun' to one-hot encoding
df_undersampling = pd.concat([df_undersampling, pd.get_dummies(df_undersampling['jam bangun'], prefix='jam_bangun')], axis=1)
df_undersampling.drop('jam bangun', axis=1, inplace=True)

def train_naive_bayes_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, train_size=0.8, random_state=0)

    model = GaussianNB(priors=[14/42, 14/42, 14/42])
    model.fit(X_train, y_train)

    pred_val = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred_val) * 100

    st.write(f"Naive Bayes Model Accuracy: {accuracy:.2f}%")

    return model

def main(model, input_values):
    # Prepare input dataframe for prediction
    input_df = pd.DataFrame(columns=df_undersampling.columns, data=[input_values])
    input_df.drop('label', axis=1, inplace=True)  # Remove label column (if exists)
    input_df['label'] = 0  # Add a placeholder for label (will be ignored in prediction)

    # Use the same column order as in training data
    input_df = input_df[df_undersampling.columns]

    predicted_label = model.predict(input_df.drop('label', axis=1))[0]
    return predicted_label

# Train the model
X = df_undersampling.drop('label', axis=1)
y = df_undersampling['label']
trained_model = train_naive_bayes_model(X, y)

# Streamlit App
st.title("BMI Prediction App")

# Sidebar for user input
st.sidebar.header("Enter Input Values")

# Prepare a dictionary to hold user inputs
user_inputs = {}
for feature in X.columns:
    if feature in ['jam_bangun_diatas_pukul_05.00_pagi', 'jam_bangun_sekitar_pukul_05.00_06.30_pagi', 'jam_bangun_diatas_pukul_06.30_pagi']:
        user_inputs[feature] = st.sidebar.checkbox(f"{feature.replace('_', ' ')}")
    else:
        user_inputs[feature] = st.sidebar.number_input(f"Enter {feature}", min_value=0)

# Predict and display the result
if st.sidebar.button("Predict BMI Label"):
    predicted_label = main(trained_model, user_inputs)
    st.success(f"Predicted BMI Label: {predicted_label}")
