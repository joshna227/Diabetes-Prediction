import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'D:\diabetes.csv')

st.title("Diabetes Prediction")

st.subheader("Training Data")
st.write(df.describe())

st.subheader("Visualization")

# Histograms
st.subheader("Histograms")
fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(12, 18))
flatten_axes = [ax for sublist in axs for ax in sublist]

for i, column in enumerate(df.columns[:-1]):
    ax = flatten_axes[i]
    ax.hist(df[column], bins=20, alpha=0.7, color='skyblue')
    ax.set_title(column)

st.pyplot(fig)
st.write("")

# Scatter Plot
st.subheader("Scatter Plot")
sns.pairplot(df, hue="Outcome", markers=["o", "s"])
st.pyplot()
st.write("")

x = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def user_report():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 846, 79)
    bmi = st.sidebar.slider("BMI", 0, 67, 20)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.0, 0.47)
    age = st.sidebar.slider("Age", 21, 88, 33)

    user_report = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    report_data = pd.DataFrame(user_report, index=[0])
    return report_data

user_data = user_report()

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

st.subheader("Accuracy:")
accuracy = accuracy_score(y_test, rf.predict(x_test))
st.write(f"{accuracy * 100:.2f}%")

user_result = rf.predict(user_data)
st.subheader("Your Report:")
output = "You Are Healthy" if user_result[0] == 0 else "You Are Not Healthy"
st.write(output)
