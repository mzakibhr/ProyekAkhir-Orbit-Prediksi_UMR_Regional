# Import Library
from turtle import width
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Linear Regression Library
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Random Forest Library
from sklearn.ensemble import RandomForestRegressor

# Evaluation
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

st.write("""
# UMR Indonesia Prediction
Aplikasi ini untuk **Tugas Akhir Orbit Future Academy AI Masterry**!""")

st.write("""
**Anggota** : Zaki Bahar, Fadlan Akmal, Haydar Rizaldy, Raihan Romzi.""")
st.write('---')

# Load Dataset UMR
df = pd.read_csv('umr_data.csv').drop(['Unnamed: 0'], axis=1)
X = df.drop(['SALARY','REGION'], axis = 1)
y = df['SALARY']

# Load Dataset Depresi
df_depresi = pd.read_csv('depresi.csv')

df = df[df["REGION"].str.contains("INDONESIA") == False]

# Animasi Data
st.header('Pergerakan UMR (Regional) 1997 - 2022')
fig = px.bar(
    df,
    x='REGION',
    y="SALARY",
    color="REGION",
    animation_frame="YEAR",
    range_y=[0, 4766460],
    labels={
        "SALARY": "UMR(Millions)",
        "REGION": "Regional"
    })
fig.update_layout(width=800, height=600, xaxis_visible=False, xaxis_showticklabels=False)
st.plotly_chart(fig)
st.write('---')

# Linear Regression
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size=0.2, random_state=4)
lin_reg = LinearRegression()
model_lr = lin_reg.fit(X_train_lr, y_train_lr)

# Random Forest
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2, random_state=4)
ran_for = RandomForestRegressor()
model_rf = ran_for.fit(X_train_rf, y_train_rf)

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Masukkan Regional dan Tahun')

def formatrupiah(uang):
    y = str(uang)
    if len(y) <= 3:
        return f'Rp{y}'
    p = y[-3:]
    q = y[:-3]
    return formatrupiah(q) + '.' + p

# User Input
def user_input_feature():
    REG = st.sidebar.selectbox('Regional', ('ACEH',
    'BALI',
    'BANTEN',
    'BENGKULU',
    'DI YOGYAKARTA',
    'DKI JAKARTA',
    'GORONTALO',
    'INDONESIA',
    'JAMBI',
    'JAWA BARAT',
    'JAWA TENGAH',
    'JAWA TIMUR',
    'KALIMANTAN BARAT',
    'KALIMANTAN SELATAN',
    'KALIMANTAN TENGAH',
    'KALIMANTAN TIMUR',
    'KALIMANTAN UTARA',
    'KEP. BANGKA BELITUNG',
    'KEP. RIAU',
    'LAMPUNG',
    'MALUKU',
    'MALUKU UTARA',
    'NUSA TENGGARA BARAT',
    'NUSA TENGGARA TIMUR',
    'PAPUA',
    'PAPUA BARAT',
    'RIAU',
    'SULAWESI BARAT',
    'SULAWESI SELATAN',
    'SULAWESI TENGAH',
    'SULAWESI TENGGARA',
    'SULAWESI UTARA',
    'SUMATERA BARAT',
    'SUMATERA SELATAN',
    'SUMATERA UTARA'))
    # REG = st.sidebar.text_input('Provinsi', 'Jawa Barat').upper()
    THN = int(st.sidebar.number_input('Tahun', 2023, 3000))

    st.header(f'UMR Regional di Provinsi {REG.capitalize()} Tahun 1997 - {THN - 1}')

    df_dep = df[df.REGION.str.contains(REG)].reset_index()
    reg_cat = df_dep._get_value(0, "REGION_Cat")
    fig = px.histogram(df_dep, x='YEAR', y="SALARY",
      labels={
          "YEAR": "Tahun",
          "SALARY": "UMR(Millions)",
          },
      )
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)
    st.write('')

    growth_reg=[0]
    growth_reg.extend(df_dep["SALARY"][i] - df_dep["SALARY"][i - 1] for i in range(1, 26))

    df_dep["growth"]=growth_reg
    fig = px.line(df_dep, x='YEAR', y="growth", 
    labels={
        "YEAR": "Tahun",
        "growth": "Pertumbuhan(Rupiah)",
        },
    )
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)
    st.write('---')

    st.header(f'Prediksi UMR {REG.capitalize()} di tahun 1997 - {THN}')
    pred = [[REG, reg_cat, int(lin_reg.predict([[reg_cat, i]])), i] for i in range(2023, THN + 1)]
    df_pred = pd.DataFrame(pred)
    df_pred.columns = ['REGION', 'REGION_Cat', 'SALARY', 'YEAR']
    df_newpred = pd.concat([df_dep, df_pred], ignore_index=True)
    prediksi = float(lin_reg.predict([[reg_cat,THN]]))
    fig = px.histogram(df_newpred, x='YEAR', y="SALARY",
    labels={
        "YEAR": "Tahun",
        "SALARY": "UMR(Millions)",
        },
    )
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)
    st.write("""
    #### Prediksi UMR : {}""".format(formatrupiah(int(prediksi))))
    st.write('---')
    
    st.header(f'Tingkat Depresi di Provinsi {REG.capitalize()}')
    cat = df_depresi.loc[df_depresi['provinsi'] == REG, 'kategori'].item()
    num = str(df_depresi.loc[df_depresi['provinsi'] == REG, 'urutan'].item())
    val = str(df_depresi.loc[df_depresi['provinsi'] == REG, 'value'].item())
    st.write("Tingkat depresi di provinsi " + '**' + REG + '**' + " masuk ke dalam kategori " + cat + ", dengan urutan ke-" + num + ", dan angka prevalensi " + val)
    fig = px.histogram(df_depresi, x='provinsi', y="value",
    labels={
        "provinsi": "Provinsi",
        "value": "Prevalensi",
        })
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)
    
df = user_input_feature()




