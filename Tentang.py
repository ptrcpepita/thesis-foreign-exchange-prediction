import streamlit as st
import pandas as pd
import numpy as np

    
st.title("💱 Sistem Prediksi Harga Penutupan dan Volatilitas Valuta Asing")
st.write("")
st.write("")

st.subheader("Tentang")
st.markdown("Aplikasi berbasis web yang dirancang untuk memberikan informasi prediktif mengenai harga penutupan mata uang asing terhadap Rupiah.")

st.write("")
st.write("")

st.subheader("Fitur")
st.markdown("""
- Prediksi harga penutupan USD/IDR, EUR/IDR, dan GBP/IDR dalam 1 hari dan 5 hari ke depan (business day)
- Prediksi volatilitas harga penutupan USD/IDR, EUR/IDR, dan GBP/IDR dalam 1 hari dan 5 hari ke depan (business day)
""")

st.write("")
st.write("---")

st.subheader("Sumber Data")
st.markdown("""
- Data utama yang digunakan, yaitu harga penutupan USD/IDR, EUR/IDR, dan GBP/IDR, diambil dari situs web Investing (https://www.investing.com/). 
- Data eksternal/variabel eksternal yang digunakan sebagai variabel tambahan untuk input model prediksi adalah data inflasi, cadangan devisa, dan suku bunga. Data ini diambil dari situs web Bank Indonesia (https://www.bi.go.id/id/).
""")

st.write("")
st.write("")

st.subheader("Model Prediksi")
st.markdown("""
- Model yang digunakan untuk memprediksi harga penutupan adalah Autoregressive Integrated Moving Average with Exogenous Variable (ARIMAX).
- Model yang digunakan untuk memprediksi volatilitas adalah ARIMAX dan Generalized Autoregressive Conditional Heteroskedasticity with Exogenous Variable (GARCHX).
""")

