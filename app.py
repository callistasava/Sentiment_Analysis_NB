
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import time

df = pd.read_csv('data_label.csv')
x = df['clean_review']
y = df['label']

model = pickle.load(open('sentiment.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_tfidf.pkl', 'rb'))

x_pred = vectorizer.transform(x.values.astype('U')).toarray()
y_pred = model.predict(x_pred)
acc = accuracy_score(y_pred,y)
acc = round((acc*100),2)

st.set_page_config(
    page_title = "Astro App Review Sentiment Analysis", 
    page_icon = "🛍"
)

#st.title('Astro Apps Review Sentiment Analysis')

with st.sidebar:
    selected = option_menu(
        menu_title = "Navigation Menu",
        options = ['Astro Apps Information','Sentiment Analysis Model Information','Single Prediction'],
        icons = ['basket-fill','box-fill','input-cursor-text','table'],
        menu_icon ='segmented-nav',
        default_index = 0,
        styles={
            "nav-link-selected":{"background-color":"#260b7d"},
        },
    )

if selected == 'Astro Apps Information':
    st.title('Astro Apps')
    st.image('astro.png')
    st.write('')
    st.write('Astro adalah layanan belanja online dengan lebih dari 1.000 produk pilihan. Belanjaanmu tiba dalam hitungan menit. Astro buka 24 jam, 7 hari, sepanjang tahun. Namun selama pandemi, Astro menyesuaikan jam operasional sesuai anjuran pemerintah. Bebas belanja kapan pun kamu mau.')
    st.write('')
    st.write('**Kelebihan Aplikasi Astro:**')
    lst = ['Keanekaragaman Produk','Layanan Pelanggan Unggulan','Pesanan Sampai Dalam Hitungan Menit','Penawaran dan Diskon Menarik','OneTalk by TapTalk.io: Solusi Layanan Komunikasi untuk Astro']
    s = ''
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
if selected == 'Sentiment Analysis Model Information':
    st.title('Reviews Sentiment Anaylsis Model')
    st.write('')
    st.write('Model ini dirancang untuk menganalisis sentimen dari ulasan pengguna aplikasi Astro. Dengan menggunakan model ini, pengguna dapat memahami apakah ulasan yang diberikan oleh pengguna lain bersifat positif atau negatif. Model analisis sentimen ini dibuat menggunakan algoritma Multinomial Naive Bayes, yang merupakan salah satu metode yang efektif dalam pengolahan teks dan analisis sentimen.')
    st.write('')
    st.write('**DataSet:**')
    st.write('Dataset yang digunakan untuk melatih model ini adalah data ulasan aplikasi Astro yang diambil langsung dari Google Playstore. Data ini mencakup berbagai ulasan dari pengguna, yang kemudian diproses dan dianalisis untuk membangun model yang akurat dan andal.')
    st.image('distribusi_label_astro.png', caption='Distribusi Jumlah Ulasan Sesuai Sentimen')
    st.write('Pada Gambar di atas memperlihatkan dataset yang berisi ulasan aplikasi Astro dari user sejumlah 733 ulasan terbagi menjadi 81 ulasan negatif dan 652 ulasan positif. Data ini nantinya akan dipisah menjadi data latih dan data uji.')
    st.write('')
    st.write('**Hasil Uji:**')
    st.image('cf-astro.png', use_column_width=True)
if selected == 'Single Prediction':
    st.title('Astro Apps Review Sentiment Analysis')
    st.markdown("""---""")
    st.title('Single-Predict Model Demo')
    coms = st.text_input('Enter your review about the Astro app')

    submit = st.button('Predict')

    if submit:
        start = time.time()
        # Transform the input text using the loaded TF-IDF vectorizer
        transformed_text = vectorizer.transform([coms]).toarray()
        #st.write('Transformed text shape:', transformed_text.shape)  # Debugging statement
        # Reshape the transformed text to 2D array
        transformed_text = transformed_text.reshape(1, -1)
        #st.write('Reshaped text shape:', transformed_text.shape)  # Debugging statement
        # Make prediction
        prediction = model.predict(transformed_text)
        end = time.time()
        st.write('Prediction time taken: ', round(end-start, 2), 'seconds')

        print(prediction[0])
        if prediction[0] == 1:
            st.title("😆 :green[**Sentimen review anda positif**]")
        else:
            st.title("🤬 :red[**Sentimen review anda negatif**]")
