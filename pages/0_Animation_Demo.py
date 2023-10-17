import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer


def main():
    st.title("Halaman TermFrequency")
    st.write("""
            Term Frequency adalah.... 
        """)

    dt = pd.read_excel("HasilPreposPTA.xlsx")

    # Ekstraksi fitur dan membentuk VSM dalam term frequency
    dt['cleaning'].fillna('', inplace=True)
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(dt['cleaning'])
    # print(count_matrix)
    count_array = count_matrix.toarray()
    df = pd.DataFrame(data=count_array, columns=vectorizer.vocabulary_.keys())
    st.dataframe(df)


# if __name__ == "__main__":
#     main()
