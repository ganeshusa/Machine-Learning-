import streamlit as st
import langchain as lg
from PyPDF2 import PdfFileWriter, PdfFileReader

OPEN_API_KEY ="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"

st.title('ChatBot Testing')

with st.sidebar:
    st.title('Your Doc')
    file= st.file_uploader('Upload your doc',type='pdf')

if file is not None:
    pdf_reader = PdfFileReader(file)
    text=""
    for page in pdf_reader.pages:
        text += page.extractText()

