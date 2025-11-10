import streamlit as st
import langchain as lg
from PyPDF2 import PdfFileWriter, PdfFileReader


st.title('ChatBot Testing')

with st.sidebar:
    st.title('Your Doc')
    file= st.file_uploader('Upload your doc',type='pdf')

if file is not None:
    pdf_reader = PdfFileReader(file)
    text=""
    for page in pdf_reader.pages:
        text += page.extractText()

