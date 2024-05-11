import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

location = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(location, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(location, device_map='auto', torch_dtype=torch.float32)

def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_text = ""
    for text in texts:
        print(text)
        final_text = final_text+text.page_content
    return final_text

def llm_pipeline(path):
    pipe_sum = pipeline(
        "summarization", model=model, tokenizer=tokenizer, max_length=500, min_length=50
    )
    input_text = file_preprocessing(path)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.set_page_config(layout="wide",page_title="PDF Summarization Application")
def main():
    st.title("Text Summarization using LLM")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Summarize"):
            filepath = "data/" + uploaded_file.name
            with open(filepath, "wb") as f:
                f.write(uploaded_file.read())
            summary = llm_pipeline(filepath)
            st.success(summary)
        if st.button("View pdf"):
            filepath = "data/" + uploaded_file.name
            with open(filepath, "wb") as f:
                f.write(uploaded_file.read())
            pdf_display = displayPDF(filepath)

if __name__ == "__main__":
    main()
