import streamlit as st
import pandas as pd
import json
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from docx import Document
import pdfplumber
from pylatexenc.latex2text import LatexNodes2Text
import gensim
import re
import requests
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# nltk preuzimanje stop rijeci na eng
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# stop rijeci na eng
nltk_stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# citanje razlicitih datoteka
def read_txt(file):
    try:
        text = file.read().decode("utf-8")
    except UnicodeDecodeError:
        file.seek(0)
        text = file.read().decode("ISO-8859-1")

    return text

def read_html(file):
    soup = BeautifulSoup(file, 'html.parser')
    return soup.get_text()

def fetch_html(url):
    response = requests.get(url)
    return response.text

def read_docx(file):
    doc = Document(file)
    return '\n'.join([p.text for p in doc.paragraphs])

def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        return '\n'.join(page.extract_text() for page in pdf.pages)

def read_epub(file):
    book = epub.read_epub(file)
    text = []
    for item in book.items:
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text.append(item.get_body_content().decode())
    return '\n'.join(text)

def read_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def read_json(file):
    data = json.load(file)
    return json.dumps(data, indent=4)

def read_latex(file):
    text = file.read().decode("utf-8")
    return LatexNodes2Text().latex_to_text(text)

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # uklanjam spec karaktere  
    text = re.sub(r'\d+', '', text)  # uklanjam brojeve
    text = text.lower()
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    tokens = [word for word in tokens if word not in nltk_stop_words]
    
    return ' '.join(tokens)

def vectorize_text(texts):
    vectorizer = CountVectorizer(stop_words=nltk_stop_words)
    X = vectorizer.fit_transform(texts)
    return X

def create_lda_model(texts, num_topics=5, passes=50):
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    return lda_model, corpus, dictionary

def plot_lda_topics(lda_model, num_words=10):
    topics = lda_model.show_topics(formatted=False, num_words=num_words)
    topic_words = []
    topic_numbers = []
    
    for topic_no, words in topics:
        for word, prob in words:
            topic_words.append((topic_no, word, prob))
            topic_numbers.append(topic_no)
    
    df = pd.DataFrame(topic_words, columns=['Topic', 'Word', 'Probability'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Probability', y='Word', hue='Topic', data=df, palette='viridis')
    plt.title('Distribucija riječi po temama')
    plt.xlabel('Vjerovatnoća')
    plt.ylabel('Riječ')
    st.pyplot(plt)

def calculate_perplexity(lda_model, corpus):
    return lda_model.log_perplexity(corpus)

def display_topics(lda_model, num_words=10):
    topics = lda_model.show_topics(formatted=True, num_words=num_words)
    for topic_no, topic in topics:
        st.write(f"**Tema {topic_no}:**")
        st.write(topic)

def assign_topics_to_documents(texts, lda_model, dictionary):
    tokenized_texts = [text.split() for text in texts]
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    topics_per_document = []
    for doc_bow in corpus:
        topic_distribution = lda_model[doc_bow]
        topics_per_document.append(topic_distribution)
    
    return topics_per_document

# gui na streamlitu
st.title('Analiza Dokumenata sa LDA Modelom')

# unos preko urla
url = st.text_input("Unesi URL za HTML stranicu (opciono):")

uploaded_files = st.file_uploader("Izaberi dokumente", type=["txt", "html", "docx", "pdf", "epub", "xlsx", "json", "latex"], accept_multiple_files=True)

if uploaded_files or url:
    # proces dokumenata
    def process_files(files):
        texts = []
        for file in files:
            file_type = file.type.split('/')[1]
            if file_type == "plain":
                text = read_txt(file)
            elif file_type == "html":
                text = read_html(file)
            elif file_type == "msword":
                text = read_docx(file)
            elif file_type == "pdf":
                text = read_pdf(file)
            elif file_type == "epub":
                text = read_epub(file)
            elif file_type == "vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                text = read_excel(file)
            elif file_type == "json":
                text = read_json(file)
            elif file_type == "latex":
                text = read_latex(file)
            else:
                continue
            # dodavanje teksta iz trenutne datoteke u listu
            texts.append(preprocess_text(text))
        return texts

    if uploaded_files:
        texts = process_files(uploaded_files)
    else:
        html_content = fetch_html(url)
        texts = [preprocess_text(read_html(html_content))]
    
    # kombinovanje tekstova
    combined_text = ' '.join(texts)

    # LDA model
    num_topics = st.slider('Broj tema', min_value=2, max_value=10, value=5)
    passes = st.slider('Broj prolaza', min_value=5, max_value=100, value=50)

    lda_model, corpus, dictionary = create_lda_model(texts, num_topics=num_topics, passes=passes)
    
    
    # prikaz tema
    #napraviti spec teme - kako uvesti korpuse za treniranje
    st.write('Teme modela:')
    display_topics(lda_model)
    
    # dodjele tema 
    #sada je samo parapleksija nema oznacenih naziva tema vec t1, t2...
    topics_per_document = assign_topics_to_documents(texts, lda_model, dictionary)
    st.write("Dodjela tema dokumentima:")
    for i, topic_distribution in enumerate(topics_per_document):
        st.write(f"Dokument {i+1}:")
        for topic_num, prob in topic_distribution:
            st.write(f"  Tema {topic_num}: {prob:.4f}")
    
    # LDA model grafik
    st.write('Vizualizacija LDA modela:')
    plot_lda_topics(lda_model)
