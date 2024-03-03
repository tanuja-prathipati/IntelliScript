import nltk
import os

# Set the path to the Intelliscript folder on your desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "IntelliScript")

# Create the nltk_data directory within the Intelliscript folder
nltk_data_path = os.path.join(desktop_path, "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Set the NLTK data path to the newly created nltk_data directory
nltk.data.path.append(nltk_data_path)

# Download the 'punkt' package
nltk.download('punkt')


from transformers import pipeline
from nltk.tokenize import sent_tokenize
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline, AutoTokenizer
import unicodedata
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

def get_summary(manual_subtitles, text, model_choice):
    if manual_subtitles:
        extractive_summary = get_extractive_summary(text)
        abstractive_summary = get_abstractive_summary(extractive_summary, model_choice)
    else:
        abstractive_summary = get_abstractive_summary(text, model_choice)
    
    return abstractive_summary

def get_abstractive_summary(extractive_summary, model_choice):
    models = {
        0: "facebook/bart-large-cnn",
        1: "t5-base"
    }
    model_name = models[model_choice]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=1024)  # Adjust max length as needed

    # Load the model for summarization
    generator = pipeline('summarization', model=model_name)
    
    # Generate abstractive summary
    abstractive_summary = generator(extractive_summary, max_length=100, min_length=5, do_sample=True, early_stopping=True)[0]['summary_text']
    
    return abstractive_summary

def get_extractive_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    req_sentences = round(len(sent_tokenize(text)) * 0.70)
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, req_sentences)
    
    ext_summary = " ".join(str(sentence) for sentence in summary)
    return ext_summary

def clean_summary(text):
    error_dict = [...]
    for error in error_dict:
        if error == text:
            return text

    irrelevant_terms = ["[music]", "[Music]", "\n", "<<", ">>"]
    sentence_list = [sentence.replace(item, "").strip() for sentence in sent_tokenize(text) for item in irrelevant_terms]
    cleaned_text = " ".join(sentence.capitalize() for sentence in sentence_list)
    normalized_text = unicodedata.normalize('NFKD', cleaned_text)
    formatted_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    return formatted_text.replace("\'", "'")

# Example usage
print(get_summary(True, "Nvidia is a leading technology company in the field of graphics processing units (GPUs) and artificial intelligence (AI). Founded in 1993, Nvidia has played a significant role in shaping the modern computing landscape.", 1))

