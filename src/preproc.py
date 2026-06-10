import pandas as pd
import re
from src.config import DATA_PATH, PROCESSED_DATA_PATH, TEXT_COLUMN

def clean_text_for_bert(text):
    text = text.lower()
    text = re.sub(r"[^\w\s.,\-_/()]", "", text)
    text = " ".join(text.split())
    return text

def preprocess_data():
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text_for_bert)
    df.to_csv(PROCESSED_DATA_PATH, index=False) 
    print(f"✅ Preprocessing finished. Context preserved for BERT.")

if __name__ == "__main__":
    preprocess_data()
    
    
    
    
    