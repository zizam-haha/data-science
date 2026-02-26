import pandas as pd
import numpy as np
import sqlite3 as sql
from transformers import pipeline
import re

# --- 1. ALLA DINA TVÄTT-FUNKTIONER ---

def clean_siffra(num_series):
    renad = num_series.astype(str).str.lower()
    siffra_map = {'en': '1', 'ett': '1', 'två': '2', 'tre': '3', 'fyra': '4', 'fem': '5'}
    for ord, nr in siffra_map.items():
        renad = renad.str.replace(ord, nr, regex=False)
    renad = renad.str.replace(r'[^\d,.]', '', regex=True)
    val = pd.to_numeric(renad.str.replace(',', '.'), errors='coerce')
    return val.mask(val <= 0)

def clean_datum(datum):
    if pd.isna(datum) or str(datum).lower() == 'nan':
        return pd.NaT
    d = str(datum).lower().strip()
    if d.startswith('20'):
        return pd.to_datetime(d, errors='coerce')
    manader = {
       'januari': 'january', 'februari': 'february', 'mars': 'march',
        'april': 'april', 'maj': 'may', 'juni': 'june',
        'juli': 'july', 'augusti': 'august', 'september': 'september',
        'oktober': 'october', 'november': 'november', 'december': 'december'
    }
    for sv, en in manader.items():
        if sv in d:
            d = d.replace(sv, en)
    return pd.to_datetime(d, dayfirst=True, errors='coerce')

def clean_product(namn):
    if pd.isna(namn): return "Okänd produkt"
    n = str(namn).lower().strip()
    if 'basilika' in n: return 'Basilika planta'
    if 'tom' in n: 
        if 'planta' in n: return 'Tomatplanta (olika sorter)'
        return 'Tomatfrön'
    if 'petunia' in n: return 'Petunia mix'
    if 'jord' in n: return 'Blomjord 50L' if '50' in n else 'Blomjord 25L'
    if 'kruka' in n or 'terrakotta' in n: return 'Lerkruka'
    if 'monstera' in n: return 'Monstera'
    return n.capitalize()

def clean_kategori(n):
    n = str(n).lower()
    if any(x in n for x in ['pers', 'ört', 'tomat', 'frö', 'påsk', 'bas', 'plant', 'sallad']):
        return 'Vår'
    if any(x in n for x in ['höst', 'ljung', 'vinter', 'tulpan']):
        return 'Höst'
    return 'Året runt'

def clean_zon(v):
    v = str(v).strip().lower()
    if '1' in v or 'storstad' in v: return 'Storstad (Zon 1)'
    if '2' in v or 'södra' in v: return 'Södra Sverige (Zon 2)'
    if '3' in v or 'mellan' in v: return 'Mellansverige (Zon 3)'
    if '4' in v: return 'Längre norrut (Zon 4)'
    if '5' in v or 'norr' in v: return 'Norrland (Zon 5)'
    return 'Okänd zon'

# --- 2. SENTIMENT ANALYS (Ladda modellen en gång) ---
print("Laddar AI-modell (BERT)...")
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment(text):
    if not text or text == 'Ingen recension' or str(text).lower() == 'nan':
        return 'Ingen recension'
    resultat = sentiment_model(str(text)[:510])[0]
    label = resultat['label']
    if label in ['1 star', '2 stars']: return 'Negativ'
    if label == '3 stars': return 'Neutral'
    return 'Positiv'

# --- 3. HUVUDPIPELINE ---
def main():
    print("\n--- Startar Gröna Gården Pipeline ---")
    
    try:
        # A. Ladda data (Tagit bort ../ för att undvika FileNotFoundError)
        df = pd.read_csv('gronagarden_data.csv')
        df_clean = df.copy()
        print("Data laddad.")

        # B. Tvätta siffror
        for col in ['vikt_gram', 'pris', 'antal']:
            if col in df_clean.columns:
                df_clean[col] = clean_siffra(df_clean[col])

        # C. Tvätta datum
        for col in ['orderdatum', 'faktiskt_leveransdatum', 'recensionsdatum']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(clean_datum)

        # D. Tvätta texter och kategorier
        df_clean['produktnamn'] = df_clean['produktnamn'].apply(clean_product)
        df_clean['säsong'] = df_clean['produktnamn'].apply(clean_kategori)
        df_clean['leveranszon'] = df_clean['leveranszon'].apply(clean_zon)
        
        # E. Hantera nollvärden
        df_clean['recension_text'] = df_clean['recension_text'].fillna('Ingen recension')
        df_clean['betyg'] = df_clean['betyg'].fillna(0)
        df_clean['leveranstid_dagar'] = (df_clean['faktiskt_leveransdatum'] - df_clean['orderdatum']).dt.days.abs()

        # F. Sentiment Analys
        print("Analyserar sentiment (BERT)...")
        df_clean['recensionssentiment'] = df_clean['recension_text'].apply(analyze_sentiment)

        # G. Spara till SQL
        print("Sparar till databas...")
        conn = sql.connect('gronagarden_data_cleaned.db')
        df_clean.to_sql('orders', conn, if_exists='replace', index=False)
        
        # H. Valideringsdata (om filen finns)
        try:
            df_val = pd.read_csv('gronagarden_validation.csv') # Borttaget ../
            df_val['recension_text'] = df_val['recension_text'].fillna('Ingen recension')
            df_val['recensionssentiment'] = df_val['recension_text'].apply(analyze_sentiment)
            df_val_final = df_val[df_val['recensionssentiment'] != 'Ingen recension'].copy()
            df_val_final.to_sql('validation_results', conn, if_exists='replace', index=False)
            print("Validering sparad till SQL.")
        except FileNotFoundError:
            print("Hittade ingen valideringsfil, hoppar över det steget.")

        conn.close()
        print("\n--- ALLT KLART! ---")
        print(df_clean[['produktnamn', 'recensionssentiment']].head())

    except Exception as e:
        print(f"Ett fel uppstod: {e}")

if __name__ == "__main__":
    main()