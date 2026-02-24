import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 as sql

df = pd.read_csv('../gronagarden_data.csv')
df_clean = df.copy()


def clean_siffra(num):
    # Gör allt till små bokstäver
    renad = num.astype(str).str.lower()
    
    # Översätter ord till siffror
    siffra_map = {'en': '1', 'ett': '1', 'två': '2', 'tre': '3', 'fyra': '4', 'fem': '5'}
    for ord, nr in siffra_map.items():
        #Går igenom varje ord i mappen
        renad = renad.str.replace(ord, nr)
    
    # Rensa bort allt utom siffror, punkt och komma
    renad = renad.str.replace(r'[^\d,.]', '', regex=True)
    
    # Fixa decimaler och gör till siffror
    return pd.to_numeric(renad.str.replace(',', '.'), errors='coerce').fillna(0)

# Använd på kolumnerna
# Här kallar du på den för dina huvudkolumner
for col in ['vikt_gram', 'pris', 'antal']:
    if col in df_clean.columns:
        df_clean[col] = clean_siffra(df_clean[col])



def clean_datum(datum):
    # Hantera tomma rutor (nullvärden)
    if pd.isna(datum) or str(datum).lower() == 'nan':
        return pd.NaT
    
    # Tvätta bort mellanslag och gör allt till små bokstäver
    d = str(datum).lower().strip()
    
    # Om datumet börjar på "20" (t.ex. 2024-01-04) tvingar vi formatet ÅÅÅÅ-MM-DD
    if d.startswith('20'):
        return pd.to_datetime(d, errors='coerce')

    # Översätt svenska månader till engelska
    manader = {
       'januari': 'january', 'februari': 'february', 'mars': 'march',
        'april': 'april', 'maj': 'may', 'juni': 'june',
        'juli': 'july', 'augusti': 'august', 'september': 'september',
        'oktober': 'october', 'november': 'november', 'december': 'december'
    }
    
    for sv, en in manader.items():
        if sv in d:
            d = d.replace(sv, en)
    
    # Omvandlar till datetime
    return pd.to_datetime(d, dayfirst=True, errors='coerce')

# --- 2. KÖR DATUMTVÄTTEN ---
for col in ['orderdatum', 'faktiskt_leveransdatum', 'recensionsdatum']:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(clean_datum)


def clean_product(namn):
    if pd.isna(namn):
        return "Okänd produkt"
    
    n = str(namn).lower().strip()
    
    # --- GRUPPERINGAR ---
    if 'basilika' in n or 'bassil' in n:
        return 'Basilika planta'
    
    # Samla ALL tomat-logik här
    if 'tom' in n: 
        if 'planta' in n:
            return 'Tomatplanta (olika sorter)'
        if 'biff' in n:
            return 'Bifftomat frön'
        if 'tiny' in n or 'tim' in n:
            return 'Tomat Tiny Tim frön'
        return 'Tomatfrön (standard)'

    if 'petunia' in n or 'petunior' in n:
        return 'Petunia mix'
        
    if 'narciss' in n or 'påsk' in n:
        return 'Påskliljor/Narcisser 15-pack'
        
    if 'tulpan' in n:
        return 'Tulpanlökar 20-pack'
        
    if 'vattenkanna' in n or 'vatten kanna' in n:
        return 'Vattenkanna 10L'
        
    if 'pelargon' in n:
        return 'Pelargon röd'
        
    if 'balkong' in n or 'plastlåda' in n:
        return 'Balkonglåda plast'
        
    if 'jord' in n:
        if '25' in n: return 'Blomjord 25L'
        if '50' in n: return 'Blomjord 50L'
        return 'Blomjord'
        
    if 'kruka' in n or 'terrakotta' in n: # Lade till terrakotta här!
        if '15' in n: return 'Lerkruka 15cm'
        if '25' in n: return 'Lerkruka 25cm'
        return 'Lerkruka'
        
    if 'sallat' in n or 'sallad' in n:
        return 'Sallad mix frön'
        
    if 'ört' in n or 'persilja' in n or 'dill' in n:
        return 'Örter mix'
        
    if 'monstera' in n:
        return 'Monstera'
        
    if 'sommar' in n or 'sommarblommor' in n:
        return 'Sommarblommor mix'
        
    if 'freds' in n:
        return 'Fredskalla'
        
    if 'gull' in n:
        return 'Gullranka'
        
    if 'gurk' in n:
        return 'Gurkfrön'

    # Om inget matchar
    return n.capitalize()

# Kör funktionen på din df_clean
df_clean['produktnamn'] = df_clean['produktnamn'].apply(clean_product)

#Städa säsonger
def clean_säsong(v):
    v = str(v).lower()
    året_runt = ['alla', 'hela året', 'all year', 'året runt']
    våren = ['vår', 'spring', 'v2024', 'april', ]
    for ord in året_runt:
        if ord in v:
            return 'Året runt'
    for ord in våren:
        if ord in v:
            return 'Våren'
    if 'höst' in v:
        return 'Hösten'
    if 'förodling' in v:
        return 'Förodling'
    return 'Okänd'
df_clean['säsong'] = df_clean['säsong'].apply(clean_säsong) 

#Städa zoner

def clean_zon(v):
    v = str(v).strip().lower()
    
    # Prioritera namn för presentationen
    if '1' in v or 'storstad' in v: return 'Storstad (Zon 1)'
    if '2' in v or 'södra' in v: return 'Södra Sverige (Zon 2)'
    if '3' in v or 'mellan' in v: return 'Mellansverige (Zon 3)'
    if '4' in v: return 'Längre norrut (Zon 4)'
    if '5' in v or 'norr' in v: return 'Norrland (Zon 5)'
    
    return 'Okänd zon'

df_clean['leveranszon'] = df_clean['leveranszon'].apply(clean_zon)


#Hantera nullvärden
#Fixa recensionerna (Text, Datum och Betyg)
df_clean['recension_text'] = df_clean['recension_text'].fillna('Ingen recension')
df_clean['recensionsdatum'] = df_clean['recensionsdatum'].fillna('Ej tillgängligt')
df_clean['betyg'] = df_clean['betyg'].fillna(0)

#Fixa önskat leveransdatum (Kopiera faktiskt datum)
df_clean['önskat_leveransdatum'] = df_clean['önskat_leveransdatum'].fillna(df_clean['faktiskt_leveransdatum'])

#Fixa säsong
df_clean['säsong'] = df_clean['säsong'].fillna('Okänd')

# Räkna ut dagarna
df_clean['leveranstid_dagar'] = (df_clean['faktiskt_leveransdatum'] - df_clean['orderdatum']).dt.days
# Gör alla dagar positiva
df_clean['leveranstid_dagar'] = df_clean['leveranstid_dagar'].abs()

# Sentiment analys
from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
def analyze_sentiment(text):
    # Om texten är "Ingen recension" eller tom, returnera direkt utan att köra AI:n
    if not text or text == 'Ingen recension' or str(text).lower() == 'nan':
        return 'Ingen recension'
    
    # Kör modellen bara på riktiga recensioner
    resultat = sentiment_model(str(text)[:510])[0]
    label = resultat['label']
    
    if label in ['1 star', '2 stars']:
        return 'Negativ'
    elif label == '3 stars':
        return 'Neutral'
    else:
        return 'Positiv'

# Kör analysen igen på din städade df_clean
df_clean['recensionssentiment'] = df_clean['recension_text'].apply(analyze_sentiment)

df_val = pd.read_csv('../gronagarden_validation.csv')

# Hantera nullvärden i valideringsdata
df_val['recension_text'] = df_val['recension_text'].fillna('Ingen recension')
df_val['säsong'] = df_val['säsong'].fillna('Okänd')

#Tvätta siffror
for col in ['vikt_gram', 'pris', 'antal']:
    if col in df_val.columns:
        df_val[col] = clean_siffra(df_val[col])

# Analysera sentiment
df_val['recensionssentiment'] = df_val['recension_text'].apply(analyze_sentiment)

# Spara till SQL
conn = sql.connect('gronagarden_data_cleaned.db')
df_val.to_sql('validation_results', conn, if_exists='replace', index=False)
conn.close()

# Visa resultatet
print("Validering klar!")
print(df_val[['recension_text', 'recensionssentiment']].sample(n=15))
df_clean = df_clean.map(lambda x: str(x) if isinstance(x, pd.Timestamp) else x)
# Anslut till databasen
conn = sql.connect('../gronagarden_data_cleaned.db')

# 2. Spara din df_clean till en tabell som heter 'orders'
df_clean.to_sql('orders', conn, if_exists='replace', index=False)

# 3. Stäng anslutningen
conn.close()
print("Datan är nu sparad i SQL databasen :)")