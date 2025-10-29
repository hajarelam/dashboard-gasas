import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Chargement des secrets + config Streamlit
# ──────────────────────────────────────────────────────────────────────────────
try:
    credentials = st.secrets["credentials"]
    ksaar_config = st.secrets["ksaar_config"]
    st.set_page_config(**ksaar_config.get('app_config', {
        'page_title': "Dashboard GASAS",
        'page_icon': "🎯",
        'layout': "wide",
        'initial_sidebar_state': "expanded"
    }))
except Exception:
    st.set_page_config(
        page_title="Dashboard GASAS",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime, timedelta
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import nltk
from textblob import TextBlob

# Debug local env (utile en dev)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# ──────────────────────────────────────────────────────────────────────────────
# HTTP session + helpers API
# ──────────────────────────────────────────────────────────────────────────────
def _build_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.6,  # 0.6, 1.2, 2.4, …
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

HTTP_SESSION = _build_session()

def _api_base():
    return ksaar_config['api_base_url'].rstrip('/')

def _ping_api():
    try:
        resp = HTTP_SESSION.get(f"{_api_base()}/health", timeout=(5, 5))
        return resp.status_code < 500
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────────────────────
# NLTK data
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk_data():
    try:
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        st.warning(f"Impossible de télécharger les ressources NLTK : {str(e)}")

download_nltk_data()

# ──────────────────────────────────────────────────────────────────────────────
# Regex / patterns
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def compile_abuse_patterns():
    abuse_keywords = [
        "sexe","bite","penis","vagin","masturb","bander","sucer","baiser",
        "jouir","éjacul","ejacul","orgasm","porno","cul","nichon","seins",
        "queue","pénis","zboub","chatte","cunni","branler","branlette","fap",
        "fellation","pipe","gode","godemichet","ken","niquer","niqué","niquee",
        "sodom","sodomie","anal","dp","orgie","orgasme","gicler","giclée",
        "cum","creampie","facial","porn","xxx","nichons","sein","boobs",
        "boobies","téton","tétons","nipple","photo","nue","nu","déshabille",
        "deshabille","montre-moi","montre moi","caméra","camera","vidéo","video",
        "snapchat","instagram","facebook","onlyfans","strip","striptease",
        "strip tease","connard","salope","pute","enculé","encule","pd","tapette",
        "nègre","negre","bougnoule","suicide","tuer","mourir","crever","adresse",
        "menace","frapper","battre","harcèle","harcele","stalker"
    ]
    pattern = re.compile('|'.join(map(re.escape, abuse_keywords)), re.IGNORECASE)
    return pattern, abuse_keywords

# ──────────────────────────────────────────────────────────────────────────────
# Utils UI / pagination
# ──────────────────────────────────────────────────────────────────────────────
def load_data_paginated(data, page_number, page_size):
    if data is None or data.empty:
        return pd.DataFrame()
    start_idx = page_number * page_size
    end_idx = start_idx + page_size
    if start_idx >= len(data):
        start_idx = 0
        st.session_state.page_number = 0
    end_idx = min(end_idx, len(data))
    return data.iloc[start_idx:end_idx].copy()

def display_pagination_controls(total_items, page_size, current_page, key_prefix=""):
    total_pages = (total_items + page_size - 1) // page_size
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if current_page > 0:
            if st.button("← Précédent", key=f"{key_prefix}prev_page"):
                if key_prefix == "calls_":
                    st.session_state.calls_page_number = current_page - 1
                else:
                    st.session_state.page_number = current_page - 1
                st.rerun()
    with col2:
        st.write(f"Page {current_page + 1} sur {total_pages}")
    with col3:
        if current_page < total_pages - 1:
            if st.button("Suivant →", key=f"{key_prefix}next_page"):
                if key_prefix == "calls_":
                    st.session_state.calls_page_number = current_page + 1
                else:
                    st.session_state.page_number = current_page + 1
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Mappings Nightline
# ──────────────────────────────────────────────────────────────────────────────
def extract_antenne(msg, dept):
    if pd.isna(msg) or pd.isna(dept) or msg is None or dept is None or msg == "" or dept == "":
        return "Inconnue"
    msg, dept = str(msg), str(dept)
    is_national = dept in ("Appels en attente (national)", "English calls (national)")
    if not is_national:
        return dept
    start_texts = [
        'as no operators online in "Nightline ',
        'from "Nightline ',
        'de "Nightline ',
        'en "Nightline '
    ]
    start_pos = None
    for text in start_texts:
        if text in msg:
            start_pos = msg.find(text) + len(text)
            break
    if start_pos is None:
        match = re.search(r'Nightline\s+([^"]+)', msg)
        if match:
            return match.group(1).strip()
        return "Inconnue"
    end_pos = msg.find('"', start_pos)
    if end_pos == -1:
        end_pos = msg.find('.', start_pos)
        if end_pos == -1:
            end_pos = len(msg)
    return msg[start_pos:end_pos].strip()

def get_normalized_antenne(antenne):
    if pd.isna(antenne) or antenne is None or antenne == "" or antenne == "Inconnue":
        return "Inconnue"
    if "Anglophone" in antenne:
        return "Paris - Anglophone"
    if "ANGERS" in antenne.upper() or "Angers" in antenne:
        return "Pays de la Loire"
    if antenne.startswith("Nightline "):
        return antenne.replace("Nightline ", "")
    return antenne

def get_operator_name(operator_id):
    if pd.isna(operator_id) or operator_id is None:
        return "Inconnu"
    try:
        operator_id = int(operator_id)
    except (ValueError, TypeError):
        return "Inconnu"
    operator_mapping = {
        1:"admin",2:"NightlineParis1",3:"NightlineParis2",4:"NightlineParis3",
        5:"NightlineParis4",6:"NightlineParis5",7:"NightlineLyon1",9:"NightlineParis6",
        12:"NightlineAnglophone1",13:"NightlineAnglophone2",14:"NightlineAnglophone3",
        16:"NightlineSaclay1",18:"NightlineSaclay3",19:"NightlineParis7",
        20:"NightlineParis8",21:"NightlineLyon2",22:"NightlineLyon3",
        26:"NightlineSaclay2",30:"NightlineSaclay4",31:"NightlineSaclay5",
        32:"NightlineSaclay6",33:"NightlineLyon4",34:"NightlineLyon5",
        35:"NightlineLyon6",36:"NightlineLyon7",37:"NightlineLyon8",
        38:"NightlineSaclay7",40:"NightlineParis9",42:"NightlineFormateur1",
        43:"NightlineAnglophone4",44:"NightlineAnglophone5",45:"NightlineParis10",
        46:"NightlineParis11",47:"NightlineToulouse1",48:"NightlineToulouse2",
        49:"NightlineToulouse3",50:"NightlineToulouse4",51:"NightlineToulouse5",
        52:"NightlineToulouse6",53:"NightlineToulouse7",54:"NightlineAngers1",
        55:"NightlineAngers2",56:"NightlineAngers3",57:"NightlineAngers4",
        58:"doubleecoute",59:"NightlineNantes1",60:"NightlineNantes2",
        61:"NightlineNantes3",62:"NightlineNantes4",63:"NightlineRouen1",
        64:"NightlineRouen2",65:"NightlineRouen3",67:"NightlineRouen4",
        68:"NightlineNantes5",69:"NightlineNantes6",70:"NightlineAngers5",
        71:"NightlineAngers6",72:"NightlineRouen5",73:"NightlineRouen6",
        74:"NightlineAngers7",75:"NightlineLyon9",76:"NightlineReims",
        77:"NightlineToulouse8",78:"NightlineToulouse9",79:"NightlineReims1",
        80:"NightlineReims2",81:"NightlineReims3",82:"NightlineReims4",
        83:"NightlineReims5",84:"NightlineLille1",85:"NightlineLille2",
        86:"NightlineLille3",87:"NightlineLille4",88:"NightlineRouen7",
        89:"NightlineRouen8",90:"NightlineRouen9",91:"NightlineRouen10",
        92:"NightlineRouen11",93:"NightlineRouen12"
    }
    return operator_mapping.get(operator_id, "Inconnu")

def get_volunteer_location(operator_name):
    if pd.isna(operator_name) or operator_name is None or operator_name == "":
        return "Autre"
    s = str(operator_name)
    if "NightlineAnglophone" in s: return "Paris_Ang"
    if "NightlineParis" in s: return "Paris"
    if "NightlineLyon" in s: return "Lyon"
    if "NightlineSaclay" in s: return "Saclay"
    if "NightlineToulouse" in s: return "Toulouse"
    if "NightlineAngers" in s: return "Angers"
    if "NightlineNantes" in s: return "Nantes"
    if "NightlineRouen" in s: return "Rouen"
    if "NightlineReims" in s: return "Reims"
    if "NightlineLille" in s: return "Lille"
    if "NightlineFormateur" in s: return "Formateur"
    if s == "admin": return "Admin"
    if s == "doubleecoute": return "Paris"
    return "Autre"

def get_antenne_from_dst(dst):
    if pd.isna(dst) or dst is None or dst == "":
        return None
    dst_str = str(dst).strip().replace("+","").replace(".0","").replace(" ","")
    if dst_str in ["33999011163","33999011065"]: return "Lille"
    if dst_str in ["33999011073"]: return "Marseille"
    if dst_str in ["33999011198","33999011066"]: return "Lyon"
    if dst_str in ["33999011201","33999011068"]: return "Paris"
    if dst_str in ["33999011263","33999011072"]: return "Toulouse"
    if dst_str in ["33999011261","33999011070"]: return "Reims"
    if dst_str in ["33999011199","33999011067"]: return "Normandie"
    if dst_str in ["33999011074"]: return "National_Fr_Hors_Zone"
    if dst_str in ["33999011262","33999011071"]: return "Saclay"
    if dst_str in ["33999011215","33999011069"]: return "Pays de la Loire"
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Data loading (Ksaar) — PATCHED (retries, timeouts, limit, max_pages)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=7200)
def get_ksaar_data(force_refresh=False, max_pages=None):
    try:
        cache_valid = (
            'chat_data' in st.session_state and
            'last_update' in st.session_state and
            (datetime.now() - st.session_state['last_update']).total_seconds() <= 300
        )
        if cache_valid and not force_refresh and not max_pages:
            return st.session_state['chat_data']

        workflow_id = "1500d159-5185-4487-be1f-fa18c6c85ec5"
        url = f"{_api_base()}/v1/workflows/{workflow_id}/records"
        auth = (ksaar_config['api_key_name'], ksaar_config['api_key_password'])

        all_records, current_page = [], 1
        LIMIT = 50
        PAGES_CAP = max_pages if isinstance(max_pages, int) and max_pages > 0 else float('inf')

        with st.spinner('Chargement des données…'):
            while current_page <= PAGES_CAP:
                params = {"page": current_page, "limit": LIMIT, "sort": "-createdAt"}
                try:
                    resp = HTTP_SESSION.get(url, params=params, auth=auth, timeout=(10, 90))
                except requests.exceptions.ReadTimeout:
                    st.warning(f"⏳ Page {current_page}: lecture trop longue (>90s). On passe à la suivante.")
                    current_page += 1
                    continue
                except requests.exceptions.ConnectTimeout:
                    st.error("Impossible de se connecter à l’API (timeout de connexion). Vérifie le réseau/VPN.")
                    break

                if resp.status_code != 200:
                    st.error(f"Erreur API (HTTP {resp.status_code}) sur page {current_page}")
                    break

                data = resp.json()
                records = data.get('results', [])
                if not records:
                    break

                for record in records:
                    record_data = {
                        'Crée le': record.get('createdAt'),
                        'Modifié le': record.get('updatedAt'),
                        'IP': record.get('IP 2', ''),
                        'pnd_time': record.get('Date complète début 2'),
                        'id_chat': record.get('Chat ID 2'),
                        'messages': record.get('Conversation complète 2', ''),
                        'last_user_message': record.get('Date complète fin 2'),
                        'last_op_message': record.get('Date complète début 2'),
                        'Message système 1': record.get('Message système 1', ''),
                        'Département Origine 2': record.get('Département Origine 2', '')
                    }

                    operator_id = record.get('Opérateur ID (API) 1')
                    operator_name = get_operator_name(operator_id)
                    record_data['Operateur_Name'] = operator_name
                    record_data['Volunteer_Location'] = get_volunteer_location(operator_name)

                    msg = record_data['Message système 1']
                    dept = record_data['Département Origine 2']
                    raw_antenne = extract_antenne(msg, dept)
                    record_data['Antenne'] = get_normalized_antenne(raw_antenne)

                    all_records.append(record_data)

                last_page = data.get('lastPage', current_page)
                if current_page >= last_page:
                    break
                current_page += 1

        if not all_records:
            st.warning("Aucun enregistrement renvoyé par l’API.")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        for col in ['Crée le','Modifié le','pnd_time','last_user_message','last_op_message']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        st.session_state['chat_data'] = df
        st.session_state['last_update'] = datetime.now()
        return df

    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=7200)
def get_calls_data(force_refresh=False):
    try:
        if 'calls_data' in st.session_state and not force_refresh:
            return st.session_state['calls_data']

        workflow_id = "deb92463-c3a5-4393-a3bf-1dd29a022cfe"
        url = f"{_api_base()}/v1/workflows/{workflow_id}/records"
        auth = (ksaar_config['api_key_name'], ksaar_config['api_key_password'])

        all_records, current_page = [], 1
        LIMIT = 50

        while True:
            params = {"page": current_page, "limit": LIMIT}
            try:
                resp = HTTP_SESSION.get(url, params=params, auth=auth, timeout=(10, 60))
            except requests.exceptions.Timeout:
                st.warning(f"⏳ Timeout page appels {current_page}, on continue.")
                current_page += 1
                continue

            if resp.status_code != 200:
                st.error(f"Erreur API appels (HTTP {resp.status_code}) page {current_page}")
                break

            data = resp.json()
            records = data.get('results', [])
            if not records:
                break

            def extract_time(timestamp):
                if timestamp:
                    try:
                        dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
                        return dt.strftime('%H:%M')
                    except:
                        return None
                return None

            for record in records:
                dst = record.get('dst', '')
                record_data = {
                    'Crée le': record.get('createdAt'),
                    'Nom': record.get('from_name', ''),
                    'Numéro': record.get('from_number', ''),
                    'Statut': record.get('disposition', ''),
                    'Code_de_cloture': record.get('Code_de_cloture', ''),
                    'Début appel': extract_time(record.get('answer')),
                    'Fin appel': extract_time(record.get('end')),
                    'dst': dst
                }
                antenne_from_dst = get_antenne_from_dst(dst)
                if antenne_from_dst:
                    record_data['Antenne'] = antenne_from_dst
                elif record.get('from_name'):
                    record_data['Antenne'] = get_normalized_antenne(record['from_name'])
                else:
                    record_data['Antenne'] = "Inconnue"
                all_records.append(record_data)

            last_page = data.get('lastPage', current_page)
            if current_page >= last_page:
                break
            current_page += 1

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        if not df.empty:
            df['Crée le'] = pd.to_datetime(df['Crée le'], errors='coerce')
            df = df[df['Crée le'] >= '2025-01-01']

        st.session_state['calls_data'] = df
        return df

    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API: {str(e)}")
        return pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# Auth simple (formulaire)
# ──────────────────────────────────────────────────────────────────────────────
def check_password():
    if 'authenticated' in st.session_state:
        return True

    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if username in credentials and password == credentials[username]:
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                if not _ping_api():
                    st.warning("🛜 API /health non joignable rapidement. Je continuerai avec des retries.")
                st.rerun()
            else:
                st.error("😕 Identifiants incorrects")
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Analyse / NLP helpers
# ──────────────────────────────────────────────────────────────────────────────
def analyze_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return "Positif"
    elif polarity < -0.3:
        return "Négatif"
    else:
        return "Neutre"

def generate_simple_summary(messages):
    user_msgs = extract_user_messages(messages)
    if len(user_msgs) <= 5:
        return "Résumé non généré (peu de messages)"
    else:
        return " ".join(user_msgs[:2]) + " [...] " + user_msgs[-1]

def cluster_chats(df, n_clusters=5):
    messages = df['messages'].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='french')
    X = vectorizer.fit_transform(messages)
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['Thème'] = model.fit_predict(X)
    return df

def extract_user_messages(messages):
    if pd.isna(messages) or messages is None:
        return []
    messages = str(messages)
    user_messages, current_message = [], ""
    is_user_message = False
    for line in messages.split('\n'):
        if line.strip().startswith('User:'):
            if current_message and is_user_message:
                user_messages.append(current_message.strip())
            current_message = line.replace('User:', '').strip()
            is_user_message = True
        elif line.strip().startswith('Operator:'):
            if current_message and is_user_message:
                user_messages.append(current_message.strip())
            current_message = ""
            is_user_message = False
        elif is_user_message and line.strip():
            current_message += " " + line.strip()
    if current_message and is_user_message:
        user_messages.append(current_message.strip())
    return user_messages

def extract_operator_messages(messages):
    if pd.isna(messages) or messages is None:
        return []
    messages = str(messages)
    operator_messages, current_message = [], ""
    is_operator_message = False
    for line in messages.split('\n'):
        if line.strip().startswith('Operator:'):
            if current_message and is_operator_message:
                operator_messages.append(current_message.strip())
            current_message = line.replace('Operator:', '').strip()
            is_operator_message = True
        elif line.strip().startswith('User:'):
            if current_message and is_operator_message:
                operator_messages.append(current_message.strip())
            current_message = ""
            is_operator_message = False
        elif is_operator_message and line.strip():
            current_message += " " + line.strip()
    if current_message and is_operator_message:
        operator_messages.append(current_message.strip())
    return operator_messages

def detect_topic_changes(user_messages, threshold=0.2, min_messages=5):
    if len(user_messages) < min_messages:
        return []
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='french')
    try:
        X = vectorizer.fit_transform(user_messages)
    except Exception:
        return []
    topic_changes = []
    for i in range(1, len(user_messages)):
        similarity = cosine_similarity(X[i:i+1], X[i-1:i])[0][0]
        if similarity < threshold:
            topic_changes.append({
                'index': i,
                'previous_message': user_messages[i-1],
                'current_message': user_messages[i],
                'similarity_score': similarity
            })
    return topic_changes

def detect_manipulation_patterns(messages):
    if pd.isna(messages) or messages is None:
        return []
    messages = str(messages)
    patterns = []

    insistence_keywords = [
        "s'il te plait","stp","svp","je t'en prie","je t'en supplie",
        "allez","aller","réponds","reponds","répond","repond"
    ]
    insistence_count = sum(1 for keyword in insistence_keywords if keyword in messages.lower())
    if insistence_count >= 3:
        patterns.append({
            'type': "Insistance excessive",
            'description': "Utilisation répétée de formules d'insistance",
            'occurrences': insistence_count
        })

    guilt_keywords = [
        "tu ne veux pas m'aider","tu refuses de m'aider","tu ne veux pas me répondre",
        "tu m'ignores","tu ne comprends pas","tu ne fais pas d'effort",
        "c'est de ta faute","à cause de toi","par ta faute"
    ]
    guilt_messages = [line for line in messages.split('\n') if any(k in line.lower() for k in guilt_keywords)]
    if guilt_messages:
        patterns.append({
            'type': "Culpabilisation",
            'description': "Tentatives de faire culpabiliser l'opérateur",
            'occurrences': len(guilt_messages),
            'examples': guilt_messages[:3]
        })

    threat_keywords = [
        "tu vas voir","tu regretteras","tu le regretteras","tu vas le regretter",
        "je vais me plaindre","je vais le dire","je sais où","je peux te trouver"
    ]
    threat_messages = [line for line in messages.split('\n') if any(k in line.lower() for k in threat_keywords)]
    if threat_messages:
        patterns.append({
            'type': "Menaces voilées",
            'description': "Utilisation de menaces indirectes",
            'occurrences': len(threat_messages),
            'examples': threat_messages[:3]
        })
    return patterns

def analyze_chat_content(messages):
    if pd.isna(messages) or messages is None or messages == "":
        return 0, [], {}, False, [], []
    messages = str(messages)
    risk_score = 0
    risk_factors = []
    problematic_phrases = {}
    operator_harassment = False

    user_messages = extract_user_messages(messages)
    operator_messages = extract_operator_messages(messages)
    if not user_messages:
        return 0, [], {}, False, [], []

    trauma_narrative_indicators = [
        "quand j'étais","dans mon enfance","j'ai été victime",
        "j'ai subi","on m'a fait","je me souviens","flashback",
        "souvenir","traumatisme","j'ai été agressé","harcelé"
    ]
    is_trauma_narrative = any(ind in messages.lower() for ind in trauma_narrative_indicators)

    mental_health_indicators = [
        "voix dans ma tête","j'entends des voix","hallucination",
        "trouble dissociatif","TDI","schizophrénie","dépression",
        "anxiété","psychiatrie","hospitalisation","thérapie"
    ]
    is_mental_health_discussion = any(ind in messages.lower() for ind in mental_health_indicators)

    suicidal_keywords = [
        "suicide","me tuer","mourir","en finir","plus envie de vivre",
        "mettre fin à mes jours","me suicider","disparaître","plus la force"
    ]
    suicidal_messages = [m for m in user_messages if any(k in m.lower() for k in suicidal_keywords)]
    if suicidal_messages:
        risk_score += 40
        risk_factors.append(f"Pensées suicidaires ({len(suicidal_messages)} occurrences)")
        problematic_phrases["Pensées suicidaires"] = suicidal_messages[:3]

    sexual_harassment_keywords = [
        "tu aimes le sexe","t'aimes sucer","tu veux baiser",
        "tu es excité","tu bandes","tu mouilles","tu te masturbes"
    ]
    harassment_messages = [m for m in user_messages if any(k in m.lower() for k in sexual_harassment_keywords)]
    if harassment_messages:
        operator_harassment = True
        risk_score += 50
        risk_factors.append(f"Harcèlement sexuel ({len(harassment_messages)} occurrences)")
        problematic_phrases["Harcèlement sexuel"] = harassment_messages[:3]

    manipulation_patterns = detect_manipulation_patterns(messages)
    if manipulation_patterns:
        risk_score += len(manipulation_patterns) * 10
        risk_factors.append(f"Patterns de manipulation ({len(manipulation_patterns)} détectés)")

    topic_changes = detect_topic_changes(user_messages)
    if len(topic_changes) > 2:
        risk_score += min(len(topic_changes) * 5, 20)
        risk_factors.append(f"Changements de sujet fréquents ({len(topic_changes)} détectés)")

    if is_trauma_narrative and not operator_harassment:
        risk_score = int(risk_score * 0.7)
        risk_factors.append("Score ajusté: récit de traumatisme")

    if is_mental_health_discussion and not operator_harassment:
        risk_score = int(risk_score * 0.8)
        risk_factors.append("Score ajusté: discussion santé mentale")

    risk_score = min(int(risk_score), 100)
    return risk_score, risk_factors, problematic_phrases, operator_harassment, manipulation_patterns, topic_changes

def get_abuse_risk_level(score):
    if score >= 80: return "Très élevé"
    if score >= 60: return "Élevé"
    if score >= 40: return "Modéré"
    if score >= 20: return "Faible"
    return "Très faible"

# ──────────────────────────────────────────────────────────────────────────────
# Génération de rapports
# ──────────────────────────────────────────────────────────────────────────────
def generate_chat_report(chat_data):
    antenne = chat_data.get('Antenne', 'Inconnue')
    volunteer_location = chat_data.get('Volunteer_Location', 'Inconnu')
    html_template = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .chat-info {{ margin: 20px 0; }}
        .messages {{ white-space: pre-wrap; background-color: white; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>Rapport de Chat Nightline</h2>
        <p><strong>ID Chat:</strong> {chat_data['id_chat']}</p>
        <p><strong>IP:</strong> {chat_data['IP']}</p>
        <p><strong>Date:</strong> {chat_data['Crée le'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['Crée le']) else 'N/A'}</p>
        <p><strong>Antenne:</strong> {antenne}</p>
        <p><strong>Bénévole:</strong> {volunteer_location}</p>
    </div>
    <div class="chat-info">
        <p><strong>Temps d'attente:</strong> {chat_data['pnd_time'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['pnd_time']) else 'N/A'}</p>
        <p><strong>Dernier message utilisateur:</strong> {chat_data['last_user_message'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['last_user_message']) else 'N/A'}</p>
        <p><strong>Dernier message opérateur:</strong> {chat_data['last_op_message'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['last_op_message']) else 'N/A'}</p>
    </div>
    <div class="messages">
        <h3>Messages:</h3>
        {chat_data['messages']}
    </div>
</body>
</html>
"""
    return html_template

def generate_chat_report_txt(chat_data):
    antenne = chat_data.get('Antenne', 'Inconnue')
    volunteer_location = chat_data.get('Volunteer_Location', 'Inconnu')
    txt_content = f"""
RAPPORT DE CHAT NIGHTLINE
=========================

ID Chat: {chat_data['id_chat']}
IP: {chat_data['IP']}
Date: {chat_data['Crée le'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['Crée le']) else 'N/A'}
Antenne: {antenne}
Bénévole: {volunteer_location}

INFORMATIONS SUPPLÉMENTAIRES
===========================
Temps d'attente: {chat_data['pnd_time'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['pnd_time']) else 'N/A'}
Dernier message utilisateur: {chat_data['last_user_message'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['last_user_message']) else 'N/A'}
Dernier message opérateur: {chat_data['last_op_message'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['last_op_message']) else 'N/A'}

MESSAGES
========
{chat_data['messages']}
"""
    return txt_content

def generate_chat_report_csv(chat_data):
    import io, csv
    antenne = chat_data.get('Antenne', 'Inconnue')
    volunteer_location = chat_data.get('Volunteer_Location', 'Inconnu')
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Champ', 'Valeur'])
    writer.writerow(['ID Chat', chat_data['id_chat']])
    writer.writerow(['IP', chat_data['IP']])
    writer.writerow(['Date', chat_data['Crée le'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['Crée le']) else 'N/A'])
    writer.writerow(['Antenne', antenne])
    writer.writerow(['Bénévole', volunteer_location])
    writer.writerow(['Temps d\'attente', chat_data['pnd_time'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['pnd_time']) else 'N/A'])
    writer.writerow(['Dernier message utilisateur', chat_data['last_user_message'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['last_user_message']) else 'N/A'])
    writer.writerow(['Dernier message opérateur', chat_data['last_op_message'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['last_op_message']) else 'N/A'])
    writer.writerow([])
    writer.writerow(['MESSAGES'])
    messages = str(chat_data['messages']).split('\n')
    for message in messages:
        writer.writerow([message])
    csv_content = output.getvalue()
    output.close()
    return csv_content

# ──────────────────────────────────────────────────────────────────────────────
# Détection abus – pipeline rapide
# ──────────────────────────────────────────────────────────────────────────────
def identify_potentially_abusive_chats(df):
    if df.empty:
        return pd.DataFrame()
    pattern, abuse_keywords = compile_abuse_patterns()
    df = df.copy()
    df['potentially_abusive'] = df['messages'].apply(
        lambda x: bool(pattern.search(str(x))) if not pd.isna(x) else False
    )
    potentially_abusive_df = df[df['potentially_abusive']].copy()

    def count_keywords(message):
        if pd.isna(message):
            return 0
        message = str(message).lower()
        matches = pattern.findall(message)
        return len(matches)

    potentially_abusive_df['preliminary_score'] = potentially_abusive_df['messages'].apply(count_keywords)
    potentially_abusive_df = potentially_abusive_df.sort_values(by='preliminary_score', ascending=False)
    return potentially_abusive_df

# ──────────────────────────────────────────────────────────────────────────────
# UI — Appels
# ──────────────────────────────────────────────────────────────────────────────
def display_calls():
    df = get_calls_data()
    if df.empty:
        st.warning("Aucune donnée d'appel n'a pu être récupérée.")
        return

    st.subheader("Filtres")

    default_start_date = datetime(2025, 1, 1)
    default_end_date = datetime.now()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Date de début",
            value=default_start_date,
            min_value=datetime(2025, 1, 1).date(),
            max_value=default_end_date,
            key="calls_start_date"
        )
    with col2:
        end_date = st.date_input(
            "Date de fin",
            value=default_end_date,
            min_value=start_date,
            max_value=default_end_date,
            key="calls_end_date"
        )

    col1, col2 = st.columns(2)
    with col1:
        start_time = st.time_input('Heure de début', value=datetime.strptime('00:00', '%H:%M').time(), key="calls_start_time")
    with col2:
        end_time = st.time_input('Heure de fin', value=datetime.strptime('23:59', '%H:%M').time(), key="calls_end_time")

    col1, col2 = st.columns(2)
    with col1:
        statuts_uniques = sorted(df['Statut'].dropna().unique().tolist())
        statut_selectionne = st.multiselect('Statut', statuts_uniques, default=statuts_uniques, key="calls_statut")
    with col2:
        codes_cloture_uniques = df['Code_de_cloture'].fillna('(vide)').unique().tolist()
        codes_cloture_uniques = sorted([code for code in codes_cloture_uniques if code])
        code_cloture_selectionne = st.multiselect('Code de clôture', codes_cloture_uniques, default=codes_cloture_uniques, key="calls_code_cloture")

    st.divider()

    filters = {
        'start_date': start_date,
        'end_date': end_date,
        'start_time': start_time,
        'end_time': end_time,
        'statut': statut_selectionne,
        'code_cloture': code_cloture_selectionne
    }

    mask = (df['Crée le'].dt.date >= filters['start_date']) & (df['Crée le'].dt.date <= filters['end_date'])
    if filters['statut']:
        mask &= df['Statut'].isin(filters['statut'])
    if filters['code_cloture']:
        code_mask = df['Code_de_cloture'].fillna('(vide)').isin(filters['code_cloture'])
        mask &= code_mask

    filtered_df = df[mask].copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre total d'appels", len(filtered_df))
    with col2:
        st.metric("Période", f"{filters['start_date'].strftime('%d/%m/%Y')} - {filters['end_date'].strftime('%d/%m/%Y')}")
    with col3:
        st.metric("Plage horaire", f"{filters['start_time'].strftime('%H:%M')} - {filters['end_time'].strftime('%H:%M')}")

    if 'calls_page_number' not in st.session_state:
        st.session_state.calls_page_number = 0
    PAGE_SIZE = 50
    total_items = len(filtered_df)
    paginated_data = load_data_paginated(filtered_df, st.session_state.calls_page_number, PAGE_SIZE)
    paginated_data['Code_de_cloture'] = paginated_data['Code_de_cloture'].fillna('(vide)')

    edited_df = st.data_editor(
        paginated_data,
        use_container_width=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Sélectionner", default=False),
            "Crée le": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
            "Nom": st.column_config.TextColumn("Antenne"),
            "Numéro": st.column_config.TextColumn("Numéro"),
            "Statut": st.column_config.TextColumn("Statut"),
            "Code_de_cloture": st.column_config.TextColumn("Code de clôture"),
            "Début appel": st.column_config.TextColumn("Heure de début"),
            "Fin appel": st.column_config.TextColumn("Heure de fin")
        },
        hide_index=True,
        num_rows="dynamic"
    )

    display_pagination_controls(total_items, PAGE_SIZE, st.session_state.calls_page_number, key_prefix="calls_")

    if st.button("Analyser les appelsSelected"):
        selected_calls = edited_df[edited_df.get("select", False)].copy()
        if not selected_calls.empty:
            st.write("### Analyse des appelsSelected")
            for _, call in selected_calls.iterrows():
                st.write("#### Détails de l'appel")
                st.write(f"Date: {call.get('Crée le')}")
                st.write(f"Antenne: {call.get('Nom')}")
                st.write(f"Numéro: {call.get('Numéro')}")
                st.write(f"Statut: {call.get('Statut')}")
                st.write(f"Code de clôture: {call.get('Code_de_cloture')}")
                if pd.notnull(call.get('Début appel')):
                    st.write(f"Heure de début: {call.get('Début appel')}")
                if pd.notnull(call.get('Fin appel')):
                    st.write(f"Heure de fin: {call.get('Fin appel')}")
                st.write("---")

    if st.sidebar.button("Rafraîchir les données d'appels", key="refresh_calls"):
        if 'calls_data' in st.session_state:
            del st.session_state['calls_data']
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# UI — Chats abus
# ──────────────────────────────────────────────────────────────────────────────
def display_abuse_analysis():
    st.title("Analyse IA des chats potentiellement abusifs")
    if 'initial_load' not in st.session_state:
        st.session_state.initial_load = True
        df = get_ksaar_data(max_pages=5)  # premier chargement rapide
        st.info("💡 Chargement initial rapide (≈ 250 derniers chats). Utilisez les filtres ou rafraîchissez pour charger plus de données.")
    else:
        df = get_ksaar_data()

    if df.empty:
        st.warning("Aucune donnée de chat n'a pu être récupérée.")
        return

    st.subheader("Filtres")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Date de début", value=datetime(2025, 1, 1).date(), key="abuse_start_date")
    with col2:
        end_date = st.date_input("Date de fin", value=datetime.now().date(), key="abuse_end_date")

    use_time_filter = st.checkbox("Activer le filtre d'heure", key="use_time_filter")
    if use_time_filter:
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.time_input("Heure de début", value=datetime.strptime('00:00', '%H:%M').time(), key="abuse_start_time")
        with col2:
            end_time = st.time_input("Heure de fin", value=datetime.strptime('23:59', '%H:%M').time(), key="abuse_end_time")
    else:
        start_time = datetime.strptime('00:00', '%H:%M').time()
        end_time = datetime.strptime('23:59', '%H:%M').time()

    col1, col2 = st.columns(2)
    with col1:
        antennes = sorted(df['Antenne'].dropna().unique().tolist())
        selected_antenne = st.multiselect('Antennes', options=['Toutes'] + antennes, default='Toutes', key="abuse_filter_antennes")
    with col2:
        benevoles = sorted(df['Volunteer_Location'].dropna().unique().tolist())
        selected_benevole = st.multiselect('Bénévoles', options=['Tous'] + benevoles, default='Tous', key="abuse_filter_benevoles")

    col1, col2 = st.columns(2)
    with col1:
        search_text = st.text_input("Rechercher des mots dans les messages", key="abuse_search_text")
    with col2:
        search_id = st.text_input("Rechercher un chat par ID", key="search_chat_id")

    st.divider()

    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df['Crée le'].dt.date >= start_date) & (filtered_df['Crée le'].dt.date <= end_date)]

    if use_time_filter:
        def convert_to_time(dt):
            try:
                if pd.isna(dt):
                    return None
                return dt.time()
            except:
                return None
        filtered_df['time_obj'] = filtered_df['Crée le'].apply(convert_to_time)
        time_mask = pd.Series(True, index=filtered_df.index)
        valid_times = filtered_df['time_obj'].notna()
        if valid_times.any():
            if start_time > end_time:
                time_mask = valid_times & ((filtered_df['time_obj'] >= start_time) | (filtered_df['time_obj'] <= end_time))
            else:
                time_mask = valid_times & ((filtered_df['time_obj'] >= start_time) & (filtered_df['time_obj'] <= end_time))
        filtered_df = filtered_df[time_mask].drop('time_obj', axis=1)

    if 'Toutes' not in selected_antenne and selected_antenne:
        filtered_df = filtered_df[filtered_df['Antenne'].isin(selected_antenne)]
    if 'Tous' not in selected_benevole and selected_benevole:
        filtered_df = filtered_df[filtered_df['Volunteer_Location'].isin(selected_benevole)]
    if search_text:
        filtered_df = filtered_df[filtered_df['messages'].str.contains(search_text, case=False, na=False)]

    if search_id:
        try:
            sid = int(search_id)
            filtered_by_id = filtered_df[filtered_df['id_chat'] == sid]
            if not filtered_by_id.empty:
                filtered_df = filtered_by_id
                st.success(f"Chat ID {sid} trouvé.")
            else:
                st.warning(f"Aucun chat avec l'ID {sid} n'a été trouvé.")
        except ValueError:
            st.error("L'ID du chat doit être un nombre entier.")

    potentially_abusive_df = identify_potentially_abusive_chats(filtered_df)
    if potentially_abusive_df.empty:
        st.warning("Aucun chat potentiellement abusif n'a été détecté avec les filtres actuels.")
        return

    st.subheader("Statistiques des chats potentiellement abusifs")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Nombre total de chats filtrés", len(filtered_df))
    with col2:
        st.metric("Chats potentiellement abusifs", len(potentially_abusive_df))

    if len(potentially_abusive_df) > 0:
        with st.expander("Répartition par antenne et bénévole", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Par antenne")
                st.bar_chart(potentially_abusive_df['Antenne'].value_counts())
            with col2:
                st.subheader("Par bénévole")
                st.bar_chart(potentially_abusive_df['Volunteer_Location'].value_counts())

    st.subheader("Liste des chats potentiellement abusifs")
    potentially_abusive_df = potentially_abusive_df.copy()
    potentially_abusive_df['select'] = False

    edited_df = st.data_editor(
        potentially_abusive_df,
        column_config={
            "select": st.column_config.CheckboxColumn("Sélectionner", default=False),
            "id_chat": st.column_config.NumberColumn("ID Chat"),
            "Crée le": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
            "Antenne": st.column_config.TextColumn("Antenne"),
            "Volunteer_Location": st.column_config.TextColumn("Bénévole"),
            "preliminary_score": st.column_config.ProgressColumn(
                "Score préliminaire", format="%d", min_value=0, max_value=20
            ),
            "messages": st.column_config.TextColumn("Aperçu du message", width="large")
        },
        column_order=[
            "select","id_chat","Crée le","Antenne","Volunteer_Location","preliminary_score","messages"
        ],
        use_container_width=True,
        hide_index=True,
        height=400
    )

    if st.button("Analyser en détail les chatsSelected", key="analyze_selected"):
        selected_chats = edited_df[edited_df["select"]].copy()
        if selected_chats.empty:
            st.warning("Veuillez sélectionner au moins un chat pour l'analyse détaillée.")
        else:
            st.subheader("Analyse détaillée des chatsSelected")
            with st.spinner("Analyse détaillée en cours..."):
                detailed_results = []
                for _, chat in selected_chats.iterrows():
                    chat_id = chat.get('id_chat')
                    original_chat_data = df[df['id_chat'] == chat_id]
                    if original_chat_data.empty:
                        st.error(f"Impossible de trouver les données complètes pour le chat {chat_id}")
                        continue
                    messages = original_chat_data.iloc[0].get('messages', '')
                    try:
                        risk_score, risk_factors, problematic_phrases, operator_harassment, manipulation_patterns, topic_changes = analyze_chat_content(messages)
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse du chat {chat_id}: {str(e)}")
                        continue

                    phrases_text = ""
                    for category, phrases in problematic_phrases.items():
                        if phrases:
                            phrases_text += f"**{category}**:\n"
                            for phrase in phrases[:3]:
                                phrases_text += f"- {phrase}\n"
                            phrases_text += "\n"

                    manipulation_text = ""
                    if manipulation_patterns:
                        for pattern in manipulation_patterns:
                            manipulation_text += f"**{pattern['type']}**: {pattern['description']}\n"
                            manipulation_text += f"Occurrences: {pattern['occurrences']}\n"
                            if 'examples' in pattern and pattern['examples']:
                                manipulation_text += "Exemples:\n"
                                for example in pattern['examples'][:2]:
                                    manipulation_text += f"- {str(example)}\n"
                            manipulation_text += "\n"

                    result_dict = {
                        'id_chat': chat_id,
                        'Crée le': original_chat_data.iloc[0].get('Crée le'),
                        'Antenne': original_chat_data.iloc[0].get('Antenne'),
                        'Volunteer_Location': original_chat_data.iloc[0].get('Volunteer_Location'),
                        'Score de risque': risk_score,
                        'Niveau de risque': get_abuse_risk_level(risk_score),
                        'Facteurs de risque': ', '.join(risk_factors),
                        'Phrases problématiques': phrases_text,
                        'Harcèlement opérateur': "Oui" if operator_harassment else "Non",
                        'Analyse contextuelle': manipulation_text,
                        'Schémas de manipulation': len(manipulation_patterns) if manipulation_patterns else 0,
                        'Changements de sujet': len(topic_changes) if topic_changes else 0,
                        'messages': messages
                    }
                    detailed_results.append(result_dict)

                detailed_df = pd.DataFrame(detailed_results)
                if not detailed_df.empty:
                    detailed_df = detailed_df.sort_values(by='Score de risque', ascending=False)
                    st.dataframe(
                        detailed_df,
                        column_config={
                            "id_chat": st.column_config.NumberColumn("ID Chat"),
                            "Crée le": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
                            "Antenne": st.column_config.TextColumn("Antenne"),
                            "Volunteer_Location": st.column_config.TextColumn("Bénévole"),
                            "Score de risque": st.column_config.ProgressColumn("Score de risque", format="%d", min_value=0, max_value=100),
                            "Niveau de risque": st.column_config.TextColumn("Niveau de risque"),
                            "Facteurs de risque": st.column_config.TextColumn("Facteurs de risque"),
                            "Phrases problématiques": st.column_config.TextColumn("Phrases problématiques", width="large"),
                            "Harcèlement opérateur": st.column_config.TextColumn("Harcèlement opérateur"),
                            "Analyse contextuelle": st.column_config.TextColumn("Analyse contextuelle", width="large"),
                            "Schémas de manipulation": st.column_config.NumberColumn("Schémas de manipulation"),
                            "Changements de sujet": st.column_config.NumberColumn("Changements de sujet"),
                            "messages": st.column_config.TextColumn("Aperçu du message", width="medium")
                        },
                        use_container_width=True,
                        hide_index=True
                    )

                    selected_chat_id = st.selectbox(
                        "Sélectionner un chat pour voir les détails complets",
                        detailed_df['id_chat'].tolist(),
                        key="selected_detailed_chat"
                    )
                    if selected_chat_id:
                        selected_chat = detailed_df[detailed_df['id_chat'] == selected_chat_id].iloc[0]
                        with st.expander(f"Détails complets du chat {selected_chat_id}", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Date:** {selected_chat['Crée le'].strftime('%d/%m/%Y %H:%M') if pd.notnull(selected_chat['Crée le']) else 'N/A'}")
                            with col2:
                                st.write(f"**Antenne:** {selected_chat['Antenne']}")
                            with col3:
                                st.write(f"**Bénévole:** {selected_chat['Volunteer_Location']}")
                            st.write(f"**Score de risque:** {selected_chat['Score de risque']} ({selected_chat['Niveau de risque']})")
                            st.write(f"**Facteurs de risque:** {selected_chat['Facteurs de risque']}")
                            st.write(f"**Harcèlement envers l'opérateur:** {selected_chat['Harcèlement opérateur']}")
                            if selected_chat['Phrases problématiques']:
                                st.subheader("Phrases problématiques détectées")
                                st.markdown(selected_chat['Phrases problématiques'])
                            if selected_chat['Analyse contextuelle']:
                                st.subheader("Analyse contextuelle")
                                st.markdown(selected_chat['Analyse contextuelle'])
                            st.subheader("Contenu complet du chat")
                            st.text_area("Messages", value=selected_chat['messages'], height=400)

                            if st.button("Générer un rapport pour ce chat", key=f"generate_report_{selected_chat_id}"):
                                chat_data = df[df['id_chat'] == selected_chat_id].iloc[0]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    html_report = generate_chat_report(chat_data)
                                    st.download_button(
                                        label=f"Télécharger en HTML",
                                        data=html_report,
                                        file_name=f"rapport_chat_abusif_{selected_chat_id}.html",
                                        mime="text/html",
                                        key=f"download_html_{selected_chat_id}"
                                    )
                                with col2:
                                    txt_report = generate_chat_report_txt(chat_data)
                                    st.download_button(
                                        label=f"Télécharger en TXT",
                                        data=txt_report,
                                        file_name=f"rapport_chat_abusif_{selected_chat_id}.txt",
                                        mime="text/plain",
                                        key=f"download_txt_{selected_chat_id}"
                                    )
                                with col3:
                                    csv_report = generate_chat_report_csv(chat_data)
                                    st.download_button(
                                        label=f"Télécharger en CSV",
                                        data=csv_report,
                                        file_name=f"rapport_chat_abusif_{selected_chat_id}.csv",
                                        mime="text/csv",
                                        key=f"download_csv_{selected_chat_id}"
                                    )

    if st.button("Générer des rapports pour les chatsSelected", key="generate_reports_selected"):
        selected_chats = edited_df[edited_df["select"]].copy()
        if selected_chats.empty:
            st.warning("Veuillez sélectionner au moins un chat pour générer des rapports.")
        else:
            for i, chat in selected_chats.iterrows():
                chat_id = chat['id_chat']
                chat_data = df[df['id_chat'] == chat_id].iloc[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    html_report = generate_chat_report(chat_data)
                    st.download_button(
                        label=f"HTML - Chat {chat_id}",
                        data=html_report,
                        file_name=f"rapport_chat_{chat_id}.html",
                        mime="text/html",
                        key=f"download_html_{chat_id}_{i}"
                    )
                with col2:
                    txt_report = generate_chat_report_txt(chat_data)
                    st.download_button(
                        label=f"TXT - Chat {chat_id}",
                        data=txt_report,
                        file_name=f"rapport_chat_{chat_id}.txt",
                        mime="text/plain",
                        key=f"download_txt_{chat_id}_{i}"
                    )
                with col3:
                    csv_report = generate_chat_report_csv(chat_data)
                    st.download_button(
                        label=f"CSV - Chat {chat_id}",
                        data=csv_report,
                        file_name=f"rapport_chat_{chat_id}.csv",
                        mime="text/csv",
                        key=f"download_csv_{chat_id}_{i}"
                    )

# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if not check_password():
        return

    st.title("Dashboard GASAS")

    if st.sidebar.button("🔄 Rafraîchir", key="refresh_button"):
        for k in ['chat_data','calls_data','abuse_analysis_results','initial_load','last_update']:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    if st.sidebar.button("🚪 Déconnexion", key="logout_button"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    tab1, tab2 = st.tabs(["Appels", "Analyse IA des abus"])
    with tab1:
        display_calls()
    with tab2:
        display_abuse_analysis()

if __name__ == "__main__":
    main()
