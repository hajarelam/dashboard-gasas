import streamlit as st

# ─────────────────────────────────────────────
# Page config (doit être la 1ère commande Streamlit)
# ─────────────────────────────────────────────
CONFIG_LOADED = False
CONFIG_ERROR = None
try:
    credentials = st.secrets["credentials"]
    ksaar_config = st.secrets["ksaar_config"]
    CONFIG_LOADED = True
    st.set_page_config(**ksaar_config.get('app_config', {
        'page_title': "Dashboard GASAS",
        'page_icon': "📊",
        'layout': "wide",
        'initial_sidebar_state': "expanded"
    }))
except Exception as e:
    CONFIG_ERROR = str(e)
    st.set_page_config(
        page_title="Dashboard GASAS",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    credentials = {}
    ksaar_config = {
        'api_base_url': '',
        'api_key_name': '',
        'api_key_password': ''
    }

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
# nltk rendu optionnel (non utilisé ici)
# from textblob import TextBlob  # import conservé si tu veux la polarité simple

# ─────────────────────────────────────────────
# Utils réseau : session avec retry/timeout
# ─────────────────────────────────────────────
def make_session():
    s = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

# ─────────────────────────────────────────────
# Regex abus : compilée une seule fois
# ─────────────────────────────────────────────
@st.cache_resource
def compile_abuse_patterns():
    abuse_keywords = [
        "sexe","bite","penis","vagin","masturb","bander","sucer","baiser","jouir",
        "éjacul","ejacul","orgasm","porno","cul","nichon","seins","queue","pénis",
        "zboub","chatte","cunni","branler","branlette","fap","fellation","pipe",
        "gode","godemichet","ken","niquer","niqué","niquee","sodom","sodomie",
        "anal","dp","orgie","orgasme","gicler","giclée","cum","creampie","facial",
        "porn","xxx","nichons","sein","boobs","boobies","téton","tétons","nipple",
        "nue","nu","déshabille","deshabille","montre-moi","montre moi","caméra",
        "camera","vidéo","video","snapchat","instagram","facebook","onlyfans",
        "strip","striptease","strip tease",
        "connard","salope","pute","enculé","encule","pd","tapette","nègre","negre",
        "bougnoule","suicide","tuer","mourir","crever","adresse","menace","frapper",
        "battre","harcèle","harcele","stalker"
    ]
    pattern = re.compile('|'.join(map(re.escape, abuse_keywords)), re.IGNORECASE)
    return pattern

# ─────────────────────────────────────────────
# Helpers Data + affichage
# ─────────────────────────────────────────────
def load_data_paginated(data: pd.DataFrame, page_number: int, page_size: int) -> pd.DataFrame:
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
            if st.button("← Précédent", key=f"{key_prefix}prev"):
                st.session_state[key_prefix + "page_number"] = current_page - 1
                st.rerun()
    with col2:
        st.write(f"Page {current_page + 1} sur {max(total_pages,1)}")
    with col3:
        if current_page < total_pages - 1:
            if st.button("Suivant →", key=f"{key_prefix}next"):
                st.session_state[key_prefix + "page_number"] = current_page + 1
                st.rerun()

# ─────────────────────────────────────────────
# Mapping opérateurs / antennes
# ─────────────────────────────────────────────
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
        20:"NightlineParis8",21:"NightlineLyon2",22:"NightlineLyon3",26:"NightlineSaclay2",
        30:"NightlineSaclay4",31:"NightlineSaclay5",32:"NightlineSaclay6",
        33:"NightlineLyon4",34:"NightlineLyon5",35:"NightlineLyon6",36:"NightlineLyon7",
        37:"NightlineLyon8",38:"NightlineSaclay7",40:"NightlineParis9",
        42:"NightlineFormateur1",43:"NightlineAnglophone4",44:"NightlineAnglophone5",
        45:"NightlineParis10",46:"NightlineParis11",47:"NightlineToulouse1",
        48:"NightlineToulouse2",49:"NightlineToulouse3",50:"NightlineToulouse4",
        51:"NightlineToulouse5",52:"NightlineToulouse6",53:"NightlineToulouse7",
        54:"NightlineAngers1",55:"NightlineAngers2",56:"NightlineAngers3",
        57:"NightlineAngers4",58:"doubleecoute",59:"NightlineNantes1",60:"NightlineNantes2",
        61:"NightlineNantes3",62:"NightlineNantes4",63:"NightlineRouen1",64:"NightlineRouen2",
        65:"NightlineRouen3",67:"NightlineRouen4",68:"NightlineNantes5",69:"NightlineNantes6",
        70:"NightlineAngers5",71:"NightlineAngers6",72:"NightlineRouen5",73:"NightlineRouen6",
        74:"NightlineAngers7",75:"NightlineLyon9",76:"NightlineReims",77:"NightlineToulouse8",
        78:"NightlineToulouse9",79:"NightlineReims1",80:"NightlineReims2",
        81:"NightlineReims3",82:"NightlineReims4",83:"NightlineReims5",84:"NightlineLille1",
        85:"NightlineLille2",86:"NightlineLille3",87:"NightlineLille4",88:"NightlineRouen7",
        89:"NightlineRouen8",90:"NightlineRouen9",91:"NightlineRouen10",
        92:"NightlineRouen11",93:"NightlineRouen12"
    }
    return operator_mapping.get(operator_id, "Inconnu")

def get_volunteer_location(operator_name):
    if pd.isna(operator_name) or not operator_name:
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

def extract_antenne(msg, dept):
    if pd.isna(msg) or pd.isna(dept) or not msg or not dept:
        return "Inconnue"
    msg = str(msg); dept = str(dept)
    is_national = dept in ("Appels en attente (national)", "English calls (national)")
    if not is_national:
        return dept
    start_texts = [
        'as no operators online in "Nightline ',
        'from "Nightline ', 'de "Nightline ', 'en "Nightline '
    ]
    start_pos = None
    for t in start_texts:
        if t in msg:
            start_pos = msg.find(t) + len(t)
            break
    if start_pos is None:
        m = re.search(r'Nightline\s+([^"]+)', msg)
        return m.group(1).strip() if m else "Inconnue"
    end_pos = msg.find('"', start_pos)
    if end_pos == -1:
        end_pos = msg.find('.', start_pos)
        if end_pos == -1: end_pos = len(msg)
    return msg[start_pos:end_pos].strip()

def get_normalized_antenne(a):
    if pd.isna(a) or not a or a == "Inconnue":
        return "Inconnue"
    s = str(a)
    if "Anglophone" in s: return "Paris - Anglophone"
    if "ANGERS" in s.upper() or "Angers" in s: return "Pays de la Loire"
    if s.startswith("Nightline "): return s.replace("Nightline ", "")
    return s

def get_antenne_from_dst(dst):
    if pd.isna(dst) or not dst: return None
    s = str(dst).strip().replace("+","").replace(".0","").replace(" ","")
    if s in ["33999011163","33999011065"]: return "Lille"
    if s in ["33999011073"]: return "Marseille"
    if s in ["33999011198","33999011066"]: return "Lyon"
    if s in ["33999011201","33999011068"]: return "Paris"
    if s in ["33999011263","33999011072"]: return "Toulouse"
    if s in ["33999011261","33999011070"]: return "Reims"
    if s in ["33999011199","33999011067"]: return "Normandie"
    if s in ["33999011074"]: return "National_Fr_Hors_Zone"
    if s in ["33999011262","33999011071"]: return "Saclay"
    if s in ["33999011215","33999011069"]: return "Pays de la Loire"
    return None

# ─────────────────────────────────────────────
# API Ksaar : chargement incrémental + cache
# ─────────────────────────────────────────────
@st.cache_data(ttl=7200)  # 2h
def get_ksaar_data(max_pages: int = 3, force_refresh: bool = False):
    """
    Charge les chats depuis Ksaar, rapidement (max_pages) puis incrémental si besoin.
    """
    try:
        # cache court-circuit si présent en session et récent (≤5 min)
        cache_valid = ('chat_data' in st.session_state and
                       'last_update_chat' in st.session_state and
                       (datetime.now() - st.session_state['last_update_chat']).total_seconds() <= 300 and
                       not force_refresh)
        if cache_valid:
            return st.session_state['chat_data']

        workflow_id = "1500d159-5185-4487-be1f-fa18c6c85ec5"
        url = f"{ksaar_config['api_base_url']}/v1/workflows/{workflow_id}/records"
        auth = (ksaar_config['api_key_name'], ksaar_config['api_key_password'])
        session = make_session()

        all_records = []
        current_page = 1
        last_page = None

        with st.spinner(f'Chargement des données (jusqu’à {max_pages} pages)...'):
            while True:
                params = {"page": current_page, "limit": 100, "sort": "-createdAt"}
                r = session.get(url, params=params, auth=auth, timeout=20)
                if r.status_code != 200:
                    st.error(f"Erreur API Ksaar: {r.status_code}")
                    break

                data = r.json()
                records = data.get('results', [])
                last_page = data.get('lastPage', current_page)

                for rec in records:
                    record_data = {
                        'Crée le': rec.get('createdAt'),
                        'Modifié le': rec.get('updatedAt'),
                        'IP': rec.get('IP 2', ''),
                        'pnd_time': rec.get('Date complète début 2'),
                        'id_chat': rec.get('Chat ID 2'),
                        'messages': rec.get('Conversation complète 2', ''),
                        'last_user_message': rec.get('Date complète fin 2'),
                        'last_op_message': rec.get('Date complète début 2'),
                        'Message système 1': rec.get('Message système 1', ''),
                        'Département Origine 2': rec.get('Département Origine 2', '')
                    }
                    operator_id = rec.get('Opérateur ID (API) 1')
                    op_name = get_operator_name(operator_id)
                    record_data['Operateur_Name'] = op_name
                    record_data['Volunteer_Location'] = get_volunteer_location(op_name)

                    msg = record_data['Message système 1']
                    dept = record_data['Département Origine 2']
                    raw_ant = extract_antenne(msg, dept)
                    record_data['Antenne'] = get_normalized_antenne(raw_ant)

                    all_records.append(record_data)

                if current_page >= last_page or current_page >= max_pages:
                    break
                current_page += 1

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        for col in ['Crée le', 'Modifié le', 'pnd_time', 'last_user_message', 'last_op_message']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        st.session_state['chat_data'] = df
        st.session_state['last_update_chat'] = datetime.now()
        st.session_state['chat_last_page'] = last_page or 1
        st.session_state['chat_loaded_pages'] = current_page
        return df

    except Exception as e:
        st.error(f"Erreur Ksaar (chats): {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=7200)
def get_calls_data(force_refresh: bool = False):
    try:
        if 'calls_data' in st.session_state and not force_refresh:
            return st.session_state['calls_data']

        workflow_id = "deb92463-c3a5-4393-a3bf-1dd29a022cfe"
        url = f"{ksaar_config['api_base_url']}/v1/workflows/{workflow_id}/records"
        auth = (ksaar_config['api_key_name'], ksaar_config['api_key_password'])
        session = make_session()

        all_records = []
        current_page = 1

        while True:
            params = {"page": current_page, "limit": 200, "sort": "-createdAt"}
            r = session.get(url, params=params, auth=auth, timeout=20)
            if r.status_code != 200:
                st.error(f"Erreur API Ksaar: {r.status_code}")
                break

            data = r.json()
            records = data.get('results', [])
            if not records:
                break

            def extract_time(ts):
                if not ts:
                    return None
                try:
                    dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ')
                    return dt.strftime('%H:%M')
                except Exception:
                    return None

            for rec in records:
                dst = rec.get('dst', '')
                record_data = {
                    'Crée le': rec.get('createdAt'),
                    'Antenne': None,  # remplie plus bas
                    'Numéro': rec.get('from_number', ''),
                    'Statut': rec.get('disposition', ''),
                    'Code_de_cloture': rec.get('Code_de_cloture', ''),
                    'Début appel': extract_time(rec.get('answer')),
                    'Fin appel': extract_time(rec.get('end')),
                    'dst': dst
                }
                ant = get_antenne_from_dst(dst)
                record_data['Antenne'] = ant if ant else "Inconnue"
                all_records.append(record_data)

            if current_page >= data.get('lastPage', 1):
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
        st.error(f"Erreur Ksaar (appels): {str(e)}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
# Auth simple
# ─────────────────────────────────────────────
def check_password():
    if st.session_state.get('authenticated') is True:
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
                st.rerun()
            else:
                st.error("😕 Identifiants incorrects")
    return False

# ─────────────────────────────────────────────
# Fonctions analyse/exports (dates robustes)
# ─────────────────────────────────────────────
def identify_potentially_abusive_chats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pattern = compile_abuse_patterns()
    tmp = df.copy()
    tmp['potentially_abusive'] = tmp['messages'].apply(
        lambda x: bool(pattern.search(str(x))) if pd.notna(x) else False
    )

    def count_keywords(message):
        if pd.isna(message):
            return 0
        matches = pattern.findall(str(message))
        return len(matches)

    tmp = tmp[tmp['potentially_abusive']].copy()
    if tmp.empty:
        return tmp
    tmp['preliminary_score'] = tmp['messages'].apply(count_keywords)
    tmp = tmp.sort_values(by='preliminary_score', ascending=False)
    return tmp

def extract_user_messages(messages: str):
    if pd.isna(messages) or messages is None:
        return []
    user_messages, current, is_user = [], "", False
    for line in str(messages).split('\n'):
        s = line.strip()
        if s.startswith('User:'):
            if current and is_user: user_messages.append(current.strip())
            current = s.replace('User:', '').strip()
            is_user = True
        elif s.startswith('Operator:'):
            if current and is_user: user_messages.append(current.strip())
            current = ""; is_user = False
        elif is_user and s:
            current += " " + s
    if current and is_user:
        user_messages.append(current.strip())
    return user_messages

def detect_topic_changes(user_messages, threshold=0.2, min_messages=5):
    if len(user_messages) < min_messages:
        return []
    try:
        X = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='french').fit_transform(user_messages)
    except Exception:
        return []
    out = []
    for i in range(1, len(user_messages)):
        sim = cosine_similarity(X[i:i+1], X[i-1:i])[0][0]
        if sim < threshold:
            out.append(i)
    return out

def get_abuse_risk_level(score):
    return ("Très élevé" if score >= 80 else
            "Élevé" if score >= 60 else
            "Modéré" if score >= 40 else
            "Faible" if score >= 20 else
            "Très faible")

def generate_chat_report(chat_data: dict) -> str:
    date_val = chat_data.get('Crée le')
    date_txt = date_val.strftime('%d/%m/%Y %H:%M') if pd.notnull(date_val) else 'N/A'
    ip_txt = chat_data.get('IP', 'N/A')
    pnd = chat_data.get('pnd_time')
    pnd_txt = pnd.strftime('%d/%m/%Y %H:%M') if pd.notnull(pnd) else 'N/A'
    lum = chat_data.get('last_user_message')
    lum_txt = lum.strftime('%d/%m/%Y %H:%M') if pd.notnull(lum) else 'N/A'
    lom = chat_data.get('last_op_message')
    lom_txt = lom.strftime('%d/%m/%Y %H:%M') if pd.notnull(lom) else 'N/A'
    antenne = chat_data.get('Antenne', 'Inconnue')
    volunteer_location = chat_data.get('Volunteer_Location', 'Inconnu')
    messages = chat_data.get('messages', '')

    return f"""
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
        <p><strong>ID Chat:</strong> {chat_data.get('id_chat','N/A')}</p>
        <p><strong>IP:</strong> {ip_txt}</p>
        <p><strong>Date:</strong> {date_txt}</p>
        <p><strong>Antenne:</strong> {antenne}</p>
        <p><strong>Bénévole:</strong> {volunteer_location}</p>
    </div>
    <div class="chat-info">
        <p><strong>Temps d'attente:</strong> {pnd_txt}</p>
        <p><strong>Dernier message utilisateur:</strong> {lum_txt}</p>
        <p><strong>Dernier message opérateur:</strong> {lom_txt}</p>
    </div>
    <div class="messages">
        <h3>Messages:</h3>
        {messages}
    </div>
</body>
</html>
"""

def generate_chat_report_txt(chat_data: dict) -> str:
    date_val = chat_data.get('Crée le')
    date_txt = date_val.strftime('%d/%m/%Y %H:%M') if pd.notnull(date_val) else 'N/A'
    ip_txt = chat_data.get('IP', 'N/A')
    pnd = chat_data.get('pnd_time')
    pnd_txt = pnd.strftime('%d/%m/%Y %H:%M') if pd.notnull(pnd) else 'N/A'
    lum = chat_data.get('last_user_message')
    lum_txt = lum.strftime('%d/%m/%Y %H:%M') if pd.notnull(lum) else 'N/A'
    lom = chat_data.get('last_op_message')
    lom_txt = lom.strftime('%d/%m/%Y %H:%M') if pd.notnull(lom) else 'N/A'
    antenne = chat_data.get('Antenne', 'Inconnue')
    volunteer_location = chat_data.get('Volunteer_Location', 'Inconnu')
    messages = chat_data.get('messages', '')

    return f"""RAPPORT DE CHAT NIGHTLINE
=========================

ID Chat: {chat_data.get('id_chat','N/A')}
IP: {ip_txt}
Date: {date_txt}
Antenne: {antenne}
Bénévole: {volunteer_location}

INFORMATIONS SUPPLÉMENTAIRES
===========================
Temps d'attente: {pnd_txt}
Dernier message utilisateur: {lum_txt}
Dernier message opérateur: {lom_txt}

MESSAGES
========
{messages}
"""

def generate_chat_report_csv(chat_data: dict) -> str:
    import io, csv
    out = io.StringIO()
    w = csv.writer(out)
    date_val = chat_data.get('Crée le')
    date_txt = date_val.strftime('%d/%m/%Y %H:%M') if pd.notnull(date_val) else 'N/A'
    ip_txt = chat_data.get('IP', 'N/A')
    pnd = chat_data.get('pnd_time')
    pnd_txt = pnd.strftime('%d/%m/%Y %H:%M') if pd.notnull(pnd) else 'N/A'
    lum = chat_data.get('last_user_message')
    lum_txt = lum.strftime('%d/%m/%Y %H:%M') if pd.notnull(lum) else 'N/A'
    lom = chat_data.get('last_op_message')
    lom_txt = lom.strftime('%d/%m/%Y %H:%M') if pd.notnull(lom) else 'N/A'
    antenne = chat_data.get('Antenne', 'Inconnue')
    volunteer_location = chat_data.get('Volunteer_Location', 'Inconnu')
    messages = str(chat_data.get('messages','')).split('\n')

    w.writerow(['Champ','Valeur'])
    w.writerow(['ID Chat', chat_data.get('id_chat','N/A')])
    w.writerow(['IP', ip_txt])
    w.writerow(['Date', date_txt])
    w.writerow(['Antenne', antenne])
    w.writerow(['Bénévole', volunteer_location])
    w.writerow(["Temps d'attente", pnd_txt])
    w.writerow(['Dernier message utilisateur', lum_txt])
    w.writerow(['Dernier message opérateur', lom_txt])
    w.writerow([])
    w.writerow(['MESSAGES'])
    for m in messages:
        w.writerow([m])
    return out.getvalue()

# ─────────────────────────────────────────────
# UI : Appels
# ─────────────────────────────────────────────
def display_calls():
    df = get_calls_data()
    if df.empty:
        st.warning("Aucune donnée d'appel n'a pu être récupérée.")
        return

    st.subheader("Filtres")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Date de début", value=datetime(2025,1,1))
    with c2:
        end_date = st.date_input("Date de fin", value=datetime.now())

    c3, c4 = st.columns(2)
    with c3:
        start_time = st.time_input('Heure de début', value=datetime.strptime('00:00','%H:%M').time())
    with c4:
        end_time = st.time_input('Heure de fin', value=datetime.strptime('23:59','%H:%M').time())

    c5, c6 = st.columns(2)
    with c5:
        statuts = sorted(df['Statut'].dropna().unique().tolist())
        statut_selectionne = st.multiselect('Statut', statuts, default=statuts)
    with c6:
        codes = df['Code_de_cloture'].fillna('(vide)').unique().tolist()
        codes = sorted([c for c in codes if c])
        code_sel = st.multiselect('Code de clôture', codes, default=codes)

    mask = (df['Crée le'].dt.date >= start_date) & (df['Crée le'].dt.date <= end_date)
    if statut_selectionne:
        mask &= df['Statut'].isin(statut_selectionne)
    if code_sel:
        mask &= df['Code_de_cloture'].fillna('(vide)').isin(code_sel)

    filtered = df[mask].copy()

    k1, k2, k3 = st.columns(3)
    with k1: st.metric("Nombre d'appels", len(filtered))
    with k2: st.metric("Période", f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
    with k3: st.metric("Plage horaire", f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")

    key_prefix = "calls_"
    if key_prefix + "page_number" not in st.session_state:
        st.session_state[key_prefix + "page_number"] = 0
    PAGE_SIZE = 50
    total_items = len(filtered)
    paged = load_data_paginated(filtered, st.session_state[key_prefix + "page_number"], PAGE_SIZE)

    # Remplir vides pour affichage
    paged['Code_de_cloture'] = paged['Code_de_cloture'].fillna('(vide)')

    st.data_editor(
        paged,
        use_container_width=True,
        column_config={
            "Crée le": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
            "Antenne": st.column_config.TextColumn("Antenne"),
            "Numéro": st.column_config.TextColumn("Numéro"),
            "Statut": st.column_config.TextColumn("Statut"),
            "Code_de_cloture": st.column_config.TextColumn("Code de clôture"),
            "Début appel": st.column_config.TextColumn("Heure de début"),
            "Fin appel": st.column_config.TextColumn("Heure de fin")
        },
        hide_index=True,
        num_rows="dynamic"
    )

    display_pagination_controls(total_items, PAGE_SIZE, st.session_state[key_prefix + "page_number"], key_prefix=key_prefix)

    if st.sidebar.button("Rafraîchir les données d'appels"):
        if 'calls_data' in st.session_state: del st.session_state['calls_data']
        st.rerun()

# ─────────────────────────────────────────────
# UI : Analyse abus (chargement incrémental)
# ─────────────────────────────────────────────
def display_abuse_analysis():
    st.title("Analyse IA des chats potentiellement abusifs")

    # Chargement initial rapide (3 pages par défaut)
    if 'initial_load' not in st.session_state:
        st.session_state.initial_load = True
        df = get_ksaar_data(max_pages=3)
        st.info("💡 Chargement initial rapide (≈300 derniers chats). Utilisez « Charger plus » pour étendre.")
    else:
        df = get_ksaar_data(max_pages=st.session_state.get('chat_loaded_pages', 3))

    if df.empty:
        st.warning("Aucune donnée de chat n'a pu être récupérée.")
        return

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Date de début", value=datetime(2025,1,1))
        start_time = st.time_input("Heure de début", value=datetime.strptime('00:00','%H:%M').time())
    with c2:
        end_date = st.date_input("Date de fin", value=datetime.now())
        end_time = st.time_input("Heure de fin", value=datetime.strptime('23:59','%H:%M').time())

    c3, c4 = st.columns(2)
    with c3:
        antennes = sorted(df['Antenne'].dropna().unique().tolist())
        selected_antenne = st.multiselect('Antennes', options=['Toutes'] + antennes, default='Toutes')
    with c4:
        benevoles = sorted(df['Volunteer_Location'].dropna().unique().tolist())
        selected_benevole = st.multiselect('Bénévoles', options=['Tous'] + benevoles, default='Tous')

    search_text = st.text_input("Rechercher dans les messages")

    mask = (df['Crée le'].dt.date >= start_date) & (df['Crée le'].dt.date <= end_date)
    # filtre heure (gestion overnight)
    def to_time(x):
        try:
            return x.time()
        except Exception:
            return None
    tmp = df.copy()
    tmp['time_obj'] = tmp['Crée le'].apply(to_time)
    if start_time > end_time:
        mask &= tmp['time_obj'].notna() & ((tmp['time_obj'] >= start_time) | (tmp['time_obj'] <= end_time))
    else:
        mask &= tmp['time_obj'].notna() & (tmp['time_obj'] >= start_time) & (tmp['time_obj'] <= end_time)

    if 'Toutes' not in selected_antenne and selected_antenne:
        mask &= tmp['Antenne'].isin(selected_antenne)
    if 'Tous' not in selected_benevole and selected_benevole:
        mask &= tmp['Volunteer_Location'].isin(selected_benevole)
    if search_text:
        mask &= tmp['messages'].str.contains(search_text, case=False, na=False)

    filtered_df = tmp[mask].drop(columns=['time_obj'])

    k1,k2,k3 = st.columns(3)
    with k1: st.metric("Nombre total de chats", len(filtered_df))
    with k2: st.metric("Période", f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
    with k3: st.metric("Plage horaire", f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")

    potentially_abusive_df = identify_potentially_abusive_chats(filtered_df)
    if potentially_abusive_df.empty:
        st.warning("Aucun chat potentiellement abusif détecté avec ces filtres.")
        # Contrôle incrémental
        _incremental_loader()
        return

    # Sélection
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
            "preliminary_score": st.column_config.ProgressColumn("Score préliminaire", format="%d", min_value=0, max_value=20),
            "messages": st.column_config.TextColumn("Aperçu du message", width="large")
        },
        column_order=["select","id_chat","Crée le","Antenne","Volunteer_Location","preliminary_score","messages"],
        use_container_width=True, hide_index=True, height=420
    )

    # Export unitaire rapide
    if st.button("Générer des rapports pour les chats sélectionnés"):
        selected = edited_df[edited_df["select"]].copy()
        if selected.empty:
            st.warning("Veuillez sélectionner au moins un chat.")
        else:
            df_full = get_ksaar_data(max_pages=st.session_state.get('chat_loaded_pages',3))
            for i, row in selected.iterrows():
                cid = row.get('id_chat')
                full_row = df_full[df_full['id_chat'] == cid]
                if full_row.empty:
                    st.warning(f"Données complètes introuvables pour chat {cid}")
                    continue
                chat_data = full_row.iloc[0].to_dict()
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.download_button("HTML", data=generate_chat_report(chat_data),
                                       file_name=f"rapport_chat_{cid}.html", mime="text/html", key=f"dl_html_{cid}")
                with c2:
                    st.download_button("TXT", data=generate_chat_report_txt(chat_data),
                                       file_name=f"rapport_chat_{cid}.txt", mime="text/plain", key=f"dl_txt_{cid}")
                with c3:
                    st.download_button("CSV", data=generate_chat_report_csv(chat_data),
                                       file_name=f"rapport_chat_{cid}.csv", mime="text/csv", key=f"dl_csv_{cid}")

    # Contrôle incrémental
    _incremental_loader()

def _incremental_loader():
    # Bouton pour charger plus de pages sans tout recharger
    last_page = st.session_state.get('chat_last_page', None)
    loaded = st.session_state.get('chat_loaded_pages', 3)
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("🔽 Charger 3 pages de plus"):
            new_pages = loaded + 3
            st.session_state['chat_loaded_pages'] = new_pages
            # purge le cache de get_ksaar_data pour relire plus de pages
            get_ksaar_data.clear()
            get_ksaar_data(max_pages=new_pages, force_refresh=True)
            st.rerun()
    with colB:
        if last_page and loaded < last_page:
            st.caption(f"Pages chargées: {loaded}/{last_page}")
        else:
            st.caption(f"Pages chargées: {loaded} (toutes les pages disponibles ont été chargées si connu)")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    if not CONFIG_LOADED:
        st.error("⚠️ ERREUR DE CONFIGURATION DES SECRETS")
        st.error(f"Erreur: {CONFIG_ERROR}")
        st.info("Vérifie Settings > Secrets dans Streamlit.")
        with st.expander("📋 Format attendu des secrets", expanded=True):
            st.code("""
[credentials]
hajar = "hajar123"
admin = "admin123"

[ksaar_config]
api_base_url = "https://api.ksaar.co"
api_key_name = "votre_api_key_name"
api_key_password = "votre_api_key_password"

[ksaar_config.app_config]
page_title = "Dashboard GASAS"
page_icon = "📊"
layout = "wide"
initial_sidebar_state = "expanded"
""", language="toml")
        return

    with st.expander("🔍 Debug", expanded=False):
        st.success("✅ Secrets chargés")
        st.write(f"API Base URL: {ksaar_config.get('api_base_url','--')}")
        st.write(f"API Key Name configuré: {'Oui' if ksaar_config.get('api_key_name') else 'Non'}")
        st.write(f"API Key Password configuré: {'Oui' if ksaar_config.get('api_key_password') else 'Non'}")
        if st.button("🧪 Tester connexion API"):
            try:
                s = make_session()
                wf = "deb92463-c3a5-4393-a3bf-1dd29a022cfe"
                url = f"{ksaar_config['api_base_url']}/v1/workflows/{wf}/records"
                r = s.get(url, params={"page":1,"limit":1}, auth=(ksaar_config['api_key_name'], ksaar_config['api_key_password']), timeout=10)
                if r.status_code == 200:
                    st.success("✅ Connexion OK")
                else:
                    st.error(f"❌ Erreur: {r.status_code}")
                    st.write(r.text[:500])
            except Exception as e:
                st.error(f"❌ Exception: {e}")

    if not check_password():
        return

    st.title("Dashboard GASAS")

    if st.sidebar.button("🔄 Rafraîchir tout"):
        for k in ["chat_data","calls_data","abuse_analysis_results","initial_load","chat_loaded_pages","chat_last_page","last_update_chat"]:
            if k in st.session_state: del st.session_state[k]
        get_ksaar_data.clear()   # clear cache_data
        get_calls_data.clear()
        st.rerun()

    if st.sidebar.button("🚪 Déconnexion"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    tab1, tab2 = st.tabs(["Appels", "Analyse IA des abus"])
    with tab1:
        display_calls()
    with tab2:
        display_abuse_analysis()

if __name__ == "__main__":
    main()
