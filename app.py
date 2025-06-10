import streamlit as st

# Au lieu d'importer depuis config.py
try:
    credentials = st.secrets["credentials"]
    ksaar_config = st.secrets["ksaar_config"]
    # Configuration de la page doit être la première commande Streamlit
    st.set_page_config(**ksaar_config.get('app_config', {
        'page_title': "Dashboard GASAS",
        'page_icon': "🎯",
        'layout': "wide",
        'initial_sidebar_state': "expanded"
    }))
except Exception as e:
    # Configuration par défaut si les secrets ne sont pas disponibles
    st.set_page_config(
        page_title="Dashboard GASAS",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

import pandas as pd
import requests
from datetime import datetime, timedelta
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import nltk
from textblob import TextBlob

# Télécharger les ressources nécessaires
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
    except Exception as e:
        st.warning(f"Impossible de télécharger les ressources NLTK : {str(e)}")

# Initialiser les ressources
download_nltk_data()

# Fonction pour charger les données par pages
def load_data_paginated(data, page_number, page_size):
    """
    Charge une portion des données pour la pagination.
    
    Args:
        data (pd.DataFrame): Le DataFrame contenant toutes les données
        page_number (int): Le numéro de la page (commence à 0)
        page_size (int): Le nombre d'éléments par page
    
    Returns:
        pd.DataFrame: Une portion du DataFrame original correspondant à la page demandée
    """
    if data is None or data.empty:
        return pd.DataFrame()
        
    start_idx = page_number * page_size
    end_idx = start_idx + page_size
    
    # S'assurer que les index sont dans les limites du DataFrame
    if start_idx >= len(data):
        start_idx = 0
        st.session_state.page_number = 0
    
    end_idx = min(end_idx, len(data))
    
    return data.iloc[start_idx:end_idx].copy()

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# === AJOUT POUR ANALYSE ÉMOTIONNELLE, RÉSUMÉ ET CLUSTERING ===
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

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
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['Thème'] = model.fit_predict(X)
    return df

def check_password():
    """Retourne `True` si l'utilisateur a entré le bon mot de passe."""
    
    # Vérifier si l'utilisateur est déjà connecté avec une session valide
    if 'authenticated' in st.session_state:
        return True
    
    # Afficher le formulaire de connexion
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if username in credentials and password == credentials[username]:
                # Stocker l'authentification dans la session
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.rerun()
            else:
                st.error("😕 Identifiants incorrects")
    
    return False

def extract_antenne(msg, dept):
    """
    Extrait l'antenne à partir du message système et du département,
    en suivant exactement la logique de la formule PowerBI.
    """
    # Vérifier les valeurs nulles
    if pd.isna(msg) or pd.isna(dept) or msg is None or dept is None or msg == "" or dept == "":
        return "Inconnue"
    
    # Convertir en chaînes de caractères pour être sûr
    msg = str(msg)
    dept = str(dept)
    
    # Vérifier si c'est national
    is_national = dept == "Appels en attente (national)" or dept == "English calls (national)"
    
    if not is_national:
        return dept
    
    # Patterns à rechercher
    start_texts = [
        'as no operators online in "Nightline ',
        'from "Nightline ',
        'de "Nightline ',
        'en "Nightline '
    ]
    
    # Trouver la position de début
    start_pos = None
    for text in start_texts:
        if text in msg:
            start_pos = msg.find(text) + len(text)
            break
    
    # Si aucun pattern n'est trouvé
    if start_pos is None:
        # Essayer avec une expression régulière plus générale
        pattern = r'Nightline\s+([^"]+)'
        match = re.search(pattern, msg)
        if match:
            extracted = match.group(1).strip()
            return extracted
        
        return "Inconnue"
    
    # Trouver la position de fin (guillemet fermant)
    end_pos = msg.find('"', start_pos)
    
    # Si pas de guillemet fermant
    if end_pos == -1:
        # Essayer de prendre jusqu'à la fin de la ligne ou un point
        end_pos = msg.find('.', start_pos)
        if end_pos == -1:
            end_pos = len(msg)
    
    # Extraire l'antenne
    extracted_antenne = msg[start_pos:end_pos].strip()
    
    # Retourner l'antenne extraite si c'est national, sinon le département
    return extracted_antenne if is_national else dept

def get_normalized_antenne(antenne):
    """Normalise le nom de l'antenne selon les règles spécifiées."""
    if pd.isna(antenne) or antenne is None or antenne == "" or antenne == "Inconnue":
        return "Inconnue"
    
    # Traitement spécial pour certaines antennes
    if "Anglophone" in antenne:
        return "Paris - Anglophone"
    elif "ANGERS" in antenne.upper() or "Angers" in antenne:
        return "Pays de la Loire"
    elif antenne.startswith("Nightline "):
        return antenne.replace("Nightline ", "")
    else:
        return antenne

def get_operator_name(operator_id):
    """
    Convertit l'ID de l'opérateur en nom standardisé,
    en suivant exactement la logique de la formule PowerBI.
    """
    if pd.isna(operator_id) or operator_id is None:
        return "Inconnu"
    
    # Convertir en entier si possible
    try:
        operator_id = int(operator_id)
    except (ValueError, TypeError):
        return "Inconnu"
    
    # Mapping des IDs vers les noms d'opérateurs
    operator_mapping = {
        1: "admin",
        2: "NightlineParis1",
        3: "NightlineParis2",
        4: "NightlineParis3",
        5: "NightlineParis4",
        6: "NightlineParis5",
        7: "NightlineLyon1",
        9: "NightlineParis6",
        12: "NightlineAnglophone1",
        13: "NightlineAnglophone2",
        14: "NightlineAnglophone3",
        16: "NightlineSaclay1",
        18: "NightlineSaclay3",
        19: "NightlineParis7",
        20: "NightlineParis8",
        21: "NightlineLyon2",
        22: "NightlineLyon3",
        26: "NightlineSaclay2",
        30: "NightlineSaclay4",
        31: "NightlineSaclay5",
        32: "NightlineSaclay6",
        33: "NightlineLyon4",
        34: "NightlineLyon5",
        35: "NightlineLyon6",
        36: "NightlineLyon7",
        37: "NightlineLyon8",
        38: "NightlineSaclay7",
        40: "NightlineParis9",
        42: "NightlineFormateur1",
        43: "NightlineAnglophone4",
        44: "NightlineAnglophone5",
        45: "NightlineParis10",
        46: "NightlineParis11",
        47: "NightlineToulouse1",
        48: "NightlineToulouse2",
        49: "NightlineToulouse3",
        50: "NightlineToulouse4",
        51: "NightlineToulouse5",
        52: "NightlineToulouse6",
        53: "NightlineToulouse7",
        54: "NightlineAngers1",
        55: "NightlineAngers2",
        56: "NightlineAngers3",
        57: "NightlineAngers4",
        58: "doubleecoute",
        59: "NightlineNantes1",
        60: "NightlineNantes2",
        61: "NightlineNantes3",
        62: "NightlineNantes4",
        63: "NightlineRouen1",
        64: "NightlineRouen2",
        65: "NightlineRouen3",
        67: "NightlineRouen4",
        68: "NightlineNantes5",
        69: "NightlineNantes6",
        70: "NightlineAngers5",
        71: "NightlineAngers6",
        72: "NightlineRouen5",
        73: "NightlineRouen6",
        74: "NightlineAngers7",
        75: "NightlineLyon9",
        76: "NightlineReims",
        77: "NightlineToulouse8",
        78: "NightlineToulouse9",
        79: "NightlineReims1",
        80: "NightlineReims2",
        81: "NightlineReims3",
        82: "NightlineReims4",
        83: "NightlineReims5",
        84: "NightlineLille1",
        85: "NightlineLille2",
        86: "NightlineLille3",
        87: "NightlineLille4",
        88: "NightlineRouen7",
        89: "NightlineRouen8",
        90: "NightlineRouen9",
        91: "NightlineRouen10",
        92: "NightlineRouen11",
        93: "NightlineRouen12"
    }
    
    # Retourner le nom correspondant à l'ID, ou "Inconnu" si l'ID n'est pas dans le mapping
    return operator_mapping.get(operator_id, "Inconnu")

def get_volunteer_location(operator_name):
    """
    Détermine la localisation du bénévole à partir du nom d'opérateur standardisé,
    en suivant exactement la logique de la formule PowerBI.
    """
    if pd.isna(operator_name) or operator_name is None or operator_name == "":
        return "Autre"
    
    # Convertir en chaîne de caractères pour être sûr
    operator_name = str(operator_name)
    
    # Vérifier les différentes chaînes de caractères dans l'ordre exact
    if "NightlineAnglophone" in operator_name:
        return "Paris_Ang"
    elif "NightlineParis" in operator_name:
        return "Paris"
    elif "NightlineLyon" in operator_name:
        return "Lyon"
    elif "NightlineSaclay" in operator_name:
        return "Saclay"
    elif "NightlineToulouse" in operator_name:
        return "Toulouse"
    elif "NightlineAngers" in operator_name:
        return "Angers"
    elif "NightlineNantes" in operator_name:
        return "Nantes"
    elif "NightlineRouen" in operator_name:
        return "Rouen"
    elif "NightlineReims" in operator_name:
        return "Reims"
    elif "NightlineLille" in operator_name:
        return "Lille"
    elif "NightlineFormateur" in operator_name:
        return "Formateur"
    elif operator_name == "admin":
        return "Admin"
    elif operator_name == "doubleecoute":
        return "Paris"
    else:
        return "Autre"

# Ajout du cache pour les données
@st.cache_data(ttl=3600)  # Cache d'une heure
def get_ksaar_data():
    """Récupère les données depuis l'API Ksaar avec le bon workflow ID."""
    try:
        # Vérifier si les données sont en cache et si elles ont plus de 5 minutes
        if ('chat_data' not in st.session_state or 
            'last_update' not in st.session_state or 
            (datetime.now() - st.session_state['last_update']).total_seconds() > 300):
            
            # Utiliser le bon workflow ID pour les chats
            workflow_id = "1500d159-5185-4487-be1f-fa18c6c85ec5"
            url = f"{ksaar_config['api_base_url']}/v1/workflows/{workflow_id}/records"
            auth = (ksaar_config['api_key_name'], ksaar_config['api_key_password'])
            
            all_records = []
            current_page = 1
            
            try:
                while True:
                    params = {
                        "page": current_page,
                        "limit": 100,
                        "sort": "-createdAt"  # Tri par date décroissante
                    }

                    response = requests.get(url, params=params, auth=auth, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        records = data.get('results', [])
                        if not records:
                            break
                            
                        for record in records:
                            # Extraire les données de base en utilisant les noms exacts des colonnes
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
                            
                            # Utiliser l'ID de l'opérateur pour déterminer le nom standardisé
                            operator_id = record.get('Opérateur ID (API) 1')
                            operator_name = get_operator_name(operator_id)
                            
                            # Utiliser le nom standardisé pour déterminer la localisation du bénévole
                            record_data['Operateur_Name'] = operator_name
                            record_data['Volunteer_Location'] = get_volunteer_location(operator_name)
                            
                            # Ajouter l'antenne
                            msg = record_data['Message système 1']
                            dept = record_data['Département Origine 2']
                            raw_antenne = extract_antenne(msg, dept)
                            record_data['Antenne'] = get_normalized_antenne(raw_antenne)
                            
                            all_records.append(record_data)
                        
                        if current_page >= data.get('lastPage', 1):
                            break
                        current_page += 1
                    else:
                        st.error(f"Erreur lors de la récupération des données: {response.status_code}")
                        return pd.DataFrame()
            except Exception as e:
                st.error(f"Erreur lors de la connexion à l'API: {str(e)}")
                return pd.DataFrame()

            if not all_records:
                st.warning("Aucun enregistrement trouvé dans la réponse de l'API.")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_records)
            
            # Conversion des colonnes de date
            date_columns = ['Crée le', 'Modifié le', 'pnd_time', 'last_user_message', 'last_op_message']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Stocker dans le cache avec l'horodatage
            st.session_state['chat_data'] = df
            st.session_state['last_update'] = datetime.now()
        
        return st.session_state['chat_data']
            
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache d'une heure
def get_calls_data():
    """Récupère les données d'appels depuis l'API Ksaar."""
    try:
        if 'calls_data' not in st.session_state:
            # Utiliser le workflow ID pour les appels
            workflow_id = "deb92463-c3a5-4393-a3bf-1dd29a022cfe"
            url = f"{ksaar_config['api_base_url']}/v1/workflows/{workflow_id}/records"
            auth = (ksaar_config['api_key_name'], ksaar_config['api_key_password'])
            
            all_records = []
            current_page = 1
            
            while True:
                params = {
                    "page": current_page,
                    "limit": 100
                }

                response = requests.get(url, params=params, auth=auth)
                
                if response.status_code == 200:
                    data = response.json()
                    records = data.get('results', [])
                    if not records:
                        break
                    
                    for record in records:
                        # Extraire les heures des timestamps
                        def extract_time(timestamp):
                            if timestamp:
                                try:
                                    # Convertir le timestamp en datetime
                                    dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
                                    # Retourner uniquement l'heure au format HH:MM
                                    return dt.strftime('%H:%M')
                                except:
                                    return None
                            return None
                        
                        record_data = {
                            'Crée le': record.get('createdAt'),
                            'Nom': record.get('from_name', ''),
                            'Numéro': record.get('from_number', ''),
                            'Statut': record.get('disposition', ''),
                            'Début appel': extract_time(record.get('answer')),
                            'Fin appel': extract_time(record.get('end'))
                        }
                        
                        # Normaliser l'antenne pour les appels aussi
                        if 'from_name' in record and record['from_name']:
                            record_data['Antenne'] = get_normalized_antenne(record['from_name'])
                        else:
                            record_data['Antenne'] = "Inconnue"
                            
                        all_records.append(record_data)
                    
                    if current_page >= data.get('lastPage', 1):
                        break
                    current_page += 1
                else:
                    st.error(f"Erreur lors de la récupération des données: {response.status_code}")
                    return pd.DataFrame()

            if not all_records:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_records)
            
            if not df.empty:
                # Conversion de la colonne 'Crée le'
                df['Crée le'] = pd.to_datetime(df['Crée le'])
                
                # Filtrer les données à partir de janvier 2025
                df = df[df['Crée le'] >= '2025-01-01']
            
            st.session_state['calls_data'] = df
        
        return st.session_state['calls_data']
            
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API: {str(e)}")
        return pd.DataFrame()

def generate_chat_report(chat_data):
    """Génère un rapport HTML pour un chat avec les informations d'antenne et de bénévole."""
    # Récupérer l'antenne et la localisation du bénévole
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
            <p><strong>Date:</strong> {chat_data['Crée le'].strftime('%d/%m/%Y %H:%M')}</p>
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
    """Génère un rapport TXT pour un chat avec les informations d'antenne et de bénévole."""
    # Récupérer l'antenne et la localisation du bénévole
    antenne = chat_data.get('Antenne', 'Inconnue')
    volunteer_location = chat_data.get('Volunteer_Location', 'Inconnu')
    
    # Créer le contenu du rapport en format texte
    txt_content = f"""
RAPPORT DE CHAT NIGHTLINE
=========================

ID Chat: {chat_data['id_chat']}
IP: {chat_data['IP']}
Date: {chat_data['Crée le'].strftime('%d/%m/%Y %H:%M')}
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
    """Génère un rapport CSV pour un chat avec les informations d'antenne et de bénévole."""
    import io
    import csv
    
    # Récupérer l'antenne et la localisation du bénévole
    antenne = chat_data.get('Antenne', 'Inconnue')
    volunteer_location = chat_data.get('Volunteer_Location', 'Inconnu')
    
    # Créer un buffer pour stocker les données CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Écrire les en-têtes
    writer.writerow(['Champ', 'Valeur'])
    
    # Écrire les informations du chat
    writer.writerow(['ID Chat', chat_data['id_chat']])
    writer.writerow(['IP', chat_data['IP']])
    writer.writerow(['Date', chat_data['Crée le'].strftime('%d/%m/%Y %H:%M')])
    writer.writerow(['Antenne', antenne])
    writer.writerow(['Bénévole', volunteer_location])
    writer.writerow(['Temps d\'attente', chat_data['pnd_time'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['pnd_time']) else 'N/A'])
    writer.writerow(['Dernier message utilisateur', chat_data['last_user_message'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['last_user_message']) else 'N/A'])
    writer.writerow(['Dernier message opérateur', chat_data['last_op_message'].strftime('%d/%m/%Y %H:%M') if pd.notnull(chat_data['last_op_message']) else 'N/A'])
    
    # Ajouter une ligne vide
    writer.writerow([])
    
    # Écrire les messages
    writer.writerow(['MESSAGES'])
    
    # Diviser les messages par ligne et les écrire
    messages = str(chat_data['messages']).split('\n')
    for message in messages:
        writer.writerow([message])
    
    # Récupérer le contenu du buffer
    csv_content = output.getvalue()
    output.close()
    
    return csv_content

def display_calls_filters():
    """Affiche les filtres pour les appels dans la sidebar avec support des antennes."""
    st.sidebar.header("Filtres des appels")
    
    # Configuration des dates par défaut
    default_start_date = datetime(2025, 1, 1)
    default_end_date = datetime.now()
    
    # Filtres de date
    start_date = st.sidebar.date_input(
        "Date de début",
        value=default_start_date,
        min_value=datetime(2025, 1, 1).date(),
        max_value=default_end_date,
        key="calls_start_date"
    )
    
    end_date = st.sidebar.date_input(
        "Date de fin",
        value=default_end_date,
        min_value=start_date,
        max_value=default_end_date,
        key="calls_end_date"
    )
    
    # Filtres d'heure
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_time = st.time_input('Heure de début', value=datetime.strptime('00:00', '%H:%M').time(), key="calls_start_time")
    with col2:
        end_time = st.time_input('Heure de fin', value=datetime.strptime('23:59', '%H:%M').time(), key="calls_end_time")
    
    # Filtres supplémentaires
    df = get_calls_data()
    if not df.empty:
        statuts_uniques = sorted(df['Statut'].unique().tolist())
        statut_selectionne = st.sidebar.multiselect(
            'Statut',
            statuts_uniques,
            default=statuts_uniques,
            key="calls_statut"
        )
        
        # Utiliser la colonne Antenne normalisée au lieu de Nom
        antennes_uniques = sorted(df['Antenne'].unique().tolist())
        antenne_selectionnee = st.sidebar.selectbox(
            'Antenne',
            ['Toutes les antennes'] + antennes_uniques,
            key="calls_antenne"
        )
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'start_time': start_time,
            'end_time': end_time,
            'statut': statut_selectionne,
            'antenne': antenne_selectionnee
        }
    return None

def display_pagination_controls(total_items, page_size, current_page, key_prefix=""):
    """
    Affiche les contrôles de pagination.
    
    Args:
        total_items (int): Nombre total d'éléments
        page_size (int): Nombre d'éléments par page
        current_page (int): Numéro de la page actuelle (commence à 0)
        key_prefix (str): Préfixe pour les clés des boutons (pour éviter les conflits)
    """
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

def display_calls():
    """Affiche les données des appels avec support des antennes."""
    df = get_calls_data()
    
    if df.empty:
        st.warning("Aucune donnée d'appel n'a pu être récupérée.")
        return
    
    # Récupérer les filtres
    filters = display_calls_filters()
    if filters:
        # Application des filtres de date et statut
        mask = (df['Crée le'].dt.date >= filters['start_date']) & \
               (df['Crée le'].dt.date <= filters['end_date'])
        
        if filters['statut']:
            mask &= df['Statut'].isin(filters['statut'])
        
        # Ajout du filtre d'antenne
        if filters['antenne'] != 'Toutes les antennes':
            mask &= (df['Antenne'] == filters['antenne'])
        
        filtered_df = df[mask].copy()
        
        # Affichage des statistiques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre total d'appels", len(filtered_df))
        with col2:
            st.metric("Période", f"{filters['start_date'].strftime('%d/%m/%Y')} - {filters['end_date'].strftime('%d/%m/%Y')}")
        with col3:
            if 'start_time' in filters and 'end_time' in filters:
                st.metric("Plage horaire", f"{filters['start_time'].strftime('%H:%M')} - {filters['end_time'].strftime('%H:%M')}")
        
        # Pagination
        if 'calls_page_number' not in st.session_state:
            st.session_state.calls_page_number = 0
        
        PAGE_SIZE = 50
        
        # Pagination des données
        total_items = len(filtered_df)
        paginated_data = load_data_paginated(filtered_df, st.session_state.calls_page_number, PAGE_SIZE)
        
        # Afficher les données paginées dans un data_editor
        edited_df = st.data_editor(
            paginated_data,
            use_container_width=True,
            column_config={
                "select": st.column_config.CheckboxColumn("Sélectionner", default=False),
                "Crée le": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
                "Antenne": st.column_config.TextColumn("Antenne"),
                "Numéro": st.column_config.TextColumn("Numéro"),
                "Statut": st.column_config.TextColumn("Statut"),
                "Début appel": st.column_config.TextColumn("Heure de début"),
                "Fin appel": st.column_config.TextColumn("Heure de fin")
            },
            hide_index=True,
            num_rows="dynamic"
        )
        
        # Afficher les contrôles de pagination avec un préfixe unique
        display_pagination_controls(total_items, PAGE_SIZE, st.session_state.calls_page_number, key_prefix="calls_")
        
        # Bouton pour analyser les appels sélectionnés
        if st.button("Analyser les appels sélectionnés"):
            selected_calls = edited_df[edited_df["select"]].copy()
            if not selected_calls.empty:
                st.write("### Analyse des appels sélectionnés")
                for _, call in selected_calls.iterrows():
                    st.write("#### Détails de l'appel")
                    st.write(f"Date: {call['Crée le']}")
                    st.write(f"Antenne: {call['Antenne']}")
                    st.write(f"Numéro: {call['Numéro']}")
                    st.write(f"Statut: {call['Statut']}")
                    if pd.notnull(call['Début appel']):
                        st.write(f"Heure de début: {call['Début appel']}")
                    if pd.notnull(call['Fin appel']):
                        st.write(f"Heure de fin: {call['Fin appel']}")
                    st.write("---")

    if st.sidebar.button("Rafraîchir les données d'appels", key="refresh_calls"):
        if 'calls_data' in st.session_state:
            del st.session_state['calls_data']
        st.rerun()

def display_chats():
    """Affiche les données des chats avec les filtres."""
    df = get_ksaar_data()
    
    if df.empty:
        st.warning("Aucune donnée de chat n'a pu être récupérée.")
        return
    
    # Créer les colonnes pour les filtres
    col1, col2 = st.columns(2)
    
    with col1:
        # Filtre de dates
        start_date = st.date_input(
            "Date de début",
            datetime.now().date() - timedelta(days=30),
            key="filter_start_date"
        )
        end_date = st.date_input(
            "Date de fin",
            datetime.now().date(),
            key="filter_end_date"
        )
        
        # Filtre d'heures
        start_time = st.time_input(
            "Heure de début",
            datetime.strptime("20:00", "%H:%M").time(),
            key="filter_start_time"
        )
        end_time = st.time_input(
            "Heure de fin",
            datetime.strptime("08:00", "%H:%M").time(),
            key="filter_end_time"
        )
    
    with col2:
        # Filtre par antenne
        antennes = sorted(df['Antenne'].dropna().unique().tolist())
        selected_antenne = st.multiselect('Antennes', options=['Toutes'] + antennes, default='Toutes', key="filter_antennes")
        
        # Filtre par bénévole
        benevoles = sorted(df['Volunteer_Location'].dropna().unique().tolist())
        selected_benevole = st.multiselect('Bénévoles', options=['Tous'] + benevoles, default='Tous', key="filter_benevoles")
    
    # Barre de recherche
    search_text = st.text_input("Rechercher dans les messages", key="search_text")
    
    # Filtrer les messages en fonction du terme de recherche
    if search_text:
        df = df[df['messages'].str.contains(search_text, case=False, na=False)]
    
    # Créer un dictionnaire de filtres
    filters = {
        'start_date': start_date,
        'end_date': end_date,
        'start_time': start_time,
        'end_time': end_time,
        'antenne': selected_antenne,
        'benevole': selected_benevole
    }
    
    # Application des filtres de date
    mask = (df['Crée le'].dt.date >= filters['start_date']) & \
           (df['Crée le'].dt.date <= filters['end_date'])
    
    # Filtrer par antenne
    if 'Toutes' not in filters['antenne'] and filters['antenne']:
        mask &= df['Antenne'].isin(filters['antenne'])
    
    # Filtrer par bénévole
    if 'Tous' not in filters['benevole'] and filters['benevole']:
        mask &= df['Volunteer_Location'].isin(filters['benevole'])
    
    filtered_df = df[mask].copy()
    
    # Affichage des statistiques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre total de chats", len(filtered_df))
    with col2:
        st.metric("Période", f"{filters['start_date'].strftime('%d/%m/%Y')} - {filters['end_date'].strftime('%d/%m/%Y')}")
    with col3:
        st.metric("Plage horaire", f"{filters['start_time'].strftime('%H:%M')} - {filters['end_time'].strftime('%H:%M')}")
    
    # Pagination
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0
    
    PAGE_SIZE = 50
    
    # Pagination des données
    total_items = len(filtered_df)
    paginated_data = load_data_paginated(filtered_df, st.session_state.page_number, PAGE_SIZE)
    
    # Afficher les données paginées dans un data_editor
    edited_df = st.data_editor(
        paginated_data,
        use_container_width=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Sélectionner", default=False),
            "Crée le": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
            "Antenne": st.column_config.TextColumn("Antenne"),
            "Volunteer_Location": st.column_config.TextColumn("Bénévole"),
            "messages": st.column_config.TextColumn("Messages", width="large"),
            "IP": st.column_config.TextColumn("IP"),
            "id_chat": st.column_config.NumberColumn("ID Chat")
        },
        hide_index=True,
        num_rows="dynamic"
    )
    
    # Afficher les contrôles de pagination avec un préfixe unique
    display_pagination_controls(total_items, PAGE_SIZE, st.session_state.page_number, key_prefix="chats_")
    
    # Bouton pour analyser les chats sélectionnés
    if st.button("Analyser les chats sélectionnés"):
        selected_chats = edited_df[edited_df["select"]].copy()
        if not selected_chats.empty:
            st.write("### Analyse des chats sélectionnés")
            for _, chat in selected_chats.iterrows():
                st.write(f"#### Chat {chat['id_chat']}")
                st.write(f"Date: {chat['Crée le']}")
                st.write(f"Antenne: {chat['Antenne']}")
                st.write(f"Bénévole: {chat['Volunteer_Location']}")
                st.write("Messages:")
                st.text_area("", chat['messages'], height=200)
                st.write("---")

def display_abuse_analysis():
    """Affiche d'abord tous les chats potentiellement abusifs, puis permet l'analyse détaillée."""
    st.title("Analyse IA des chats potentiellement abusifs")
    
    # Récupérer les données de chat
    df = get_ksaar_data()
    
    if df.empty:
        st.warning("Aucune donnée de chat n'a pu être récupérée.")
        return
    
    # Créer une sidebar pour les filtres
    st.sidebar.header("Filtres d'analyse des abus")
    
    # Filtres de date
    st.sidebar.subheader("Filtres de date")
    date_col1, date_col2 = st.sidebar.columns(2)
    with date_col1:
        start_date = st.date_input("Date de début", value=datetime(2025, 1, 1).date(), key="abuse_start_date")
    with date_col2:
        end_date = st.date_input("Date de fin", value=datetime.now().date(), key="abuse_end_date")
    
    # Filtres d'heure
    st.sidebar.subheader("Filtres d'heure")
    use_time_filter = st.sidebar.checkbox("Activer le filtre d'heure", key="use_time_filter")
    
    if use_time_filter:
        time_col1, time_col2 = st.sidebar.columns(2)
        with time_col1:
            start_time = st.time_input("Heure de début", value=datetime.strptime('00:00', '%H:%M').time(), key="abuse_start_time")
        with time_col2:
            end_time = st.time_input("Heure de fin", value=datetime.strptime('23:59', '%H:%M').time(), key="abuse_end_time")
    else:
        # Valeurs par défaut si le filtre n'est pas activé (toute la journée)
        start_time = datetime.strptime('00:00', '%H:%M').time()
        end_time = datetime.strptime('23:59', '%H:%M').time()
    
    # Filtres par antenne et bénévole
    st.sidebar.subheader("Filtres par antenne et bénévole")
    
    # Filtre par antenne
    antennes = sorted(df['Antenne'].dropna().unique().tolist())
    selected_antenne = st.sidebar.multiselect(
        'Antennes', 
        options=['Toutes'] + antennes, 
        default='Toutes', 
        key="abuse_filter_antennes"
    )
    
    # Filtre par bénévole
    benevoles = sorted(df['Volunteer_Location'].dropna().unique().tolist())
    selected_benevole = st.sidebar.multiselect(
        'Bénévoles', 
        options=['Tous'] + benevoles, 
        default='Tous', 
        key="abuse_filter_benevoles"
    )
    
    # Recherche de mots dans les messages
    st.sidebar.subheader("Recherche dans les messages")
    search_text = st.sidebar.text_input("Rechercher des mots dans les messages", key="abuse_search_text")
    
    # Recherche par ID de chat
    search_id = st.sidebar.text_input("Rechercher un chat par ID", key="search_chat_id")
    
    # Appliquer les filtres de base
    filtered_df = df.copy()
    
    # Filtre par date
    filtered_df = filtered_df[(filtered_df['Crée le'].dt.date >= start_date) & 
                             (filtered_df['Crée le'].dt.date <= end_date)]
    
    # Appliquer le filtre d'heure seulement si activé
    if use_time_filter:
        # Fonction pour convertir en objet time
        def convert_to_time(dt):
            try:
                if pd.isna(dt):
                    return None
                return dt.time()
            except:
                return None
        
        # Créer des colonnes temporaires pour le filtrage des heures
        filtered_df['time_obj'] = filtered_df['Crée le'].apply(convert_to_time)
        
        # Appliquer le filtre d'heure
        time_mask = pd.Series(True, index=filtered_df.index)
        valid_times = filtered_df['time_obj'].notna()
        
        if valid_times.any():
            # Gérer le cas où l'heure de début est après l'heure de fin (période nocturne)
            if start_time > end_time:
                # La plage horaire s'étend sur deux jours (ex: de 21:00 à 00:00)
                time_mask = valid_times & (
                    (filtered_df['time_obj'] >= start_time) | 
                    (filtered_df['time_obj'] <= end_time)
                )
            else:
                # Plage horaire normale dans la même journée
                time_mask = valid_times & (
                    (filtered_df['time_obj'] >= start_time) & 
                    (filtered_df['time_obj'] <= end_time)
                )
        
        filtered_df = filtered_df[time_mask]
        
        # Supprimer la colonne temporaire
        filtered_df = filtered_df.drop('time_obj', axis=1)
    
    # Filtre par antenne
    if 'Toutes' not in selected_antenne and selected_antenne:
        filtered_df = filtered_df[filtered_df['Antenne'].isin(selected_antenne)]
    
    # Filtre par bénévole
    if 'Tous' not in selected_benevole and selected_benevole:
        filtered_df = filtered_df[filtered_df['Volunteer_Location'].isin(selected_benevole)]
    
    # Filtre par recherche de texte dans les messages
    if search_text:
        filtered_df = filtered_df[filtered_df['messages'].str.contains(search_text, case=False, na=False)]
    
    # Filtre par ID de chat
    if search_id:
        try:
            search_id = int(search_id)
            filtered_by_id = filtered_df[filtered_df['id_chat'] == search_id]
            if not filtered_by_id.empty:
                filtered_df = filtered_by_id
                st.success(f"Chat ID {search_id} trouvé.")
            else:
                st.warning(f"Aucun chat avec l'ID {search_id} n'a été trouvé.")
        except ValueError:
            st.error("L'ID du chat doit être un nombre entier.")
    
    # Identifier les chats potentiellement abusifs
    potentially_abusive_df = identify_potentially_abusive_chats(filtered_df)
    
    if potentially_abusive_df.empty:
        st.warning("Aucun chat potentiellement abusif n'a été détecté avec les filtres actuels.")
        return
    
    # Afficher les statistiques
    st.subheader("Statistiques des chats potentiellement abusifs")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Nombre total de chats filtrés", len(filtered_df))
    with col2:
        st.metric("Chats potentiellement abusifs", len(potentially_abusive_df))
    
    # Afficher des statistiques par antenne et bénévole
    if len(potentially_abusive_df) > 0:
        with st.expander("Statistiques par antenne et bénévole", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Répartition par antenne")
                antenne_counts = potentially_abusive_df['Antenne'].value_counts()
                st.bar_chart(antenne_counts)
            
            with col2:
                st.subheader("Répartition par bénévole")
                benevole_counts = potentially_abusive_df['Volunteer_Location'].value_counts()
                st.bar_chart(benevole_counts)
    
    # Afficher la liste des chats potentiellement abusifs
    st.subheader("Liste des chats potentiellement abusifs")
    
    # Ajouter une colonne de sélection
    potentially_abusive_df['select'] = False
    
    # Afficher le tableau avec les colonnes disponibles
    edited_df = st.data_editor(
        potentially_abusive_df,
        column_config={
            "select": st.column_config.CheckboxColumn("Sélectionner", default=False),
            "id_chat": st.column_config.NumberColumn("ID Chat"),
            "Crée le": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
            "Antenne": st.column_config.TextColumn("Antenne"),
            "Volunteer_Location": st.column_config.TextColumn("Bénévole"),
            "preliminary_score": st.column_config.ProgressColumn(
                "Score préliminaire",
                format="%d",
                min_value=0,
                max_value=20,
            ),
            "messages": st.column_config.TextColumn("Aperçu du message", width="large")
        },
        column_order=[
            "select", "id_chat", "Crée le", "Antenne", "Volunteer_Location", 
            "preliminary_score", "messages"
        ],
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Bouton pour analyser en détail les chats sélectionnés
    if st.button("Analyser en détail les chats sélectionnés", key="analyze_selected"):
        selected_chats = edited_df[edited_df["select"]].copy()
        
        if selected_chats.empty:
            st.warning("Veuillez sélectionner au moins un chat pour l'analyse détaillée.")
        else:
            st.subheader("Analyse détaillée des chats sélectionnés")
            
            with st.spinner("Analyse détaillée en cours..."):
                # Analyser chaque chat sélectionné
                detailed_results = []
                
                for _, chat in selected_chats.iterrows():
                    chat_id = chat.get('id_chat')
                    # Récupérer les données complètes du chat depuis le DataFrame original
                    original_chat_data = df[df['id_chat'] == chat_id]
                    
                    if original_chat_data.empty:
                        st.error(f"Impossible de trouver les données complètes pour le chat {chat_id}")
                        continue
                    
                    # Utiliser les données complètes du chat
                    messages = original_chat_data.iloc[0].get('messages', '')
                    
                    # Utiliser la fonction d'analyse contextuelle améliorée
                    try:
                        risk_score, risk_factors, problematic_phrases, operator_harassment, manipulation_patterns, topic_changes = analyze_chat_content(messages)
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse du chat {chat_id}: {str(e)}")
                        continue
                    
                    # Convertir le dictionnaire de phrases problématiques en texte formaté
                    phrases_text = ""
                    for category, phrases in problematic_phrases.items():
                        if phrases:
                            phrases_text += f"**{category}**:\n"
                            for phrase in phrases[:3]:  # Limiter à 3 phrases par catégorie
                                phrases_text += f"- {phrase}\n"
                            phrases_text += "\n"
                    
                    # Convertir les patterns de manipulation en texte formaté
                    manipulation_text = ""
                    if manipulation_patterns:
                        for pattern in manipulation_patterns:
                            manipulation_text += f"**{pattern['type']}**: {pattern['description']}\n"
                            manipulation_text += f"Occurrences: {pattern['occurrences']}\n"
                            if 'examples' in pattern and pattern['examples']:
                                manipulation_text += "Exemples:\n"
                                for example in pattern['examples'][:2]:  # Limiter à 2 exemples
                                    if isinstance(example, dict) and 'message' in example:
                                        manipulation_text += f"- {example['message']}\n"
                                    else:
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
                        'messages': messages  # Contenu complet du chat
                    }
                    
                    detailed_results.append(result_dict)
                
                # Créer le DataFrame des résultats détaillés
                detailed_df = pd.DataFrame(detailed_results)
                
                if not detailed_df.empty:
                    # Trier par score de risque
                    detailed_df = detailed_df.sort_values(by='Score de risque', ascending=False)
                    
                    # Afficher les résultats détaillés
                    st.dataframe(
                        detailed_df,
                        column_config={
                            "id_chat": st.column_config.NumberColumn("ID Chat"),
                            "Crée le": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
                            "Antenne": st.column_config.TextColumn("Antenne"),
                            "Volunteer_Location": st.column_config.TextColumn("Bénévole"),
                            "Score de risque": st.column_config.ProgressColumn(
                                "Score de risque",
                                format="%d",
                                min_value=0,
                                max_value=100,
                            ),
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
                    
                    # Permettre de voir les détails d'un chat analysé
                    if not detailed_df.empty:
                        selected_chat_id = st.selectbox(
                            "Sélectionner un chat pour voir les détails complets",
                            detailed_df['id_chat'].tolist(),
                            key="selected_detailed_chat"
                        )
                        
                        if selected_chat_id:
                            selected_chat = detailed_df[detailed_df['id_chat'] == selected_chat_id].iloc[0]
                            
                            with st.expander(f"Détails complets du chat {selected_chat_id}", expanded=True):
                                # Afficher les informations du chat
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Date:** {selected_chat['Crée le'].strftime('%d/%m/%Y %H:%M')}")
                                with col2:
                                    st.write(f"**Antenne:** {selected_chat['Antenne']}")
                                with col3:
                                    st.write(f"**Bénévole:** {selected_chat['Volunteer_Location']}")
                                
                                st.write(f"**Score de risque:** {selected_chat['Score de risque']} ({selected_chat['Niveau de risque']})")
                                st.write(f"**Facteurs de risque:** {selected_chat['Facteurs de risque']}")
                                st.write(f"**Harcèlement envers l'opérateur:** {selected_chat['Harcèlement opérateur']}")
                                
                                # Afficher les phrases problématiques
                                if selected_chat['Phrases problématiques']:
                                    st.subheader("Phrases problématiques détectées")
                                    st.markdown(selected_chat['Phrases problématiques'])
                                
                                # Afficher l'analyse contextuelle
                                if selected_chat['Analyse contextuelle']:
                                    st.subheader("Analyse contextuelle")
                                    st.markdown(selected_chat['Analyse contextuelle'])
                                
                                # Afficher le contenu complet du chat
                                st.subheader("Contenu complet du chat")
                                
                                # Afficher le contenu du chat dans un format plus lisible
                                chat_content = selected_chat['messages']
                                st.text_area("Messages", value=chat_content, height=400)
                                
                                # Bouton pour générer un rapport
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
                else:
                    st.warning("Aucun résultat d'analyse détaillée n'a été généré.")
    
    # Bouton pour générer des rapports pour les chats sélectionnés
    if st.button("Générer des rapports pour les chats sélectionnés", key="generate_reports_selected"):
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

def identify_potentially_abusive_chats(df):
    """
    Fonction pour identifier rapidement les chats potentiellement abusifs
    en utilisant des mots-clés et des patterns simples.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Liste étendue de mots-clés pour la détection préliminaire
    abuse_keywords = [
        # Contenu sexuel explicite
        "sexe", "bite", "penis", "vagin", "masturb", "bander", "sucer", "baiser",
        "jouir", "éjacul", "ejacul", "orgasm", "porno", "cul", "nichon", "seins",
        "sexe", "bite", "queue", "pénis", "penis", "zboub", "vagin", "chatte", "cunni",
        "masturb", "branler", "branlette", "fap", "fellation", "pipe", "sucer",
        "bander", "gode", "godemichet", "baiser", "ken", "niquer", "niqué", "niquee",
        "sodom", "sodomie", "anal", "dp", "orgie", "orgasm", "orgasme", "jouir",
        "gicler", "giclée", "éjacul", "ejacul", "cum", "creampie", "facial", "porno",
        "porn", "xxx", "cul", "nichon", "nichons", "sein", "seins", "boobs", "boobies",
        "téton", "tétons", "nipple",
        
        # Demandes inappropriées
        "photo", "nue", "nu", "déshabille", "deshabille", "montre-moi", "montre moi",
        "caméra", "camera", "vidéo", "video", "snapchat", "instagram", "facebook", 
        "onlyfans", "strip", "striptease", "strip tease",
        
        # Harcèlement et menaces
        "connard", "salope", "pute", "enculé", "encule", "pd", "tapette", "nègre",
        "negre", "bougnoule", "suicide", "tuer", "mourir", "crever", "adresse",
        "menace", "frapper", "battre", "harcèle", "harcele", "stalker",
        
        # Comportements suspects
        "je te surveille", "je sais où tu es", "je sais ou tu es", "je t'observe",
        "je vais te retrouver", "je connais ton adresse", "donne-moi ton adresse",
        "donne moi ton adresse", "adresse ip", "ip address", "gps", "géolocalisation",
        "share location", "send location", "where you live", "gps coordinates",
        "dox", "doxx", "doxxing", "docx"
    ]
    
    # Créer une colonne pour indiquer si le chat contient des mots-clés abusifs
    df['potentially_abusive'] = df['messages'].str.lower().apply(
        lambda x: any(keyword in str(x).lower() for keyword in abuse_keywords) if not pd.isna(x) else False
    )
    
    # Filtrer les chats potentiellement abusifs
    potentially_abusive_df = df[df['potentially_abusive']].copy()
    
    # Ajouter une colonne pour le score préliminaire (nombre de mots-clés trouvés)
    def count_keywords(message):
        if pd.isna(message):
            return 0
        message = str(message).lower()
        return sum(1 for keyword in abuse_keywords if keyword in message)
    
    potentially_abusive_df['preliminary_score'] = potentially_abusive_df['messages'].apply(count_keywords)
    
    # Trier par score préliminaire décroissant
    potentially_abusive_df = potentially_abusive_df.sort_values(by='preliminary_score', ascending=False)
    
    return potentially_abusive_df

def extract_user_messages(messages):
    """Extrait les messages de l'utilisateur d'une conversation."""
    if pd.isna(messages) or messages is None:
        return []
    
    messages = str(messages)
    user_messages = []
    current_message = ""
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
    """Extrait les messages de l'opérateur d'une conversation."""
    if pd.isna(messages) or messages is None:
        return []
    
    messages = str(messages)
    operator_messages = []
    current_message = ""
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
    """Détecte les changements brusques de sujet dans les messages."""
    if len(user_messages) < min_messages:
        return []
    
    # Vectoriser les messages
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='french')
    try:
        X = vectorizer.fit_transform(user_messages)
    except:
        return []
    
    # Calculer la similarité entre messages consécutifs
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
    """Détecte les patterns de manipulation dans les messages."""
    if pd.isna(messages) or messages is None:
        return []
    
    messages = str(messages)
    patterns = []
    
    # Pattern 1: Insistance excessive
    insistence_keywords = [
        "s'il te plait", "stp", "svp", "je t'en prie", "je t'en supplie",
        "allez", "aller", "réponds", "reponds", "répond", "repond"
    ]
    
    insistence_count = sum(1 for keyword in insistence_keywords if keyword in messages.lower())
    if insistence_count >= 3:
        patterns.append({
            'type': "Insistance excessive",
            'description': "Utilisation répétée de formules d'insistance",
            'occurrences': insistence_count
        })
    
    # Pattern 2: Culpabilisation
    guilt_keywords = [
        "tu ne veux pas m'aider", "tu refuses de m'aider", "tu ne veux pas me répondre",
        "tu m'ignores", "tu ne comprends pas", "tu ne fais pas d'effort",
        "c'est de ta faute", "à cause de toi", "par ta faute"
    ]
    
    guilt_messages = []
    for line in messages.split('\n'):
        if any(keyword in line.lower() for keyword in guilt_keywords):
            guilt_messages.append(line)
    
    if guilt_messages:
        patterns.append({
            'type': "Culpabilisation",
            'description': "Tentatives de faire culpabiliser l'opérateur",
            'occurrences': len(guilt_messages),
            'examples': guilt_messages[:3]
        })
    
    # Pattern 3: Menaces voilées
    threat_keywords = [
        "tu vas voir", "tu regretteras", "tu le regretteras", "tu vas le regretter",
        "je vais me plaindre", "je vais le dire", "je sais où", "je peux te trouver"
    ]
    
    threat_messages = []
    for line in messages.split('\n'):
        if any(keyword in line.lower() for keyword in threat_keywords):
            threat_messages.append(line)
    
    if threat_messages:
        patterns.append({
            'type': "Menaces voilées",
            'description': "Utilisation de menaces indirectes",
            'occurrences': len(threat_messages),
            'examples': threat_messages[:3]
        })
    
    return patterns

def analyze_chat_content(messages):
    """
    Analyse contextuelle avancée du contenu d'un chat pour détecter des signes d'abus.
    """
    if pd.isna(messages) or messages is None or messages == "":
        return 0, [], {}, False, [], []
    
    messages = str(messages)
    risk_score = 0
    risk_factors = []
    problematic_phrases = {}
    operator_harassment = False
    
    # Extraire les messages de l'utilisateur et de l'opérateur
    user_messages = extract_user_messages(messages)
    operator_messages = extract_operator_messages(messages)
    
    if not user_messages:
        return 0, [], {}, False, [], []
    
    # 1. ANALYSE DU CONTEXTE GLOBAL
    
    # Détecter si la personne parle principalement de traumatismes passés
    trauma_narrative_indicators = [
        "quand j'étais", "dans mon enfance", "j'ai été victime", 
        "j'ai subi", "on m'a fait", "je me souviens", "flashback",
        "souvenir", "traumatisme", "j'ai été agressé", "harcelé"
    ]
    
    is_trauma_narrative = any(indicator in messages.lower() for indicator in trauma_narrative_indicators)
    
    # Détecter si la personne parle de troubles mentaux
    mental_health_indicators = [
        "voix dans ma tête", "j'entends des voix", "hallucination", 
        "trouble dissociatif", "TDI", "schizophrénie", "dépression",
        "anxiété", "psychiatrie", "hospitalisation", "thérapie"
    ]
    
    is_mental_health_discussion = any(indicator in messages.lower() for indicator in mental_health_indicators)
    
    # 2. DÉTECTION DE CONTENU SUICIDAIRE ET AUTOMUTILATOIRE
    suicidal_keywords = [
        "suicide", "me tuer", "mourir", "en finir", "plus envie de vivre",
        "mettre fin à mes jours", "me suicider", "disparaître", "plus la force"
    ]
    
    suicidal_messages = []
    for message in user_messages:
        if any(keyword in message.lower() for keyword in suicidal_keywords):
            suicidal_messages.append(message)
    
    if suicidal_messages:
        risk_score += 40
        risk_factors.append(f"Pensées suicidaires ({len(suicidal_messages)} occurrences)")
        problematic_phrases["Pensées suicidaires"] = suicidal_messages[:3]
    
    # 3. DÉTECTION DE HARCÈLEMENT SEXUEL
    sexual_harassment_keywords = [
        "tu aimes le sexe", "t'aimes sucer", "tu veux baiser",
        "tu es excité", "tu bandes", "tu mouilles", "tu te masturbes"
    ]
    
    harassment_messages = []
    for message in user_messages:
        if any(keyword in message.lower() for keyword in sexual_harassment_keywords):
            harassment_messages.append(message)
    
    if harassment_messages:
        operator_harassment = True
        risk_score += 50
        risk_factors.append(f"Harcèlement sexuel ({len(harassment_messages)} occurrences)")
        problematic_phrases["Harcèlement sexuel"] = harassment_messages[:3]
    
    # 4. DÉTECTION DE MANIPULATION
    manipulation_patterns = detect_manipulation_patterns(messages)
    if manipulation_patterns:
        risk_score += len(manipulation_patterns) * 10
        risk_factors.append(f"Patterns de manipulation ({len(manipulation_patterns)} détectés)")
    
    # 5. DÉTECTION DE CHANGEMENTS DE SUJET BRUSQUES
    topic_changes = detect_topic_changes(user_messages)
    if len(topic_changes) > 2:
        risk_score += min(len(topic_changes) * 5, 20)
        risk_factors.append(f"Changements de sujet fréquents ({len(topic_changes)} détectés)")
    
    # Ajustements du score en fonction du contexte
    if is_trauma_narrative and not operator_harassment:
        risk_score *= 0.7
        risk_factors.append("Score ajusté: récit de traumatisme")
    
    if is_mental_health_discussion and not operator_harassment:
        risk_score *= 0.8
        risk_factors.append("Score ajusté: discussion sur la santé mentale")
    
    # Limiter le score à 100
    risk_score = min(int(risk_score), 100)
    
    return risk_score, risk_factors, problematic_phrases, operator_harassment, manipulation_patterns, topic_changes

def get_abuse_risk_level(score):
    """Retourne le niveau de risque en fonction du score."""
    if score >= 80:
        return "Très élevé"
    elif score >= 60:
        return "Élevé"
    elif score >= 40:
        return "Modéré"
    elif score >= 20:
        return "Faible"
    else:
        return "Très faible"

def main():
    """Fonction principale de l'application."""
    
    if not check_password():
        return
    
    st.title("Dashboard GASAS")
    
    # Ajouter un bouton de rafraîchissement manuel
    if st.sidebar.button("🔄 Rafraîchir les données", key="refresh_button"):
        # Effacer le cache des données
        if 'chat_data' in st.session_state:
            del st.session_state['chat_data']
        if 'calls_data' in st.session_state:
            del st.session_state['calls_data']
        if 'abuse_analysis_results' in st.session_state:
            del st.session_state['abuse_analysis_results']
        st.rerun()
    
    # Ajouter le bouton de déconnexion
    if st.sidebar.button("🚪 Déconnexion", key="logout_button"):
        # Effacer toutes les données de session
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Créer les onglets
    tab1, tab2, tab3 = st.tabs(["Chats", "Appels", "Analyse IA des abus"])
    
    with tab1:
        display_chats()
    with tab2:
        display_calls()
    with tab3:
        display_abuse_analysis()

if __name__ == "__main__":
    main()
