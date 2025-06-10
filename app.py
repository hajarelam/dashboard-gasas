import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# Au lieu d'importer depuis config.py
credentials = st.secrets["credentials"]
ksaar_config = st.secrets["ksaar_config"]


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

def display_pagination_controls(total_items, page_size, current_page):
    total_pages = (total_items + page_size - 1) // page_size
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_page > 0:
            if st.button("← Précédent"):
                st.session_state.page_number = current_page - 1
                st.rerun()
    
    with col2:
        st.write(f"Page {current_page + 1} sur {total_pages}")
    
    with col3:
        if current_page < total_pages - 1:
            if st.button("Suivant →"):
                st.session_state.page_number = current_page + 1
                st.rerun()

def display_calls():
    if 'calls_page_number' not in st.session_state:
        st.session_state.calls_page_number = 0
    
    PAGE_SIZE = 50
    
    # Charger les données avec cache
    data = get_calls_data()
    
    # Appliquer les filtres
    filters = display_calls_filters()
    if filters:
        # Application des filtres de date et statut
        mask = (data['Crée le'].dt.date >= filters['start_date']) & \
               (data['Crée le'].dt.date <= filters['end_date']) & \
               (data['Statut'].isin(filters['statut']))
        
        # Convertir les heures en objets time pour la comparaison
        def convert_to_time(time_str):
            try:
                if pd.isna(time_str):
                    return None
                return datetime.strptime(time_str, '%H:%M').time()
            except:
                return None
        
        # Appliquer la conversion aux colonnes d'heure
        data['Début appel_time'] = data['Début appel'].apply(convert_to_time)
        data['Fin appel_time'] = data['Fin appel'].apply(convert_to_time)
        
        # Filtrer par heure
        time_mask = pd.Series(True, index=data.index)
        valid_times = data['Début appel_time'].notna() & data['Fin appel_time'].notna()
        
        if valid_times.any():
            # Gérer le cas où l'heure de début est après l'heure de fin (période nocturne)
            if filters['start_time'] > filters['end_time']:
                # La plage horaire s'étend sur deux jours (ex: de 21:00 à 00:00)
                time_mask = valid_times & (
                    ((data['Début appel_time'] >= filters['start_time']) | 
                     (data['Début appel_time'] <= filters['end_time'])) &
                    ((data['Fin appel_time'] >= filters['start_time']) | 
                     (data['Fin appel_time'] <= filters['end_time']))
                )
            else:
                # Un appel est dans la plage horaire seulement si :
                # - son heure de début est dans la plage ET
                # - son heure de fin est dans la plage
                time_mask = valid_times & (
                    (data['Début appel_time'] >= filters['start_time']) & 
                    (data['Début appel_time'] <= filters['end_time']) &
                    (data['Fin appel_time'] >= filters['start_time']) & 
                    (data['Fin appel_time'] <= filters['end_time'])
                )
        
        mask &= time_mask
        
        # Ajout du filtre d'antenne (utiliser la colonne Antenne normalisée)
        if filters['antenne'] != 'Toutes les antennes':
            mask = mask & (data['Antenne'] == filters['antenne'])
        
        filtered_df = data[mask].copy()
        
        # Supprimer les colonnes temporaires utilisées pour le filtrage
        filtered_df = filtered_df.drop(['Début appel_time', 'Fin appel_time'], axis=1)
        
        # Pagination
        total_items = len(filtered_df)
        paginated_data = load_data_paginated(filtered_df, st.session_state.calls_page_number, PAGE_SIZE)
        
        # Afficher les données paginées
        for index, row in paginated_data.iterrows():
            st.write(f"**Chat {row['id_chat']}**")
            st.write(f"Date: {row['Crée le']}")
            st.write(f"Antenne: {row['Antenne']}")
            st.write(f"Statut: {row['Statut']}")
            st.write("---")
        
        # Afficher les contrôles de pagination
        display_pagination_controls(total_items, PAGE_SIZE, st.session_state.calls_page_number)

    if st.sidebar.button("Rafraîchir les données d'appels", key="refresh_calls"):
        if 'calls_data' in st.session_state:
            del st.session_state['calls_data']
        st.rerun()

def display_chats():
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0
    
    PAGE_SIZE = 50  # Nombre d'éléments par page
    
    # Charger les données avec cache
    data = get_ksaar_data()
    
    # Appliquer les filtres
    start_date = datetime.now().date() - timedelta(days=30)
    end_date = datetime.now().date()
    start_time = datetime.strptime('00:00', '%H:%M').time()
    end_time = datetime.strptime('23:59', '%H:%M').time()
    
    st.subheader("Filtres")
    
    # Définir les colonnes avant de les utiliser
    col1, col2 = st.columns(2)
    
    with col1:
        # Utiliser des clés uniques pour chaque élément
        start_date = st.date_input('Date de début', value=start_date, key="filter_start_date")
        end_date = st.date_input('Date de fin', value=end_date, key="filter_end_date")
        
        start_time = st.time_input('Heure de début', value=start_time, key="filter_chat_start_time")
        end_time = st.time_input('Heure de fin', value=end_time, key="filter_chat_end_time")
    
    with col2:
        antennes = sorted(data['Antenne'].dropna().unique().tolist())
        selected_antenne = st.multiselect('Antennes', options=['Toutes'] + antennes, default='Toutes', key="filter_antennes")
        
        benevoles = sorted(data['Volunteer_Location'].dropna().unique().tolist())
        selected_benevole = st.multiselect('Bénévoles', options=['Tous'] + benevoles, default='Tous', key="filter_benevoles")
        
        search_text = st.text_input('Rechercher dans les messages', key="filter_search_text")
    
    # Filtrer les messages en fonction du terme de recherche
    if search_text:
        data = data[data['messages'].str.contains(search_text, case=False, na=False)]
    
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
    mask = (data['Crée le'].dt.date >= filters['start_date']) & \
           (data['Crée le'].dt.date <= filters['end_date'])
    
    # Convertir les heures en objets time pour la comparaison
    def convert_to_time(dt):
        try:
            if pd.isna(dt):
                return None
            return dt.time()
        except:
            return None
    
    # Créer des colonnes temporaires pour le filtrage des heures
    data['time_obj'] = data['Crée le'].apply(convert_to_time)
    
    # Appliquer le filtre d'heure
    time_mask = pd.Series(True, index=data.index)
    valid_times = data['time_obj'].notna()
    
    if valid_times.any():
        # Gérer le cas où l'heure de début est après l'heure de fin (période nocturne)
        if filters['start_time'] > filters['end_time']:
            # La plage horaire s'étend sur deux jours (ex: de 21:00 à 00:00)
            time_mask = valid_times & (
                (data['time_obj'] >= filters['start_time']) | 
                (data['time_obj'] <= filters['end_time'])
            )
        else:
            # Plage horaire normale dans la même journée
            time_mask = valid_times & (
                (data['time_obj'] >= filters['start_time']) & 
                (data['time_obj'] <= filters['end_time'])
            )
    
    mask &= time_mask
    
    # Filtrer par antenne
    if 'Toutes' not in filters['antenne'] and filters['antenne']:
        mask &= data['Antenne'].isin(filters['antenne'])
    
    # Filtrer par bénévole
    if 'Tous' not in filters['benevole'] and filters['benevole']:
        mask &= data['Volunteer_Location'].isin(filters['benevole'])
    
    filtered_df = data[mask].copy()
    
    # Supprimer la colonne temporaire
    filtered_df = filtered_df.drop('time_obj', axis=1)
    
    filtered_df['select'] = False
    
    # Formatage des heures pour l'affichage (HH:MM)
    filtered_df['last_op_msg_time'] = filtered_df['last_op_message'].dt.strftime('%H:%M')
    filtered_df['last_user_msg_time'] = filtered_df['last_user_message'].dt.strftime('%H:%M')
    
    # Pagination
    total_items = len(filtered_df)
    paginated_data = load_data_paginated(filtered_df, st.session_state.page_number, PAGE_SIZE)
    
    # Afficher les données paginées
    for index, row in paginated_data.iterrows():
        st.write(f"**Chat {row['id_chat']}**")
        st.write(f"Date: {row['Crée le']}")
        st.write(f"Antenne: {row['Antenne']}")
        st.write(f"Statut: {row['Statut']}")
        st.write("---")
    
    # Afficher les contrôles de pagination
    display_pagination_controls(total_items, PAGE_SIZE, st.session_state.page_number)
    
    # Afficher les statistiques
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Nombre total de chats filtrés", len(filtered_df))
    with col2:
        st.metric("Chats potentiellement abusifs", len(identify_potentially_abusive_chats(filtered_df)))
    
    # Afficher des statistiques par antenne et bénévole
    if len(filtered_df) > 0:
        with st.expander("Statistiques par antenne et bénévole", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Répartition par antenne")
                antenne_counts = filtered_df['Antenne'].value_counts()
                st.bar_chart(antenne_counts)
            
            with col2:
                st.subheader("Répartition par bénévole")
                benevole_counts = filtered_df['Volunteer_Location'].value_counts()
                st.bar_chart(benevole_counts)
    
    # Afficher la liste des chats potentiellement abusifs
    st.subheader("Liste des chats potentiellement abusifs")
    
    # Ajouter une colonne de sélection
    filtered_df['select'] = False
    
    # Afficher le tableau avec les colonnes disponibles
    edited_df = st.data_editor(
        filtered_df,
        column_config={
            "select": st.column_config.CheckboxColumn("Sélectionner", default=False),
            "Crée le": st.column_config.DatetimeColumn("Crée le", format="DD/MM/YYYY HH:mm"),
            "IP": st.column_config.TextColumn("IP"),
            "last_op_msg_time": st.column_config.TextColumn("Début du chat"),
            "id_chat": st.column_config.NumberColumn("ID Chat"),
            "messages": st.column_config.TextColumn("Messages", width="large"),
            "last_user_msg_time": st.column_config.TextColumn("Fin du chat"),
            "Antenne": st.column_config.TextColumn("Antenne"),
            "Volunteer_Location": st.column_config.TextColumn("Bénévole")
        },
        column_order=[
            "select", "Crée le", "IP", "Antenne", "Volunteer_Location", "last_op_msg_time", 
            "id_chat", "messages", "last_user_msg_time"
        ],
        height=500,
        num_rows="dynamic",
        key="chat_table",
        hide_index=True
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
                    original_chat_data = data[data['id_chat'] == chat_id]
                    
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
                                    chat_data = data[data['id_chat'] == selected_chat_id].iloc[0]
                                    
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
                chat_data = data[data['id_chat'] == chat_id].iloc[0]
                
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

def extract_user_messages(messages):
    """Extrait les messages de l'utilisateur à partir du texte complet de la conversation."""
    if pd.isna(messages) or messages is None or messages == "":
        return []
    
    messages = str(messages)
    lines = messages.split('\n')
    user_messages = []
    
    for line in lines:
        if line.strip().startswith("Visiteur:"):
            # Extraire le contenu du message (après "Visiteur:")
            message_content = line.strip()[len("Visiteur:"):].strip()
            if message_content:  # Ne pas ajouter les messages vides
                user_messages.append(message_content)
    
    return user_messages

def extract_operator_messages(messages):
    """Extrait les messages de l'opérateur à partir du texte complet de la conversation."""
    if pd.isna(messages) or messages is None or messages == "":
        return []
    
    messages = str(messages)
    lines = messages.split('\n')
    operator_messages = []
    
    for line in lines:
        if line.strip().startswith("Opérateur:"):
            # Extraire le contenu du message (après "Opérateur:")
            message_content = line.strip()[len("Opérateur:"):].strip()
            if message_content:  # Ne pas ajouter les messages vides
                operator_messages.append(message_content)
    
    return operator_messages

def is_talking_about_past_harassment(message):
    """Détermine si le message parle de harcèlement passé (vécu par l'utilisateur) plutôt que d'un harcèlement actif."""
    past_indicators = [
        "j'ai été harcelé", "j'ai vécu du harcèlement", "j'ai subi", 
        "dans mon passé", "quand j'étais", "à l'école", "durant ma scolarité",
        "avant", "autrefois", "dans le passé", "j'ai connu", "j'ai traversé"
    ]
    
    message_lower = message.lower()
    return any(indicator in message_lower for indicator in past_indicators)

def is_talking_about_personal_experience(message):
    """Détermine si le message parle d'une expérience personnelle plutôt que d'une demande d'information."""
    experience_indicators = [
        "j'ai", "je suis", "j'utilise", "je fais", "mon", "ma", "mes",
        "pour moi", "je me sens", "je pense", "je crois", "je trouve"
    ]
    
    message_lower = message.lower()
    return any(indicator in message_lower for indicator in experience_indicators)

def contains_positive_emotion(message):
    """Détecte si le message contient des expressions d'émotions positives."""
    positive_emotions = [
        "content", "heureu", "bien", "super", "génial", "cool", "aime", 
        "plaisir", "merci", "reconnaissant", "soulagé", "espoir", "confiance"
    ]
    
    message_lower = message.lower()
    return any(emotion in message_lower for emotion in positive_emotions)

def detect_topic_changes(user_messages, threshold=0.2, min_messages=5):
    """
    Détecte les changements significatifs de sujet dans les messages de l'utilisateur.
    Version améliorée avec un seuil plus bas et un minimum de messages requis.
    """
    if len(user_messages) < min_messages:
        return []
    
    # Créer un vectoriseur TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='french')
    
    try:
        # Transformer les messages en vecteurs TF-IDF
        tfidf_matrix = vectorizer.fit_transform(user_messages)
        
        # Calculer la similarité cosinus entre messages consécutifs
        similarities = []
        topic_changes = []
        
        for i in range(1, len(user_messages)):
            # Calculer la similarité entre le message actuel et le précédent
            similarity = cosine_similarity(
                tfidf_matrix[i:i+1], 
                tfidf_matrix[i-1:i]
            )[0][0]
            
            similarities.append(similarity)
            
            # Si la similarité est inférieure au seuil ET que le message est suffisamment long
            # pour éviter de détecter des changements sur des messages courts
            if similarity < threshold and len(user_messages[i]) > 20 and len(user_messages[i-1]) > 20:
                topic_changes.append({
                    'index': i,
                    'message': user_messages[i],
                    'similarity': similarity
                })
        
        # Filtrer les changements trop proches les uns des autres (dans une fenêtre de 3 messages)
        filtered_changes = []
        if topic_changes:
            filtered_changes.append(topic_changes[0])
            for change in topic_changes[1:]:
                if change['index'] - filtered_changes[-1]['index'] > 3:
                    filtered_changes.append(change)
        
        return filtered_changes
    except:
        # En cas d'erreur (par exemple, si les messages sont trop courts)
        return []

def detect_manipulation_patterns(messages):
    """
    Détecte les schémas de manipulation dans les messages.
    Version améliorée avec une meilleure détection contextuelle.
    """
    if pd.isna(messages) or messages is None or messages == "":
        return []
    
    manipulation_patterns = []
    
    # Extraire les messages de l'utilisateur
    user_messages = extract_user_messages(messages)
    
    # Patterns de manipulation à rechercher (version améliorée)
    gaslighting_phrases = [
        "tu te trompes complètement", "tu imagines des choses", "ce n'est jamais arrivé", 
        "tu es fou", "tu inventes tout", "tu exagères toujours", "tu es trop sensible",
        "personne ne te croira"
    ]
    
    isolation_phrases = [
        "ne fais pas confiance à", "ils te mentent tous", "ils ne comprennent pas",
        "je suis le seul qui", "personne d'autre ne peut", "ils ne t'aiment pas vraiment",
        "ils parlent dans ton dos", "ils se moquent de toi"
    ]
    
    guilt_phrases = [
        "après tout ce que j'ai fait pour toi", "si tu m'aimais vraiment", "c'est entièrement ta faute",
        "tu me dois au moins", "regarde ce que tu m'as fait faire", "tu me rends malheureux",
        "tu me déçois toujours", "j'ai tout sacrifié pour toi"
    ]
    
    # Vérifier les patterns de gaslighting avec contexte
    gaslighting_matches = []
    for i, message in enumerate(user_messages):
        message_lower = message.lower()
        for phrase in gaslighting_phrases:
            if phrase in message_lower and not is_talking_about_past_harassment(message):
                gaslighting_matches.append({
                    'index': i,
                    'message': message,
                    'pattern': 'gaslighting',
                    'phrase': phrase
                })
    
    if len(gaslighting_matches) >= 2:  # Au moins 2 occurrences pour confirmer un pattern
        manipulation_patterns.append({
            'type': 'Gaslighting',
            'description': 'Tentatives de faire douter l\'opérateur de sa perception ou de sa mémoire',
            'occurrences': len(gaslighting_matches),
            'examples': gaslighting_matches[:2]  # Limiter à 2 exemples
        })
    
    # Vérifier les patterns d'isolation avec contexte
    isolation_matches = []
    for i, message in enumerate(user_messages):
        message_lower = message.lower()
        for phrase in isolation_phrases:
            if phrase in message_lower and not is_talking_about_past_harassment(message):
                isolation_matches.append({
                    'index': i,
                    'message': message,
                    'pattern': 'isolation',
                    'phrase': phrase
                })
    
    if len(isolation_matches) >= 2:  # Au moins 2 occurrences pour confirmer un pattern
        manipulation_patterns.append({
            'type': 'Isolation',
            'description': 'Tentatives d\'isoler l\'opérateur ou de créer de la méfiance envers les autres',
            'occurrences': len(isolation_matches),
            'examples': isolation_matches[:2]
        })
    
    # Vérifier les patterns de culpabilisation avec contexte
    guilt_matches = []
    for i, message in enumerate(user_messages):
        message_lower = message.lower()
        for phrase in guilt_phrases:
            if phrase in message_lower and not is_talking_about_past_harassment(message):
                guilt_matches.append({
                    'index': i,
                    'message': message,
                    'pattern': 'guilt',
                    'phrase': phrase
                })
    
    if len(guilt_matches) >= 2:  # Au moins 2 occurrences pour confirmer un pattern
        manipulation_patterns.append({
            'type': 'Culpabilisation',
            'description': 'Tentatives de faire culpabiliser l\'opérateur',
            'occurrences': len(guilt_matches),
            'examples': guilt_matches[:2]
        })
    
    # Détecter les changements brusques de sujet (avec seuil amélioré)
    topic_changes = detect_topic_changes(user_messages, threshold=0.2, min_messages=5)
    if len(topic_changes) >= 3:  # Au moins 3 changements brusques pour confirmer un pattern
        manipulation_patterns.append({
            'type': 'Changements de sujet',
            'description': 'Changements brusques de sujet qui peuvent indiquer une tentative de manipulation',
            'occurrences': len(topic_changes),
            'examples': topic_changes[:2]
        })
    
    # Détecter les cycles d'abus (alternance entre agression et réconciliation)
    aggression_phrases = ["tu es stupide", "tu es inutile", "je déteste", "tu m'énerves", "ferme ta gueule"]
    reconciliation_phrases = ["je suis désolé", "pardonne-moi", "je ne voulais pas", "je t'aime", "tu es important"]
    
    aggression_indices = []
    reconciliation_indices = []
    
    for i, message in enumerate(user_messages):
        message_lower = message.lower()
        
        # Vérifier les phrases d'agression
        if any(phrase in message_lower for phrase in aggression_phrases):
            aggression_indices.append(i)
        
        # Vérifier les phrases de réconciliation
        if any(phrase in message_lower for phrase in reconciliation_phrases):
            reconciliation_indices.append(i)
    
    # Détecter les cycles (agression suivie de réconciliation)
    cycles = []
    for agg_idx in aggression_indices:
        for rec_idx in reconciliation_indices:
            # Si la réconciliation suit l'agression dans un délai raisonnable (1-3 messages)
            if 1 <= rec_idx - agg_idx <= 3:
                cycles.append({
                    'aggression_index': agg_idx,
                    'reconciliation_index': rec_idx,
                    'aggression_message': user_messages[agg_idx],
                    'reconciliation_message': user_messages[rec_idx]
                })
    
    if len(cycles) >= 2:  # Au moins 2 cycles pour confirmer un pattern
        manipulation_patterns.append({
            'type': 'Cycles d\'abus',
            'description': 'Alternance entre agression et réconciliation, typique des relations abusives',
            'occurrences': len(cycles),
            'examples': cycles[:2]
        })
    
    return manipulation_patterns

def analyze_chat_content(messages):
    """
    Analyse contextuelle avancée du contenu d'un chat pour détecter des signes d'abus.
    Version améliorée avec une meilleure catégorisation des phrases problématiques.
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
    
    # Analyse du sentiment et résumé
    sentiment = analyze_sentiment(" ".join(user_messages)) if user_messages else "Neutre"
    summary = generate_simple_summary(messages)
    emotions = [analyze_sentiment(msg) for msg in user_messages] if user_messages else []
    emotion_counts = Counter(emotions)
    
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
    
    # Détecter si la personne parle de sa sexualité dans un contexte légitime
    legitimate_sexual_discussion_indicators = [
        "orientation sexuelle", "identité de genre", "difficulté dans ma relation",
        "problème avec ma partenaire", "problème avec mon partenaire",
        "je ne sais pas comment gérer", "je me pose des questions sur",
        "je suis perdu", "je cherche des conseils", "je ne sais pas quoi faire"
    ]
    
    is_legitimate_sexual_discussion = any(indicator in messages.lower() for indicator in legitimate_sexual_discussion_indicators)
    
    # 2. DÉTECTION DE CONTENU SUICIDAIRE ET AUTOMUTILATOIRE
    
    # Mots-clés pour détecter les pensées suicidaires
    suicidal_keywords = [
        "suicide", "me tuer", "mourir", "en finir", "plus envie de vivre",
        "mettre fin à mes jours", "me suicider", "disparaître", "plus la force",
        "veux mourir", "veut mourir", "veux me suicider", "veut me suicider",
        "veux en finir", "veut en finir", "veux plus vivre", "veut plus vivre"
    ]
    
    # Vérifier si les messages contiennent des pensées suicidaires
    suicidal_messages = []
    for message in user_messages:
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in suicidal_keywords):
            suicidal_messages.append(message)
    
    if suicidal_messages:
        risk_score += 40  # Score élevé pour les pensées suicidaires
        risk_factors.append(f"Pensées suicidaires ({len(suicidal_messages)} occurrences)")
        problematic_phrases["Pensées suicidaires"] = suicidal_messages[:3]
    
    # 3. DÉTECTION DE HARCÈLEMENT SEXUEL ENVERS L'OPÉRATEUR
    
    # Mots-clés pour détecter le harcèlement sexuel envers l'opérateur
    sexual_harassment_keywords = [
        "tu aimes le sexe", "t'aimes sucer", "tu veux baiser", "tu préfères avaler",
        "tu es excité", "tu bandes", "tu mouilles", "tu te masturbes",
        "montre-moi tes seins", "je peux te montrer ma bite", "je te montre mon sexe",
        "tu veux qu'on baise", "on peut se branler", "tu es sexy", "je veux te baiser"
    ]
    
    # Vérifier si les messages contiennent du harcèlement sexuel envers l'opérateur
    sexual_harassment_messages = []
    for message in user_messages:
        message_lower = message.lower()
        # Vérifier que ce n'est pas une phrase suicidaire déjà détectée
        if any(keyword in message_lower for keyword in sexual_harassment_keywords) and message not in suicidal_messages:
            sexual_harassment_messages.append(message)
    
    if sexual_harassment_messages:
        operator_harassment = True
        risk_score += 50  # Score très élevé pour le harcèlement sexuel envers l'opérateur
        risk_factors.append(f"Harcèlement sexuel envers l'opérateur ({len(sexual_harassment_messages)} occurrences)")
        problematic_phrases["Harcèlement sexuel"] = sexual_harassment_messages[:3]
    
    # 4. DÉTECTION DE CONTENU SEXUEL EXPLICITE INAPPROPRIÉ
    
    # Mots-clés pour détecter le contenu sexuel explicite inapproprié
    explicit_sexual_content_keywords = [
        "je me masturbe", "je bande", "je suis excité", "je suis en train de",
        "je me caresse", "je jouis", "je vais jouir", "je suis bandé",
        "mon pénis", "ma bite", "mon gland", "je suce", "je me fais sucer",
        "happy ending", "je me soulage", "je me branle"
    ]
    
    # Vérifier si les messages contiennent du contenu sexuel explicite inapproprié
    explicit_sexual_content_messages = []
    for message in user_messages:
        message_lower = message.lower()
        # Vérifier que ce n'est pas une phrase déjà détectée dans une autre catégorie
        if (any(keyword in message_lower for keyword in explicit_sexual_content_keywords) and 
            message not in suicidal_messages and 
            message not in sexual_harassment_messages and
            not is_legitimate_sexual_discussion and 
            not is_trauma_narrative):
            explicit_sexual_content_messages.append(message)
    
    if explicit_sexual_content_messages:
        risk_score += min(len(explicit_sexual_content_messages) * 10, 40)  # Max 40 points
        risk_factors.append(f"Contenu sexuel explicite inapproprié ({len(explicit_sexual_content_messages)} occurrences)")
        problematic_phrases["Contenu sexuel explicite"] = explicit_sexual_content_messages[:3]
    
    # 5. DÉTECTION DE MENACES ET COMPORTEMENTS AGRESSIFS
    
    # Mots-clés pour détecter les menaces et comportements agressifs
    aggressive_keywords = [
        "je vais te tuer", "je sais où tu habites", "je vais te retrouver",
        "je vais te faire mal", "je vais te frapper", "je vais te violer",
        "je te surveille", "je t'observe", "je connais ton adresse",
        "je vais venir", "je vais te faire payer", "tu vas regretter"
    ]
    
    # Vérifier si les messages contiennent des menaces ou comportements agressifs
    aggressive_messages = []
    for message in user_messages:
        message_lower = message.lower()
        # Vérifier que ce n'est pas une phrase déjà détectée dans une autre catégorie
        if (any(keyword in message_lower for keyword in aggressive_keywords) and 
            message not in suicidal_messages and 
            message not in sexual_harassment_messages and
            message not in explicit_sexual_content_messages):
            aggressive_messages.append(message)
    
    if aggressive_messages:
        risk_score += 45  # Score élevé pour les menaces
        risk_factors.append(f"Menaces ou comportements agressifs ({len(aggressive_messages)} occurrences)")
        problematic_phrases["Menaces et comportements agressifs"] = aggressive_messages[:3]
    
    # 6. ANALYSE DU COMPORTEMENT GLOBAL
    
    # Détecter si l'utilisateur ignore les avertissements de l'opérateur
    warnings_from_operator = [
        "mauvais usage", "mettre fin", "raccrocher", "pas à l'aise", 
        "comportement inapproprié", "ne pas continuer", "je ne peux pas"
    ]
    
    operator_warnings = []
    for message in operator_messages:
        message_lower = message.lower()
        if any(warning in message_lower for warning in warnings_from_operator):
            operator_warnings.append(message)
    
    # Vérifier si l'utilisateur continue son comportement après un avertissement
    if operator_warnings:
        # Trouver l'index du premier avertissement
        first_warning_index = None
        for i, message in enumerate(operator_messages):
            message_lower = message.lower()
            if any(warning in message_lower for warning in warnings_from_operator):
                first_warning_index = i
                break
        
        # Vérifier si l'utilisateur continue son comportement après l'avertissement
        if first_warning_index is not None and first_warning_index < len(user_messages) - 1:
            messages_after_warning = user_messages[first_warning_index + 1:]
            
            # Vérifier si les messages après l'avertissement contiennent du contenu problématique
            problematic_after_warning = []
            for message in messages_after_warning:
                message_lower = message.lower()
                if (any(keyword in message_lower for keyword in explicit_sexual_content_keywords + 
                        sexual_harassment_keywords + aggressive_keywords)):
                    problematic_after_warning.append(message)
            
            if problematic_after_warning:
                risk_score += 30  # Score élevé pour ignorer les avertissements
                risk_factors.append("Ignore les avertissements de l'opérateur")
                problematic_phrases["Ignore les avertissements"] = problematic_after_warning[:3]
    
    # 7. AJUSTEMENTS FINAUX
    
    # Réduire le score global si c'est principalement un récit de traumatisme
    if is_trauma_narrative and risk_score > 0 and not operator_harassment:
        risk_score *= 0.5  # Réduire de 50%
        risk_factors.append("Score ajusté: récit de traumatisme personnel")
    
    # Réduire le score global si c'est une discussion sur la santé mentale
    if is_mental_health_discussion and risk_score > 0 and not operator_harassment:
        risk_score *= 0.7  # Réduire de 30%
        risk_factors.append("Score ajusté: discussion sur la santé mentale")
    
    # Réduire le score global si c'est une discussion légitime sur la sexualité
    if is_legitimate_sexual_discussion and risk_score > 0 and not operator_harassment:
        risk_score *= 0.6  # Réduire de 40%
        risk_factors.append("Score ajusté: discussion légitime sur la sexualité")
    
    # Limiter le score maximum à 100
    risk_score = min(int(risk_score), 100)
    
    # Détection des patterns de manipulation et changements de sujet
    manipulation_patterns = detect_manipulation_patterns(messages)
    topic_changes = detect_topic_changes(user_messages, threshold=0.2, min_messages=5)
    
    return risk_score, risk_factors, problematic_phrases, operator_harassment, manipulation_patterns, topic_changes

def get_abuse_risk_level(score):
    """Convertit un score de risque en niveau de risque textuel avec des seuils ajustés."""
    if score < 15:
        return "Faible"
    elif score < 35:
        return "Modéré"
    elif score < 60:
        return "Élevé"
    else:
        return "Très élevé"

def analyze_chats():
    """Analyse tous les chats et retourne un DataFrame avec les résultats, incluant l'analyse contextuelle."""
    df = get_ksaar_data()
    
    if df.empty:
        return pd.DataFrame()
    
    # Créer un DataFrame pour les résultats
    results = []
    
    # Analyser chaque conversation
    for _, row in df.iterrows():
        messages = row.get('messages', '')
        
        # Utiliser la nouvelle fonction d'analyse contextuelle
        try:
            risk_score, risk_factors, problematic_phrases, operator_harassment, manipulation_patterns, topic_changes = analyze_chat_content(messages)
        except ValueError:
            # Fallback si l'analyse échoue
            risk_score, risk_factors = analyze_chat_content(messages)[:2]
            problematic_phrases = {}
            operator_harassment = False
            manipulation_patterns = []
            topic_changes = []
        
        if risk_score > 0:  # Ne garder que les conversations avec un risque non nul
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
                'id_chat': row.get('id_chat'),
                'Crée le': row.get('Crée le'),
                'Antenne': row.get('Antenne'),
                'Volunteer_Location': row.get('Volunteer_Location'),
                'Score de risque': risk_score,
                'Niveau de risque': get_abuse_risk_level(risk_score),
                'Facteurs de risque': ', '.join(risk_factors),
                'messages': messages[:500] + '...' if len(str(messages)) > 500 else messages
            }
            
            # Ajouter les nouvelles colonnes
            if problematic_phrases:
                result_dict['Phrases problématiques'] = phrases_text
            
            result_dict['Harcèlement opérateur'] = "Oui" if operator_harassment else "Non"
            
            # Ajouter les résultats de l'analyse contextuelle
            if manipulation_patterns:
                result_dict['Analyse contextuelle'] = manipulation_text
                result_dict['Schémas de manipulation'] = len(manipulation_patterns)
            else:
                result_dict['Analyse contextuelle'] = "Aucun schéma de manipulation détecté"
                result_dict['Schémas de manipulation'] = 0
            
            if topic_changes:
                result_dict['Changements de sujet'] = len(topic_changes)
            else:
                result_dict['Changements de sujet'] = 0
            
            
            result_dict["Sentiment global"] = sentiment
            result_dict["Résumé"] = summary
            result_dict["Répartition émotions"] = dict(emotion_counts)
    results.append(result_dict)
    
    # Créer le DataFrame et trier par score de risque
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = cluster_chats(results_df)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Score de risque', ascending=False)
    
    return results_df

def identify_potentially_abusive_chats(df):
    """
    Fonction préliminaire pour identifier rapidement les chats potentiellement abusifs
    en utilisant des mots-clés et des patterns simples.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Liste étendue de mots-clés pour la détection préliminaire
    abuse_keywords = [
        # Contenu sexuel explicite
        "sexe", "bite", "penis", "vagin", "masturb", "bander", "sucer", "baiser",
        "jouir", "éjacul", "ejacul", "orgasm", "porno", "cul", "nichon", "seins","sexe", "bite", "queue", "pénis", "penis", "zboub", "vagin", "chatte", "cunni",
        "masturb", "branler", "branlette", "fap", "fellation", "pipe", "sucer",
        "bander", "gode", "godemichet", "baiser", "ken", "niquer", "niqué", "niquee",
        "sodom", "sodomie", "anal", "dp", "orgie", "orgasm", "orgasme", "jouir",
        "gicler", "giclée", "éjacul", "ejacul", "cum", "creampie", "facial", "porno",
        "porn", "xxx", "cul", "nichon", "nichons", "sein", "seins", "boobs", "boobies",
        "téton", "tétons", "nipple", "rimming", "69", "doggystyle", "photo", "photos", "pic", "pics", "send pic", "send pics", "nude", "nudes",
        "déshabille", "deshabille", "dégage tes fringues", "montre-moi", "montre moi",
        "montre-ta", "montre ta", "montre-tes", "montre tes", "caméra", "camera",
        "webcam", "vidéo", "video", "snap", "snapchat", "insta", "instagram", "facebook",
        "onlyfans", "strip", "striptease", "strip tease",
        
        
        # Harcèlement et menaces
        "connard", "salope", "pute", "enculé", "encule", "pd", "tapette", "nègre", "bite" , "bit"
        "negre", "bougnoule", "suicide", "tuer", "mourir", "crever", "adresse",
        "menace", "frapper", "battre", "harcèle", "harcele", "stalker", "connard", "connasse", "conasse", "con", "sale con", "salope", "salopes",
        "pute", "putain", "putes", "enculé", "encule", "enculer", "ntm", "nique ta mère",
        "fdp", "fils de pute", "merde", "batard", "bâtard", "mongol", "mongoloïde",
        "débil", "retard", "retarded", "idiot", "imbécile", "moron", "asshole",
        "bitch", "slut", "whore", "motherfucker", "fuck", "fucking", "stfu",
        "kys", "kill yourself", "go die", "die", "suicide", "hang yourself",
        "pends-toi", "pend toi", "crever", "tuer", "je vais te tuer", "je vais te crever",
        "menace", "menacer", "adresse", "je connais ton adresse", "je vais venir",
        "frapper", "battre", "je vais te frapper", "je vais te battre", "stalker", "nègre", "negre", "negro", "nigger", "nigga", "youpin", "youpine", "kike",
        "juifde", "yid", "bougnoule", "bougnoul", "raghead", "terrorist", "islamiste",
        "paki", "pakis", "wog", "chink", "chinetoque", "gook", "tranny", "trannie",
        "fag", "faggot", "tapette", "pd", "tarlouze", "gitan", "tsigane", "zingaro",
        "roma", "rom", "romanos", "spic","pédophilie", "pedophilie", "pédophile", "pedophile", "pedo", "map",
        "underage", "mineur", "mineure", "minor", "teen", "preteena", "preteens",
        "young girl", "young boy", "loli", "shota", "end my life", "slit wrists", "slit your wrists", "cutting", "self-harm",
        "selfharm", "auto-mutilation", "automutilation", "noose", "rope"
        
        # Manipulation
        "tu te trompes", "tu imagines", "personne ne te croira", "tu es fou",
        "après tout ce que j'ai fait", "si tu m'aimais", "c'est ta faute",
        "tu me dois", "regarde ce que tu m'as fait faire", "si tu pars", "tu ne peux pas vivre sans moi", "c'est pour ton bien",
        "tu le voulais", "tu seras responsable", "tu me triggers",  "photo", "photos", "pic", "pics", "send pic", "send pics", "nude", "nudes",
        "déshabille", "deshabille", "dégage tes fringues", "montre-moi", "montre moi",
         "montre-ta", "montre ta", "montre-tes", "montre tes", "caméra", "camera",
        "webcam", "vidéo", "video", "snap", "snapchat", "insta", "instagram", "facebook",
        "onlyfans", "strip", "striptease", "strip tease",
        
        # Demandes inappropriées
        "photo", "nue", "nu", "déshabille", "deshabille", "montre-moi", "montre moi",
        "caméra", "camera", "vidéo", "video", "snapchat", "instagram", "facebook", 
        
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

def main():
    st.set_page_config(**ksaar_config['app_config'])
    
    if check_password():
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
        
        # Sélecteur pour choisir entre Chats, Appels et Analyse IA des abus
        page = st.sidebar.radio("Navigation", ["Chats", "Appels", "Analyse IA des abus"], key="navigation")
        
        # Afficher le contenu en fonction de la sélection
        if page == "Chats":
            tab1, _, _ = st.tabs(["Chats", "Appels", "Analyse IA des abus"])
            with tab1:
                display_chats()
        elif page == "Appels":
            _, tab2, _ = st.tabs(["Chats", "Appels", "Analyse IA des abus"])
            with tab2:
                display_calls()
        else:
            _, _, tab3 = st.tabs(["Chats", "Appels", "Analyse IA des abus"])
            with tab3:
                display_abuse_analysis()

if __name__ == "__main__":
    main()
