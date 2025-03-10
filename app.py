import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import hashlib

# Au lieu d'importer depuis config.py
credentials = st.secrets["credentials"]
ksaar_config = st.secrets["ksaar_config"]

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

def get_ksaar_data():
    """Récupère les données depuis l'API Ksaar."""
    try:
        # Vérifier si les données sont en cache et si elles ont plus de 5 minutes
        if ('chat_data' not in st.session_state or 
            'last_update' not in st.session_state or 
            (datetime.now() - st.session_state['last_update']).total_seconds() > 300):
            
            url = f"{ksaar_config['api_base_url']}/v1/workflows/{ksaar_config['workflow_id']}/records"
            auth = (ksaar_config['api_key_name'], ksaar_config['api_key_password'])
            
            all_records = []
            current_page = 1
            
            while True:
                params = {
                    "page": current_page,
                    "limit": 100,
                    "sort": "-createdAt"  # Tri par date décroissante
                }

                response = requests.get(url, params=params, auth=auth)
                
                if response.status_code == 200:
                    data = response.json()
                    records = data.get('results', [])
                    if not records:
                        break
                        
                    for record in records:
                        record_data = {
                            'Crée le': record.get('createdAt'),
                            'Modifié le': record.get('updatedAt'),
                            'IP': record.get('ip'),
                            'pnd_time': record.get('pnd_time'),
                            'id_chat': record.get('id_chat'),
                            'messages': record.get('messages'),
                            'last_user_message': record.get('last_user_msg_time'),
                            'last_op_message': record.get('cls_time')
                        }
                        all_records.append(record_data)
                    
                    if current_page >= data.get('lastPage', 1):
                        break
                    current_page += 1
                else:
                    st.error(f"Erreur lors de la récupération des données: {response.status_code}")
                    return pd.DataFrame()

            if not all_records:
                st.warning("Aucun enregistrement trouvé dans la réponse de l'API")
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

def get_calls_data():
    """Récupère les données d'appels depuis l'API Ksaar."""
    try:
        if 'calls_data' not in st.session_state:
            url = f"{ksaar_config['api_base_url']}/v1/workflows/deb92463-c3a5-4393-a3bf-1dd29a022cfe/records"
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
    """Génère un rapport HTML pour un chat."""
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

def main():
    st.set_page_config(**ksaar_config['app_config'])
    
    # Supprimer le rafraîchissement automatique qui cause la déconnexion
    # st.markdown("""<meta http-equiv="refresh" content="300">""", unsafe_allow_html=True)
    
    if check_password():
        st.title("Dashboard GASAS")
        
        # Ajouter un bouton de rafraîchissement manuel
        if st.sidebar.button("🔄 Rafraîchir les données"):
            # Effacer le cache des données
            if 'chat_data' in st.session_state:
                del st.session_state['chat_data']
            if 'calls_data' in st.session_state:
                del st.session_state['calls_data']
            st.rerun()
        
        # Ajouter le bouton de déconnexion
        if st.sidebar.button("🚪 Déconnexion"):
            # Effacer toutes les données de session
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Sélecteur pour choisir entre Chats et Appels
        page = st.sidebar.radio("Navigation", ["Chats", "Appels"])
        
        # Effacer les anciens filtres de la sidebar
        st.sidebar.empty()
        
        # Afficher le contenu en fonction de la sélection
        if page == "Chats":
            tab1, _ = st.tabs(["Chats", "Appels"])
            with tab1:
                display_chats()
        else:
            _, tab2 = st.tabs(["Chats", "Appels"])
            with tab2:
                display_calls()

def display_calls_filters():
    """Affiche les filtres pour les appels dans la sidebar."""
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
            default=statuts_uniques
        )
        
        antennes_uniques = sorted(df['Nom'].unique().tolist())
        antenne_selectionnee = st.sidebar.selectbox(
            'Antenne',
            ['Toutes les antennes'] + antennes_uniques
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

def display_calls():
    """Affiche les données des appels."""
    df = get_calls_data()
    
    if df.empty:
        st.warning("Aucune donnée d'appel n'a pu être récupérée.")
        return
    
    # Récupérer les filtres
    filters = display_calls_filters()
    if filters:
        # Application des filtres de date et statut
        mask = (df['Crée le'].dt.date >= filters['start_date']) & \
               (df['Crée le'].dt.date <= filters['end_date']) & \
               (df['Statut'].isin(filters['statut']))
        
        # Convertir les heures en objets time pour la comparaison
        def convert_to_time(time_str):
            try:
                if pd.isna(time_str):
                    return None
                return datetime.strptime(time_str, '%H:%M').time()
            except:
                return None
        
        # Appliquer la conversion aux colonnes d'heure
        df['Début appel_time'] = df['Début appel'].apply(convert_to_time)
        df['Fin appel_time'] = df['Fin appel'].apply(convert_to_time)
        
        # Filtrer par heure
        time_mask = pd.Series(True, index=df.index)
        valid_times = df['Début appel_time'].notna() & df['Fin appel_time'].notna()
        
        if valid_times.any():
            # Un appel est dans la plage horaire seulement si :
            # - son heure de début est dans la plage ET
            # - son heure de fin est dans la plage
            time_mask = valid_times & (
                (df['Début appel_time'] >= filters['start_time']) & 
                (df['Début appel_time'] <= filters['end_time']) &
                (df['Fin appel_time'] >= filters['start_time']) & 
                (df['Fin appel_time'] <= filters['end_time'])
            )
        
        mask &= time_mask
        
        # Ajout du filtre d'antenne
        if filters['antenne'] != 'Toutes les antennes':
            mask = mask & (df['Nom'] == filters['antenne'])
        
        filtered_df = df[mask].copy()
        
        # Supprimer les colonnes temporaires utilisées pour le filtrage
        filtered_df = filtered_df.drop(['Début appel_time', 'Fin appel_time'], axis=1)
        
        # Affichage des statistiques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre total d'appels", len(filtered_df))
        with col2:
            st.metric("Période", f"{filters['start_date'].strftime('%d/%m/%Y')} - {filters['end_date'].strftime('%d/%m/%Y')}")
        with col3:
            st.metric("Plage horaire", f"{filters['start_time'].strftime('%H:%M')} - {filters['end_time'].strftime('%H:%M')}")
        
        # Affichage des données
        st.subheader("Liste des appels")
        
        st.data_editor(
            filtered_df,
            use_container_width=True,
            column_config={
                "Crée le": st.column_config.DatetimeColumn("Crée le", format="DD/MM/YYYY HH:mm"),
                "Nom": st.column_config.TextColumn("Antenne"),
                "Numéro": st.column_config.TextColumn("Numéro"),
                "Statut": st.column_config.TextColumn("Statut"),
                "Début appel": st.column_config.TextColumn("Heure de début"),
                "Fin appel": st.column_config.TextColumn("Heure de fin")
            },
            column_order=["Crée le", "Nom", "Numéro", "Statut", "Début appel", "Fin appel"],
            height=500,
            num_rows="dynamic",
            key="calls_table",
            hide_index=True,
            disabled=True
        )

    if st.sidebar.button("Rafraîchir les données d'appels"):
        if 'calls_data' in st.session_state:
            del st.session_state['calls_data']
        st.rerun()

def display_chats_filters(df):
    """Affiche les filtres pour les chats dans la sidebar."""
    st.sidebar.header("Filtres des chats")
    
    # Configuration des dates par défaut
    default_start_date = datetime(2025, 1, 1)
    default_end_date = datetime.now()
    
    # Filtres de date
    start_date = st.sidebar.date_input(
        "Date de début",
        value=default_start_date,
        min_value=datetime(2025, 1, 1).date(),
        max_value=default_end_date,
        key="chat_start_date"
    )
    
    end_date = st.sidebar.date_input(
        "Date de fin",
        value=default_end_date,
        min_value=start_date,
        max_value=default_end_date,
        key="chat_end_date"
    )
    
    # Filtres d'heure
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_time = st.time_input('Heure de début', value=datetime.strptime('00:00', '%H:%M').time(), key="chat_start_time")
    with col2:
        end_time = st.time_input('Heure de fin', value=datetime.strptime('23:59', '%H:%M').time(), key="chat_end_time")
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'start_time': start_time,
        'end_time': end_time
    }

def display_chats():
    """Affiche les données des chats."""
    df = get_ksaar_data()
    
    if df.empty:
        st.warning("Aucune donnée n'a pu être récupérée.")
        return
    
    # Champ de recherche
    search_term = st.sidebar.text_input("Rechercher dans les messages", "")
    
    # Filtrer les messages en fonction du terme de recherche
    if search_term:
        df = df[df['messages'].str.contains(search_term, case=False, na=False)]
    
    # Configuration des dates par défaut
    default_start_date = datetime(2025, 1, 1)
    default_end_date = datetime.now()
    
    # Filtres dans la barre latérale pour les chats
    st.sidebar.header("Filtres des chats")
    
    # Filtres de date et heure
    start_date = st.sidebar.date_input(
        "Date de début",
        value=default_start_date,
        min_value=datetime(2025, 1, 1).date(),
        max_value=default_end_date,
        key="chat_start_date"
    )
    
    end_date = st.sidebar.date_input(
        "Date de fin",
        value=default_end_date,
        min_value=start_date,
        max_value=default_end_date,
        key="chat_end_date"
    )
    
    # Filtres d'heure
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_time = st.time_input('Heure de début', value=datetime.strptime('00:00', '%H:%M').time(), key="chat_start_time")
    with col2:
        end_time = st.time_input('Heure de fin', value=datetime.strptime('23:59', '%H:%M').time(), key="chat_end_time")
    
    # Application des filtres de date
    mask = (df['Crée le'].dt.date >= start_date) & \
           (df['Crée le'].dt.date <= end_date)
    
    # Convertir les heures en objets time pour la comparaison
    def convert_to_time(dt):
        try:
            if pd.isna(dt):
                return None
            return dt.time()
        except:
            return None
    
    # Créer des colonnes temporaires pour le filtrage des heures
    df['last_op_msg_time_obj'] = df['last_op_message'].apply(convert_to_time)
    df['last_user_msg_time_obj'] = df['last_user_message'].apply(convert_to_time)
    
    # Appliquer le filtre d'heure
    time_mask = pd.Series(True, index=df.index)
    valid_times = df['last_op_msg_time_obj'].notna() & df['last_user_msg_time_obj'].notna()
    
    if valid_times.any():
        time_mask = valid_times & (
            # L'heure de début (last_op_msg_time) doit être après ou égale à l'heure de début sélectionnée
            (df['last_op_msg_time_obj'] >= start_time) & 
            # L'heure de fin (last_user_msg_time) doit être avant ou égale à l'heure de fin sélectionnée
            (df['last_user_msg_time_obj'] <= end_time)
        )
    
    mask &= time_mask
    filtered_df = df[mask].copy()
    filtered_df['select'] = False
    
    # Formatage des heures pour l'affichage (HH:MM)
    filtered_df['last_op_msg_time'] = df['last_op_message'].dt.strftime('%H:%M')
    filtered_df['last_user_msg_time'] = df['last_user_message'].dt.strftime('%H:%M')
    
    # Affichage des statistiques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre total de chats", len(filtered_df))
    with col2:
        st.metric("Période", f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
    with col3:
        st.metric("Plage horaire", f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
    
    # Affichage des données
    st.subheader("Liste des chats")
    
    edited_df = st.data_editor(
        filtered_df,
        use_container_width=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Sélectionner", default=False),
            "Crée le": st.column_config.DatetimeColumn("Crée le", format="DD/MM/YYYY HH:mm"),
            "IP": st.column_config.TextColumn("IP"),
            "last_op_msg_time": st.column_config.TextColumn("Début du chat"),
            "id_chat": st.column_config.NumberColumn("ID Chat"),
            "messages": st.column_config.TextColumn("Messages", width="large"),
            "last_user_msg_time": st.column_config.TextColumn("Fin du chat")
        },
        column_order=[
            "select", "Crée le", "IP", "last_op_msg_time", "id_chat", 
            "messages", "last_user_msg_time"
        ],
        height=500,
        num_rows="dynamic",
        key="chat_table",
        hide_index=True
    )
    
    # Bouton pour générer le rapport uniquement pour les chats sélectionnés
    if st.button("Générer rapport(s) HTML"):
        selected_chats = edited_df[edited_df["select"]].copy()
        if not selected_chats.empty:
            for _, chat_data in selected_chats.iterrows():
                html_report = generate_chat_report(chat_data)
                st.download_button(
                    label=f"Télécharger le rapport pour le chat {chat_data['id_chat']}",
                    data=html_report,
                    file_name=f"rapport_chat_{chat_data['id_chat']}.html",
                    mime="text/html",
                    key=f"download_{chat_data['id_chat']}"
                )
        else:
            st.warning("Veuillez sélectionner au moins un chat pour générer un rapport.")

    if st.sidebar.button("Rafraîchir les données"):
        if 'chat_data' in st.session_state:
            del st.session_state['chat_data']
        st.rerun()

if __name__ == "__main__":
    main() 
