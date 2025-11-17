import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, date
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import nltk

# ==========================
# CONFIG & SECRETS
# ==========================

CONFIG_LOADED = False
CONFIG_ERROR = None

try:
    credentials = st.secrets["credentials"]
    ksaar_config = st.secrets["ksaar_config"]
    CONFIG_LOADED = True

    st.set_page_config(
        **ksaar_config.get(
            "app_config",
            {
                "page_title": "Dashboard GASAS",
                "page_icon": "📊",
                "layout": "wide",
                "initial_sidebar_state": "expanded",
            },
        )
    )
except Exception as e:
    CONFIG_ERROR = str(e)
    st.set_page_config(
        page_title="Dashboard GASAS",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    credentials = {}
    ksaar_config = {
        "api_base_url": "",
        "api_key_name": "",
        "api_key_password": "",
    }

# ==========================
# RESSOURCES PARTAGÉES
# ==========================

@st.cache_resource
def download_nltk_data():
    try:
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
    except Exception as e:
        st.warning(f"Impossible de télécharger les ressources NLTK : {e}")


download_nltk_data()


@st.cache_resource
def compile_abuse_patterns():
    abuse_keywords = [
        # sexual / insult / threat / suicide, etc.
        "sexe", "bite", "penis", "pénis", "vagin", "chatte", "masturb",
        "branler", "baiser", "ken", "niquer", "sodom", "anal", "orgasm",
        "porno", "porn", "xxx", "cul", "nichon", "sein", "boobs", "téton",
        "photo nue", "photo nu", "déshabille", "deshabille", "caméra",
        "camera", "video", "vidéo", "snapchat", "instagram", "onlyfans",
        "strip", "striptease",
        "connard", "salope", "pute", "enculé", "encule", "pd", "tapette",
        "nègre", "negre", "bougnoule",
        "suicide", "me tuer", "me suicider", "en finir", "plus envie de vivre",
        "mourir", "mettre fin à mes jours",
        "adresse", "je sais où tu", "je peux te trouver", "je vais venir",
        "je vais te retrouver",
        "harcèle", "harcele", "stalker", "menace", "frapper", "battre",
    ]
    pattern = re.compile("|".join(map(re.escape, abuse_keywords)), re.IGNORECASE)
    return pattern, abuse_keywords


# ==========================
# UTILITAIRES
# ==========================

def check_password() -> bool:
    """Petit login basique basé sur st.secrets['credentials']."""
    if st.session_state.get("authenticated", False):
        return True

    st.title("Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if username in credentials and password == credentials[username]:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("😕 Identifiants incorrects")

    return False


def load_data_paginated(df: pd.DataFrame, page_number: int, page_size: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    start_idx = page_number * page_size
    end_idx = start_idx + page_size

    if start_idx >= len(df):
        start_idx = 0
        end_idx = min(page_size, len(df))

    end_idx = min(end_idx, len(df))
    return df.iloc[start_idx:end_idx].copy()


def display_pagination_controls(total_items, page_size, current_page, key_prefix: str):
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if current_page > 0:
            if st.button("← Précédent", key=f"{key_prefix}_prev"):
                st.session_state[key_prefix + "_page"] = current_page - 1
                st.rerun()

    with col2:
        st.write(f"Page {current_page + 1} / {total_pages}")

    with col3:
        if current_page < total_pages - 1:
            if st.button("Suivant →", key=f"{key_prefix}_next"):
                st.session_state[key_prefix + "_page"] = current_page + 1
                st.rerun()


# ==========================
# MAPPINGS (ANTENNES / OPÉRATEURS)
# ==========================

def extract_antenne(msg, dept):
    """Logique proche de ta version Power BI."""
    if pd.isna(msg) or pd.isna(dept) or not msg or not dept:
        return "Inconnue"

    msg = str(msg)
    dept = str(dept)

    is_national = dept in ["Appels en attente (national)", "English calls (national)"]
    if not is_national:
        return dept

    start_texts = [
        'as no operators online in "Nightline ',
        'from "Nightline ',
        'de "Nightline ',
        'en "Nightline ',
    ]

    start_pos = None
    for text in start_texts:
        if text in msg:
            start_pos = msg.find(text) + len(text)
            break

    if start_pos is None:
        m = re.search(r'Nightline\s+([^"]+)', msg)
        if m:
            return m.group(1).strip()
        return "Inconnue"

    end_pos = msg.find('"', start_pos)
    if end_pos == -1:
        end_pos = msg.find(".", start_pos)
        if end_pos == -1:
            end_pos = len(msg)

    return msg[start_pos:end_pos].strip() or "Inconnue"


def get_normalized_antenne(antenne: str) -> str:
    if pd.isna(antenne) or not antenne:
        return "Inconnue"

    antenne = str(antenne)
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

    mapping = {
        1: "admin", 2: "NightlineParis1", 3: "NightlineParis2", 4: "NightlineParis3",
        5: "NightlineParis4", 6: "NightlineParis5", 7: "NightlineLyon1", 9: "NightlineParis6",
        12: "NightlineAnglophone1", 13: "NightlineAnglophone2", 14: "NightlineAnglophone3",
        16: "NightlineSaclay1", 18: "NightlineSaclay3", 19: "NightlineParis7",
        20: "NightlineParis8", 21: "NightlineLyon2", 22: "NightlineLyon3",
        26: "NightlineSaclay2", 30: "NightlineSaclay4", 31: "NightlineSaclay5",
        32: "NightlineSaclay6", 33: "NightlineLyon4", 34: "NightlineLyon5",
        35: "NightlineLyon6", 36: "NightlineLyon7", 37: "NightlineLyon8",
        38: "NightlineSaclay7", 40: "NightlineParis9", 42: "NightlineFormateur1",
        43: "NightlineAnglophone4", 44: "NightlineAnglophone5", 45: "NightlineParis10",
        46: "NightlineParis11", 47: "NightlineToulouse1", 48: "NightlineToulouse2",
        49: "NightlineToulouse3", 50: "NightlineToulouse4", 51: "NightlineToulouse5",
        52: "NightlineToulouse6", 53: "NightlineToulouse7", 54: "NightlineAngers1",
        55: "NightlineAngers2", 56: "NightlineAngers3", 57: "NightlineAngers4",
        58: "doubleecoute", 59: "NightlineNantes1", 60: "NightlineNantes2",
        61: "NightlineNantes3", 62: "NightlineNantes4", 63: "NightlineRouen1",
        64: "NightlineRouen2", 65: "NightlineRouen3", 67: "NightlineRouen4",
        68: "NightlineNantes5", 69: "NightlineNantes6", 70: "NightlineAngers5",
        71: "NightlineAngers6", 72: "NightlineRouen5", 73: "NightlineRouen6",
        74: "NightlineAngers7", 75: "NightlineLyon9", 76: "NightlineReims",
        77: "NightlineToulouse8", 78: "NightlineToulouse9", 79: "NightlineReims1",
        80: "NightlineReims2", 81: "NightlineReims3", 82: "NightlineReims4",
        83: "NightlineReims5", 84: "NightlineLille1", 85: "NightlineLille2",
        86: "NightlineLille3", 87: "NightlineLille4", 88: "NightlineRouen7",
        89: "NightlineRouen8", 90: "NightlineRouen9", 91: "NightlineRouen10",
        92: "NightlineRouen11", 93: "NightlineRouen12",
    }
    return mapping.get(operator_id, "Inconnu")


def get_volunteer_location(operator_name: str) -> str:
    if pd.isna(operator_name) or not operator_name:
        return "Autre"

    name = str(operator_name)
    if "NightlineAnglophone" in name:
        return "Paris_Ang"
    if "NightlineParis" in name:
        return "Paris"
    if "NightlineLyon" in name:
        return "Lyon"
    if "NightlineSaclay" in name:
        return "Saclay"
    if "NightlineToulouse" in name:
        return "Toulouse"
    if "NightlineAngers" in name:
        return "Angers"
    if "NightlineNantes" in name:
        return "Nantes"
    if "NightlineRouen" in name:
        return "Rouen"
    if "NightlineReims" in name:
        return "Reims"
    if "NightlineLille" in name:
        return "Lille"
    if "NightlineFormateur" in name:
        return "Formateur"
    if name == "admin":
        return "Admin"
    if name == "doubleecoute":
        return "Paris"
    return "Autre"


def get_antenne_from_dst(dst):
    if pd.isna(dst) or dst is None or dst == "":
        return None

    dst_str = str(dst).strip().replace("+", "").replace(".0", "").replace(" ", "")

    if dst_str in ["33999011163", "33999011065"]:
        return "Lille"
    if dst_str in ["33999011073"]:
        return "Marseille"
    if dst_str in ["33999011198", "33999011066"]:
        return "Lyon"
    if dst_str in ["33999011201", "33999011068"]:
        return "Paris"
    if dst_str in ["33999011263", "33999011072"]:
        return "Toulouse"
    if dst_str in ["33999011261", "33999011070"]:
        return "Reims"
    if dst_str in ["33999011199", "33999011067"]:
        return "Normandie"
    if dst_str in ["33999011074"]:
        return "National_Fr_Hors_Zone"
    if dst_str in ["33999011262", "33999011071"]:
        return "Saclay"
    if dst_str in ["33999011215", "33999011069"]:
        return "Pays de la Loire"
    return None


# ==========================
# CHARGEMENT DES DONNÉES
# ==========================

@st.cache_data(ttl=300)
def get_ksaar_chats():
    """Récupère les chats + pré-calcul des flags abusifs."""
    if not ksaar_config.get("api_base_url"):
        st.error("API base URL non configurée (secrets.ksaar_config.api_base_url manquant).")
        return pd.DataFrame()

    workflow_id = "1500d159-5185-4487-be1f-fa18c6c85ec5"  # chats
    url = f"{ksaar_config['api_base_url']}/v1/workflows/{workflow_id}/records"
    auth = (ksaar_config["api_key_name"], ksaar_config["api_key_password"])

    all_records = []
    current_page = 1
    pattern, abuse_keywords = compile_abuse_patterns()

    while True:
        params = {"page": current_page, "limit": 100, "sort": "-createdAt"}
        try:
            resp = requests.get(url, params=params, auth=auth, timeout=30)
        except Exception as e:
            st.error(f"Erreur de connexion à l'API Chats : {e}")
            break

        if resp.status_code != 200:
            st.error(f"Erreur API Chats (status {resp.status_code}) pour la page {current_page}")
            try:
                st.text(f"Réponse brute : {resp.text[:500]}")
            except Exception:
                pass
            break

        data = resp.json()
        records = data.get("results", [])
        if not records:
            # st.info(f"Aucun enregistrement de chat pour la page {current_page}.")
            break

        for record in records:
            rd = {
                "Crée le": record.get("createdAt"),
                "Modifié le": record.get("updatedAt"),
                "IP": record.get("IP 2", ""),
                "pnd_time": record.get("Date complète début 2"),
                "id_chat": record.get("Chat ID 2"),
                "messages": record.get("Conversation complète 2", ""),
                "last_user_message": record.get("Date complète fin 2"),
                "last_op_message": record.get("Date complète début 2"),
                "Message système 1": record.get("Message système 1", ""),
                "Département Origine 2": record.get("Département Origine 2", ""),
            }
            op_id = record.get("Opérateur ID (API) 1")
            op_name = get_operator_name(op_id)
            rd["Operateur_Name"] = op_name
            rd["Volunteer_Location"] = get_volunteer_location(op_name)

            raw_antenne = extract_antenne(rd["Message système 1"], rd["Département Origine 2"])
            rd["Antenne"] = get_normalized_antenne(raw_antenne)

            all_records.append(rd)

        if current_page >= data.get("lastPage", 1):
            break
        current_page += 1

    if not all_records:
        st.warning("Ksaar : la requête Chats a réussi mais aucun enregistrement n'a été retourné.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    for col in ["Crée le", "Modifié le", "pnd_time", "last_user_message", "last_op_message"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["messages"] = df["messages"].astype(str)
    df["messages_lower"] = df["messages"].str.lower()

    def is_abusive(text: str) -> bool:
        if not text:
            return False
        return bool(pattern.search(text))

    df["potentially_abusive"] = df["messages_lower"].apply(is_abusive)

    def abuse_score(text: str) -> int:
        if not text:
            return 0
        matches = pattern.findall(text)
        return len(matches)

    df["preliminary_score"] = df["messages_lower"].apply(abuse_score)

    return df



@st.cache_data(ttl=600)
def get_ksaar_calls():
    """Récupère les appels."""
    if not ksaar_config.get("api_base_url"):
        st.error("API base URL non configurée (secrets.ksaar_config.api_base_url manquant).")
        return pd.DataFrame()

    workflow_id = "deb92463-c3a5-4393-a3bf-1dd29a022cfe"  # appels
    url = f"{ksaar_config['api_base_url']}/v1/workflows/{workflow_id}/records"
    auth = (ksaar_config["api_key_name"], ksaar_config["api_key_password"])

    all_records = []
    current_page = 1

    def extract_time(ts):
        if not ts:
            return None
        try:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
            return dt.strftime("%H:%M")
        except Exception:
            return None

    while True:
        params = {"page": current_page, "limit": 100}
        try:
            resp = requests.get(url, params=params, auth=auth, timeout=30)
        except Exception as e:
            st.error(f"Erreur de connexion à l'API Appels : {e}")
            break

        if resp.status_code != 200:
            st.error(f"Erreur API Appels (status {resp.status_code}) pour la page {current_page}")
            try:
                st.text(f"Réponse brute : {resp.text[:500]}")
            except Exception:
                pass
            break

        data = resp.json()
        records = data.get("results", [])
        if not records:
            # st.info(f"Aucun enregistrement d'appel pour la page {current_page}.")
            break

        for record in records:
            dst = record.get("dst", "")
            rec = {
                "Crée le": record.get("createdAt"),
                "Nom": record.get("from_name", ""),
                "Numéro": record.get("from_number", ""),
                "Statut": record.get("disposition", ""),
                "Code_de_cloture": record.get("Code_de_cloture", ""),
                "Début appel": extract_time(record.get("answer")),
                "Fin appel": extract_time(record.get("end")),
                "dst": dst,
            }

            antenne_from_dst = get_antenne_from_dst(dst)
            if antenne_from_dst:
                rec["Antenne"] = antenne_from_dst
            elif record.get("from_name"):
                rec["Antenne"] = get_normalized_antenne(record["from_name"])
            else:
                rec["Antenne"] = "Inconnue"

            all_records.append(rec)

        if current_page >= data.get("lastPage", 1):
            break
        current_page += 1

    if not all_records:
        st.warning("Ksaar : la requête Appels a réussi mais aucun enregistrement n'a été retourné.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["Crée le"] = pd.to_datetime(df["Crée le"], errors="coerce")

    # ⚠️ TEMPORAIREMENT : on enlève le filtre sur 2025
    # df = df[df["Crée le"] >= "2025-01-01"]

    return df


# ==========================
# ANALYSE IA DES CHATS
# ==========================

def extract_user_messages(messages: str):
    if not messages:
        return []
    msgs = []
    current = ""
    is_user = False
    for line in messages.split("\n"):
        s = line.strip()
        if s.startswith("User:"):
            if current and is_user:
                msgs.append(current.strip())
            current = s.replace("User:", "").strip()
            is_user = True
        elif s.startswith("Operator:"):
            if current and is_user:
                msgs.append(current.strip())
            current = ""
            is_user = False
        elif is_user and s:
            current += " " + s
    if current and is_user:
        msgs.append(current.strip())
    return msgs


def extract_operator_messages(messages: str):
    if not messages:
        return []
    msgs = []
    current = ""
    is_op = False
    for line in messages.split("\n"):
        s = line.strip()
        if s.startswith("Operator:"):
            if current and is_op:
                msgs.append(current.strip())
            current = s.replace("Operator:", "").strip()
            is_op = True
        elif s.startswith("User:"):
            if current and is_op:
                msgs.append(current.strip())
            current = ""
            is_op = False
        elif is_op and s:
            current += " " + s
    if current and is_op:
        msgs.append(current.strip())
    return msgs


def detect_topic_changes(user_messages, threshold=0.2, min_messages=5):
    if len(user_messages) < min_messages:
        return []
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, stop_words="french")
    try:
        X = vectorizer.fit_transform(user_messages)
    except Exception:
        return []

    changes = []
    for i in range(1, len(user_messages)):
        sim = cosine_similarity(X[i:i+1], X[i-1:i])[0][0]
        if sim < threshold:
            changes.append(
                {
                    "index": i,
                    "previous_message": user_messages[i - 1],
                    "current_message": user_messages[i],
                    "similarity_score": sim,
                }
            )
    return changes


def detect_manipulation_patterns(messages: str):
    if not messages:
        return []
    patterns = []

    insistence_keywords = [
        "s'il te plait", "stp", "svp", "je t'en prie", "je t'en supplie",
        "allez", "réponds", "repond", "répond", "reponds",
    ]
    insistence_count = sum(1 for k in insistence_keywords if k in messages.lower())
    if insistence_count >= 3:
        patterns.append(
            {
                "type": "Insistance excessive",
                "description": "Utilisation répétée de formulations insistantes",
                "occurrences": insistence_count,
            }
        )

    guilt_keywords = [
        "tu ne veux pas m'aider", "tu refuses de m'aider", "tu ne veux pas me répondre",
        "tu m'ignores", "c'est de ta faute", "à cause de toi",
    ]
    guilt_msgs = [l for l in messages.split("\n") if any(k in l.lower() for k in guilt_keywords)]
    if guilt_msgs:
        patterns.append(
            {
                "type": "Culpabilisation",
                "description": "Tentatives de faire culpabiliser l'opérateur",
                "occurrences": len(guilt_msgs),
                "examples": guilt_msgs[:3],
            }
        )

    threat_keywords = [
        "tu vas voir", "tu regretteras", "je vais me plaindre", "je sais où tu",
        "je peux te trouver",
    ]
    threat_msgs = [l for l in messages.split("\n") if any(k in l.lower() for k in threat_keywords)]
    if threat_msgs:
        patterns.append(
            {
                "type": "Menaces voilées",
                "description": "Menaces plus ou moins directes",
                "occurrences": len(threat_msgs),
                "examples": threat_msgs[:3],
            }
        )

    return patterns


def analyze_chat_content(messages: str):
    """Analyse avancée, mais uniquement pour les chats sélectionnés (donc peu nombreux)."""
    if not messages:
        return 0, [], {}, False, [], []

    msgs = str(messages)
    risk_score = 0
    risk_factors = []
    problematic_phrases = {}
    operator_harassment = False

    user_msgs = extract_user_messages(msgs)

    suicidal_keywords = [
        "suicide", "me tuer", "me suicider", "en finir", "mettre fin à mes jours",
        "plus envie de vivre", "mourir",
    ]
    suicidal_msgs = [
        m for m in user_msgs if any(k in m.lower() for k in suicidal_keywords)
    ]
    if suicidal_msgs:
        risk_score += 40
        risk_factors.append(f"Pensées suicidaires ({len(suicidal_msgs)} occur.)")
        problematic_phrases["Pensées suicidaires"] = suicidal_msgs[:3]

    harass_keywords = [
        "tu aimes le sexe", "tu veux baiser", "tu es excité", "tu mouilles",
        "tu bandes", "t'aimes sucer", "tu te masturbes",
    ]
    harass_msgs = [
        m for m in user_msgs if any(k in m.lower() for k in harass_keywords)
    ]
    if harass_msgs:
        operator_harassment = True
        risk_score += 50
        risk_factors.append(f"Harcèlement sexuel ({len(harass_msgs)} occur.)")
        problematic_phrases["Harcèlement sexuel"] = harass_msgs[:3]

    manipulation_patterns = detect_manipulation_patterns(msgs)
    if manipulation_patterns:
        risk_score += len(manipulation_patterns) * 10
        risk_factors.append(f"Patterns de manipulation ({len(manipulation_patterns)})")

    topic_changes = detect_topic_changes(user_msgs)
    if len(topic_changes) > 2:
        risk_score += min(len(topic_changes) * 5, 20)
        risk_factors.append(f"Changements de sujet fréquents ({len(topic_changes)})")

    risk_score = min(int(risk_score), 100)
    return risk_score, risk_factors, problematic_phrases, operator_harassment, manipulation_patterns, topic_changes


def get_abuse_risk_level(score: int) -> str:
    if score >= 80:
        return "Très élevé"
    if score >= 60:
        return "Élevé"
    if score >= 40:
        return "Modéré"
    if score >= 20:
        return "Faible"
    return "Très faible"


# ==========================
# AFFICHAGE : APPELS
# ==========================

def display_calls():
    df = get_ksaar_calls()
    if df.empty:
        st.warning("Aucune donnée d'appel.")
        return

    st.subheader("Filtres appels")

    default_start = max(df["Crée le"].min().date(), date.today() - timedelta(days=7))
    default_end = df["Crée le"].max().date()

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Date de début", value=default_start)
    with c2:
        end_date = st.date_input("Date de fin", value=default_end, min_value=start_date)

    c3, c4 = st.columns(2)
    with c3:
        start_time = st.time_input("Heure de début", value=datetime.strptime("00:00", "%H:%M").time())
    with c4:
        end_time = st.time_input("Heure de fin", value=datetime.strptime("23:59", "%H:%M").time())

    c5, c6 = st.columns(2)
    with c5:
        statuts = sorted(df["Statut"].dropna().unique().tolist())
        statut_sel = st.multiselect("Statut", statuts, default=statuts)
    with c6:
        codes = df["Code_de_cloture"].fillna("(vide)").unique().tolist()
        codes = sorted(codes)
        code_sel = st.multiselect("Code de clôture", codes, default=codes)

    mask = (df["Crée le"].dt.date >= start_date) & (df["Crée le"].dt.date <= end_date)

    # filtre heure
    def to_time(dt):
        try:
            return dt.time()
        except Exception:
            return None

    tmp = df.copy()
    tmp["time"] = tmp["Crée le"].apply(to_time)
    if start_time <= end_time:
        mask &= (tmp["time"] >= start_time) & (tmp["time"] <= end_time)
    else:
        # plage type 21h–06h
        mask &= (tmp["time"] >= start_time) | (tmp["time"] <= end_time)

    if statut_sel:
        mask &= tmp["Statut"].isin(statut_sel)
    if code_sel:
        mask &= tmp["Code_de_cloture"].fillna("(vide)").isin(code_sel)

    fdf = tmp[mask].copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Nb appels", len(fdf))
    with c2:
        st.metric("Période", f"{start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}")
    with c3:
        st.metric("Plage horaire", f"{start_time.strftime('%H:%M')} → {end_time.strftime('%H:%M')}")

    if "calls_page" not in st.session_state:
        st.session_state["calls_page"] = 0

    PAGE_SIZE = 50
    page = st.session_state["calls_page"]
    paginated = load_data_paginated(fdf, page, PAGE_SIZE)

    paginated["Code_de_cloture"] = paginated["Code_de_cloture"].fillna("(vide)")
    paginated["select"] = False

    edited = st.data_editor(
        paginated,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "select": st.column_config.CheckboxColumn("Sélectionner", default=False),
            "Crée le": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
            "Nom": st.column_config.TextColumn("Nom (origine)"),
            "Antenne": st.column_config.TextColumn("Antenne"),
            "Numéro": st.column_config.TextColumn("Numéro"),
            "Statut": st.column_config.TextColumn("Statut"),
            "Code_de_cloture": st.column_config.TextColumn("Code de clôture"),
            "Début appel": st.column_config.TextColumn("Heure début"),
            "Fin appel": st.column_config.TextColumn("Heure fin"),
        },
    )

    display_pagination_controls(len(fdf), PAGE_SIZE, page, key_prefix="calls")

    if st.button("Analyser les appels sélectionnés"):
        sel = edited[edited["select"]]
        if sel.empty:
            st.warning("Aucun appel sélectionné.")
        else:
            st.markdown("### Détails des appels sélectionnés")
            for _, row in sel.iterrows():
                st.write(f"**Date :** {row['Crée le']}")
                st.write(f"**Antenne :** {row['Antenne']}")
                st.write(f"**Numéro :** {row['Numéro']}")
                st.write(f"**Statut :** {row['Statut']}")
                st.write(f"**Code de clôture :** {row['Code_de_cloture']}")
                st.write(f"**Heure début :** {row['Début appel']}")
                st.write(f"**Heure fin :** {row['Fin appel']}")
                st.write("---")

    if st.sidebar.button("🔄 Rafraîchir les appels"):
        get_ksaar_calls.clear()
        st.experimental_rerun()


# ==========================
# AFFICHAGE : ANALYSE IA ABUS
# ==========================

def display_abuse_analysis():
    st.title("Analyse IA des chats potentiellement abusifs")

    df = get_ksaar_chats()
    if df.empty:
        st.warning("Aucune donnée de chat.")
        return

    c1, c2 = st.columns(2)
    with c1:
        default_start = max(df["Crée le"].min().date(), date.today() - timedelta(days=30))
        start_date = st.date_input("Date de début", value=default_start)
    with c2:
        end_date = st.date_input("Date de fin", value=df["Crée le"].max().date(), min_value=start_date)

    use_time_filter = st.checkbox("Filtrer par heure", value=False)
    if use_time_filter:
        c3, c4 = st.columns(2)
        with c3:
            start_time = st.time_input("Heure de début", value=datetime.strptime("00:00", "%H:%M").time())
        with c4:
            end_time = st.time_input("Heure de fin", value=datetime.strptime("23:59", "%H:%M").time())
    else:
        start_time = datetime.strptime("00:00", "%H:%M").time()
        end_time = datetime.strptime("23:59", "%H:%M").time()

    c5, c6 = st.columns(2)
    with c5:
        antennes = sorted(df["Antenne"].dropna().unique().tolist())
        sel_ant = st.multiselect("Antennes", ["Toutes"] + antennes, default=["Toutes"])
    with c6:
        benevoles = sorted(df["Volunteer_Location"].dropna().unique().tolist())
        sel_ben = st.multiselect("Bénévoles", ["Tous"] + benevoles, default=["Tous"])

    c7, c8 = st.columns(2)
    with c7:
        search_text = st.text_input("Recherche texte dans les messages")
    with c8:
        search_id = st.text_input("Rechercher par ID chat")

    # filtre date / heure
    filtered = df.copy()
    filtered = filtered[
        (filtered["Crée le"].dt.date >= start_date)
        & (filtered["Crée le"].dt.date <= end_date)
    ]

    if use_time_filter:
        def to_time(dt):
            try:
                return dt.time()
            except Exception:
                return None

        filtered["time"] = filtered["Crée le"].apply(to_time)
        if start_time <= end_time:
            mask_time = (filtered["time"] >= start_time) & (filtered["time"] <= end_time)
        else:
            mask_time = (filtered["time"] >= start_time) | (filtered["time"] <= end_time)
        filtered = filtered[mask_time].drop(columns=["time"])

    if "Toutes" not in sel_ant:
        filtered = filtered[filtered["Antenne"].isin(sel_ant)]
    if "Tous" not in sel_ben:
        filtered = filtered[filtered["Volunteer_Location"].isin(sel_ben)]

    if search_text:
        filtered = filtered[filtered["messages_lower"].str.contains(search_text.lower(), na=False)]

    if search_id:
        try:
            cid = int(search_id)
            tmp = filtered[filtered["id_chat"] == cid]
            if tmp.empty:
                st.warning(f"Aucun chat avec l'ID {cid}")
            else:
                filtered = tmp
                st.success(f"Chat {cid} trouvé.")
        except ValueError:
            st.error("ID doit être un entier.")

    abusive_df = filtered[filtered["potentially_abusive"]].copy()

    if abusive_df.empty:
        st.warning("Aucun chat potentiellement abusif avec ces filtres.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Chats filtrés", len(filtered))
    with c2:
        st.metric("Chats potentiellement abusifs", len(abusive_df))

    with st.expander("Répartition par antenne / bénévole"):
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Par antenne")
            st.bar_chart(abusive_df["Antenne"].value_counts())
        with c4:
            st.subheader("Par bénévole")
            st.bar_chart(abusive_df["Volunteer_Location"].value_counts())

    st.subheader("Liste des chats potentiellement abusifs")

    abusive_df = abusive_df.sort_values("preliminary_score", ascending=False)
    abusive_df["select"] = False

    max_rows = st.slider("Nombre max de lignes à afficher", 10, 300, 100)
    abusive_display = abusive_df.head(max_rows)

    edited = st.data_editor(
        abusive_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Sélectionner", default=False),
            "id_chat": st.column_config.NumberColumn("ID Chat"),
            "Crée le": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY HH:mm"),
            "Antenne": st.column_config.TextColumn("Antenne"),
            "Volunteer_Location": st.column_config.TextColumn("Bénévole"),
            "preliminary_score": st.column_config.ProgressColumn(
                "Score mots-clés",
                min_value=0,
                max_value=50,
            ),
            "messages": st.column_config.TextColumn("Messages", width="large"),
        },
        column_order=[
            "select", "id_chat", "Crée le", "Antenne",
            "Volunteer_Location", "preliminary_score", "messages",
        ],
    )

    if st.button("Analyser en détail les chats sélectionnés"):
        selected = edited[edited["select"]]
        if selected.empty:
            st.warning("Sélectionne au moins un chat.")
            return

        results = []
        with st.spinner("Analyse détaillée..."):
            for _, row in selected.iterrows():
                cid = row["id_chat"]
                full_chat = df[df["id_chat"] == cid]
                if full_chat.empty:
                    continue
                chat_row = full_chat.iloc[0]
                messages = chat_row["messages"]

                score, factors, phrases, harass, patterns, changes = analyze_chat_content(messages)

                phr_text = ""
                for cat, lst in phrases.items():
                    if lst:
                        phr_text += f"**{cat}**\n"
                        for p in lst[:3]:
                            phr_text += f"- {p}\n"
                        phr_text += "\n"

                results.append(
                    {
                        "id_chat": cid,
                        "Crée le": chat_row["Crée le"],
                        "Antenne": chat_row["Antenne"],
                        "Volunteer_Location": chat_row["Volunteer_Location"],
                        "IP": chat_row.get("IP", ""),
                        "Score de risque": score,
                        "Niveau de risque": get_abuse_risk_level(score),
                        "Facteurs de risque": ", ".join(factors),
                        "Phrases problématiques": phr_text,
                        "Harcèlement opérateur": "Oui" if harass else "Non",
                        "Nb patterns manipulation": len(patterns),
                        "Nb changements de sujet": len(changes),
                        "messages": messages,
                    }
                )

        if not results:
            st.warning("Pas de résultats d'analyse.")
            return

        res_df = pd.DataFrame(results).sort_values("Score de risque", ascending=False)

        st.subheader("Résultats de l'analyse détaillée")
        st.dataframe(
            res_df[[
                "id_chat", "Crée le", "Antenne", "Volunteer_Location",
                "Score de risque", "Niveau de risque", "Facteurs de risque",
                "Harcèlement opérateur", "Nb patterns manipulation",
                "Nb changements de sujet",
            ]],
            use_container_width=True,
        )

        selected_id = st.selectbox(
            "Voir le détail complet d'un chat",
            res_df["id_chat"].tolist(),
        )

        if selected_id:
            sel = res_df[res_df["id_chat"] == selected_id].iloc[0]
            st.markdown(f"### Chat {selected_id}")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**Date :** {sel['Crée le'].strftime('%d/%m/%Y %H:%M')}")
            with c2:
                st.write(f"**Antenne :** {sel['Antenne']}")
            with c3:
                st.write(f"**Bénévole :** {sel['Volunteer_Location']}")

            st.write(f"**IP :** {sel.get('IP', 'N/A')}")
            st.write(f"**Score :** {sel['Score de risque']} ({sel['Niveau de risque']})")
            st.write(f"**Facteurs de risque :** {sel['Facteurs de risque']}")
            st.write(f"**Harcèlement envers l'opérateur :** {sel['Harcèlement opérateur']}")

            if sel["Phrases problématiques"]:
                st.subheader("Phrases problématiques détectées")
                st.markdown(sel["Phrases problématiques"])

            st.subheader("Contenu du chat")
            st.text_area("Messages", sel["messages"], height=350)

            # ===== Téléchargement du chat sélectionné =====
            chat_text = (
                f"Chat ID : {sel['id_chat']}\n"
                f"Date : {sel['Crée le'].strftime('%d/%m/%Y %H:%M')}\n"
                f"Antenne : {sel['Antenne']}\n"
                f"Bénévole : {sel['Volunteer_Location']}\n"
                f"IP : {sel.get('IP', 'N/A')}\n"
                f"Score de risque : {sel['Score de risque']} ({sel['Niveau de risque']})\n"
                f"Facteurs de risque : {sel['Facteurs de risque']}\n"
                f"Harcèlement opérateur : {sel['Harcèlement opérateur']}\n"
                "\n"
                "===== MESSAGES =====\n\n"
                f"{sel['messages']}"
            )

            st.download_button(
                label="📥 Télécharger ce chat (.txt)",
                data=chat_text,
                file_name=f"chat_{sel['id_chat']}.txt",
                mime="text/plain",
            )

    if st.sidebar.button("🔄 Rafraîchir les chats / analyse"):
        get_ksaar_chats.clear()
        st.experimental_rerun()


# ==========================
# MAIN
# ==========================

def main():
    if not CONFIG_LOADED:
        st.error("⚠️ Erreur de configuration des secrets")
        st.error(CONFIG_ERROR)
        return

    with st.expander("🔍 Debug config", expanded=False):
        st.write(f"API base URL : {ksaar_config.get('api_base_url', 'N/A')}")
        st.write(f"API key name configurée : {bool(ksaar_config.get('api_key_name'))}")
        st.write(f"API key password configuré : {bool(ksaar_config.get('api_key_password'))}")

    if not check_password():
        return

    st.sidebar.title("Navigation")
    if st.sidebar.button("🚪 Déconnexion"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

    tab1, tab2 = st.tabs(["📞 Appels", "🧠 Analyse IA des abus"])

    with tab1:
        display_calls()

    with tab2:
        display_abuse_analysis()


if __name__ == "__main__":
    main()
