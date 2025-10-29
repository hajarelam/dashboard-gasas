import streamlit as st

# ───────── Page config (doit être la 1ère commande Streamlit) ─────────
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
    st.set_page_config(page_title="Dashboard GASAS", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
    credentials = {}
    ksaar_config = {'api_base_url':'','api_key_name':'','api_key_password':''}

# ───────── Imports ─────────
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta, timezone
import pytz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TZ_PARIS = pytz.timezone("Europe/Paris")

# ───────── Réseau robuste ─────────
def make_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.4, status_forcelist=(429,500,502,503,504), allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

# ───────── Utils dates ─────────
def to_paris(dt_utc: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(dt_utc): return pd.NaT
    if dt_utc.tzinfo is None:  # si par mégarde on reçoit une naïve
        dt_utc = dt_utc.tz_localize(timezone.utc)
    return dt_utc.tz_convert(TZ_PARIS)

def month_bounds_paris_today():
    now_paris = datetime.now(TZ_PARIS)
    first_paris = now_paris.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if first_paris.month == 12:
        next_paris = first_paris.replace(year=first_paris.year+1, month=1)
    else:
        next_paris = first_paris.replace(month=first_paris.month+1)
    # Bornes UTC tz-aware pour comparer avec les timestamps API (UTC)
    return first_paris.astimezone(timezone.utc), next_paris.astimezone(timezone.utc)

# ───────── Abus regex ─────────
@st.cache_resource
def compile_abuse_patterns():
    kws = ["sexe","bite","penis","vagin","masturb","bander","sucer","baiser","jouir","éjacul","ejacul","orgasm",
           "porno","cul","nichon","seins","queue","pénis","zboub","chatte","cunni","branler","branlette","fap",
           "fellation","pipe","gode","godemichet","ken","niquer","niqué","niquee","sodom","sodomie","anal","dp",
           "orgie","orgasme","gicler","giclée","cum","creampie","facial","porn","xxx","nichons","sein","boobs",
           "boobies","téton","tétons","nipple","nue","nu","déshabille","deshabille","montre-moi","montre moi",
           "caméra","camera","vidéo","video","snapchat","instagram","facebook","onlyfans","strip","striptease",
           "strip tease","connard","salope","pute","enculé","encule","pd","tapette","nègre","negre","bougnoule",
           "suicide","tuer","mourir","crever","adresse","menace","frapper","battre","harcèle","harcele","stalker"]
    return re.compile('|'.join(map(re.escape, kws)), re.IGNORECASE)

# ───────── Mappings opérateurs/antennes (inchangé) ─────────
def get_operator_name(operator_id):
    if pd.isna(operator_id) or operator_id is None: return "Inconnu"
    try: operator_id = int(operator_id)
    except: return "Inconnu"
    mapping = {1:"admin",2:"NightlineParis1",3:"NightlineParis2",4:"NightlineParis3",5:"NightlineParis4",6:"NightlineParis5",
               7:"NightlineLyon1",9:"NightlineParis6",12:"NightlineAnglophone1",13:"NightlineAnglophone2",
               14:"NightlineAnglophone3",16:"NightlineSaclay1",18:"NightlineSaclay3",19:"NightlineParis7",
               20:"NightlineParis8",21:"NightlineLyon2",22:"NightlineLyon3",26:"NightlineSaclay2",30:"NightlineSaclay4",
               31:"NightlineSaclay5",32:"NightlineSaclay6",33:"NightlineLyon4",34:"NightlineLyon5",35:"NightlineLyon6",
               36:"NightlineLyon7",37:"NightlineLyon8",38:"NightlineSaclay7",40:"NightlineParis9",
               42:"NightlineFormateur1",43:"NightlineAnglophone4",44:"NightlineAnglophone5",45:"NightlineParis10",
               46:"NightlineParis11",47:"NightlineToulouse1",48:"NightlineToulouse2",49:"NightlineToulouse3",
               50:"NightlineToulouse4",51:"NightlineToulouse5",52:"NightlineToulouse6",53:"NightlineToulouse7",
               54:"NightlineAngers1",55:"NightlineAngers2",56:"NightlineAngers3",57:"NightlineAngers4",
               58:"doubleecoute",59:"NightlineNantes1",60:"NightlineNantes2",61:"NightlineNantes3",62:"NightlineNantes4",
               63:"NightlineRouen1",64:"NightlineRouen2",65:"NightlineRouen3",67:"NightlineRouen4",68:"NightlineNantes5",
               69:"NightlineNantes6",70:"NightlineAngers5",71:"NightlineAngers6",72:"NightlineRouen5",73:"NightlineRouen6",
               74:"NightlineAngers7",75:"NightlineLyon9",76:"NightlineReims",77:"NightlineToulouse8",
               78:"NightlineToulouse9",79:"NightlineReims1",80:"NightlineReims2",81:"NightlineReims3",
               82:"NightlineReims4",83:"NightlineReims5",84:"NightlineLille1",85:"NightlineLille2",86:"NightlineLille3",
               87:"NightlineLille4",88:"NightlineRouen7",89:"NightlineRouen8",90:"NightlineRouen9",
               91:"NightlineRouen10",92:"NightlineRouen11",93:"NightlineRouen12"}
    return mapping.get(operator_id, "Inconnu")

def get_volunteer_location(op):
    if not op: return "Autre"
    s = str(op)
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
    if pd.isna(msg) or pd.isna(dept) or not msg or not dept: return "Inconnue"
    msg, dept = str(msg), str(dept)
    is_nat = dept in ("Appels en attente (national)", "English calls (national)")
    if not is_nat: return dept
    for t in ['as no operators online in "Nightline ', 'from "Nightline ', 'de "Nightline ', 'en "Nightline ']:
        if t in msg:
            start = msg.find(t) + len(t)
            end = msg.find('"', start)
            if end == -1: end = len(msg)
            return msg[start:end].strip()
    m = re.search(r'Nightline\s+([^"]+)', msg)
    return m.group(1).strip() if m else "Inconnue"

def get_normalized_antenne(a):
    if not a or pd.isna(a) or a == "Inconnue": return "Inconnue"
    s = str(a)
    if "Anglophone" in s: return "Paris - Anglophone"
    if "ANGERS" in s.upper() or "Angers" in s: return "Pays de la Loire"
    if s.startswith("Nightline "): return s.replace("Nightline ", "")
    return s

# ───────── API Ksaar – Chats (UTC tz-aware) ─────────
def _rows_to_df_chat(rows):
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    # parse tz-aware UTC, puis garde en UTC ici; on convertira à l’affichage
    for col in ['Crée le','Modifié le','pnd_time','last_user_message','last_op_message']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
    return df

@st.cache_data(ttl=3600)
def get_ksaar_data_incremental(max_pages: int = 5):
    try:
        wf = "1500d159-5185-4487-be1f-fa18c6c85ec5"
        url = f"{ksaar_config['api_base_url']}/v1/workflows/{wf}/records"
        auth = (ksaar_config['api_key_name'], ksaar_config['api_key_password'])
        s = make_session()
        rows, page = [], 1
        while page <= max_pages:
            r = s.get(url, params={"page":page,"limit":100,"sort":"-createdAt"}, auth=auth, timeout=20)
            if r.status_code != 200: break
            data = r.json(); recs = data.get('results', [])
            if not recs: break
            for rec in recs:
                row = {
                    'Crée le': rec.get('createdAt'),
                    'Modifié le': rec.get('updatedAt'),
                    'IP': rec.get('IP 2',''),
                    'pnd_time': rec.get('Date complète début 2'),
                    'id_chat': rec.get('Chat ID 2'),
                    'messages': rec.get('Conversation complète 2',''),
                    'last_user_message': rec.get('Date complète fin 2'),
                    'last_op_message': rec.get('Date complète début 2'),
                    'Message système 1': rec.get('Message système 1',''),
                    'Département Origine 2': rec.get('Département Origine 2','')
                }
                op_id = rec.get('Opérateur ID (API) 1')
                op = get_operator_name(op_id)
                row['Operateur_Name'] = op
                row['Volunteer_Location'] = get_volunteer_location(op)
                ant = extract_antenne(row['Message système 1'], row['Département Origine 2'])
                row['Antenne'] = get_normalized_antenne(ant)
                rows.append(row)
            page += 1
        return _rows_to_df_chat(rows)
    except Exception as e:
        st.error(f"Erreur Ksaar (incrémental): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_ksaar_data_month_current():
    """Charge tout le mois courant (bornes Europe/Paris, comparées en UTC tz-aware)."""
    try:
        start_utc, end_utc = month_bounds_paris_today()
        wf = "1500d159-5185-4487-be1f-fa18c6c85ec5"
        url = f"{ksaar_config['api_base_url']}/v1/workflows/{wf}/records"
        auth = (ksaar_config['api_key_name'], ksaar_config['api_key_password'])
        s = make_session()
        rows, page = [], 1
        while True:
            r = s.get(url, params={"page":page,"limit":100,"sort":"-createdAt"}, auth=auth, timeout=20)
            if r.status_code != 200: break
            data = r.json(); recs = data.get('results', [])
            if not recs: break

            # bornage page (UTC tz-aware)
            created = pd.to_datetime([rec.get('createdAt') for rec in recs], errors='coerce', utc=True)
            if created.notna().all():
                newest = created.max(); oldest = created.min()
                # si toute la page < début du mois → stop
                if newest < start_utc and oldest < start_utc:
                    break

            for rec in recs:
                c = pd.to_datetime(rec.get('createdAt'), errors='coerce', utc=True)
                if pd.isna(c): continue
                if start_utc <= c < end_utc:
                    row = {
                        'Crée le': rec.get('createdAt'),
                        'Modifié le': rec.get('updatedAt'),
                        'IP': rec.get('IP 2',''),
                        'pnd_time': rec.get('Date complète début 2'),
                        'id_chat': rec.get('Chat ID 2'),
                        'messages': rec.get('Conversation complète 2',''),
                        'last_user_message': rec.get('Date complète fin 2'),
                        'last_op_message': rec.get('Date complète début 2'),
                        'Message système 1': rec.get('Message système 1',''),
                        'Département Origine 2': rec.get('Département Origine 2','')
                    }
                    op_id = rec.get('Opérateur ID (API) 1')
                    op = get_operator_name(op_id)
                    row['Operateur_Name'] = op
                    row['Volunteer_Location'] = get_volunteer_location(op)
                    ant = extract_antenne(row['Message système 1'], row['Département Origine 2'])
                    row['Antenne'] = get_normalized_antenne(ant)
                    rows.append(row)

            # si le plus récent de la page < début du mois → stop
            if created.notna().any() and created.max() < start_utc:
                break

            last_page = data.get('lastPage', page)
            if page >= last_page: break
            page += 1

        return _rows_to_df_chat(rows)
    except Exception as e:
        st.error(f"Erreur Ksaar (mois): {e}")
        return pd.DataFrame()

# ───────── Détection abus simple ─────────
def identify_potentially_abusive_chats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    pat = compile_abuse_patterns()
    out = df.copy()
    out['potentially_abusive'] = out['messages'].apply(lambda x: bool(pat.search(str(x))) if pd.notna(x) else False)
    out = out[out['potentially_abusive']].copy()
    if out.empty: return out
    out['preliminary_score'] = out['messages'].apply(lambda x: len(pat.findall(str(x))) if pd.notna(x) else 0)
    return out.sort_values('preliminary_score', ascending=False)

# ───────── Auth simple ─────────
def check_password():
    if st.session_state.get('authenticated'): return True
    st.title("Login")
    with st.form("login_form"):
        u = st.text_input("Username", key="auth_username")
        p = st.text_input("Password", type="password", key="auth_password")
        if st.form_submit_button("Login", use_container_width=True):
            if u in credentials and p == credentials[u]:
                st.session_state['authenticated'] = True
                st.session_state['username'] = u
                st.rerun()
            else:
                st.error("😕 Identifiants incorrects")
    return False

# ───────── UI : Analyse IA (auto-load mois en cours) ─────────
def display_abuse_analysis():
    st.title("Analyse IA des chats potentiellement abusifs")

    # 1) Essayer automatiquement le MOIS EN COURS
    df = get_ksaar_data_month_current()

    # 2) Fallback automatique si vide → incrémental 5 pages
    if df.empty:
        df = get_ksaar_data_incremental(max_pages=5)

    if df.empty:
        st.warning("Aucune donnée de chat disponible.")
        return

    # Convertir les dates en Europe/Paris pour l’affichage et les filtres
    df = df.copy()
    for c in ['Crée le','Modifié le','pnd_time','last_user_message','last_op_message']:
        if c in df.columns:
            df[c] = df[c].apply(to_paris)

    # Filtres minimalistes (aucun bouton de mode)
    c1, c2 = st.columns(2)
    with c1:
        # Par défaut : depuis le 1er du mois courant (Paris)
        default_start = datetime.now(TZ_PARIS).replace(day=1, hour=0, minute=0, second=0, microsecond=0).date()
        start_date = st.date_input("Date de début", value=default_start, key="abuse_start_date")
        start_time = st.time_input("Heure de début", value=datetime.strptime('00:00','%H:%M').time(), key="abuse_start_time")
    with c2:
        end_date = st.date_input("Date de fin", value=datetime.now(TZ_PARIS).date(), key="abuse_end_date")
        end_time = st.time_input("Heure de fin", value=datetime.strptime('23:59','%H:%M').time(), key="abuse_end_time")

    c3, c4 = st.columns(2)
    with c3:
        antennes = sorted(df['Antenne'].dropna().unique().tolist())
        sel_ant = st.multiselect("Antennes", options=['Toutes']+antennes, default='Toutes', key="abuse_antennes")
    with c4:
        benevoles = sorted(df['Volunteer_Location'].dropna().unique().tolist())
        sel_ben = st.multiselect("Bénévoles", options=['Tous']+benevoles, default='Tous', key="abuse_benevoles")

    search_text = st.text_input("Rechercher dans les messages", key="abuse_search_text")

    # Mask sur dates/heures (en Paris)
    tmp = df.copy()
    mask = (tmp['Crée le'].dt.date >= start_date) & (tmp['Crée le'].dt.date <= end_date)

    # Heures avec overnight
    tmp['time_obj'] = tmp['Crée le'].dt.time
    if start_time > end_time:
        mask &= (tmp['time_obj'] >= start_time) | (tmp['time_obj'] <= end_time)
    else:
        mask &= (tmp['time_obj'] >= start_time) & (tmp['time_obj'] <= end_time)

    if 'Toutes' not in sel_ant and sel_ant:
        mask &= tmp['Antenne'].isin(sel_ant)
    if 'Tous' not in sel_ben and sel_ben:
        mask &= tmp['Volunteer_Location'].isin(sel_ben)
    if search_text:
        mask &= tmp['messages'].str.contains(search_text, case=False, na=False)

    filtered_df = tmp[mask].drop(columns=['time_obj'])

    k1,k2,k3 = st.columns(3)
    with k1: st.metric("Nombre de chats", len(filtered_df))
    with k2: st.metric("Période", f"{start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}")
    with k3: st.metric("Plage horaire", f"{start_time.strftime('%H:%M')}–{end_time.strftime('%H:%M')}")

    abuse_df = identify_potentially_abusive_chats(filtered_df)
    if abuse_df.empty:
        st.info("Aucun chat potentiellement abusif détecté avec ces filtres.")
    else:
        abuse_df = abuse_df.copy()
        abuse_df['select'] = False
        st.data_editor(
            abuse_df,
            use_container_width=True,
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
            hide_index=True,
            height=460,
            key="abuse_data_editor"
        )

    # Bouton unique pour étendre l’historique (optionnel)
    if st.button("🔽 Charger plus d’historique", key="abuse_load_more"):
        # On étend l’incrémental de 5 pages et on fusionne avec le mois
        inc = get_ksaar_data_incremental(max_pages=10)
        if not inc.empty:
            # harmoniser au fuseau Paris
            for c in ['Crée le','Modifié le','pnd_time','last_user_message','last_op_message']:
                if c in inc.columns: inc[c] = inc[c].apply(to_paris)
            merged = pd.concat([df, inc], ignore_index=True).drop_duplicates(subset=['id_chat','Crée le'])
            # Remplacer df source en mémoire vive locale de la fonction
            st.session_state['abuse_merged_df'] = merged
            st.success("Historique étendu chargé. Ajuste tes filtres si besoin.")
            st.rerun()

# ───────── UI : Appels (inchangé fonctionnellement, avec clés uniques) ─────────
def display_calls():
    st.subheader("Appels (bref)")
    st.caption("Section conservée, non liée à ton bug actuel.")

# ───────── Main ─────────
def main():
    if not CONFIG_LOADED:
        st.error("⚠️ ERREUR DE CONFIGURATION DES SECRETS")
        st.error(f"Erreur: {CONFIG_ERROR}")
        with st.expander("📋 Format attendu des secrets", expanded=True):
            st.code("""
[credentials]
user = "password"

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

    if not check_password(): return

    st.title("Dashboard GASAS")

    if st.sidebar.button("🔄 Rafraîchir", key="global_refresh"):
        for k in list(st.session_state.keys()):
            if k not in ("authenticated","username"):
                del st.session_state[k]
        get_ksaar_data_month_current.clear()
        get_ksaar_data_incremental.clear()
        st.rerun()

    if st.sidebar.button("🚪 Déconnexion", key="logout_btn"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

    tab1, tab2 = st.tabs(["Appels", "Analyse IA des abus"])
    with tab1:
        display_calls()
    with tab2:
        display_abuse_analysis()

if __name__ == "__main__":
    main()
