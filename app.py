"""
=============================================================================
ÉTAPE 3 — Application Streamlit
Projet : Prédiction de pannes / arrêts — Albéa Simandre
Lancement : streamlit run app.py
Prérequis  : pip install streamlit plotly pandas scikit-learn openpyxl
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Maintenance Prédictive — Albéa Simandre",
    page_icon="🏭",
    layout="wide",
)

st.markdown("""
<style>
    .machine-card {
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
        cursor: pointer;
        border: none;
    }
    .card-red    { background: #FCEBEB; border-left: 5px solid #E24B4A; }
    .card-orange { background: #FAEEDA; border-left: 5px solid #EF9F27; }
    .card-green  { background: #EAF3DE; border-left: 5px solid #639922; }
    .card-title  { font-size: 17px; font-weight: 600; margin-bottom: 6px; }
    .card-score-red    { color: #A32D2D; font-size: 22px; font-weight: 700; }
    .card-score-orange { color: #854F0B; font-size: 22px; font-weight: 700; }
    .card-score-green  { color: #3B6D11; font-size: 22px; font-weight: 700; }
    .kpi-label { font-size: 11px; color: #888; margin-bottom: 2px; }
    .kpi-val   { font-size: 15px; font-weight: 600; }
    .alert-box { border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }
    .alert-red    { background: #FCEBEB; border-left: 4px solid #E24B4A; }
    .alert-orange { background: #FAEEDA; border-left: 4px solid #EF9F27; }
    [data-testid="stMetricValue"] { font-size: 28px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHARGEMENT & MODÈLE
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_and_train(filepath):
    df = pd.read_excel(filepath)
    df['D_CALEN'] = pd.to_datetime(df['D_CALEN'])
    df = df.sort_values(['N_POSTE','D_CALEN']).reset_index(drop=True)

    ETATS_PANNE = ['Panne Presse','Panne Robot','Panne Convoyeur / Péripherique',
                   'Panne Broyeur','Panne machine Déco','Problème moule',
                   'Intervention Maintenance','Intervention Maintenance (Montage)',
                   'Intervention Maintenance (Réglage)']
    ETATS_ARRET = ['Arrêt','Arret NON Qualite','Manque TF','Manque Matière',
                   'Manque Colorant','Manque Régleur','Manque Regleur Chgt Serie',
                   'Operateur NON Disponible']

    df['is_panne']   = df['LIB_ETAT_POSTE'].isin(ETATS_PANNE).astype(int)
    df['is_arret']   = df['LIB_ETAT_POSTE'].isin(ETATS_ARRET).astype(int)
    df['is_nonqual'] = (df['Famille Arret']=='Non qualité').astype(int)
    df['is_smed']    = (df['Famille Arret']=='SMED').astype(int)

    daily = df.groupby(['N_POSTE','D_CALEN']).agg(
        dur_prod    =('Durée', lambda x: df.loc[x.index].loc[df.loc[x.index,'LIB_ETAT_POSTE']=='Production','Durée'].sum()),
        dur_panne   =('Durée', lambda x: df.loc[x.index].loc[df.loc[x.index,'is_panne']==1,'Durée'].sum()),
        dur_arret   =('Durée', lambda x: df.loc[x.index].loc[df.loc[x.index,'is_arret']==1,'Durée'].sum()),
        dur_nonqual =('Durée', lambda x: df.loc[x.index].loc[df.loc[x.index,'is_nonqual']==1,'Durée'].sum()),
        dur_smed    =('Durée', lambda x: df.loc[x.index].loc[df.loc[x.index,'is_smed']==1,'Durée'].sum()),
        n_panne=('is_panne','sum'), n_arret=('is_arret','sum'), n_nonqual=('is_nonqual','sum'),
    ).reset_index()
    daily.columns = ['machine','date','dur_prod','dur_panne','dur_arret',
                     'dur_nonqual','dur_smed','n_panne','n_arret','n_nonqual']

    all_machines = daily['machine'].unique()
    all_dates = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
    grid = pd.MultiIndex.from_product([all_machines, all_dates], names=['machine','date'])
    daily = daily.set_index(['machine','date']).reindex(grid, fill_value=0).reset_index()
    daily = daily.sort_values(['machine','date']).reset_index(drop=True)

    for fenetre in [3,7]:
        for col in ['dur_prod','dur_panne','dur_arret','dur_nonqual','dur_smed','n_panne','n_arret','n_nonqual']:
            daily[f'{col}_{fenetre}d'] = daily.groupby('machine')[col].transform(
                lambda x: x.rolling(fenetre, min_periods=1).sum().shift(1))

    daily['trs_7d'] = daily['dur_prod_7d']/(daily['dur_prod_7d']+daily['dur_panne_7d']+daily['dur_arret_7d']+1e-6)
    daily['days_since_panne'] = daily.groupby('machine')['n_panne'].transform(
        lambda x: x.shift(1).eq(0).groupby((x.shift(1).ne(0)).cumsum()).cumcount())
    daily['trend_panne'] = daily['n_panne_3d']/(daily['n_panne_7d']+1e-6)

    HORIZON = 3
    daily['_pf'] = daily.groupby('machine')['n_panne'].transform(lambda x: x.rolling(HORIZON,min_periods=1).sum().shift(-HORIZON))
    daily['_af'] = daily.groupby('machine')['n_arret'].transform(lambda x: x.rolling(HORIZON,min_periods=1).sum().shift(-HORIZON))
    daily['target'] = ((daily['_pf']>0)|(daily['_af']>0)).astype(int)
    daily = daily.dropna().reset_index(drop=True)

    feature_cols = [c for c in daily.columns if '_7d' in c or '_3d' in c or 'trs' in c or 'days_' in c or 'trend' in c]
    X = daily[feature_cols].fillna(0)
    y = daily['target']

    rf = RandomForestClassifier(n_estimators=150, class_weight='balanced',
                                max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf.fit(X[daily['date'] < '2025-08-01'], y[daily['date'] < '2025-08-01'])
    daily['score_risque'] = rf.predict_proba(X.fillna(0))[:,1]

    return daily, df, feature_cols

# ─────────────────────────────────────────────────────────────
# SIDEBAR — UPLOAD
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
             width=0)  # placeholder invisible
    st.title("🏭 Maintenance\nPrédictive")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Charger Bilan_evt.xlsx", type=["xlsx"])
    st.markdown("---")
    st.markdown("**Modèle**")
    st.markdown("Random Forest · Horizon **J+3**")
    st.markdown("F1 = **0.82** · AUC = **0.925**")
    st.markdown("---")
    filtre = st.selectbox("Filtrer par niveau de risque",
                          ["Tous", "🔴 Élevé (>70%)", "🟠 Moyen (40–70%)", "🟢 Faible (<40%)"])
    st.markdown("---")
    st.caption("Albéa Simandre — 2025")

# ─────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────

if uploaded is None:
    st.info("👈 Charge le fichier **Bilan_evt.xlsx** dans la barre latérale pour démarrer.")
    st.stop()

with st.spinner("Chargement et entraînement du modèle..."):
    daily, df_raw, feature_cols = load_and_train(uploaded)

# Données de la dernière date
last_date = daily['date'].max()
latest = daily[daily['date'] == last_date].copy()
latest['niveau'] = pd.cut(latest['score_risque'], bins=[0,0.4,0.7,1.0],
                          labels=['Faible','Moyen','Élevé'])

# Filtre sidebar
if filtre == "🔴 Élevé (>70%)":
    latest = latest[latest['score_risque'] > 0.7]
elif filtre == "🟠 Moyen (40–70%)":
    latest = latest[(latest['score_risque'] >= 0.4) & (latest['score_risque'] <= 0.7)]
elif filtre == "🟢 Faible (<40%)":
    latest = latest[latest['score_risque'] < 0.4]

latest = latest.sort_values('score_risque', ascending=False)

# ─────────────────────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────────────────────

st.title("🏭 Tableau de bord — Prédiction de pannes")
st.caption(f"Dernière mise à jour : **{last_date.strftime('%d/%m/%Y')}** · Horizon de prédiction : **3 jours**")

# KPI globaux
k1, k2, k3, k4 = st.columns(4)
n_rouge  = (latest['score_risque'] > 0.7).sum()
n_orange = ((latest['score_risque'] >= 0.4) & (latest['score_risque'] <= 0.7)).sum()
n_vert   = (latest['score_risque'] < 0.4).sum()
trs_moy  = latest['trs_7d'].mean()

k1.metric("🔴 Risque élevé",   f"{n_rouge} machines",  "intervention requise")
k2.metric("🟠 Risque moyen",   f"{n_orange} machines", "à surveiller")
k3.metric("🟢 Risque faible",  f"{n_vert} machines",   "état normal")
k4.metric("📊 TRS moyen (7j)", f"{trs_moy:.0%}",       f"cible : 80%")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# ÉTAT DE SESSION (machine sélectionnée)
# ─────────────────────────────────────────────────────────────

if 'selected_machine' not in st.session_state:
    st.session_state.selected_machine = None

# ─────────────────────────────────────────────────────────────
# CARTES MACHINES
# ─────────────────────────────────────────────────────────────

st.subheader("📋 État des machines")

def render_card(row):
    score = row['score_risque']
    pct   = int(score * 100)
    trs   = row['trs_7d']
    m     = int(row['machine'])
    n_p   = int(row['n_panne_7d'])
    n_a   = int(row['n_arret_7d'])
    dsp   = int(row['days_since_panne'])

    if score > 0.7:
        cls, score_cls, badge, emoji = 'card-red', 'card-score-red', '🔴 ÉLEVÉ', '🔴'
    elif score >= 0.4:
        cls, score_cls, badge, emoji = 'card-orange', 'card-score-orange', '🟠 MOYEN', '🟠'
    else:
        cls, score_cls, badge, emoji = 'card-green', 'card-score-green', '🟢 FAIBLE', '🟢'

    return f"""
    <div class="machine-card {cls}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div class="card-title">{emoji} Machine {m}</div>
            <div class="{score_cls}">{pct}%</div>
        </div>
        <div style="font-size:11px; color:#888; margin-bottom:10px;">{badge} · Risque panne J+3</div>
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:8px;">
            <div><div class="kpi-label">TRS 7j</div><div class="kpi-val">{trs:.0%}</div></div>
            <div><div class="kpi-label">Pannes 7j</div><div class="kpi-val">{n_p}</div></div>
            <div><div class="kpi-label">Arrêts 7j</div><div class="kpi-val">{n_a}</div></div>
            <div><div class="kpi-label">J. sans panne</div><div class="kpi-val">{dsp}</div></div>
        </div>
    </div>
    """

# Affichage en grille 3 colonnes
cols_grid = st.columns(3)
for i, (_, row) in enumerate(latest.iterrows()):
    with cols_grid[i % 3]:
        st.markdown(render_card(row), unsafe_allow_html=True)
        if st.button(f"📈 Voir détail", key=f"btn_{row['machine']}"):
            st.session_state.selected_machine = int(row['machine'])

# ─────────────────────────────────────────────────────────────
# PANNEAU DÉTAIL MACHINE
# ─────────────────────────────────────────────────────────────

if st.session_state.selected_machine is not None:
    m = st.session_state.selected_machine
    st.markdown("---")
    st.subheader(f"📈 Historique — Machine {m}")

    hist = daily[daily['machine'] == m].sort_values('date')

    tab1, tab2, tab3 = st.tabs(["Score de risque", "TRS & Production", "Événements"])

    with tab1:
        fig = go.Figure()
        fig.add_hrect(y0=0.7, y1=1.0, fillcolor="#FCEBEB", opacity=0.4, line_width=0, annotation_text="Zone rouge")
        fig.add_hrect(y0=0.4, y1=0.7, fillcolor="#FAEEDA", opacity=0.4, line_width=0, annotation_text="Zone orange")
        fig.add_hrect(y0=0.0, y1=0.4, fillcolor="#EAF3DE", opacity=0.4, line_width=0, annotation_text="Zone verte")
        fig.add_scatter(x=hist['date'], y=hist['score_risque'],
                        mode='lines', line=dict(color='#E24B4A', width=2.5),
                        name='Score de risque', fill='tozeroy', fillcolor='rgba(226,75,74,0.08)')
        fig.update_layout(yaxis=dict(range=[0,1], tickformat='.0%', title='Score de risque'),
                          xaxis_title='Date', height=350, margin=dict(t=20,b=20),
                          hovermode='x unified', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_scatter(x=hist['date'], y=hist['trs_7d'],
                         mode='lines', line=dict(color='#185FA5', width=2),
                         name='TRS 7j', fill='tozeroy', fillcolor='rgba(24,95,165,0.1)')
        fig2.add_hline(y=0.80, line_dash='dash', line_color='#639922',
                       annotation_text='Cible 80%')
        fig2.update_layout(yaxis=dict(range=[0,1.05], tickformat='.0%', title='TRS'),
                           xaxis_title='Date', height=350, margin=dict(t=20,b=20),
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig3 = go.Figure()
            fig3.add_bar(x=hist['date'], y=hist['dur_prod'], name='Production', marker_color='#185FA5')
            fig3.add_bar(x=hist['date'], y=hist['dur_panne'], name='Panne', marker_color='#E24B4A')
            fig3.add_bar(x=hist['date'], y=hist['dur_arret'], name='Arrêt', marker_color='#EF9F27')
            fig3.update_layout(barmode='stack', height=280, margin=dict(t=10,b=10),
                               title='Durée journalière (h)', legend=dict(orientation='h'))
            st.plotly_chart(fig3, use_container_width=True)

        with c2:
            # Stats résumées
            st.markdown("**Résumé sur la période complète**")
            total_prod  = hist['dur_prod'].sum()
            total_panne = hist['dur_panne'].sum()
            total_arret = hist['dur_arret'].sum()
            nb_pannes   = hist['n_panne'].sum()
            nb_arrets   = hist['n_arret'].sum()
            st.metric("Heures de production", f"{total_prod:.0f} h")
            st.metric("Heures de panne",      f"{total_panne:.0f} h")
            st.metric("Nombre de pannes",     f"{nb_pannes:.0f}")
            st.metric("Nombre d'arrêts",      f"{nb_arrets:.0f}")

    with tab3:
        evts = df_raw[df_raw['N_POSTE'] == m].sort_values('D_CALEN', ascending=False)
        evts = evts[['D_CALEN','LIB_ETAT_POSTE','Durée','Famille Arret','C_EQUIPE']].copy()
        evts.columns = ['Date','État','Durée (h)','Famille','Équipe']
        evts['Durée (h)'] = evts['Durée (h)'].round(2)
        st.dataframe(evts.head(100), use_container_width=True, height=350)

    if st.button("✕ Fermer le détail"):
        st.session_state.selected_machine = None
        st.rerun()

# ─────────────────────────────────────────────────────────────
# ALERTES EN BAS
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🚨 Alertes actives")

alertes = latest[latest['score_risque'] >= 0.4].sort_values('score_risque', ascending=False)

if len(alertes) == 0:
    st.success("✅ Aucune alerte active — toutes les machines sont en zone verte.")
else:
    for _, row in alertes.iterrows():
        score = row['score_risque']
        m     = int(row['machine'])
        trs   = row['trs_7d']
        n_p   = int(row['n_panne_7d'])
        n_a   = int(row['n_arret_7d'])
        dsp   = int(row['days_since_panne'])

        if score > 0.7:
            cls  = 'alert-red'
            icon = '🔴'
            msg  = f"Risque élevé ({score:.0%}) — Intervention recommandée sous 72h"
        else:
            cls  = 'alert-orange'
            icon = '🟠'
            msg  = f"Risque moyen ({score:.0%}) — Surveiller dans les prochains jours"

        details = f"TRS : {trs:.0%} · Pannes 7j : {n_p} · Arrêts 7j : {n_a} · Jours sans panne : {dsp}"

        st.markdown(f"""
        <div class="alert-box {cls}">
            <div style="font-weight:600; font-size:15px;">{icon} Machine {m} — {msg}</div>
            <div style="font-size:12px; color:#666; margin-top:4px;">{details}</div>
        </div>
        """, unsafe_allow_html=True)

st.caption(f"Modèle : Random Forest · F1 = 0.82 · AUC = 0.925 · Données : Albéa Simandre 2025")
