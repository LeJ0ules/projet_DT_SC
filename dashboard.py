import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import warnings
from pathlib import Path
import requests
warnings.filterwarnings('ignore')

root_dir = Path.cwd()

API_URL = "http://localhost:8000"

# ─────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Marketing ROI Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────
# CSS personnalisé
# ─────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        border-left: 4px solid #667eea;
        padding-left: 10px;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Chargement des données et modèles
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(root_dir / 'data' / 'Dummy_Data_HSS.csv')
    df = df[['TV', 'Radio', 'Social Media', 'Influencer', 'Sales']].dropna()
    df['Total_Budget'] = df['TV'] + df['Radio'] + df['Social Media']
    df['ROI'] = df['Sales'] / df['Total_Budget']
    return df

@st.cache_data(ttl=30)
def get_available_models():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        data = r.json()
        return data.get("available_models", [])
    except Exception as e:
        st.error(f"API inaccessible : {e}")
        return []

@st.cache_data
def load_results():
    try:
        comparison = pd.read_csv(root_dir / 'models_output' / 'model_comparison.csv')
        impact     = pd.read_csv(root_dir / 'models_output' / 'marginal_impact.csv')
        return comparison, impact
    except:
        return None, None

@st.cache_data
def load_feature_names():
    with open(root_dir / 'pipeline' / 'feature_names.json') as f:
        return json.load(f)

# Chargement
df             = load_data()
available_models = get_available_models()
comparison_df, impact_df = load_results()
feature_names  = load_feature_names()

INFLUENCER_TYPES = ['Mega', 'Macro', 'Micro', 'Nano']
COLORS = {
    'TV':           '#e74c3c',
    'Radio':        '#3498db',
    'Social Media': '#2ecc71',
    'Mega':         '#e74c3c',
    'Macro':        '#3498db',
    'Micro':        '#2ecc71',
    'Nano':         '#f39c12',
}

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
col1, col2 = st.columns([1, 6])

with col1:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=70)

with col2:
    st.title("Marketing ROI Dashboard")
    st.caption("Projet Data Science — EFREI 2025-26")

st.markdown("---")

# ─────────────────────────────────────────
# NAVIGATION HORIZONTALE
# ─────────────────────────────────────────
page1, page2, page3, page4 = st.tabs([
    "📊 Vue Générale",
    "🤖 Comparaison Modèles",
    "🎯 Simulateur Budgétaire",
    "🔍 Impact Marginal & SHAP"
])

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — VUE GÉNÉRALE
# ═══════════════════════════════════════════════════════════════
with page1 :
    st.title("📊 Vue Générale du Dataset Marketing")
    st.markdown("Analyse exploratoire des campagnes marketing multicanal.")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Campagnes analysées", f"{len(df):,}")
    with col2:
        st.metric("Budget TV moyen", f"{df['TV'].mean():.1f} M€")
    with col3:
        st.metric("Ventes moyennes", f"{df['Sales'].mean():.1f} M€")
    with col4:
        st.metric("ROI moyen", f"{df['ROI'].mean():.2f}x")

    st.markdown("---")

    # Distribution des budgets
    st.markdown('<div class="section-title">Distribution des budgets par canal</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for canal in ['TV', 'Radio', 'Social Media']:
            fig.add_trace(go.Box(
                y=df[canal],
                name=canal,
                marker_color=COLORS[canal],
                boxmean=True
            ))
        fig.update_layout(
            title="Distribution des budgets par canal",
            yaxis_title="Budget (M€)",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Budget moyen par canal
        budget_means = pd.DataFrame({
            'Canal': ['TV', 'Radio', 'Social Media'],
            'Budget moyen (M€)': [df['TV'].mean(), df['Radio'].mean(), df['Social Media'].mean()],
            'Part (%)': [
                df['TV'].mean() / df['Total_Budget'].mean() * 100,
                df['Radio'].mean() / df['Total_Budget'].mean() * 100,
                df['Social Media'].mean() / df['Total_Budget'].mean() * 100,
            ]
        })
        fig = px.pie(
            budget_means,
            values='Budget moyen (M€)',
            names='Canal',
            title="Répartition moyenne du budget marketing",
            color='Canal',
            color_discrete_map=COLORS,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Ventes par influencer
    st.markdown('<div class="section-title">Ventes par type d\'Influenceur</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        inf_stats = df.groupby('Influencer')['Sales'].agg(['mean', 'median', 'std']).reset_index()
        inf_stats.columns = ['Influencer', 'Moyenne', 'Médiane', 'Std']
        fig = px.bar(
            inf_stats,
            x='Influencer',
            y='Moyenne',
            color='Influencer',
            color_discrete_map=COLORS,
            title="Ventes moyennes par type d'influenceur",
            error_y='Std',
            text='Moyenne'
        )
        fig.update_traces(texttemplate='%{text:.1f} M€', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="Ventes moyennes (M€)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distribution Influencer
        inf_count = df['Influencer'].value_counts().reset_index()
        inf_count.columns = ['Influencer', 'Count']
        fig = px.pie(
            inf_count,
            values='Count',
            names='Influencer',
            title="Distribution des types d'influenceur",
            color='Influencer',
            color_discrete_map=COLORS,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Scatter TV vs Sales
    st.markdown('<div class="section-title">Relations entre budgets et ventes</div>',
                unsafe_allow_html=True)

    canal_select = st.selectbox("Choisir un canal", ['TV', 'Radio', 'Social Media'])

    fig = px.scatter(
        df,
        x=canal_select,
        y='Sales',
        color='Influencer',
        color_discrete_map=COLORS,
        trendline='ols',
        trendline_scope='overall',
        title=f"Budget {canal_select} vs Ventes — coloré par type d'influenceur",
        labels={canal_select: f'Budget {canal_select} (M€)', 'Sales': 'Ventes (M€)'},
        opacity=0.6,
        height=500
    )
    corr = df[canal_select].corr(df['Sales'])
    fig.add_annotation(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=f'r = {corr:.4f}',
        showarrow=False,
        bgcolor='white',
        bordercolor='gray',
        font=dict(size=13)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Insight box
    st.markdown(f"""
    <div class="insight-box">
    💡 <b>Insight clé :</b> Le budget TV présente une corrélation quasi parfaite avec les ventes
    (r = {df['TV'].corr(df['Sales']):.4f}), ce que confirme l'analyse SHAP avec un impact moyen
    de 64M€. C'est le levier principal d'allocation budgétaire.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — COMPARAISON MODÈLES
# ═══════════════════════════════════════════════════════════════
with page2 :
    st.title("🤖 Comparaison des Modèles")
    st.markdown("Évaluation comparative des 5 modèles entraînés.")

    if comparison_df is not None:
        # Tableau comparatif
        st.markdown('<div class="section-title">Tableau des performances</div>',
                    unsafe_allow_html=True)

        # Mise en forme conditionnelle
        styled = comparison_df.style\
            .highlight_max(subset=['R² Test', 'R² Train', 'CV R²'],
                           color='#d4edda')\
            .highlight_min(subset=['MAE Test', 'RMSE Test'],
                           color='#d4edda')\
            .format(precision=4)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("🟢 Vert = meilleure valeur pour ce critère")

        st.markdown("---")

        # Graphiques comparatifs
        col1, col2, col3 = st.columns(3)

        metrics_list = [
            ('R² Test',   'plus haut = mieux'),
            ('MAE Test',  'plus bas = mieux'),
            ('RMSE Test', 'plus bas = mieux'),
        ]

        model_colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12']

        for col, (metric, hint) in zip(
            [col1, col2, col3], metrics_list
        ):
            with col:
                fig = px.bar(
                    comparison_df,
                    x='Modèle',
                    y=metric,
                    color='Modèle',
                    color_discrete_sequence=model_colors,
                    title=f"{metric}<br><sup>{hint}</sup>",
                    text=metric
                )
                fig.update_traces(
                    texttemplate='%{text:.4f}',
                    textposition='outside'
                )
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    xaxis_tickangle=-20
                )
                st.plotly_chart(fig, use_container_width=True)

        # Analyse overfitting
        st.markdown('<div class="section-title">Analyse du surapprentissage (Overfitting)</div>',
                    unsafe_allow_html=True)

        if 'R² Train' in comparison_df.columns and 'R² Test' in comparison_df.columns:
            comparison_df['Gap R²'] = (comparison_df['R² Train'] - comparison_df['R² Test']).round(4)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='R² Train',
                x=comparison_df['Modèle'],
                y=comparison_df['R² Train'],
                marker_color='steelblue',
                opacity=0.8
            ))
            fig.add_trace(go.Bar(
                name='R² Test',
                x=comparison_df['Modèle'],
                y=comparison_df['R² Test'],
                marker_color='coral',
                opacity=0.8
            ))
            fig.update_layout(
                barmode='group',
                title="R² Train vs Test — détection de l'overfitting",
                yaxis_title="R²",
                height=400,
                legend=dict(orientation='h', y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                comparison_df[['Modèle', 'R² Train', 'R² Test', 'Gap R²']],
                use_container_width=True,
                hide_index=True
            )

        # Modèle recommandé
        st.markdown('<div class="section-title">Modèle final recommandé</div>',
                    unsafe_allow_html=True)

        best_idx = comparison_df['R² Test'].idxmax()
        best_model_name = comparison_df.loc[best_idx, 'Modèle']
        best_r2  = comparison_df.loc[best_idx, 'R² Test']
        best_mae = comparison_df.loc[best_idx, 'MAE Test']

        st.success(f"""
        **✅ Modèle sélectionné : {best_model_name}**

        - R² Test  : **{best_r2:.4f}** (variance expliquée)
        - MAE Test : **{best_mae:.4f} M€** (erreur moyenne)

        Ce modèle offre le meilleur compromis performance / stabilité / interprétabilité
        dans le contexte d'optimisation du ROI marketing.
        """)

    else:
        st.warning("⚠️ Fichier model_comparison.csv non trouvé. Lance d'abord le notebook de modélisation.")


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — SIMULATEUR BUDGÉTAIRE
# ═══════════════════════════════════════════════════════════════
with page3 :
    st.title("🎯 Simulateur Budgétaire")
    st.markdown("Saisissez un scénario budgétaire et obtenez une prédiction de ventes en temps réel.")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="section-title">Paramètres du scénario</div>',
                    unsafe_allow_html=True)

        tv_budget = st.slider(
            "📺 Budget TV (M€)",
            min_value=float(df['TV'].min()),
            max_value=float(df['TV'].max()),
            value=float(df['TV'].mean()),
            step=1.0,
            help="Budget alloué à la publicité TV"
        )
        radio_budget = st.slider(
            "📻 Budget Radio (M€)",
            min_value=float(df['Radio'].min()),
            max_value=float(df['Radio'].max()),
            value=float(df['Radio'].mean()),
            step=0.5,
            help="Budget alloué à la publicité Radio"
        )
        social_budget = st.slider(
            "📱 Budget Social Media (M€)",
            min_value=float(df['Social Media'].min()),
            max_value=float(df['Social Media'].max()),
            value=float(df['Social Media'].mean()),
            step=0.1,
            help="Budget alloué au Social Media"
        )
        influencer_type = st.selectbox(
            "🌟 Type d'Influenceur",
            INFLUENCER_TYPES,
            help="Type d'influenceur mobilisé pour la campagne"
        )
        model_choice = st.selectbox(
            "🤖 Modèle de prédiction",
            available_models
        )

    with col_result:
        st.markdown('<div class="section-title">Prédiction en temps réel</div>',
                    unsafe_allow_html=True)

        # Préparation de l'input
        input_data = pd.DataFrame({
            'TV':           [tv_budget],
            'Radio':        [radio_budget],
            'Social Media': [social_budget],
            'Influencer':   [influencer_type]
        })

        # Preprocessing + prédiction
        try:
            payload = {
                "TV":           tv_budget,
                "Radio":        radio_budget,
                "Social_Media": social_budget,
                "Influencer":   influencer_type,
                "model":        model_choice
            }   
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            prediction     = result["prediction"]["sales_predicted_M€"]
            roi_estimated  = result["prediction"]["roi_estimated"]
            total_budget   = result["prediction"]["total_budget_M€"]
            avg_sales      = df['Sales'].mean()
            avg_roi        = df['ROI'].mean()
            delta_sales    = prediction - avg_sales
            delta_roi      = roi_estimated - avg_roi

            # Métriques principales
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "🎯 Ventes prédites",
                    f"{prediction:.1f} M€",
                    delta=f"{delta_sales:+.1f} M€ vs moyenne",
                    delta_color="normal"
                )
                st.metric("💰 Budget total", f"{total_budget:.1f} M€")

            with col_b:
                st.metric(
                    "📈 ROI estimé",
                    f"{roi_estimated:.2f}x",
                    delta=f"{delta_roi:+.2f}x vs moyenne",
                    delta_color="normal"
                )
                st.metric("🌟 Influenceur", influencer_type)

            st.markdown("---")

            # Jauge de performance
            percentile = (df['Sales'] < prediction).mean() * 100
            st.markdown(f"**Cette campagne serait meilleure que {percentile:.0f}% des campagnes historiques**")
            st.progress(percentile / 100)

            # Répartition du budget
            fig = px.pie(
                values=[tv_budget, radio_budget, social_budget],
                names=['TV', 'Radio', 'Social Media'],
                title="Répartition du budget saisi",
                color_discrete_map=COLORS,
                hole=0.4
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de joindre l'API. Vérifie qu'elle tourne sur " + API_URL)
        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur API ({e.response.status_code}) : {e.response.json().get('detail', str(e))}")
        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")

    # Comparaison de scénarios
    st.markdown("---")
    st.markdown('<div class="section-title">Simulation : effet d\'une variation de budget</div>',
                unsafe_allow_html=True)

    canal_sim = st.selectbox("Canal à faire varier", ['TV', 'Radio', 'Social Media'])
    variation_range = st.slider("Plage de variation (%)", -50, 100, (-20, 50))

    try:
        variations = np.linspace(variation_range[0], variation_range[1], 30)
        predictions_sim = []

        for pct in variations:
            sim_tv     = tv_budget     * (1 + pct/100) if canal_sim == 'TV'           else tv_budget
            sim_radio  = radio_budget  * (1 + pct/100) if canal_sim == 'Radio'        else radio_budget
            sim_social = social_budget * (1 + pct/100) if canal_sim == 'Social Media' else social_budget

            payload = {
                "TV": max(0.01, sim_tv),
                "Radio": max(0.01, sim_radio),
                "Social_Media": max(0.01, sim_social),
                "Influencer": influencer_type,
                "model": model_choice
            }
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            predictions_sim.append(r.json()["prediction"]["sales_predicted_M€"])

        sim_df = pd.DataFrame({
            'Variation (%)': variations,
            'Ventes prédites (M€)': predictions_sim
        })

        avg_sales = df['Sales'].mean()

        fig = px.line(
            sim_df,
            x='Variation (%)',
            y='Ventes prédites (M€)',
            title=f"Impact d'une variation du budget {canal_sim} sur les ventes prédites",
            markers=True
        )
        fig.add_vline(x=0, line_dash='dash', line_color='gray',
                      annotation_text='Budget actuel')
        fig.add_hline(y=avg_sales, line_dash='dot', line_color='red',
                      annotation_text=f'Moyenne historique ({avg_sales:.0f} M€)')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur simulation : {e}")


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — IMPACT MARGINAL & SHAP
# ═══════════════════════════════════════════════════════════════
with page4 :
    st.title("🔍 Impact Marginal & Interprétabilité SHAP")
    st.markdown("Analyse de la contribution de chaque canal marketing aux ventes.")

    # Impact marginal
    st.markdown('<div class="section-title">Impact Marginal par Canal</div>',
                unsafe_allow_html=True)
    st.markdown("*Augmentation moyenne des ventes pour +1 unité standardisée de budget par canal.*")

    if impact_df is not None:
        # Graphique impact marginal par modèle
        canaux = ['TV', 'Radio', 'Social Media']
        model_colors_list = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12']

        fig = go.Figure()
        for i, row in impact_df.iterrows():
            color = model_colors_list[i % len(model_colors_list)]
            fig.add_trace(go.Bar(
                name=row['Modèle'],
                x=canaux,
                y=[row.get(c, 0) for c in canaux],
                marker_color=color,
                opacity=0.85
            ))

        fig.update_layout(
            barmode='group',
            title="Impact marginal par canal — comparaison entre modèles",
            yaxis_title="ΔSales moyen (M€)",
            height=450,
            legend=dict(orientation='h', y=1.1)
        )
        fig.add_hline(y=0, line_color='black', line_width=0.8)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(impact_df, use_container_width=True, hide_index=True)

    # SHAP
    st.markdown("---")
    st.markdown('<div class="section-title">Analyse SHAP — Importance des variables</div>',
                unsafe_allow_html=True)

    # Importance SHAP manuelle (valeurs issues du notebook SHAP)
    shap_data = {
        'Variable': ['TV', 'Radio', 'Social Media', 'Macro', 'Nano', 'Mega', 'Micro'],
        'Impact moyen |φ| (M€)': [64.19, 14.52, 0.84, 0.21, 0.15, 0.14, 0.14]
    }
    shap_df = pd.DataFrame(shap_data).sort_values('Impact moyen |φ| (M€)', ascending=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.bar(
            shap_df,
            x='Impact moyen |φ| (M€)',
            y='Variable',
            orientation='h',
            title="SHAP — Importance moyenne absolue par variable",
            color='Impact moyen |φ| (M€)',
            color_continuous_scale='RdYlGn',
            text='Impact moyen |φ| (M€)'
        )
        fig.update_traces(texttemplate='%{text:.2f} M€', textposition='outside')
        fig.update_layout(
            height=400,
            coloraxis_showscale=False,
            xaxis_title="Impact moyen sur les ventes (M€)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Interprétation métier")
        st.markdown("""
        | Canal | Impact | Recommandation |
        |-------|--------|---------------|
        | 📺 TV | 64.19 M€ | **Levier principal** |
        | 📻 Radio | 14.52 M€ | Levier secondaire |
        | 📱 Social | 0.84 M€ | Impact limité |
        | 🌟 Influencer | < 0.25 M€ | Rôle marginal |
        """)

        st.markdown("""
        <div class="insight-box">
        💡 <b>Recommandation CMO :</b><br>
        Prioriser le budget TV — son impact marginal est
        <b>4x supérieur</b> à Radio et <b>76x supérieur</b>
        à Social Media. Le type d'influenceur n'est pas
        un levier décisif sur ce dataset.
        </div>
        """, unsafe_allow_html=True)

    # Images SHAP générées
    st.markdown("---")
    st.markdown('<div class="section-title">Visualisations SHAP détaillées</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Beeswarm", "Importance", "Dependence", "Explication locale"
    ])

    shap_images = {
        "Beeswarm":           (root_dir / 'shap_output' / 'shap_beeswarm.png',       "Vision globale détaillée — chaque point est une campagne"),
        "Importance":         (root_dir / 'shap_output' / 'shap_importance_bar.png', "Importance moyenne absolue par variable"),
        "Dependence":         (root_dir / 'shap_output' / 'shap_dependence.png',     "Impact marginal visuel — effet de saturation par canal"),
        "Explication locale": (root_dir / 'shap_output' / 'shap_local.png',          "Contribution par variable pour 3 campagnes précises"),
    }

    for tab, (label, (img_path, caption)) in zip(
        [tab1, tab2, tab3, tab4], shap_images.items()
    ):
        with tab:
            try:
                st.image(img_path, caption=caption, use_container_width=True)
            except:
                st.info(f"Image {img_path} non trouvée. Lance d'abord le notebook SHAP.")

    # Conclusion
    st.markdown("---")
    st.markdown("""
    <div class="insight-box">
    📌 <b>Conclusion de l'analyse SHAP :</b><br><br>
    L'analyse SHAP confirme que le budget <b>TV est le déterminant principal des ventes</b>
    avec un impact moyen de 64M€, soit 4x supérieur à Radio et 76x supérieur à Social Media.
    Le type d'influenceur (Mega, Macro, Micro, Nano) joue un rôle quasi négligeable.
    <br><br>
    Un CMO cherchant à maximiser son ROI devrait donc <b>prioriser l'allocation budgétaire TV</b>
    avant tout autre canal, tout en maintenant une présence Radio pour un effet complémentaire.
    </div>
    """, unsafe_allow_html=True)