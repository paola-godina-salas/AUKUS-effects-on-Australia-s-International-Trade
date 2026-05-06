# requirements.txt
# streamlit>=1.32.0
# pandas>=2.0.0
# plotly>=5.18.0
# scikit-learn>=1.3.0
# matplotlib>=3.7.0
# numpy>=1.24.0
# scipy>=1.11.0

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AUKUS Trade Intelligence",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  h1, h2, h3, .big-title {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0a0f1e;
    border-right: 1px solid #1e2a45;
  }
  section[data-testid="stSidebar"] * {
    color: #c8d6f0 !important;
  }
  section[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    padding: 4px 0;
  }

  /* KPI cards */
  .kpi-card {
    background: linear-gradient(135deg, #0d1b35 0%, #162544 100%);
    border: 1px solid #1e3a6e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 8px 0 4px;
  }
  .kpi-label {
    font-size: 0.78rem;
    color: #7a93c0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .kpi-delta-pos { color: #4caf7d; }
  .kpi-delta-neg { color: #e05252; }
  .kpi-delta-neu { color: #5b9bd5; }

  /* News cards */
  .news-card {
    background: #0d1b35;
    border: 1px solid #1e3a6e;
    border-left: 4px solid #3a7bd5;
    border-radius: 10px;
    padding: 18px 20px;
    height: 100%;
    transition: border-left-color 0.2s;
  }
  .news-card:hover { border-left-color: #5b9bd5; }
  .news-card-tag {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3a7bd5;
    font-weight: 600;
    margin-bottom: 6px;
  }
  .news-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #e8f0ff;
    margin-bottom: 8px;
    line-height: 1.4;
  }
  .news-card-body {
    font-size: 0.82rem;
    color: #8aa3cc;
    line-height: 1.6;
    margin-bottom: 12px;
  }
  .news-card-link {
    font-size: 0.78rem;
    color: #3a7bd5;
    text-decoration: none;
    font-weight: 500;
  }

  /* Section divider */
  .section-divider {
    border: none;
    border-top: 1px solid #1e3a6e;
    margin: 28px 0;
  }

  /* RQ badge */
  .rq-badge {
    display: inline-block;
    background: #1e3a6e;
    color: #7ab3f0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }

  /* Main background */
  .stApp { background-color: #060c1a; }
  .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("trade_data.csv")
    df["year"] = df["year"].astype(int)
    df["trade_value"] = pd.to_numeric(df["trade_value"], errors="coerce").fillna(0)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ `trade_data.csv` not found. Export `df_clean` from your notebook with `df_clean.to_csv('trade_data.csv', index=False)` and place it in the same directory as this app.")
    st.stop()

# ── Constants ─────────────────────────────────────────────────────────────────
COLORS = {
    "China": "#E53935",
    "AUKUS Members": "#1565C0",
    "Non-Affiliated": "#FB8C00",
    "United States": "#1976D2",
    "United Kingdom": "#0D47A1",
    "Japan": "#F57C00",
    "South Korea": "#E65100",
}
AUKUS_LINE = dict(x0=2021, x1=2021, line=dict(color="white", width=1.5, dash="dash"))
EXPORT_LINE = dict(x0=2024, x1=2024, line=dict(color="#9e9e9e", width=1.5, dash="dot"))

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,53,0.6)",
    font=dict(family="DM Sans", color="#c8d6f0"),
    title_font=dict(family="Syne", color="#e8f0ff"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3a6e", borderwidth=1),
    xaxis=dict(gridcolor="#1e2a45", zeroline=False),
    yaxis=dict(gridcolor="#1e2a45", zeroline=False),
)

def add_event_lines(fig):
    fig.add_vline(x=2021, line_dash="dash", line_color="white", line_width=1.5,
                  annotation_text="AUKUS 2021", annotation_position="top",
                  annotation_font=dict(color="white", size=11))
    fig.add_vline(x=2024, line_dash="dot", line_color="#9e9e9e", line_width=1.5,
                  annotation_text="Export Controls 2024", annotation_position="top",
                  annotation_font=dict(color="#9e9e9e", size=11))
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px;'>
      <div style='font-family: Syne; font-size: 1.3rem; font-weight: 800;
                  color: #e8f0ff; letter-spacing: -0.02em; line-height: 1.2;'>
        🌏 AUKUS<br>Trade Intelligence
      </div>
      <div style='font-size: 0.7rem; color: #4a6a9e; margin-top: 6px;
                  text-transform: uppercase; letter-spacing: 0.1em;'>
        Australia · 2017–2025
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Why This Matters",
         "Total Trade Overview",
         "High-Risk Commodity Flows",
         "Trade Regime Clustering",
         "About This Dashboard"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e3a6e; margin: 16px 0;'>", unsafe_allow_html=True)
    flow_filter = st.selectbox("Trade Flow", ["Both", "Import", "Export"])
    st.markdown("""
    <div style='font-size: 0.7rem; color: #4a6a9e; margin-top: 20px;
                text-transform: uppercase; letter-spacing: 0.08em;'>
      Data: UN Comtrade · AUS Reporter<br>Partners: US · UK · CN · JP · KR
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — WHY THIS MATTERS
# ─────────────────────────────────────────────────────────────────────────────
if page == "Why This Matters":
    st.markdown("""
    <h1 style='font-family: Syne; font-size: 2.4rem; font-weight: 800;
               color: #e8f0ff; margin-bottom: 4px;'>
      The Age of AI & Export Controls
    </h1>
    <p style='color: #4a6a9e; font-size: 1rem; margin-bottom: 28px;'>
      How AUKUS is reshaping Australia's strategic trade — and why every supply chain dependent firm should care.
    </p>
    """, unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("""
        <div class='kpi-card'>
          <div class='kpi-label'>China · Total Trade Change</div>
          <div class='kpi-value kpi-delta-pos'>+34.0%</div>
          <div style='font-size:0.78rem; color:#4caf7d;'>↑ Pre → Post AUKUS &nbsp;·&nbsp; p = 0.029 ✅</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class='kpi-card'>
          <div class='kpi-label'>Non-Affiliated · Strategic Goods</div>
          <div class='kpi-value kpi-delta-neg'>−32.6%</div>
          <div style='font-size:0.78rem; color:#e05252;'>↓ Pre → Post AUKUS &nbsp;·&nbsp; p = 0.015 ✅</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class='kpi-card'>
          <div class='kpi-label'>Cluster Model · Breakpoint</div>
          <div class='kpi-value kpi-delta-neu'>2022</div>
          <div style='font-size:0.78rem; color:#5b9bd5;'>Model-confirmed regime shift · No label needed</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Two-column: framing text + timeline ───────────────────────────────────
    col_text, col_timeline = st.columns([1.1, 0.9], gap="large")

    with col_text:
        st.markdown("""
        <h2 style='font-family: Syne; font-size: 1.4rem; color: #e8f0ff;
                   margin-bottom: 14px;'>The New Geopolitics of Supply Chains</h2>
        <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8; margin-bottom: 14px;'>
          Export control regimes used to be a compliance concern for a narrow set of defence contractors.
          <strong style='color: #c8d6f0;'>In the age of AI, they are a supply chain risk for any firm
          that sources semiconductors, integrated circuits, radar components, or advanced computing hardware.</strong>
        </p>
        <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8; margin-bottom: 14px;'>
          The goods tracked in this dashboard — HS codes 8541, 8542, 8526, 8543, 9014 —
          are the physical infrastructure of artificial intelligence. AUKUS-linked export controls
          sit within a broader global pattern: the US CHIPS Act, Dutch ASML export restrictions,
          and Australia's Defence Trade Controls Amendment Act of 2024 have collectively redefined
          which firms can access which technologies and from which partners.
        </p>
        <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8;'>
          This dashboard surfaces how those pressures are <em>already visible</em> in Australia's trade data —
          giving procurement teams, risk analysts, and policy practitioners an evidence base
          for decisions that used to rely on intuition alone.
        </p>
        """, unsafe_allow_html=True)

    with col_timeline:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        fig_tl, ax_tl = plt.subplots(figsize=(6, 2.6))
        fig_tl.patch.set_facecolor("#0d1b35")
        ax_tl.set_facecolor("#0d1b35")

        events = [
            (2021, "AUKUS\nAnnounced", "#3a7bd5"),
            (2022, "Submarine\nDeal Confirmed", "#5b9bd5"),
            (2024, "Export Controls\nEnacted", "#e05252"),
            (2025, "Data\nCutoff", "#4a6a9e"),
        ]
        ax_tl.axhline(0.5, color="#1e3a6e", linewidth=2, zorder=1)
        for x, label, color in events:
            ax_tl.scatter(x, 0.5, s=120, color=color, zorder=3, linewidths=0)
            ax_tl.text(x, 0.82, label, ha="center", va="bottom", fontsize=7.5,
                       color="#c8d6f0", fontweight="600", linespacing=1.5)
            ax_tl.plot([x, x], [0.5, 0.75], color=color, linewidth=1, alpha=0.6, zorder=2)

        ax_tl.set_xlim(2019.5, 2025.8)
        ax_tl.set_ylim(0, 1.3)
        ax_tl.set_xticks([2021, 2022, 2024, 2025])
        ax_tl.set_xticklabels([2021, 2022, 2024, 2025],
                               color="#4a6a9e", fontsize=8)
        ax_tl.set_yticks([])
        for spine in ax_tl.spines.values():
            spine.set_visible(False)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig_tl, use_container_width=True)
        plt.close(fig_tl)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Why Businesses Should Care: news cards ────────────────────────────────
    st.markdown("""
    <h2 style='font-family: Syne; font-size: 1.4rem; color: #e8f0ff; margin-bottom: 20px;'>
      Why Businesses Should Care
    </h2>
    """, unsafe_allow_html=True)

    cards = [
        {
            "tag": "AI & National Security",
            "title": "AI, National Security & the Global Technology Race",
            "body": "The Hudson Institute argues that US export controls are now the defining instrument in the AI race — framing chip access restrictions as strategic weapons following China's DeepSeek breakthrough.",
            "url": "https://www.hudson.org/national-security-defense/ai-national-security-global-technology-race-how-us-export-controls-define-nury-turkel",
            "link_text": "Hudson Institute →",
        },
        {
            "tag": "Compliance Trends 2026",
            "title": "2026 Global Trade Compliance Trends",
            "body": "Visual Compliance identifies tightening export control regimes as the top compliance challenge of 2026, with AI-adjacent goods facing the greatest regulatory scrutiny across major trading blocs.",
            "url": "https://www.visualcompliance.com/blog/2026-global-trade-compliance-trends/",
            "link_text": "Visual Compliance →",
        },
        {
            "tag": "Supply Chain Exposure",
            "title": "Strategic Goods Declines Hit Neutral Partners First",
            "body": "Non-Affiliated partners Japan and South Korea saw a statistically significant -32.6% drop in strategic goods trade post-AUKUS. Procurement teams should audit whether key supplier relationships are exposed.",
            "url": "#",
            "link_text": "See RQ2 Analysis →",
        },
        {
            "tag": "Market Access Risk",
            "title": "China Trade Grew — But Strategic Goods Flatlined",
            "body": "Despite total trade with China rising +34%, strategic goods showed no statistically significant change. Firms relying on Chinese-sourced advanced components face a stable but compliance-heavy environment.",
            "url": "#",
            "link_text": "See Commodity Flows →",
        },
        {
            "tag": "AI Infrastructure Policy",
            "title": "Defence Policy Now Intersects with Commercial Tech",
            "body": "Semiconductors, integrated circuits, and radar systems — the goods most affected by AUKUS controls — are the same components underpinning global AI infrastructure. The line between defence and commercial is gone.",
            "url": "#",
            "link_text": "See Clustering Analysis →",
        },
    ]

    # ── Row 1: first 3 cards ──────────────────────────────────────────────────
    row1 = st.columns(3, gap="medium")
    for col, card in zip(row1, cards[:3]):
        with col:
            st.markdown(f"""
            <div class='news-card'>
              <div class='news-card-tag'>{card["tag"]}</div>
              <div class='news-card-title'>{card["title"]}</div>
              <div class='news-card-body'>{card["body"]}</div>
              <a class='news-card-link' href='{card["url"]}' target='_blank'>{card["link_text"]}</a>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Row 2: last 2 cards, centred with spacer columns ──────────────────────
    _, r2c1, r2c2, _ = st.columns([0.5, 1, 1, 0.5], gap="medium")
    for col, card in zip([r2c1, r2c2], cards[3:]):
        with col:
            st.markdown(f"""
            <div class='news-card'>
              <div class='news-card-tag'>{card["tag"]}</div>
              <div class='news-card-title'>{card["title"]}</div>
              <div class='news-card-body'>{card["body"]}</div>
              <a class='news-card-link' href='{card["url"]}' target='_blank'>{card["link_text"]}</a>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Featured article previews ─────────────────────────────────────────────
    st.markdown("""
    <h2 style='font-family: Syne; font-size: 1.4rem; color: #e8f0ff; margin-bottom: 18px;'>
      Featured Articles
    </h2>
    """, unsafe_allow_html=True)

    fa1, fa2 = st.columns(2, gap="large")

    with fa1:
        st.markdown("""
        <div style='background:#0d1b35; border:1px solid #1e3a6e; border-radius:12px; overflow:hidden;'>
          <div style='background: linear-gradient(135deg, #1a2e55 0%, #0d1b35 100%);
                      padding: 28px 24px 20px; border-bottom: 1px solid #1e3a6e;'>
            <div style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em;
                        color:#3a7bd5; font-weight:700; margin-bottom:10px;'>
              Hudson Institute · National Security
            </div>
            <div style='font-family: Syne; font-size:1.15rem; font-weight:700;
                        color:#e8f0ff; line-height:1.4; margin-bottom:12px;'>
              AI, National Security & the Global Technology Race: How US Export Controls Define the Future of Innovation
            </div>
            <div style='font-size:0.8rem; color:#8aa3cc; line-height:1.7;'>
              President Trump called China's DeepSeek AI a "wake-up call" for the US technology sector.
              This Hudson Institute analysis argues that export controls on semiconductors and AI chips
              are now the primary instrument in the global technology race — with direct consequences
              for firms operating across AUKUS and non-AUKUS supply chains.
            </div>
          </div>
          <div style='padding: 14px 24px; display:flex; align-items:center; justify-content:space-between;'>
            <span style='font-size:0.75rem; color:#4a6a9e;'>Hudson Institute · 2025</span>
            <a href='https://www.hudson.org/national-security-defense/ai-national-security-global-technology-race-how-us-export-controls-define-nury-turkel'
               target='_blank'
               style='font-size:0.78rem; color:#3a7bd5; font-weight:600; text-decoration:none;'>
              Read Full Article →
            </a>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with fa2:
        st.markdown("""
        <div style='background:#0d1b35; border:1px solid #1e3a6e; border-radius:12px; overflow:hidden;'>
          <div style='background: linear-gradient(135deg, #1a2e55 0%, #0d1b35 100%);
                      padding: 28px 24px 20px; border-bottom: 1px solid #1e3a6e;'>
            <div style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em;
                        color:#3a7bd5; font-weight:700; margin-bottom:10px;'>
              Visual Compliance · Trade Compliance 2026
            </div>
            <div style='font-family: Syne; font-size:1.15rem; font-weight:700;
                        color:#e8f0ff; line-height:1.4; margin-bottom:12px;'>
              2026 Global Trade Compliance Trends for Compliance Leaders
            </div>
            <div style='font-size:0.8rem; color:#8aa3cc; line-height:1.7;'>
              Tightening export control regimes are identified as the top compliance challenge of 2026,
              with AI-adjacent goods — precisely the HS codes tracked in this dashboard — facing the
              greatest regulatory scrutiny across major trading blocs. Essential reading for any firm
              navigating cross-border technology trade in the current environment.
            </div>
          </div>
          <div style='padding: 14px 24px; display:flex; align-items:center; justify-content:space-between;'>
            <span style='font-size:0.75rem; color:#4a6a9e;'>Visual Compliance · Jan 2026</span>
            <a href='https://www.visualcompliance.com/blog/2026-global-trade-compliance-trends/'
               target='_blank'
               style='font-size:0.78rem; color:#3a7bd5; font-weight:600; text-decoration:none;'>
              Read Full Article →
            </a>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Further reading expander ──────────────────────────────────────────────
    with st.expander("📰 Further Reading — Export Controls in the AI Era"):
        st.markdown("""
        | Source | Description |
        |--------|-------------|
        | [US CHIPS and Science Act (Congress.gov)](https://www.congress.gov/bill/117th-congress/house-bill/4346) | Full text of the 2022 legislation restricting semiconductor manufacturing investment and exports |
        | [Australian Defence Trade Controls Amendment Act (APH)](https://www.aph.gov.au/Parliamentary_Business/Bills_Legislation/Bills_Search_Results/Result?bId=r7161) | The 2024 legislation directly shaping the export controls tracked in this dashboard |
        | [ASML Export Restrictions — Reuters](https://www.reuters.com/technology/asml-gets-export-licenses-some-china-shipments-restricted-2023-2024-01-30/) | Dutch government restricts ASML chip-equipment exports to China, a parallel policy to AUKUS controls |
        | [Supply Chain Decoupling — Peterson Institute](https://www.piie.com/research/piie-charts/us-china-decoupling-semiconductors) | Peterson Institute analysis of US-China semiconductor decoupling and its economic costs |
        | [AI Chip Export Controls — MIT Technology Review](https://www.technologyreview.com/2023/10/17/1081719/the-us-chip-export-controls-are-already-failing/) | Critical assessment of whether export controls on AI chips are achieving their strategic goals |
        | [Hudson Institute: AI & Export Controls](https://www.hudson.org/national-security-defense/ai-national-security-global-technology-race-how-us-export-controls-define-nury-turkel) | How US export controls define the future of AI innovation and national security |
        | [2026 Trade Compliance Trends](https://www.visualcompliance.com/blog/2026-global-trade-compliance-trends/) | Key compliance challenges for trade-dependent businesses in the current regulatory environment |
        """)


    # ── RQ answer blurb ───────────────────────────────────────────────────────
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='font-family: Syne; font-size: 1.4rem; color: #e8f0ff; margin-bottom: 14px;'>
      So — Does AUKUS Affect Trade?
    </h2>
    <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8; margin-bottom: 14px;'>
      The short answer is: <strong style='color: #c8d6f0;'>yes, but not in the way you might expect.</strong>
      Total trade with China has grown significantly since AUKUS was announced, rising 34% in average
      annual value from the pre-AUKUS period to the post-AUKUS period — despite the agreement being
      explicitly designed to counter Chinese influence in the Indo-Pacific. This suggests that broad
      economic ties between Australia and China have remained resilient to geopolitical pressure at
      the aggregate level.
    </p>
    <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8; margin-bottom: 14px;'>
      The more telling story emerges when you look beneath the surface at <em>strategic goods</em> —
      the semiconductors, radar systems, and integrated circuits that are the physical backbone of
      AI and modern defence. Here, the pattern is different: Non-Affiliated partners like Japan and
      South Korea saw a statistically significant <strong style='color: #e05252;'>-32.6% decline</strong>
      in strategic goods trade, while China's strategic goods trade barely moved.
      This points to export control spillover effects — regulations aimed at restricting
      technology transfer may be disrupting trade with neutral partners more visibly than with
      the intended target.
    </p>
    <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8;'>
      For businesses, the implication is clear: <strong style='color: #c8d6f0;'>the risk is not just
      in trading with adversaries — it is in the compliance burden placed on otherwise routine
      supply chain relationships with allied and neutral partners.</strong> Navigate to the
      Total Trade Overview and High-Risk Commodity Flows pages to explore the data in depth.
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — TOTAL TRADE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Total Trade Overview":
    st.markdown("""
    <h1 style='font-family: Syne; font-size: 2rem; font-weight: 800; color: #e8f0ff;'>
      Total Trade Overview
    </h1>""", unsafe_allow_html=True)

    st.info("**RQ1:** Did total trade volumes change post-AUKUS across AUKUS members, China, and non-affiliated partners — and do the patterns differ meaningfully between these groups?")

    total = df[df["commodity"] == "TOTAL"].copy()
    if flow_filter != "Both":
        total = total[total["flow"] == flow_filter]

    annual = total.groupby(["year", "partner", "partner_group"])["trade_value"].sum().reset_index()

    # Main line chart
    fig_line = px.line(
        annual, x="year", y=annual["trade_value"] / 1e9,
        color="partner",
        color_discrete_map=COLORS,
        markers=True,
        labels={"y": "Trade Value (USD Billions)", "x": "Year"},
        title="Australia Total Trade by Partner (2017–2025)",
    )
    fig_line = add_event_lines(fig_line)
    fig_line.update_layout(**PLOTLY_LAYOUT, height=420)
    fig_line.update_traces(line_width=2.5, marker_size=7)
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    col_bar, col_table = st.columns(2, gap="large")

    # Pre/Post grouped bar
    with col_bar:
        grp_avg = (
            total[total["aukus_period"] != "Transition (2021)"]
            .groupby(["aukus_period", "partner_group"])["trade_value"]
            .mean()
            .reset_index()
        )
        fig_bar = px.bar(
            grp_avg, x="partner_group", y=grp_avg["trade_value"] / 1e9,
            color="aukus_period",
            barmode="group",
            color_discrete_map={
                "Pre-AUKUS (2017-2020)": "#2d4a7a",
                "Post-AUKUS (2022-2025)": "#3a7bd5",
            },
            labels={"y": "Avg Annual Trade (USD B)", "x": "Partner Group"},
            title="Avg Annual Trade: Pre vs Post AUKUS",
        )
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Stats table
    with col_table:
        st.markdown("<br>", unsafe_allow_html=True)
        stats_data = {
            "Partner Group":   ["AUKUS Members", "China", "Non-Affiliated"],
            "Pre-AUKUS Avg":   ["$23.03B", "$135.09B", "$31.53B"],
            "Post-AUKUS Avg":  ["$29.76B", "$181.00B", "$45.23B"],
            "% Change":        ["+29.2%", "+34.0%", "+43.5%"],
            "p-value":         [0.5054, 0.0286, 0.0379],
            "Significant":     ["⚠️ Directional", "✅ Yes", "✅ Yes"],
        }
        st.dataframe(
            pd.DataFrame(stats_data),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Mann-Whitney U test · Pre (2017–2020) vs Post (2022–2025) · p < 0.05 = significant")

    # ── Mann-Whitney explanation ───────────────────────────────────────────────
    with st.expander("ℹ️ About the Statistical Test — Mann-Whitney U"):
        st.markdown("""
        **What test is being performed?**
        A **Mann-Whitney U test** compares two independent groups to determine whether one group's
        values tend to be systematically higher or lower than the other's. Unlike a t-test, it does
        not assume a normal distribution, making it suitable for small trade samples like this one
        (4 pre-AUKUS annual observations vs. 4 post-AUKUS annual observations).

        **How to read the results:**
        - The test compares the annual trade values from the Pre-AUKUS period (2017–2020)
          against the Post-AUKUS period (2022–2025) for each partner group separately.
        - A **p-value below 0.05** indicates the difference between the two periods is statistically
          significant — meaning it is unlikely to be due to chance alone.
        - A **p-value above 0.05** means the difference is directional but not statistically
          confirmed at the 95% confidence level, and should be interpreted with caution.

        **What the results tell us here:**
        - **China (+34.0%, p = 0.029):** The increase in total trade with China post-AUKUS is
          statistically significant. Trade grew meaningfully, despite geopolitical tension.
        - **Non-Affiliated (+43.5%, p = 0.038):** Japan and South Korea also saw significant total
          trade growth — likely reflecting broader post-COVID economic recovery and Australia's
          deeper engagement with Indo-Pacific partners.
        - **AUKUS Members (+29.2%, p = 0.505):** While trade with the US and UK grew directionally,
          the result is not statistically significant. With only two countries in this group, the
          variance is high and the signal is weaker.

        *Note: With only 4 observations per period, all results should be interpreted directionally.
        Statistical significance here confirms a consistent trend, not a causal relationship with AUKUS.*
        """)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Import/Export split — full width, one per row
    st.markdown("<h3 style='font-family: Syne; color: #e8f0ff;'>Import vs Export Breakdown</h3>", unsafe_allow_html=True)
    for flow_type in ["Import", "Export"]:
        subset = total[total["flow"] == flow_type]
        ann = subset.groupby(["year", "partner"])["trade_value"].sum().reset_index()
        fig_f = px.line(ann, x="year", y=ann["trade_value"] / 1e9,
                        color="partner", color_discrete_map=COLORS,
                        markers=True, title=f"Australia {flow_type}s by Partner (2017–2025)",
                        labels={"y": "USD Billions", "x": "Year"})
        fig_f = add_event_lines(fig_f)
        fig_f.update_layout(**PLOTLY_LAYOUT, height=420)
        fig_f.update_traces(line_width=2.5, marker_size=7)
        st.plotly_chart(fig_f, use_container_width=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── RQ1 answer blurb ──────────────────────────────────────────────────────
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='font-family: Syne; font-size: 1.3rem; color: #e8f0ff; margin-bottom: 14px;'>
      Answering RQ1: Do Total Trade Patterns Differ Between Groups?
    </h2>
    <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8; margin-bottom: 14px;'>
      Yes — total trade volumes changed post-AUKUS for all three partner groups, but the patterns
      differ in both magnitude and statistical strength. <strong style='color: #c8d6f0;'>China and
      Non-Affiliated partners both showed statistically significant increases</strong> (p &lt; 0.05),
      while the increase for AUKUS Members, though directionally positive at +29.2%, did not reach
      statistical significance — likely due to the small sample of two countries in that group.
    </p>
    <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8; margin-bottom: 14px;'>
      Notably, Non-Affiliated partners (Japan and South Korea) showed the <em>largest</em> proportional
      increase in total trade at +43.5%, suggesting Australia's economic engagement with the broader
      Indo-Pacific has deepened in the post-AUKUS period — consistent with Australia's pivot toward
      regional partnerships beyond the Anglosphere.
    </p>
    <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8;'>
      The growth in total trade with China (+34.0%) is particularly significant from a policy
      perspective: despite AUKUS being explicitly framed as a counterweight to Chinese influence,
      aggregate trade volumes continued to grow. This does <em>not</em> mean AUKUS had no effect —
      but it does suggest that economic interdependence with China has remained resilient at the
      headline level. Whether this resilience holds when examining <strong style='color: #c8d6f0;'>
      strategically sensitive goods specifically</strong> is answered in the High-Risk Commodity
      Flows section.
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — HIGH-RISK COMMODITY FLOWS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "High-Risk Commodity Flows":
    st.markdown("""
    <h1 style='font-family: Syne; font-size: 2rem; font-weight: 800; color: #e8f0ff;'>
      High-Risk Commodity Flows
    </h1>""", unsafe_allow_html=True)

    st.info("**RQ2:** Did strategic goods trade change post-AUKUS, and does the direction of change differ between AUKUS members, China, and non-affiliated partners?")

    strat = df[df["commodity"] != "TOTAL"].copy()
    if flow_filter != "Both":
        strat = strat[strat["flow"] == flow_filter]

    all_commodities = sorted(strat["commodity"].unique())
    selected_commodities = st.multiselect(
        "Filter by Commodity (default = all)",
        options=all_commodities,
        default=all_commodities,
    )
    strat = strat[strat["commodity"].isin(selected_commodities)]

    # Stacked bar: pre/post by partner group
    strat_avg = (
        strat[strat["aukus_period"] != "Transition (2021)"]
        .groupby(["aukus_period", "partner_group", "commodity"])["trade_value"]
        .mean()
        .reset_index()
    )
    fig_sb = px.bar(
        strat_avg, x="partner_group", y=strat_avg["trade_value"] / 1e9,
        color="commodity", facet_col="aukus_period",
        barmode="stack",
        labels={"y": "Avg Annual Trade (USD B)", "x": "Partner Group"},
        title="Strategic Goods Trade by Partner Group: Pre vs Post AUKUS",
        height=420,
    )
    fig_sb.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_sb, use_container_width=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col_hm, col_st = st.columns(2, gap="large")

    # Heatmap: % change
    with col_hm:
        pre = (strat[strat["aukus_period"] == "Pre-AUKUS (2017-2020)"]
               .groupby(["partner_group", "commodity"])["trade_value"].mean())
        post = (strat[strat["aukus_period"] == "Post-AUKUS (2022-2025)"]
                .groupby(["partner_group", "commodity"])["trade_value"].mean())
        pct = ((post - pre) / pre.replace(0, np.nan) * 100).reset_index()
        pct.columns = ["partner_group", "commodity", "pct_change"]
        pivot_hm = pct.pivot(index="partner_group", columns="commodity", values="pct_change")

        fig_hm = go.Figure(data=go.Heatmap(
            z=pivot_hm.values,
            x=pivot_hm.columns.tolist(),
            y=pivot_hm.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=np.round(pivot_hm.values, 1),
            texttemplate="%{text}%",
            hovertemplate="%{y} · %{x}<br>Change: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="% Change", tickfont=dict(color="#c8d6f0"), title_font=dict(color="#c8d6f0")),
        ))
        fig_hm.update_layout(**PLOTLY_LAYOUT, height=320,
                              title="% Change in Strategic Goods Trade (Pre → Post AUKUS)")
        fig_hm.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_hm, use_container_width=True)

    # Stats table
    with col_st:
        st.markdown("<br><br>", unsafe_allow_html=True)
        stats2 = {
            "Partner Group":  ["AUKUS Members", "China", "Non-Affiliated"],
            "Pre-AUKUS Avg":  ["$2.49B", "$11.44B", "$0.21B"],
            "Post-AUKUS Avg": ["$2.67B", "$11.77B", "$0.14B"],
            "% Change":       ["+7.1%", "+2.9%", "-32.6%"],
            "p-value":        [0.7209, 0.8857, 0.0148],
            "Significant":    ["⚠️ Directional", "⚠️ Directional", "✅ Yes"],
        }
        st.dataframe(pd.DataFrame(stats2), hide_index=True, use_container_width=True)

    st.warning("⚠️ **Key Finding:** Strategic goods trade with Non-Affiliated partners (Japan & South Korea) declined **-32.6%** post-AUKUS (p = 0.0148). This is the most statistically significant shift in the dataset and may reflect export control spillover effects on non-AUKUS, non-China partners — a potential unintended consequence of the regulatory regime.")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Strategic goods line over time
    strat_time = strat.groupby(["year", "partner"])["trade_value"].sum().reset_index()
    fig_sl = px.line(strat_time, x="year", y=strat_time["trade_value"] / 1e9,
                     color="partner", color_discrete_map=COLORS, markers=True,
                     title="Strategic Goods Trade Over Time by Partner",
                     labels={"y": "USD Billions", "x": "Year"})
    fig_sl = add_event_lines(fig_sl)
    fig_sl.update_layout(**PLOTLY_LAYOUT, height=380)
    st.plotly_chart(fig_sl, use_container_width=True)

    # ── RQ2 answer blurb ──────────────────────────────────────────────────────
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='font-family: Syne; font-size: 1.3rem; color: #e8f0ff; margin-bottom: 14px;'>
      Answering RQ2: Did Strategic Goods Trade Change — And Does It Vary By Partner?
    </h2>
    <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8; margin-bottom: 14px;'>
      Yes — but the direction and significance of change differs sharply between partner groups,
      revealing a more nuanced picture than total trade volumes alone would suggest.
      <strong style='color: #c8d6f0;'>The only statistically significant change in strategic
      goods trade was a -32.6% decline with Non-Affiliated partners</strong> (Japan and South Korea,
      p = 0.0148). This is the most striking finding in the dataset: partners with no direct
      involvement in AUKUS saw the largest and most statistically reliable shift in strategic goods
      trade with Australia.
    </p>
    <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8; margin-bottom: 14px;'>
      In contrast, strategic goods trade with <strong style='color: #c8d6f0;'>China changed very
      little</strong> — rising only +2.9% (p = 0.886, not significant). This is counterintuitive
      given that AUKUS export controls were partly designed to limit sensitive technology flows
      toward adversarial actors. One interpretation is that pre-existing trade structures with China
      in these categories are difficult to rapidly unwind, or that the controls have not yet fully
      permeated bilateral trade flows at the commodity level.
    </p>
    <p style='color: #8aa3cc; font-size: 0.92rem; line-height: 1.8;'>
      For businesses, this divergence has important operational implications.
      <strong style='color: #e05252;'>Firms sourcing AI-critical components from Japan or South
      Korea should be actively assessing whether new compliance requirements are affecting their
      supplier relationships.</strong> The data suggests these neutral partners may be bearing
      a disproportionate share of the regulatory friction generated by AUKUS-linked export controls —
      an unintended consequence that warrants attention from both procurement teams and
      trade policy practitioners.
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — TRADE REGIME CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Trade Regime Clustering":
    st.markdown("""
    <h1 style='font-family: Syne; font-size: 2rem; font-weight: 800; color: #e8f0ff;'>
      Trade Regime Clustering
    </h1>""", unsafe_allow_html=True)

    st.info("**RQ3:** Can an unsupervised clustering model identify distinct trade regimes in Australia's annual trading patterns without a predetermined breakpoint, and does the model's natural grouping align with the pre- and post-AUKUS periods?")

    st.markdown("""
    <p style='color: #8aa3cc; font-size: 0.9rem; line-height: 1.7; margin-bottom: 8px;'>
      K-Means clustering (k=2) was applied to annual trade data using strategic goods composition
      by partner as features — <strong style='color: #c8d6f0;'>without providing the model any
      information about AUKUS timing.</strong> The question is whether the algorithm independently
      identifies the same structural break that the AUKUS timeline would predict.
    </p>
    """, unsafe_allow_html=True)

    strat_all = df[df["commodity"] != "TOTAL"].copy()

    with st.spinner("Running clustering models…"):

        # ── Cluster 1: Year-level ─────────────────────────────────────────────
        pivot_year = (strat_all
                      .groupby(["year", "commodity"])["trade_value"]
                      .sum()
                      .unstack(fill_value=0))

        scaler = StandardScaler()
        X_year = scaler.fit_transform(pivot_year)
        km_year = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_year)
        labels_year = km_year.labels_

        # Ensure cluster 0 = pre-AUKUS (lower years)
        years = pivot_year.index.tolist()
        mean_year_c0 = np.mean([y for y, l in zip(years, labels_year) if l == 0])
        mean_year_c1 = np.mean([y for y, l in zip(years, labels_year) if l == 1])
        if mean_year_c0 > mean_year_c1:
            labels_year = 1 - labels_year

        sil_year = silhouette_score(X_year, labels_year)

        total_by_year = (df[df["commodity"] == "TOTAL"]
                         .groupby("year")["trade_value"].sum() / 1e9)

        year_df = pd.DataFrame({
            "year": years,
            "cluster": labels_year,
            "regime": ["Pre-AUKUS Regime" if l == 0 else "Post-AUKUS Regime" for l in labels_year],
            "total_trade": [total_by_year.get(y, 0) for y in years],
            "aukus_period": ["Pre-AUKUS" if y <= 2020 else ("Transition" if y == 2021 else "Post-AUKUS")
                             for y in years],
        })

        # ── Cluster 2: Strategic goods composition per cluster ────────────────
        cluster_comp = strat_all.copy()
        cluster_comp["regime"] = cluster_comp["year"].map(
            dict(zip(year_df["year"], year_df["regime"])))
        comp_avg = (cluster_comp.groupby(["regime", "commodity"])["trade_value"]
                    .mean().reset_index())

        # ── Cluster 3: Partner profiles (post-AUKUS only) ─────────────────────
        post_strat = strat_all[strat_all["aukus_period"] == "Post-AUKUS (2022-2025)"]
        pivot_partner = (post_strat
                         .groupby(["partner", "commodity"])["trade_value"]
                         .mean()
                         .unstack(fill_value=0))

        X_partner = scaler.fit_transform(pivot_partner)
        km_partner = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_partner)

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_partner)

        partner_df = pd.DataFrame({
            "partner": pivot_partner.index.tolist(),
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "cluster": km_partner.labels_.astype(str),
        })

    # ── Viz 1: Year clusters ──────────────────────────────────────────────────
    st.markdown("<h3 style='font-family: Syne; color: #e8f0ff; margin-top: 12px;'>① Year-Level Regime Detection</h3>", unsafe_allow_html=True)

    fig_yc = px.scatter(
        year_df, x="year", y="total_trade",
        color="regime",
        color_discrete_map={"Pre-AUKUS Regime": "#2d4a7a", "Post-AUKUS Regime": "#3a7bd5"},
        size="total_trade", size_max=22,
        text="year",
        labels={"total_trade": "Total Trade (USD Billions)", "year": "Year"},
        title="K-Means Year Clustering: Unsupervised Regime Detection",
        hover_data={"aukus_period": True, "total_trade": ":.1f"},
    )
    fig_yc.add_vline(x=2021.5, line_dash="dash", line_color="#4a6a9e", line_width=1)
    fig_yc.update_traces(textposition="top center", textfont=dict(color="#c8d6f0", size=11))
    fig_yc.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig_yc, use_container_width=True)

    st.success(f"✅ **Without being told where the break occurred**, the model assigned 2017–2021 to Cluster 0 (Pre-AUKUS Regime) and 2022–2025 to Cluster 1 (Post-AUKUS Regime) — aligning precisely with the AUKUS timeline. Silhouette score: **{sil_year:.3f}**")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Viz 2: Composition per cluster ───────────────────────────────────────
    st.markdown("<h3 style='font-family: Syne; color: #e8f0ff;'>② Strategic Goods Composition by Regime</h3>", unsafe_allow_html=True)

    all_comp_commodities = sorted(comp_avg["commodity"].unique())
    selected_comp = st.multiselect(
        "Filter commodities (default = all)",
        options=all_comp_commodities,
        default=all_comp_commodities,
        key="comp_commodity_filter",
    )
    comp_filtered = comp_avg[comp_avg["commodity"].isin(selected_comp)].copy()

    # Ensure Pre-AUKUS Regime always renders before Post-AUKUS Regime
    comp_filtered["regime"] = pd.Categorical(
        comp_filtered["regime"],
        categories=["Pre-AUKUS Regime", "Post-AUKUS Regime"],
        ordered=True,
    )
    comp_filtered = comp_filtered.sort_values("regime")

    fig_comp = px.bar(
        comp_filtered, x="commodity", y=comp_filtered["trade_value"] / 1e9,
        color="regime",
        barmode="group",
        color_discrete_map={"Pre-AUKUS Regime": "#2d4a7a", "Post-AUKUS Regime": "#3a7bd5"},
        labels={"y": "Avg Annual Trade (USD B)", "x": ""},
        title="What Changed? Strategic Goods Composition Per Regime",
        category_orders={"regime": ["Pre-AUKUS Regime", "Post-AUKUS Regime"]},
    )
    fig_comp.update_layout(**PLOTLY_LAYOUT, height=380)
    fig_comp.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Viz 3: Partner profile clustering ────────────────────────────────────
    st.markdown("<h3 style='font-family: Syne; color: #e8f0ff;'>③ Partner Profiles — Strategic Goods Composition (Post-AUKUS)</h3>", unsafe_allow_html=True)

    fig_pca = px.scatter(
        partner_df, x="PC1", y="PC2",
        color="cluster",
        text="partner",
        color_discrete_sequence=["#3a7bd5", "#e05252", "#4caf7d"],
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        title="Partner Clustering by Strategic Goods Mix (PCA · Post-AUKUS Only)",
    )
    fig_pca.update_traces(textposition="top center", marker_size=16,
                          textfont=dict(color="#e8f0ff", size=12))
    fig_pca.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.info("🔍 **Interpretation:** This clustering asks whether trading partners group naturally by *what* strategic goods they exchange with Australia — independent of their AUKUS status. Partners that cluster together share a similar strategic goods composition profile with Australia, which may reflect underlying industrial or security alignment.")

    # ── Model quality ─────────────────────────────────────────────────────────
    with st.expander("📐 Model Quality Indicators"):
        q1, q2 = st.columns(2)
        with q1:
            st.dataframe(pd.DataFrame({
                "Metric": ["K-Means k", "Silhouette Score (year clusters)",
                           "Inertia (year clusters)", "PCA Variance Explained"],
                "Value": [2, f"{sil_year:.3f}", f"{km_year.inertia_:.1f}",
                          f"{sum(pca.explained_variance_ratio_)*100:.1f}%"],
            }), hide_index=True)
        with q2:
            st.caption("""
            **Interpreting silhouette score:** Range –1 to +1. Above 0.5 = strong structure.
            With only 9 annual data points, results should be interpreted directionally.
            The clean temporal split (all pre-2022 in one cluster, all post-2022 in another)
            is the substantive finding — not the precise score.
            """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "About This Dashboard":
    st.markdown("""
    <h1 style='font-family: Syne; font-size: 2rem; font-weight: 800; color: #e8f0ff;'>
      About This Dashboard
    </h1>""", unsafe_allow_html=True)

    col_about1, col_about2 = st.columns(2, gap="large")

    with col_about1:
        st.markdown("""
        <h2 style='font-family: Syne; font-size: 1.2rem; color: #e8f0ff;'>Project Overview</h2>
        <p style='color: #8aa3cc; font-size: 0.9rem; line-height: 1.8;'>
          This dashboard investigates how the AUKUS trilateral security agreement —
          and its associated export control legislation — has affected Australia's
          trade patterns with five key partners: the United States, United Kingdom,
          China, Japan, and South Korea.<br><br>
          In the age of AI, export controls are no longer a niche compliance issue.
          The strategic goods tracked here — semiconductors, integrated circuits,
          radar systems — are the same inputs powering global AI infrastructure.
          This project makes those shifts visible and quantifiable for decision-makers.
        </p>
        """, unsafe_allow_html=True)

    with col_about2:
        st.markdown("""
        <h2 style='font-family: Syne; font-size: 1.2rem; color: #e8f0ff;'>Research Questions</h2>
        """, unsafe_allow_html=True)
        for rq, text in [
            ("RQ1", "Did total trade volumes change post-AUKUS across AUKUS members, China, and non-affiliated partners — and do the patterns differ meaningfully between these groups?"),
            ("RQ2", "Did strategic goods trade change post-AUKUS, and does the direction of change differ between AUKUS members, China, and non-affiliated partners?"),
            ("RQ3", "Can an unsupervised clustering model identify distinct trade regimes in Australia's annual trading patterns without a predetermined breakpoint, and does the model's natural grouping align with the pre- and post-AUKUS periods?"),
        ]:
            st.markdown(f"""
            <div style='margin-bottom: 14px;'>
              <span class='rq-badge'>{rq}</span>
              <p style='color: #8aa3cc; font-size: 0.88rem; line-height: 1.6; margin: 4px 0 0 0;'>{text}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # PDID expanders
    st.markdown("<h2 style='font-family: Syne; font-size: 1.2rem; color: #e8f0ff;'>PDID Framework</h2>", unsafe_allow_html=True)

    pdid = {
        "🔍 Problem": [
            "AUKUS was announced in September 2021 as a trilateral security partnership between Australia, the US, and the UK",
            "The agreement introduced export controls on strategic and dual-use goods in 2024 via the Defence Trade Controls Amendment Act",
            "Research question: does a major defence policy realignment leave a measurable footprint in trade data?",
            "Focus narrowed to five partners (US, UK, China, Japan, South Korea) and a curated set of strategic HS codes",
        ],
        "📦 Data": [
            "Source: UN Comtrade Database — Australia as reporting country, 2017–2025 (annual)",
            "Strategic goods: HS 8411, 8471, 8517, 8526, 8541, 8542, 8543, 8802, 8803, 8806, 8906, 9014, 9031 plus TOTAL aggregate",
            "Key cleaning steps: duplicate HS code descriptions harmonised across classification years; CIF used for imports, FOB for exports",
            "Partners grouped as AUKUS Members (US+UK), China (standalone), Non-Affiliated (Japan+South Korea)",
        ],
        "💡 Insights": [
            "RQ1: Total trade grew with all groups post-AUKUS, but only China (+34%, p=0.029) and Non-Affiliated (+43.5%, p=0.038) were statistically significant",
            "RQ2: The only statistically significant strategic goods shift was a -32.6% decline in Non-Affiliated trade (p=0.015) — a potential unintended consequence of export controls",
            "RQ3: K-Means (k=2) independently partitioned years 2017–2021 vs 2022–2025 without any temporal labelling, confirming a structural trade regime shift",
            "Key tension: total trade with China grew while strategic goods remained flat — suggesting volume resilience but controlled technology flow",
        ],
        "🚀 Deployment": [
            "Built with Streamlit for accessible, interactive deployment without requiring coding knowledge from end users",
            "Deployed via Streamlit Community Cloud connected to a public GitHub repository",
            "All clustering runs live in the app using scikit-learn; data loaded from a single CSV (UN Comtrade export)",
            "Designed for non-technical stakeholders: plain-English callouts, KPI cards, and structured narrative tabs",
        ],
    }

    for title, bullets in pdid.items():
        with st.expander(title):
            for b in bullets:
                st.markdown(f"- {b}")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.info("**Data Source:** UN Comtrade Database. Reporting country: Australia. Partners: US, UK, China, Japan, South Korea. Period: 2017–2025 (annual). HS codes tracked: 8411, 8471, 8517, 8526, 8541, 8542, 8543, 8802, 8803, 8806, 8906, 9014, 9031, and TOTAL aggregates.")

    st.warning("**Limitations:** Correlation with AUKUS does not establish causation — post-COVID global trade recovery affects all post-2021 figures. 2025 data may be partial. Small sample size (9 annual observations per partner) limits statistical power of Mann-Whitney tests. Clustering results should be interpreted directionally given the small n.")

    with st.expander("📋 Variable Glossary"):
        st.markdown("""
        | Variable | Definition |
        |----------|-----------|
        | `partner_group` | Categorical grouping: AUKUS Members (US+UK), China, Non-Affiliated (Japan+South Korea) |
        | `trade_value` | USD value of trade: CIF (cost, insurance, freight) for imports; FOB (free on board) for exports |
        | `aukus_period` | Pre-AUKUS (2017–2020), Transition (2021), Post-AUKUS (2022–2025) |
        | `strategic goods` | HS-coded commodities with dual-use or defence relevance (semiconductors, radar, aircraft parts, etc.) |
        | `TOTAL commodity` | UN Comtrade aggregate row representing all goods traded — used for total trade analysis |
        | `CIF value` | Import valuation including cost of goods + insurance + freight to destination |
        | `FOB value` | Export valuation of goods at point of departure, excluding insurance and freight |
        """)
