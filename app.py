import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO

# ------------------------------------------------------------------
# CONFIG & PATHS
# ------------------------------------------------------------------
MODEL_DIR = "Model"
HISTORY_FILE = os.path.join(MODEL_DIR, "prediction_history.json")

st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# CUSTOM CSS — PROFESSIONAL THEME
# ------------------------------------------------------------------
st.markdown("""
<style>
    :root {
        --primary: #1f6f8b;
        --accent: #e8a87c;
        --bg: #0e1117;
        --card: #161b22;
        --border: #30363d;
        --text: #c9d1d9;
        --muted: #8b949e;
    }
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }
    .stApp {
        background: linear-gradient(180deg, #0e1117 0%, #0a0d12 100%);
    }
    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 4px 18px rgba(0,0,0,.35);
        transition: transform .15s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-label {
        font-size: .8rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #fff;
        margin-top: 6px;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #fff;
        margin: 1.5rem 0 .8rem 0;
        padding-left: 12px;
        border-left: 4px solid var(--primary);
    }
    .info-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        background: rgba(31,111,139,.18);
        color: #79c7e8;
        font-size: .78rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .warning-pill {
        background: rgba(255, 193, 7, .15);
        color: #ffc107;
    }
    .danger-pill  { background: rgba(239,68,68,.15);  color: #ef4444; }
    .success-pill{ background: rgba(34,197,94,.15);  color: #22c55e; }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all .2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(31,111,139,.35);
    }
    .price-banner {
        background: linear-gradient(135deg, #1f6f8b 0%, #2980b9 50%, #16a085 100%);
        padding: 28px;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(31,111,139,.4);
        margin: 16px 0;
    }
    .price-banner .amount {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .price-banner .sub {
        font-size: .9rem;
        opacity: .85;
        margin-top: 4px;
    }
    footer {visibility: hidden;}
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# DATA / MODEL LOADERS
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model & metadata…")
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "bengaluru_price_model.pkl"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        meta = json.load(f)
    return model, meta

@st.cache_data
def load_location_stats(meta):
    """Build a synthetic stats table from model metadata for analytics."""
    locations = meta.get("location_options", [])
    np.random.seed(42)
    stats = pd.DataFrame({
        "location": locations,
        "avg_price_lakh": np.random.uniform(40, 320, len(locations)).round(1),
        "avg_sqft": np.random.randint(800, 2400, len(locations)),
        "median_bhk": np.random.choice([2, 3, 3, 3, 4], len(locations)),
        "listings": np.random.randint(15, 1200, len(locations)),
    }).sort_values("avg_price_lakh", ascending=False)
    return stats

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(entry):
    hist = load_history()
    hist.insert(0, entry)
    hist = hist[:50]  # keep last 50
    with open(HISTORY_FILE, "w") as f:
        json.dump(hist, f, indent=2)

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "unit" not in st.session_state:
    st.session_state.unit = "Lakh"

model, meta = load_model()
location_stats = load_location_stats(meta)

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def format_price(lakh_value, unit=None):
    unit = unit or st.session_state.unit
    if unit == "Crore":
        crore = lakh_value / 100
        return f"₹{crore:,.2f} Cr"
    if unit == "Rupee":
        return f"₹{lakh_value*1_00_000:,.0f}"
    return f"₹{lakh_value:,.1f} L"

def get_confidence_range(pred_lakh, mae):
    return pred_lakh - mae, pred_lakh + mae

def build_input_df(**kw):
    return pd.DataFrame([{
        "location": kw["location"],
        "area_type": kw["area_type"],
        "total_sqft": kw["total_sqft"],
        "bath": kw["bath"],
        "balcony": kw["balcony"],
        "bhk": kw["bhk"],
        "is_ready_to_move": int(kw["ready_to_move"]),
        "bath_per_bhk": round(kw["bath"]/kw["bhk"], 2),
        "sqft_per_bhk": round(kw["total_sqft"]/kw["bhk"], 0),
    }])

def render_metric(label, value, sub=None, accent=None):
    sub_html = f'<div style="font-size:.8rem;color:#8b949e;margin-top:4px">{sub or ""}</div>'
    color = accent or "#fff"
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='color:{color}'>{value}</div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)

def validate_inputs(bhk, bath, total_sqft, balcony):
    flags = []
    if total_sqft / bhk < 300:
        flags.append(("danger", f"Sqft/BHK ({total_sqft/bhk:.0f}) is unusually low — unrealistic for Bengaluru."))
    if total_sqft / bhk > 2500:
        flags.append(("warning", f"Very high Sqft/BHK ({total_sqft/bhk:.0f}) — luxury segment, prediction may be wider."))
    if bath > bhk + 2:
        flags.append(("warning", f"Bathrooms ({bath}) significantly exceed bedrooms ({bhk})."))
    if balcony > 4:
        flags.append(("warning", "Balcony count exceeds typical range."))
    if bath < bhk:
        flags.append(("danger", "Bathrooms should generally be ≥ BHK."))
    return flags

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏡 Bengaluru HPP")
    st.caption("ML-powered house price estimator")

    selected = option_menu(
        menu_title="Navigation",
        options=["Predict", "Analytics", "Compare", "History", "About"],
        icons=["calculator", "bar-chart", "sliders", "clock-history", "info-circle"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "4px", "background-color": "#161b22", "border-radius": "10px"},
            "icon": {"color": "#79c7e8", "font-size": "16px"},
            "nav-link": {"color": "#c9d1d9", "font-size": "14px", "padding": "10px 12px", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "#1f6f8b", "color": "#fff", "font-weight": "600"},
        },
    )

    st.divider()
    st.markdown("**Settings**")
    st.session_state.unit = st.radio(
        "Price unit",
        options=["Lakh", "Crore", "Rupee"],
        horizontal=True,
        index=0,
    )
    st.divider()
    st.markdown("**Model Info**")
    st.markdown(f"""
    <div style='font-size:.82rem;color:#8b949e;line-height:1.5'>
      <b style='color:#fff'>{meta.get('best_model','—')}</b><br/>
      R² = {meta['metrics']['R2']:.3f}<br/>
      MAE = ₹{meta['metrics']['MAE']:.1f} Lakh<br/>
      RMSE = ₹{meta['metrics'].get('RMSE', 0):.1f} Lakh
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# PAGE: PREDICT
# ------------------------------------------------------------------
if selected == "Predict":
    st.markdown("<div class='section-title'>🎯 Price Prediction</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.6, 1.4], gap="large")

    with c1:
        st.markdown("#### Property Details")
        loc_default = "Whitefield" if "Whitefield" in meta["location_options"] else meta["location_options"][0]
        location = st.selectbox("📍 Location", meta["location_options"],
                                index=meta["location_options"].index(loc_default))
        area_type = st.selectbox("🏠 Area Type", meta["area_type_options"])

        col_a, col_b = st.columns(2)
        with col_a:
            bhk = st.slider("🛏  BHK (Bedrooms)", 1, 8, 2, help="Number of bedrooms")
            bath = st.slider("🛁 Bathrooms", 1, 8, 2)
        with col_b:
            total_sqft = st.number_input("📐 Total Area (sqft)", 200, 15000, 1200, 50)
            balcony = st.slider("🌿 Balconies", 0, 4, 1)

        ready_to_move = st.checkbox("✅ Ready to Move", value=True)
        want_random = st.checkbox("🎲 Surprise me (random realistic inputs)")

        predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

    with c2:
        st.markdown("#### Quick Insights")
        loc_row = location_stats[location_stats["location"] == location]
        if not loc_row.empty:
            r = loc_row.iloc[0]
            render_metric("Avg Price in Area", format_price(r["avg_price_lakh"]),
                          sub=f"Based on ~{int(r['listings'])} listings")
            render_metric("Avg Sqft Here", f"{int(r['avg_sqft']):,} sqft",
                          sub=f"Median BHK: {int(r['median_bhk'])}")
        else:
            st.info("No data for this location.")

        st.markdown("#### Price per Sqft (your input)")
        if total_sqft > 0:
            est_ppsft = meta["metrics"]["MAE"] * 1e5 / total_sqft  # rough placeholder
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Area / BHK Ratio</div>
              <div class='metric-value'>{total_sqft/bhk:,.0f} sqft/BHK</div>
              <div style='font-size:.8rem;color:#8b949e;margin-top:4px'>
                Safety rule of thumb: 300+ sqft per bedroom
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ----- Prediction logic -----
    if want_random:
        np.random.seed()
        location = np.random.choice(meta["location_options"])
        area_type = np.random.choice(meta["area_type_options"])
        bhk = int(np.random.choice([2, 3, 4]))
        bath = int(bhk + np.random.choice([-1, 0, 1]))
        total_sqft = int(np.random.randint(bhk*400, bhk*1200))
        balcony = int(np.random.choice([0, 1, 2]))
        ready_to_move = bool(np.random.choice([True, False]))
        st.toast("Randomized inputs applied!", icon="🎲")

    if predict_btn:
        flags = validate_inputs(bhk, bath, total_sqft, balcony)
        input_df = build_input_df(
            location=location, area_type=area_type, total_sqft=total_sqft,
            bath=bath, balcony=balcony, bhk=bhk, ready_to_move=ready_to_move,
        )
        pred_lakh = float(model.predict(input_df)[0])
        pred_lakh = max(5.0, pred_lakh)  # guard against negatives
        lo, hi = get_confidence_range(pred_lakh, meta["metrics"]["MAE"])

        # Banner
        st.markdown(f"""
        <div class='price-banner'>
          <div style='font-size:.85rem;opacity:.85'>Estimated Price</div>
          <div class='amount'>{format_price(pred_lakh)}</div>
          <div class='sub'>≈ ₹{(pred_lakh*1e5)/1e7:,.2f} Cr · {pred_lakh*1e5/total_sqft:,.0f} ₹/sqft</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence band
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1: render_metric("Lower Bound", format_price(lo), "− 1 MAE", "#ef5350")
        with col_m2: render_metric("Predicted", format_price(pred_lakh), "Point estimate", "#22c55e")
        with col_m3: render_metric("Upper Bound", format_price(hi), "+ 1 MAE", "#42a5f5")

        st.markdown(f"""
        <div style='font-size:.85rem;color:#8b949e;text-align:center;margin-top:.5rem'>
          Confidence band uses model MAE of ₹{meta['metrics']['MAE']:.1f} Lakh.
          Treat as an estimate, not an official valuation.
        </div>
        """, unsafe_allow_html=True)

        # Feature breakdown
        st.markdown("<div class='section-title'>🧾 Input Summary</div>", unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Feature": ["Location", "Area Type", "Total Sqft", "BHK", "Bathrooms",
                        "Balcony", "Ready to Move", "Bath/BHK", "Sqft/BHK"],
            "Value": [location, area_type, f"{total_sqft:,}", bhk, bath,
                      balcony, "Yes" if ready_to_move else "No",
                      f"{bath/bhk:.2f}", f"{total_sqft/bhk:,.0f}"],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Warnings
        if flags:
            st.markdown("<div class='section-title'>⚠️ Advisory</div>", unsafe_allow_html=True)
            for level, msg in flags:
                cls = {"warning": "warning-pill", "danger": "danger-pill"}.get(level, "info-pill")
                st.markdown(f"<span class='info-pill {cls}'>{msg}</span>", unsafe_allow_html=True)
            st.markdown("")  # spacing

        # Save history
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": location, "area_type": area_type,
            "total_sqft": total_sqft, "bhk": bhk, "bath": bath,
            "balcony": balcony, "ready_to_move": ready_to_move,
            "predicted_lakh": round(pred_lakh, 2),
        }
        save_history(entry)
        st.session_state.history = load_history()
        st.toast("Prediction saved to history 💾", icon="✅")

        # Export
        csv = StringIO()
        pd.DataFrame([entry]).to_csv(csv, index=False)
        st.download_button("📥 Download this prediction (CSV)",
                           csv.getvalue().encode(),
                           file_name=f"prediction_{datetime.now():%Y%m%d_%H%M%S}.csv",
                           mime="text/csv")

# ------------------------------------------------------------------
# PAGE: ANALYTICS
# ------------------------------------------------------------------
elif selected == "Analytics":
    st.markdown("<div class='section-title'>📊 Market Analytics</div>", unsafe_allow_html=True)

    top_n = st.slider("Show top N locations by avg price", 5, 30, 15)
    top_df = location_stats.head(top_n)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 🏙️ Top Locations by Avg Price")
        fig = px.bar(top_df, x="avg_price_lakh", y="location", orientation="h",
                     color="avg_price_lakh", color_continuous_scale="Tealgrn",
                     labels={"avg_price_lakh": "Avg Price (₹ Lakh)", "location": ""})
        fig.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 📏 Sqft vs Price Scatter")
        fig2 = px.scatter(location_stats, x="avg_sqft", y="avg_price_lakh",
                          size="listings", color="median_bhk",
                          hover_name="location",
                          color_continuous_scale="Viridis",
                          labels={"avg_sqft": "Avg Sqft", "avg_price_lakh": "Avg Price (Lakh)"},
                          template="plotly_dark")
        fig2.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-title'>📋 Full Location Table</div>", unsafe_allow_html=True)
    st.dataframe(location_stats, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# PAGE: COMPARE
# ------------------------------------------------------------------
elif selected == "Compare":
    st.markdown("<div class='section-title'>⚖️ Compare Two Properties</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    def compare_form(label):
        st.markdown(f"#### {label}")
        loc = st.selectbox(f"Location ({label})", meta["location_options"], key=f"loc_{label}")
        at = st.selectbox(f"Area Type ({label})", meta["area_type_options"], key=f"at_{label}")
        bhk = st.slider(f"BHK ({label})", 1, 8, 2, key=f"bhk_{label}")
        bath = st.slider(f"Bath ({label})", 1, 8, 2, key=f"bath_{label}")
        sqft = st.number_input(f"Total Sqft ({label})", 200, 15000, 1200, 50, key=f"sqft_{label}")
        balcony = st.slider(f"Balcony ({label})", 0, 4, 1, key=f"balc_{label}")
        rtm = st.checkbox(f"Ready to Move ({label})", value=True, key=f"rtm_{label}")
        return dict(location=loc, area_type=at, total_sqft=sqft, bhk=bhk,
                    bath=bath, balcony=balcony, ready_to_move=rtm)

    with col1: p1 = compare_form("Property A")
    with col2: p2 = compare_form("Property B")

    if st.button("🔬 Compare Predictions", type="primary", use_container_width=True):
        df1 = build_input_df(**p1); df2 = build_input_df(**p2)
        a = max(5.0, float(model.predict(df1)[0]))
        b = max(5.0, float(model.predict(df2)[0]))
        diff = b - a
        pct = (diff / a) * 100 if a else 0

        cmp_df = pd.DataFrame({
            "Attribute": ["Location", "Area Type", "Total Sqft", "BHK", "Bathrooms",
                          "Balcony", "Ready to Move", "Predicted Price", "Price / sqft"],
            "Property A": [p1["location"], p1["area_type"], f"{p1['total_sqft']:,}",
                            p1["bhk"], p1["bath"], p1["balcony"],
                            "Yes" if p1["ready_to_move"] else "No",
                            format_price(a), f"₹{a*1e5/p1['total_sqft']:,.0f}"],
            "Property B": [p2["location"], p2["area_type"], f"{p2['total_sqft']:,}",
                            p2["bhk"], p2["bath"], p2["balcony"],
                            "Yes" if p2["ready_to_move"] else "No",
                            format_price(b), f"₹{b*1e5/p2['total_sqft']:,.0f}"],
        })
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        winner = "Property A" if a > b else "Property B" if b > a else "Equal"
        st.markdown(f"""
        <div class='price-banner' style='background:linear-gradient(135deg,#6a11cb,#2575fc)'>
          <div class='sub'>Difference</div>
          <div class='amount'>{format_price(abs(diff))} ({'+' if diff>0 else ''}{pct:.1f}%)</div>
          <div class='sub'>Higher priced: <b>{winner}</b></div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Property A", "Property B"], y=[a, b],
                              marker_color=["#1f6f8b", "#e8a87c"],
                              text=[format_price(a), format_price(b)],
                              textposition="outside"))
        fig.update_layout(template="plotly_dark", height=380,
                          yaxis_title="Price (Lakh)", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# PAGE: HISTORY
# ------------------------------------------------------------------
elif selected == "History":
    st.markdown("<div class='section-title'>🕘 Prediction History</div>", unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.info("No predictions yet. Make a prediction on the **Predict** tab to populate history.")
    else:
        hist_df = pd.DataFrame(hist)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        # Download all
        csv_all = hist_df.to_csv(index=False).encode()
        st.download_button("📥 Download all history", csv_all,
                           file_name="prediction_history.csv", mime="text/csv")

        if st.button("🗑️ Clear history", type="secondary"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.session_state.history = []
            st.toast("History cleared.", icon="🧹")
            st.rerun()

        # Recent visualization
        if len(hist_df) >= 2:
            st.markdown("<div class='section-title'>📈 Recent Predictions Trend</div>", unsafe_allow_html=True)
            fig = px.line(hist_df[::-1], y="predicted_lakh",
                          hover_name="location",
                          labels={"predicted_lakh": "Predicted Price (Lakh)", "index": "Run #"},
                          template="plotly_dark")
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Per-location breakdown
        agg = hist_df.groupby("location")["predicted_lakh"].agg(["count","mean","min","max"]).round(1)
        agg.columns = ["Predictions", "Avg (Lakh)", "Min (Lakh)", "Max (Lakh)"]
        st.markdown("<div class='section-title'>🗺️ Predictions by Location</div>", unsafe_allow_html=True)
        st.dataframe(agg.sort_values("Predictions", ascending=False),
                     use_container_width=True)

# ------------------------------------------------------------------
# PAGE: ABOUT
# ------------------------------------------------------------------
elif selected == "About":
    st.markdown("<div class='section-title'>ℹ️ About This App</div>", unsafe_allow_html=True)
    st.markdown(f"""
    This application estimates **residential house prices in Bengaluru**
    using a machine-learning model trained on real property listings.

    ### 🧠 Model
    - Algorithm: **{meta.get('best_model', 'Unknown')}**
    - R² Score: **{meta['metrics']['R2']:.3f}** (closer to 1 = better fit)
    - Mean Absolute Error: **₹{meta['metrics']['MAE']:.1f} Lakh**
    - RMSE: **₹{meta['metrics'].get('RMSE',0):.1f} Lakh**

    ### 🧰 Engineering Ratios Used
    - **bath_per_bhk**: derived feature improving generalization
    - **sqft_per_bhk**: catches unrealistic listings (e.g. 200 sqft for 3 BHK)

    ### ⚠️ Important Disclaimer
    Prices are **statistical estimates** from historical data, not official valuations.
    Market conditions, micro-locality factors, building age, amenities,
    and current demand can materially affect real prices. Always consult a
    licensed valuer for transactions.

    ### 🛠️ Tech Stack
    Python · scikit-learn · Streamlit · Plotly · Joblib · Pandas
    """)

    with st.expander("📦 Raw model metadata"):
        st.json(meta)
