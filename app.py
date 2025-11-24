import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fredapi import Fred

st.set_page_config(page_title="Client Risk Dashboard", layout="wide")

st.title("Client Risk Explorer – US vs Foreign 🌍")

st.markdown(
    """
Upload a client file or use the sample dataset.  
Risk scores incorporate **FRED macro stress** for US clients and  
use **NAICS codes** to group clients into economic sectors.
"""
)

# ---------- FRED integration ----------
@st.cache_data
def get_fred_macro():
    """
    Fetch macro indicators from FRED and create a simple 'macro stress' score.
    You can refine this later depending on legal/pricing use cases.
    """
    fred = Fred(api_key=st.secrets["FRED_API_KEY"])

    nfcfi = fred.get_series("NFCI")      # Chicago Fed National Financial Conditions Index
    unrate = fred.get_series("UNRATE")   # US Unemployment Rate

    recent_nfcfi = float(nfcfi.iloc[-1])
    recent_unrate = float(unrate.iloc[-1])

    # Demo formula (replace with your own scaling later)
    macro_stress = (recent_nfcfi + (recent_unrate / 10.0)) / 2.0

    return {
        "macro_stress": macro_stress,
        "nfcfi": recent_nfcfi,
        "unrate": recent_unrate,
    }

# Load FRED data
try:
    fred_data = get_fred_macro()
except Exception:
    fred_data = None


# ---------- File upload / sample data ----------
st.sidebar.header("Data Source")

uploaded = st.sidebar.file_uploader("Upload CSV or Excel client file", type=["csv", "xlsx"])

if uploaded is not None:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
else:
    st.sidebar.info("No file uploaded — using sample data.")
    np.random.seed(0)
    df = pd.DataFrame({
        "Client": [f"Client {chr(65+i)}" for i in range(20)],
        "Country": np.random.choice(["US", "UK", "DE", "CN", "BR"], size=20),
        "NAICS": np.random.choice([2211, 3254, 4234, 5241, 5411, 5613], size=20),
        "BaseRisk": np.random.uniform(0, 100, size=20)
    })


# ---------- NAICS to Sector Mapping ----------
def map_naics_to_sector(naics_code):
    try:
        code2 = int(str(int(naics_code))[:2])
    except (ValueError, TypeError):
        return "Unknown"

    if 11 <= code2 <= 21:
        return "Natural Resources & Mining"
    elif code2 == 22:
        return "Utilities"
    elif code2 == 23:
        return "Construction"
    elif 31 <= code2 <= 33:
        return "Manufacturing"
    elif code2 == 42:
        return "Wholesale Trade"
    elif 44 <= code2 <= 45:
        return "Retail Trade"
    elif 48 <= code2 <= 49:
        return "Transportation & Warehousing"
    elif code2 == 51:
        return "Information"
    elif code2 == 52:
        return "Finance & Insurance"
    elif code2 == 53:
        return "Real Estate & Rental"
    elif code2 == 54:
        return "Professional, Scientific & Technical"
    elif code2 == 55:
        return "Management of Companies"
    elif code2 == 56:
        return "Admin & Support"
    elif 61 <= code2 <= 62:
        return "Education & Healthcare"
    elif 71 <= code2 <= 72:
        return "Arts, Entertainment & Accommodation"
    elif 81 <= code2 <= 92:
        return "Other Services & Public Admin"
    else:
        return "Other / Unknown"


required_cols = ["Client", "Country", "NAICS", "BaseRisk"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your data is missing required columns: {missing}")
    st.stop()

# Derive Sector + Region
df["Sector"] = df["NAICS"].apply(map_naics_to_sector)
df["Region"] = np.where(df["Country"].str.upper() == "US", "US", "Foreign")


# ---------- Macro Adjustment for US Clients ----------
if fred_data is not None:
    macro_stress = fred_data["macro_stress"]
    df["Macro_Adj"] = np.where(df["Region"] == "US", macro_stress * 10.0, 0.0)
else:
    df["Macro_Adj"] = 0.0

df["RiskScore"] = df["BaseRisk"] + df["Macro_Adj"]


# ---------- Filters ----------
st.sidebar.header("Filters")

region_choice = st.sidebar.radio("Client Region", ["All", "US only", "Foreign only"])

filtered = df.copy()
if region_choice == "US only":
    filtered = filtered[filtered["Region"] == "US"]
elif region_choice == "Foreign only":
    filtered = filtered[filtered["Region"] == "Foreign"]

available_sectors = sorted(filtered["Sector"].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect(
    "Sectors (NAICS Groups)",
    options=available_sectors,
    default=available_sectors
)

if selected_sectors:
    filtered = filtered[filtered["Sector"].isin(selected_sectors)]


# ---------- Top-Level Metrics ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Client Counts")
    st.metric("Total (filtered)", len(filtered))

with col2:
    st.subheader("FRED Indicators")
    if fred_data:
        st.metric("NFCI", f"{fred_data['nfcfi']:.3f}")
        st.metric("UNRATE", f"{fred_data['unrate']:.2f}%")
    else:
        st.info("FRED API key not configured.")

with col3:
    st.subheader("Macro Stress")
    if fred_data:
        st.metric("Macro Stress (demo)", f"{fred_data['macro_stress']:.3f}")
    else:
        st.write("N/A")


# ---------- Charts ----------
st.subheader("Average Risk by Sector & Region")

if filtered.empty:
    st.warning("No clients after filtering.")
else:
    grouped = (
        filtered
        .groupby(["Region", "Sector"], as_index=False)["RiskScore"]
        .mean()
    )

    fig = px.bar(
        grouped,
        x="Sector",
        y="RiskScore",
        color="Region",
        barmode="group",
        title="Average Client Risk by Sector & Region",
        labels={"RiskScore": "Risk Score"}
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


# ---------- Table ----------
st.subheader("Client-Level Detail")
st.dataframe(filtered, use_container_width=True)
