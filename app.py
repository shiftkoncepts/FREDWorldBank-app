import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fredapi import Fred

st.set_page_config(page_title="Client Risk Analyzer – Global", layout="wide")

st.title("Client Risk Analyzer – Global 🌍")

st.markdown(
    """
Upload a client dataset and explore risk by **country** and **sector**.

**RiskScore = BaseRisk + FRED macro adjustment (applied to US clients) + World Bank–style country risk (optional).**

Your file should contain:

**Required columns**
- `Client` – client name  
- `Country` – country code or name (e.g. US, CN, UK…)  
- `BaseRisk` – internal client risk score (0–100 or your own scale)  
- One of: `NAICS` **or** `SIC` – industry classification  

**Optional**
- `WB_CountryRisk` – numeric country risk score (e.g. derived from World Bank indicators)
"""
)

# ---------- FRED integration (for US macro) ----------
@st.cache_data
def get_fred_macro():
    """Fetch macro indicators from FRED and create a simple 'macro stress' score."""
    fred = Fred(api_key=st.secrets["FRED_API_KEY"])

    nfcfi = fred.get_series("NFCI")      # Chicago Fed National Financial Conditions Index
    unrate = fred.get_series("UNRATE")   # US Unemployment Rate

    recent_nfcfi = float(nfcfi.iloc[-1])
    recent_unrate = float(unrate.iloc[-1])

    # Demo formula (tweak later if you like)
    macro_stress = (recent_nfcfi + (recent_unrate / 10.0)) / 2.0

    return {
        "macro_stress": macro_stress,
        "nfcfi": recent_nfcfi,
        "unrate": recent_unrate,
    }

try:
    fred_data = get_fred_macro()
except Exception:
    fred_data = None


# ---------- File upload / sample fallback ----------
st.sidebar.header("Data Source")

uploaded = st.sidebar.file_uploader("Upload CSV or Excel client file", type=["csv", "xlsx"])

if uploaded is not None:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
else:
    st.sidebar.info("No file uploaded — using a 10-client US/China example.")
    df = pd.DataFrame({
        "Client": [
            "Google", "Apple", "Ford", "Goldman Sachs", "UnitedHealth",
            "Alibaba", "Tencent", "Evergrande", "PetroChina", "BYD"
        ],
        "Country": ["US", "US", "US", "US", "US", "CN", "CN", "CN", "CN", "CN"],
        "NAICS": [519130, 334111, 336110, 523110, 621111, 454110, 511210, 236220, 211120, 336320],
        "BaseRisk": [35.2, 22.8, 48.1, 40.6, 25.4, 55.3, 38.7, 92.4, 67.9, 44.2],
        # Example World Bank–style country risk scores
        "WB_CountryRisk": [8.0, 8.0, 8.0, 8.0, 8.0, 18.0, 18.0, 25.0, 18.0, 18.0],
    })


# ---------- Sector mapping helpers (NAICS or SIC) ----------
def sector_from_two_digit(code2: int) -> str:
    """Map a 2-digit industry code (NAICS or SIC group) to a broad sector label."""
    if 1 <= code2 <= 9 or 10 <= code2 <= 14 or 11 <= code2 <= 21:
        return "Natural Resources & Mining"
    if code2 == 22:
        return "Utilities"
    if code2 == 23 or 15 <= code2 <= 17:
        return "Construction"
    if 31 <= code2 <= 33 or 20 <= code2 <= 39:
        return "Manufacturing"
    if code2 == 42 or 50 <= code2 <= 51:
        return "Wholesale Trade"
    if 44 <= code2 <= 45 or 52 <= code2 <= 59:
        return "Retail Trade"
    if 48 <= code2 <= 49:
        return "Transportation & Warehousing"
    if code2 == 51 or code2 == 48:
        return "Information"
    if code2 == 52 or 60 <= code2 <= 67:
        return "Finance & Insurance"
    if code2 == 53 or 65 <= code2 <= 67:
        return "Real Estate & Rental"
    if code2 == 54 or 70 <= code2 <= 73:
        return "Professional, Scientific & Technical"
    if code2 == 55:
        return "Management of Companies"
    if code2 == 56 or 72 <= code2 <= 73:
        return "Admin & Support"
    if 61 <= code2 <= 62 or 80 <= code2 <= 89:
        return "Education & Healthcare"
    if 71 <= code2 <= 72 or 78 <= code2 <= 79:
        return "Arts, Entertainment & Accommodation"
    if 81 <= code2 <= 92 or 90 <= code2 <= 99:
        return "Other Services & Public Admin"
    return "Other / Unknown"


def infer_sector(row):
    """Use NAICS if present, otherwise SIC, to derive a sector."""
    # Prefer NAICS if available
    if "NAICS" in row and not pd.isna(row["NAICS"]):
        try:
            code2 = int(str(int(row["NAICS"]))[:2])
            return sector_from_two_digit(code2)
        except Exception:
            pass

    # Fallback to SIC
    if "SIC" in row and not pd.isna(row["SIC"]):
        try:
            code2 = int(str(int(row["SIC"]))[:2])
            return sector_from_two_digit(code2)
        except Exception:
            pass

    return "Other / Unknown"


# ---------- Validate minimal required cols ----------
base_required = ["Client", "Country", "BaseRisk"]
missing_basic = [c for c in base_required if c not in df.columns]

if missing_basic:
    st.error(f"Your data is missing required columns: {missing_basic}")
    st.stop()

if "NAICS" not in df.columns and "SIC" not in df.columns:
    st.error("Your data must contain either a `NAICS` column or a `SIC` column.")
    st.stop()

# Optional World Bank country risk column
if "WB_CountryRisk" not in df.columns:
    df["WB_CountryRisk"] = 0.0

# ---------- Derive sector and apply risk logic ----------
df["Sector"] = df.apply(infer_sector, axis=1)

# Normalize types
df["Country"] = df["Country"].astype(str)
df["BaseRisk"] = pd.to_numeric(df["BaseRisk"], errors="coerce").fillna(0.0)
df["WB_CountryRisk"] = pd.to_numeric(df["WB_CountryRisk"], errors="coerce").fillna(0.0)

# FRED macro adjustment for US clients only
if fred_data is not None:
    macro_stress = fred_data["macro_stress"]
    df["Macro_Adj"] = np.where(df["Country"].str.upper() == "US", macro_stress * 10.0, 0.0)
else:
    df["Macro_Adj"] = 0.0

# Final RiskScore
df["RiskScore"] = df["BaseRisk"] + df["Macro_Adj"] + df["WB_CountryRisk"]


# ---------- Filters (by country and sector) ----------
st.sidebar.header("Filters")

all_countries = sorted(df["Country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=all_countries,
    default=all_countries,
)

filtered = df.copy()
if selected_countries:
    filtered = filtered[filtered["Country"].isin(selected_countries)]

available_sectors = sorted(filtered["Sector"].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect(
    "Sectors (NAICS / SIC groups)",
    options=available_sectors,
    default=available_sectors
)

if selected_sectors:
    filtered = filtered[filtered["Sector"].isin(selected_sectors)]


# ---------- Top-Level Metrics ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Clients")
    st.metric("Total (filtered)", len(filtered))
    if not filtered.empty:
        st.metric("Avg RiskScore", f"{filtered['RiskScore'].mean():.1f}")

with col2:
    st.subheader("FRED – US Macro (used for US clients)")
    if fred_data:
        st.metric("NFCI", f"{fred_data['nfcfi']:.3f}")
        st.metric("UNRATE", f"{fred_data['unrate']:.2f}%")
    else:
        st.info("FRED API key not configured.")

with col3:
    st.subheader("World Bank Country Risk (all countries)")
    if not filtered.empty:
        avg_country_risk = filtered["WB_CountryRisk"].mean()
        st.metric("Avg WB_CountryRisk", f"{avg_country_risk:.1f}")
    else:
        st.write("N/A")


# ---------- Charts ----------
if filtered.empty:
    st.warning("No clients after applying filters.")
else:
    st.subheader("Average Risk by Sector")
    sector_group = (
        filtered.groupby("Sector", as_index=False)["RiskScore"]
        .mean()
        .sort_values("RiskScore", ascending=False)
    )
    fig_sector = px.bar(
        sector_group,
        x="Sector",
        y="RiskScore",
        labels={"RiskScore": "Avg Risk Score"},
        title="Average Client Risk by Sector",
    )
    fig_sector.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_sector, use_container_width=True)

    st.subheader("Average Risk by Country")
    country_group = (
        filtered.groupby("Country", as_index=False)["RiskScore"]
        .mean()
        .sort_values("RiskScore", ascending=False)
    )
    fig_country = px.bar(
        country_group,
        x="Country",
        y="RiskScore",
        labels={"RiskScore": "Avg Risk Score"},
        title="Average Client Risk by Country",
    )
    st.plotly_chart(fig_country, use_container_width=True)

# ---------- Table ----------
st.subheader("Client-Level Detail")
st.dataframe(filtered, use_container_width=True)
