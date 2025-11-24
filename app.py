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

**RiskScore = BaseRisk + Sector-specific FRED macro adjustment (US clients) + World Bank–style country risk (optional).**

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

# ---------- FRED: overall snapshot for US macro ----------
@st.cache_data
def get_fred_overall():
    """Overall US macro snapshot (for display only)."""
    fred = Fred(api_key=st.secrets["FRED_API_KEY"])

    nfcfi = fred.get_series("NFCI")      # Financial Conditions Index
    unrate = fred.get_series("UNRATE")   # Unemployment rate

    recent_nfcfi = float(nfcfi.iloc[-1])
    recent_unrate = float(unrate.iloc[-1])

    return {
        "nfcfi": recent_nfcfi,
        "unrate": recent_unrate,
    }


# ---------- FRED: sector-specific series ----------
# For each broad sector label, choose a FRED series + whether "high is good" or "high is bad".
FRED_SECTOR_INFO = {
    "Manufacturing":                   {"series": "INDPRO",      "good_high": True},   # Industrial production
    "Retail Trade":                    {"series": "RSXFS",       "good_high": True},   # Retail sales excl. autos
    "Wholesale Trade":                 {"series": "RSXFS",       "good_high": True},   # Approx by retail activity
    "Finance & Insurance":             {"series": "STLFSI4",     "good_high": False},  # Financial stress index
    "Information":                     {"series": "INDPRO",      "good_high": True},   # Use broad production proxy
    "Construction":                    {"series": "HOUST",       "good_high": True},   # Housing starts
    "Real Estate & Rental":            {"series": "HOUST",       "good_high": True},   # Same as construction
    "Natural Resources & Mining":      {"series": "IPMINE",      "good_high": True},   # Mining industrial production
    "Transportation & Warehousing":    {"series": "TSIFRGHT",    "good_high": True},   # Freight TSI
    "Education & Healthcare":          {"series": "CPIMEDSL",    "good_high": False},  # Medical CPI (high = stress)
    "Admin & Support":                 {"series": "UNRATE",      "good_high": False},  # Labor market slack
    "Professional, Scientific & Technical": {"series": "INDPRO", "good_high": True},
    "Management of Companies":         {"series": "INDPRO",      "good_high": True},
    "Arts, Entertainment & Accommodation": {"series": "UNRATE",  "good_high": False},
    "Other Services & Public Admin":   {"series": "UNRATE",      "good_high": False},
    "Utilities":                       {"series": "INDPRO",      "good_high": True},
}


@st.cache_data
def get_sector_stress(series_id: str, good_high: bool) -> float:
    """
    Get a sector stress score from a FRED time series.

    Steps:
    - Pull the full series.
    - Compute z-score of the latest value vs history.
    - If 'good_high' is True, invert sign so high activity = lower stress.
    - Cap to [-3, 3] to avoid extreme outliers.
    """
    try:
        fred = Fred(api_key=st.secrets["FRED_API_KEY"])
    except Exception:
        return 0.0

    try:
        series = fred.get_series(series_id)
    except Exception:
        return 0.0

    s = series.dropna()
    if len(s) < 10:
        return 0.0

    last = s.iloc[-1]
    mean = s.mean()
    std = s.std()
    if std == 0 or pd.isna(std):
        return 0.0

    z = (last - mean) / std
    # If high values are good (e.g., more production), then high = low stress.
    if good_high:
        stress = -z
    else:
        stress = z

    # Cap the stress to keep it reasonable
    stress = float(max(min(stress, 3.0), -3.0))
    return stress


# Try to get overall FRED info (for metrics).
try:
    fred_overall = get_fred_overall()
except Exception:
    fred_overall = None


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
            "Google", "Apple", "Ford Motor", "Goldman Sachs", "UnitedHealth",
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
if "WB_Coun_
