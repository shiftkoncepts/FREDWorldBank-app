import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fredapi import Fred
import requests

st.set_page_config(page_title="Client Risk Analyzer – Global", layout="wide")

st.title("Client Risk Analyzer – Global 🌍")

st.markdown(
    """
Upload a client dataset and explore risk by **country** and **sector**.

**RiskScore = BaseRisk + sector-specific FRED macro (US clients) + World Bank country macro (all countries).**

Your file should contain:

**Required columns**
- `Client` – client name  
- `Country` – country code or name (e.g. US, CN, UK…)  
- `BaseRisk` – internal client risk score (0–100 or your own scale)  
- One of: `NAICS` **or** `SIC` – industry classification  

World Bank and FRED data are fetched live; no API key is needed for World Bank.
"""
)

# ------------- Helpers: country code normalization -------------
def normalize_country_code(val: str):
    """Convert various country strings to ISO2 if possible."""
    if pd.isna(val):
        return None
    s = str(val).strip().upper()

    manual = {
        "US": "US",
        "USA": "US",
        "UNITED STATES": "US",
        "CHINA": "CN",
        "P.R. CHINA": "CN",
        "PEOPLE'S REPUBLIC OF CHINA": "CN",
        "PEOPLES REPUBLIC OF CHINA": "CN",
        "UK": "GB",
        "UNITED KINGDOM": "GB",
    }
    if s in manual:
        return manual[s]
    if len(s) == 2:
        return s
    return None  # unknown format; WB call will be skipped


# ------------- FRED overall snapshot (for display only) -------------
@st.cache_data
def get_fred_overall():
    fred = Fred(api_key=st.secrets["FRED_API_KEY"])
    nfcfi = fred.get_series("NFCI")      # Financial Conditions Index
    unrate = fred.get_series("UNRATE")   # Unemployment rate
    return {
        "nfcfi": float(nfcfi.iloc[-1]),
        "unrate": float(unrate.iloc[-1]),
    }


# ------------- FRED sector-specific configuration -------------
FRED_SECTOR_INFO = {
    "Manufacturing":                        {"series": "INDPRO",   "good_high": True},   # Industrial production
    "Retail Trade":                         {"series": "RSXFS",    "good_high": True},   # Retail sales excl. autos
    "Wholesale Trade":                      {"series": "RSXFS",    "good_high": True},
    "Finance & Insurance":                  {"series": "STLFSI4",  "good_high": False},  # Financial stress index
    "Information":                          {"series": "INDPRO",   "good_high": True},
    "Construction":                         {"series": "HOUST",    "good_high": True},   # Housing starts
    "Real Estate & Rental":                 {"series": "HOUST",    "good_high": True},
    "Natural Resources & Mining":           {"series": "IPMINE",   "good_high": True},
    "Transportation & Warehousing":         {"series": "TSIFRGHT", "good_high": True},   # Freight TSI
    "Education & Healthcare":               {"series": "CPIMEDSL", "good_high": False},  # Medical CPI (high = stress)
    "Admin & Support":                      {"series": "UNRATE",   "good_high": False},
    "Professional, Scientific & Technical": {"series": "INDPRO",   "good_high": True},
    "Management of Companies":              {"series": "INDPRO",   "good_high": True},
    "Arts, Entertainment & Accommodation":  {"series": "UNRATE",   "good_high": False},
    "Other Services & Public Admin":        {"series": "UNRATE",   "good_high": False},
    "Utilities":                            {"series": "INDPRO",   "good_high": True},
}

@st.cache_data
def get_sector_stress(series_id: str, good_high: bool) -> float:
    """
    Get a sector stress score from a FRED time series.

    - Pull full series
    - Compute z-score of latest vs history
    - If good_high=True, invert sign so high level = low stress
    - Cap to [-3, 3] and scale lightly
    """
    try:
        fred = Fred(api_key=st.secrets["FRED_API_KEY"])
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
    stress = -z if good_high else z
    stress = float(max(min(stress, 3.0), -3.0))  # cap
    return stress * 2.0  # scale so adj is roughly -6 to +6


# ------------- World Bank: fetch indicators & build country stress -------------
WB_INDICATORS = {
    "gdp_growth":       "NY.GDP.MKTP.KD.ZG",   # GDP growth (%)
    "inflation":        "FP.CPI.TOTL.ZG",      # Inflation (%)
    "external_debt_gni":"DT.DOD.DECT.GN.ZS",   # External debt (% of GNI)
    "pol_stability":    "PV.PSAV.PT",          # Political stability index (-2.5 to 2.5)
}

def wb_latest_value(country_code: str, indicator: str):
    """Fetch latest non-null value for a given WB indicator & country."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return np.nan
        entries = data[1]
    except Exception:
        return np.nan

    for entry in entries:
        val = entry.get("value")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return np.nan

@st.cache_data
def get_world_bank_country_stress(country_codes):
    """
    For each ISO2 country code, fetch WB indicators and compute a stress score.

    Stress is a weighted combo of z-scores:
      + inflation_z
      + external_debt_z
      - gdp_growth_z
      - pol_stability_z
    """
    rows = []
    for cc in country_codes:
        if cc is None:
            continue
        metrics = {}
        for key, ind in WB_INDICATORS.items():
            metrics[key] = wb_latest_value(cc.lower(), ind)
        row = {"CountryCode": cc}
        row.update(metrics)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["CountryCode", "WB_Stress"])

    wb_df = pd.DataFrame(rows)

    # Compute z-scores for each metric across available countries
    for col in ["gdp_growth", "inflation", "external_debt_gni", "pol_stability"]:
        s = wb_df[col]
        mean = s.mean()
        std = s.std()
        if std == 0 or pd.isna(std):
            wb_df[col + "_z"] = 0.0
        else:
            wb_df[col + "_z"] = (s - mean) / std

    # Higher inflation/external_debt_z => more stress
    # Higher gdp_growth_z / pol_stability_z => less stress
    wb_df["WB_Stress"] = (
        0.3 * wb_df["inflation_z"] +
        0.3 * wb_df["external_debt_gni_z"] +
        0.2 * (-wb_df["gdp_growth_z"]) +
        0.2 * (-wb_df["pol_stability_z"])
    )

    # Cap and scale a bit (roughly -6 to +6)
    wb_df["WB_Stress"] = wb_df["WB_Stress"].clip(-3, 3) * 2.0

    return wb_df[["CountryCode", "WB_Stress"]]


# Try to get overall FRED snapshot
try:
    fred_overall = get_fred_overall()
except Exception:
    fred_overall = None


# ------------- File upload / sample fallback -------------
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
    })


# ------------- Sector mapping (NAICS or SIC) -------------
def sector_from_two_digit(code2: int) -> str:
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
    if "NAICS" in row and not pd.isna(row["NAICS"]):
        try:
            code2 = int(str(int(row["NAICS"]))[:2])
            return sector_from_two_digit(code2)
        except Exception:
            pass
    if "SIC" in row and not pd.isna(row["SIC"]):
        try:
            code2 = int(str(int(row["SIC"]))[:2])
            return sector_from_two_digit(code2)
        except Exception:
            pass
    return "Other / Unknown"


# ------------- Validate minimum columns -------------
base_required = ["Client", "Country", "BaseRisk"]
missing_basic = [c for c in base_required if c not in df.columns]
if missing_basic:
    st.error(f"Your data is missing required columns: {missing_basic}")
    st.stop()

if "NAICS" not in df.columns and "SIC" not in df.columns:
    st.error("Your data must contain either a `NAICS` column or a `SIC` column.")
    st.stop()

# ------------- Derive sector, normalize types -------------
df["Sector"] = df.apply(infer_sector, axis=1)
df["CountryRaw"] = df["Country"].astype(str)
df["CountryCode"] = df["CountryRaw"].apply(normalize_country_code)

df["BaseRisk"] = pd.to_numeric(df["BaseRisk"], errors="coerce").fillna(0.0)

# ------------- World Bank: compute country stress and map to clients -------------
unique_codes = sorted({c for c in df["CountryCode"].unique() if c is not None})
if unique_codes:
    wb_df = get_world_bank_country_stress(unique_codes)
    wb_map = wb_df.set_index("CountryCode")["WB_Stress"].to_dict()
    df["WB_CountryRisk"] = df["CountryCode"].map(wb_map).fillna(0.0)
else:
    df["WB_CountryRisk"] = 0.0

# ------------- FRED sector macro: apply to US clients only -------------
df["Macro_Adj"] = 0.0
if fred_overall is not None:
    us_mask = df["CountryCode"] == "US"
    us_sectors = df.loc[us_mask, "Sector"].dropna().unique().tolist()
    sector_adj = {}
    for sector in us_sectors:
        info = FRED_SECTOR_INFO.get(sector)
        if not info:
            sector_adj[sector] = 0.0
            continue
        sector_adj[sector] = get_sector_stress(info["series"], info["good_high"])
    df.loc[us_mask, "Macro_Adj"] = df.loc[us_mask, "Sector"].map(sector_adj).fillna(0.0)

# ------------- Final RiskScore -------------
df["RiskScore"] = df["BaseRisk"] + df["Macro_Adj"] + df["WB_CountryRisk"]

# ------------- Filters -------------
st.sidebar.header("Filters")

all_countries = sorted(df["CountryRaw"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=all_countries,
    default=all_countries,
)

filtered = df.copy()
if selected_countries:
    filtered = filtered[filtered["CountryRaw"].isin(selected_countries)]

available_sectors = sorted(filtered["Sector"].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect(
    "Sectors (NAICS / SIC groups)",
    options=available_sectors,
    default=available_sectors,
)
if selected_sectors:
    filtered = filtered[filtered["Sector"].isin(selected_sectors)]

# ------------- Top metrics -------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Clients")
    st.metric("Total (filtered)", len(filtered))
    if not filtered.empty:
        st.metric("Avg RiskScore", f"{filtered['RiskScore'].mean():.1f}")

with col2:
    st.subheader("FRED – US Macro Snapshot")
    if fred_overall:
        st.metric("NFCI", f"{fred_overall['nfcfi']:.3f}")
        st.metric("UNRATE", f"{fred_overall['unrate']:.2f}%")
    else:
        st.info("FRED API key not configured or FRED unavailable.")

with col3:
    st.subheader("World Bank Country Stress")
    if not filtered.empty:
        st.metric("Avg WB_CountryRisk", f"{filtered['WB_CountryRisk'].mean():.2f}")
    else:
        st.write("N/A")

# ------------- Charts -------------
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
        filtered.groupby("CountryRaw", as_index=False)["RiskScore"]
        .mean()
        .sort_values("RiskScore", ascending=False)
    )
    fig_country = px.bar(
        country_group,
        x="CountryRaw",
        y="RiskScore",
        labels={"RiskScore": "Avg Risk Score", "CountryRaw": "Country"},
        title="Average Client Risk by Country",
    )
    st.plotly_chart(fig_country, use_container_width=True)

# ------------- Table -------------
st.subheader("Client-Level Detail")
show_cols = ["Client", "CountryRaw", "BaseRisk", "Sector",
             "Macro_Adj", "WB_CountryRisk", "RiskScore"]
show_cols = [c for c in show_cols if c in filtered.columns]
filtered = filtered.rename(columns={"CountryRaw": "Country"})
st.dataframe(filtered[show_cols], use_container_width=True)
