import io
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="UIO Demand Forecast Model", layout="wide")

st.title("UIO Demand Forecast Model")
st.caption("Calculates compact age/product weighted rates first, then applies them to UIO. Includes demand by year, product, and location.")


REQUIRED_UIO_COLS = ["CBSA", "Forecast Year", "Vehicle Age", "Calibrated Retained UIO Rounded"]
REQUIRED_VMT_COLS = ["Annual VMT", "VMT Probability"]
PRODUCT_COL = "Product Category"


def clean_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def normalize_probability(series: pd.Series) -> pd.Series:
    s = clean_numeric(series)
    if s.dropna().max() > 1:
        s = s / 100
    return s


def parse_mileage_band(label: str) -> Optional[Tuple[int, int]]:
    nums = re.findall(r"\d[\d,]*", str(label))
    if len(nums) < 2:
        return None
    return int(nums[0].replace(",", "")), int(nums[1].replace(",", ""))


def build_mileage_bands_from_headers(headers) -> pd.DataFrame:
    rows = []
    for h in headers:
        parsed = parse_mileage_band(h)
        if parsed:
            rows.append({"Min Miles": parsed[0], "Max Miles": parsed[1], "Mileage Band": str(h).strip()})
    bands = pd.DataFrame(rows).drop_duplicates().sort_values("Min Miles").reset_index(drop=True)
    if bands.empty:
        raise ValueError("No mileage band headers could be parsed. Expected headers like '0 To 9,999'.")
    return bands


def assign_mileage_band(miles: pd.Series, bands: pd.DataFrame, cap_to_max_band: bool = True) -> pd.Series:
    """Assign cumulative miles to available mileage bands.

    If cap_to_max_band is True, any cumulative mileage above the highest
    available mileage band is assigned to the highest band instead of
    returning 'Above Max Mileage'. This prevents high-VMT scenarios from
    failing when the incidence/units files stop at a maximum mileage band.
    """
    bands = bands.sort_values("Min Miles").reset_index(drop=True)
    bins = list(bands["Min Miles"]) + [bands["Max Miles"].iloc[-1] + 1]
    labels = list(bands["Mileage Band"])

    assigned = pd.cut(
        miles,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    ).astype("object")

    if cap_to_max_band:
        max_miles = bands["Max Miles"].iloc[-1]
        max_band = bands["Mileage Band"].iloc[-1]
        min_miles = bands["Min Miles"].iloc[0]
        min_band = bands["Mileage Band"].iloc[0]
        assigned = assigned.where(miles <= max_miles, max_band)
        assigned = assigned.where(miles >= min_miles, min_band)
        assigned = assigned.fillna(max_band)
    else:
        assigned = assigned.fillna("Above Max Mileage")

    return assigned


def normalize_rate_table(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if PRODUCT_COL not in df.columns:
        raise ValueError(f"Missing required column: {PRODUCT_COL}")

    mileage_cols = [c for c in df.columns if c != PRODUCT_COL]
    long_df = df.melt(
        id_vars=[PRODUCT_COL],
        value_vars=mileage_cols,
        var_name="Mileage Band",
        value_name=value_name,
    )

    long_df[PRODUCT_COL] = long_df[PRODUCT_COL].astype(str).str.strip()
    long_df["Mileage Band"] = long_df["Mileage Band"].astype(str).str.strip()
    long_df[value_name] = clean_numeric(long_df[value_name])

    if value_name == "Incidence Rate" and long_df[value_name].dropna().max() > 1:
        long_df[value_name] = long_df[value_name] / 100

    return long_df


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def excel_bytes(sheets: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            sheet_name = name[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                ws.set_column(idx, idx, min(max(len(str(col)) + 2, 12), 42))
    return output.getvalue()


@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def run_forecast(
    uio_df: pd.DataFrame,
    incidence_wide: pd.DataFrame,
    units_wide: pd.DataFrame,
    vmt_df: pd.DataFrame,
    age0_factor: float,
    make_location_product_output: bool,
    make_location_age_output: bool,
    cap_to_max_band: bool,
):
    uio = uio_df.copy()
    uio.columns = [str(c).strip() for c in uio.columns]

    missing = [c for c in REQUIRED_UIO_COLS if c not in uio.columns]
    if missing:
        raise ValueError(f"UIO file is missing required columns: {missing}")

    uio["CBSA"] = uio["CBSA"].astype(str).str.strip()
    uio["Forecast Year"] = clean_numeric(uio["Forecast Year"]).astype("Int64")
    uio["Vehicle Age"] = clean_numeric(uio["Vehicle Age"]).astype("Int64")
    uio["UIO"] = clean_numeric(uio["Calibrated Retained UIO Rounded"]).fillna(0)

    uio = uio.dropna(subset=["Forecast Year", "Vehicle Age"])
    uio["Forecast Year"] = uio["Forecast Year"].astype(int)
    uio["Vehicle Age"] = uio["Vehicle Age"].astype(int)

    vmt = vmt_df.copy()
    vmt.columns = [str(c).strip() for c in vmt.columns]
    missing = [c for c in REQUIRED_VMT_COLS if c not in vmt.columns]
    if missing:
        raise ValueError(f"VMT file is missing required columns: {missing}")

    vmt["Annual VMT"] = clean_numeric(vmt["Annual VMT"])
    vmt["VMT Probability"] = normalize_probability(vmt["VMT Probability"])
    vmt = vmt.dropna(subset=["Annual VMT", "VMT Probability"])

    if not np.isclose(vmt["VMT Probability"].sum(), 1.0, atol=0.0001):
        raise ValueError(f"VMT probabilities must sum to 100%. Current sum: {vmt['VMT Probability'].sum():.2%}")

    incidence_wide = incidence_wide.copy()
    incidence_wide.columns = [str(c).strip() for c in incidence_wide.columns]

    units_wide = units_wide.copy()
    units_wide.columns = [str(c).strip() for c in units_wide.columns]

    if PRODUCT_COL not in incidence_wide.columns:
        raise ValueError(f"Incidence table missing column: {PRODUCT_COL}")
    if PRODUCT_COL not in units_wide.columns:
        raise ValueError(f"Units table missing column: {PRODUCT_COL}")

    incidence_headers = [c for c in incidence_wide.columns if c != PRODUCT_COL]
    bands = build_mileage_bands_from_headers(incidence_headers)

    inc_long = normalize_rate_table(incidence_wide, "Incidence Rate")
    units_long = normalize_rate_table(units_wide, "Units Per Repair")

    ages = pd.DataFrame({"Vehicle Age": sorted(uio["Vehicle Age"].unique())})
    ages["_key"] = 1
    vmt["_key"] = 1
    age_vmt = ages.merge(vmt, on="_key").drop(columns="_key")

    age_vmt["Cumulative Miles"] = np.where(
        age_vmt["Vehicle Age"] == 0,
        age_vmt["Annual VMT"] * age0_factor,
        (age_vmt["Vehicle Age"] + age0_factor) * age_vmt["Annual VMT"],
    )
    max_band_miles = bands["Max Miles"].max()
    max_band_label = bands.loc[bands["Max Miles"].idxmax(), "Mileage Band"]
    age_vmt["Above Max Band Flag"] = age_vmt["Cumulative Miles"] > max_band_miles

    age_vmt["Mileage Band"] = assign_mileage_band(
        age_vmt["Cumulative Miles"],
        bands,
        cap_to_max_band=cap_to_max_band
    )

    products = pd.DataFrame({PRODUCT_COL: sorted(incidence_wide[PRODUCT_COL].astype(str).str.strip().unique())})
    age_vmt["_key"] = 1
    products["_key"] = 1
    bridge = age_vmt.merge(products, on="_key").drop(columns="_key")

    bridge = bridge.merge(inc_long, on=[PRODUCT_COL, "Mileage Band"], how="left")
    bridge = bridge.merge(units_long, on=[PRODUCT_COL, "Mileage Band"], how="left")

    missing_inc = int(bridge["Incidence Rate"].isna().sum())
    missing_units = int(bridge["Units Per Repair"].isna().sum())
    if missing_inc or missing_units:
        raise ValueError(
            f"Missing rate data. Missing incidence rows: {missing_inc:,}. "
            f"Missing units rows: {missing_units:,}. Check product/mileage-band headers."
        )

    bridge["Weighted Incident Rate Component"] = bridge["VMT Probability"] * bridge["Incidence Rate"]
    bridge["Weighted Piece Rate Component"] = bridge["VMT Probability"] * bridge["Incidence Rate"] * bridge["Units Per Repair"]

    age_product_rates = (
        bridge.groupby(["Vehicle Age", PRODUCT_COL], as_index=False)
        .agg(
            Weighted_Incident_Rate=("Weighted Incident Rate Component", "sum"),
            Weighted_Piece_Rate=("Weighted Piece Rate Component", "sum"),
        )
    )

    uio_year_age = (
        uio.groupby(["Forecast Year", "Vehicle Age"], as_index=False)
        .agg(UIO=("UIO", "sum"))
    )

    year_product = uio_year_age.merge(age_product_rates, on="Vehicle Age", how="left")
    year_product["Forecast Incidents"] = year_product["UIO"] * year_product["Weighted_Incident_Rate"]
    year_product["Forecast Piece Demand"] = year_product["UIO"] * year_product["Weighted_Piece_Rate"]

    demand_by_year_product = (
        year_product.groupby(["Forecast Year", PRODUCT_COL], as_index=False)
        .agg(
            Forecast_Incidents=("Forecast Incidents", "sum"),
            Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
        )
    )

    uio_location_year_age = (
        uio.groupby(["CBSA", "Forecast Year", "Vehicle Age"], as_index=False)
        .agg(UIO=("UIO", "sum"))
    )

    location_year_age_product = uio_location_year_age.merge(age_product_rates, on="Vehicle Age", how="left")
    location_year_age_product["Forecast Incidents"] = (
        location_year_age_product["UIO"] * location_year_age_product["Weighted_Incident_Rate"]
    )
    location_year_age_product["Forecast Piece Demand"] = (
        location_year_age_product["UIO"] * location_year_age_product["Weighted_Piece_Rate"]
    )

    demand_by_location = (
        location_year_age_product.groupby(["CBSA", "Forecast Year"], as_index=False)
        .agg(
            Forecast_Incidents=("Forecast Incidents", "sum"),
            Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
        )
    )

    demand_by_location_product = None
    if make_location_product_output:
        demand_by_location_product = (
            location_year_age_product.groupby(["CBSA", "Forecast Year", PRODUCT_COL], as_index=False)
            .agg(
                Forecast_Incidents=("Forecast Incidents", "sum"),
                Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
            )
        )

    demand_by_location_age = None
    if make_location_age_output:
        demand_by_location_age = (
            location_year_age_product.groupby(["CBSA", "Forecast Year", "Vehicle Age"], as_index=False)
            .agg(
                Forecast_Incidents=("Forecast Incidents", "sum"),
                Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
            )
        )

    uio_age = (
        uio.groupby(["Vehicle Age"], as_index=False)
        .agg(UIO=("UIO", "sum"))
    )
    age_product_output = uio_age.merge(age_product_rates, on="Vehicle Age", how="left")
    age_product_output["Forecast Incidents"] = age_product_output["UIO"] * age_product_output["Weighted_Incident_Rate"]
    age_product_output["Forecast Piece Demand"] = age_product_output["UIO"] * age_product_output["Weighted_Piece_Rate"]

    demand_by_age_product = (
        age_product_output.groupby(["Vehicle Age", PRODUCT_COL], as_index=False)
        .agg(
            Forecast_Incidents=("Forecast Incidents", "sum"),
            Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
        )
    )

    diagnostics = {
        "uio_rows": len(uio),
        "location_count": uio["CBSA"].nunique(),
        "product_count": products[PRODUCT_COL].nunique(),
        "age_product_rate_rows": len(age_product_rates),
        "location_rows": len(demand_by_location),
        "total_piece_demand": demand_by_year_product["Forecast_Piece_Demand"].sum(),
        "above_max_age_vmt_rows": int(age_vmt.get("Above Max Band Flag", pd.Series(dtype=bool)).sum()),
        "max_mileage_band": max_band_label,
        "max_mileage_band_miles": int(max_band_miles),
    }

    return {
        "diagnostics": diagnostics,
        "age_vmt_bridge": age_vmt.drop(columns=[c for c in ["_key"] if c in age_vmt.columns]),
        "age_product_rates": age_product_rates,
        "demand_by_year_product": demand_by_year_product,
        "demand_by_location": demand_by_location,
        "demand_by_location_product": demand_by_location_product,
        "demand_by_location_age": demand_by_location_age,
        "demand_by_age_product": demand_by_age_product,
    }


st.subheader("1. Upload files")

c1, c2 = st.columns(2)
with c1:
    uio_upload = st.file_uploader("UIO_Age_Summary.csv", type=["csv"])
    incidence_upload = st.file_uploader("Incidence_By_Mileage.csv", type=["csv"])
with c2:
    units_upload = st.file_uploader("Units_Per_Repair.csv", type=["csv"])
    vmt_upload = st.file_uploader("VMT_Distribution.csv optional", type=["csv"])

default_vmt = pd.DataFrame(
    {"Annual VMT": [8000, 10000, 12000, 15000], "VMT Probability": [0.15, 0.25, 0.40, 0.20]}
)

try:
    vmt_df = read_csv_cached(vmt_upload.getvalue()) if vmt_upload else default_vmt
except Exception as e:
    st.error(f"Could not read VMT file: {e}")
    st.stop()

st.subheader("2. VMT assumptions")
st.write("Edit the assumptions below. The probabilities must sum to 100%.")

edited_vmt = st.data_editor(
    vmt_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Annual VMT": st.column_config.NumberColumn("Annual VMT", min_value=0, step=500),
        "VMT Probability": st.column_config.NumberColumn("VMT Probability", min_value=0.0, step=0.01, format="%.2f"),
    },
)

edited_vmt["Annual VMT"] = clean_numeric(edited_vmt["Annual VMT"])
edited_vmt["VMT Probability"] = normalize_probability(edited_vmt["VMT Probability"])
prob_sum = edited_vmt["VMT Probability"].sum()

st.write(f"Probability sum: **{prob_sum:.2%}**")

st.sidebar.header("Settings")
age0_factor = st.sidebar.slider("Age 0 mileage factor", 0.0, 1.0, 0.5, 0.05)

cap_to_max_band = st.sidebar.checkbox(
    "Cap mileage above max band to highest band",
    value=True,
    help="Recommended. High VMT scenarios can create cumulative mileage above the highest incidence/units mileage band. This caps those rows to the highest available band instead of failing."
)

make_location_product_output = st.sidebar.checkbox(
    "Create location/product output",
    value=True,
    help="Output demand by CBSA, forecast year, and product category."
)

make_location_age_output = st.sidebar.checkbox(
    "Create location/age output",
    value=False,
    help="Output demand by CBSA, forecast year, and vehicle age."
)

if not np.isclose(prob_sum, 1.0, atol=0.0001):
    st.warning("Fix the VMT probabilities before running.")
    st.stop()

st.subheader("3. Run model")

if not all([uio_upload, incidence_upload, units_upload]):
    st.info("Upload the UIO, incidence, and units-per-repair files to run.")
    st.stop()

if st.button("Run demand forecast", type="primary"):
    try:
        with st.spinner("Running forecast..."):
            uio_df = read_csv_cached(uio_upload.getvalue())
            incidence_df = read_csv_cached(incidence_upload.getvalue())
            units_df = read_csv_cached(units_upload.getvalue())

            result = run_forecast(
                uio_df=uio_df,
                incidence_wide=incidence_df,
                units_wide=units_df,
                vmt_df=edited_vmt,
                age0_factor=age0_factor,
                make_location_product_output=make_location_product_output,
                make_location_age_output=make_location_age_output,
                cap_to_max_band=cap_to_max_band,
            )

        st.success("Forecast complete.")

        d = result["diagnostics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("UIO Rows", f"{d['uio_rows']:,}")
        m2.metric("Locations", f"{d['location_count']:,}")
        m3.metric("Product Categories", f"{d['product_count']:,}")
        m4.metric("Total Piece Demand", f"{d['total_piece_demand']:,.0f}")

        if d.get("above_max_age_vmt_rows", 0) > 0 and cap_to_max_band:
            st.warning(
                f"{d['above_max_age_vmt_rows']:,} age/VMT outcomes exceeded the highest mileage band "
                f"({d['max_mileage_band']}, max {d['max_mileage_band_miles']:,} miles). "
                "They were capped to the highest available mileage band."
            )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Year/Product Demand",
            "Location Demand",
            "Age/Product Rates",
            "Other Outputs"
        ])

        with tab1:
            st.subheader("Demand by Forecast Year and Product Category")
            st.dataframe(result["demand_by_year_product"], use_container_width=True, height=520)

        with tab2:
            st.subheader("Demand by Location")
            st.caption("Aggregated by CBSA and forecast year across all product categories.")
            st.dataframe(result["demand_by_location"], use_container_width=True, height=420)

            if result["demand_by_location_product"] is not None:
                st.subheader("Demand by Location and Product Category")
                st.caption("Aggregated by CBSA, forecast year, and product category.")
                st.dataframe(result["demand_by_location_product"], use_container_width=True, height=520)

        with tab3:
            st.subheader("Age/Product Weighted Rates")
            st.dataframe(result["age_product_rates"], use_container_width=True, height=520)

        with tab4:
            st.subheader("Demand by Age and Product")
            st.dataframe(result["demand_by_age_product"], use_container_width=True, height=420)

            if result["demand_by_location_age"] is not None:
                st.subheader("Demand by Location and Vehicle Age")
                st.dataframe(result["demand_by_location_age"], use_container_width=True, height=420)

            st.subheader("Age/VMT Bridge")
            st.dataframe(result["age_vmt_bridge"], use_container_width=True, height=320)

        st.subheader("Downloads")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.download_button(
                "Download year/product demand CSV",
                data=csv_bytes(result["demand_by_year_product"]),
                file_name="demand_by_year_product.csv",
                mime="text/csv",
            )

        with c2:
            st.download_button(
                "Download location demand CSV",
                data=csv_bytes(result["demand_by_location"]),
                file_name="demand_by_location.csv",
                mime="text/csv",
            )

        with c3:
            if result["demand_by_location_product"] is not None:
                st.download_button(
                    "Download location/product demand CSV",
                    data=csv_bytes(result["demand_by_location_product"]),
                    file_name="demand_by_location_product.csv",
                    mime="text/csv",
                )

        sheets = {
            "Demand_Year_Product": result["demand_by_year_product"],
            "Demand_Location": result["demand_by_location"],
            "Demand_Age_Product": result["demand_by_age_product"],
            "Age_Product_Rates": result["age_product_rates"],
            "Age_VMT_Bridge": result["age_vmt_bridge"],
        }

        if result["demand_by_location_product"] is not None:
            sheets["Demand_Location_Product"] = result["demand_by_location_product"]

        if result["demand_by_location_age"] is not None:
            sheets["Demand_Location_Age"] = result["demand_by_location_age"]

        st.download_button(
            "Download all Excel outputs",
            data=excel_bytes(sheets),
            file_name="demand_forecast_outputs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    st.caption("Model will not process the data until you click the run button.")
