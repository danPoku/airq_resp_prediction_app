import os
import streamlit as st
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.pyfunc import PyFuncModel
import altair as alt

from feature_engineering import climate_clean_transform

# Constants
POLLUTANT_COLS = ["co","no","no2","o3","so2","pm2_5","pm10","nh3"]
RESP_DISEASE_COLS = [
    "Acute Bronchitis (J20)",
    "Asthma (J45)",
    "Bronchiolitis (J21)",
    "Chronic Obstructive Pulmonary Disease (J44)",
    "Influenza (J09-J11)",
    "Pneumonia (J12-J18)",
    "Upper Respiratory Tract Infection (J00-J06)",
]
AQ_MODEL_NAME, AQ_MODEL_VERSION = "AirQBoost", "0.1.1"
RESP_MODEL_NAME, RESP_MODEL_VERSION = "PulmoPulse", "0.1.1"
MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/danpoku/canmlpipeline.mlflow"
)
st.write(secrets)
# --- Setup Functions ---
def setup_tracking():
    mlflow.set_tracking_uri(MLFLOW_URI)

@st.cache_resource
def load_model(model_uri: str) -> PyFuncModel:
    return mlflow.pyfunc.load_model(model_uri)


def validate_schema(df: pd.DataFrame, model: PyFuncModel):
    sig = model.metadata.signature
    expected = [inp.name for inp in sig.inputs]
    missing = set(expected) - set(df.columns)
    extra = set(df.columns) - set(expected)
    return expected, missing, extra


def paginate_df(df: pd.DataFrame, rows_key: str, page_key: str) -> pd.DataFrame:
    rows = st.number_input("Rows per page", min_value=5, max_value=50, value=10, key=rows_key)
    total = (len(df) + rows - 1) // rows
    page = st.number_input("Page", min_value=1, max_value=total, value=1, key=page_key)
    start, end = (page - 1) * rows, (page - 1) * rows + rows
    return df.iloc[start:end]

# --- Sidebar Input ---
def get_climate_data():
    st.sidebar.header("Climate Data Source")
    source = st.sidebar.radio(
        "Choose data input method:",
        ["Upload CSV", "Fetch from API"],
        index=0,
        key="climate_data_source"
    )
    df = None
    if source == "Fetch from API":
        url = st.sidebar.text_input("Enter CSV GET URL:", key="api_url_input")
        if st.sidebar.button("Fetch Data", key="api_fetch_btn"):
            try:
                df = pd.read_csv(url)
                st.sidebar.success("Data fetched successfully.")
            except Exception as e:
                st.sidebar.error(f"Error fetching data: {e}")
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload climate CSV", type=["csv"], key="csv_uploader"
        )
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.sidebar.success("File uploaded successfully.")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
    return df

# --- Display Sections ---
def show_climate_section(df: pd.DataFrame):
    st.subheader("Climate Data")
    # df['date'] = pd.to_datetime(df['date'])
    return paginate_df(df, "climate_rows", "climate_pages"), df


def show_aq_section(climate_df: pd.DataFrame, aq_model: PyFuncModel):
    st.subheader("Air Quality Prediction")
    exp, miss, extra = validate_schema(climate_df, aq_model)
    if miss:
        st.error(f"Missing for AQ: {miss}")
        st.stop()
    if extra:
        st.warning("Less important climate features ignored.")
    df_input = climate_df[exp].astype(float)
    preds = aq_model.predict(df_input)
    df_out = pd.DataFrame(preds, columns=POLLUTANT_COLS)
    df_out.insert(0, 'date', climate_df['date'].values)
    st.subheader("AQ Predictions")
    st.dataframe(paginate_df(df_out, "aq_rows", "aq_pages"))
    return df_out


def show_resp_section(climate_df: pd.DataFrame, df_preds_aq: pd.DataFrame, resp_model: PyFuncModel):
    st.subheader("Respiratory Disease Predictions")
    df_in = pd.concat([
        climate_df.reset_index(drop=True),
        df_preds_aq.drop(columns=['date']).reset_index(drop=True)
    ], axis=1)
    exp, miss, extra = validate_schema(df_in, resp_model)
    if miss:
        st.error(f"Missing for RESP: {miss}")
        st.stop()
    if extra:
        st.warning("Less important features ignored for RESP.")
    df_input = df_in[exp].astype(float)
    preds = resp_model.predict(df_input)
    df_out = pd.DataFrame(preds, columns=RESP_DISEASE_COLS)
    df_out = df_out.round().astype(int)
    df_out.insert(0, 'date', climate_df['date'].values)
    st.subheader("RESP Predictions (Paginated)")
    st.dataframe(paginate_df(df_out, "resp_rows", "resp_pages"))
    return df_out


def plot_time_series(df: pd.DataFrame, id_var: str, value_vars: list, title: str):
    st.subheader(title)
    df_melt = df.melt(
        id_vars=id_var,
        value_vars=value_vars,
        var_name='Category',
        value_name='Value'
    )
    selected = st.multiselect(
        "Select to display", options=value_vars, default=value_vars, key=f"{title}_select"
    )
    filtered = df_melt[df_melt['Category'].isin(selected)]
    scale = st.radio(
        "Y-axis scale", ["linear", "log"], index=0, key=f"{title}_scale"
    )
    legend = alt.selection_multi(fields=['Category'], bind='legend')
    chart = alt.Chart(filtered).mark_line().encode(
        x=alt.X(f'{id_var}:T', axis=alt.Axis(format='%b %d', labelAngle=-45)),
        y=alt.Y('Value:Q', scale=alt.Scale(type=scale)),
        color=alt.Color('Category:N'),
        opacity=alt.condition(legend, alt.value(1), alt.value(0.2))
    ).add_selection(legend).properties(width=800, height=400)
    st.altair_chart(chart, use_container_width=True)

# --- Main ---

def main():
    st.title("Accra Air Quality and Respiratory Forecasting")
    setup_tracking()
    # Show models in sidebar
    st.sidebar.subheader("Models")
    st.sidebar.write(f"**{AQ_MODEL_NAME}** v{AQ_MODEL_VERSION}")
    st.sidebar.write(f"**{RESP_MODEL_NAME}** v{RESP_MODEL_VERSION}")
    # Get climate data
    df_raw = get_climate_data()
    if df_raw is None:
        st.info("Please upload or fetch climate data to begin.")
        return
    # Display and retrieve full df
    raw_page, df_full = show_climate_section(df_raw)

    # Prepare data
    climate_df = climate_clean_transform(df_full.copy())

    # Load models
    aq_model = load_model("runs:/e81a7b1389ab485d8b4de63607008f3d/model_artifact")
    resp_model = load_model("runs:/9b84e0378ccf42379b208c11b8116b6e/model_artifact")

    # Air Quality
    df_preds_aq = show_aq_section(climate_df, aq_model)
    plot_time_series(df_preds_aq, 'date', POLLUTANT_COLS, "Air Quality Forecast Time Series")

    # Respiratory
    df_preds_resp = show_resp_section(climate_df, df_preds_aq, resp_model)
    plot_time_series(df_preds_resp, 'date', RESP_DISEASE_COLS, "Respiratory Forecast Time Series")

if __name__ == '__main__':
    main()