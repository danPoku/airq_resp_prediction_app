import os
from typing import Tuple, List, Set
from datetime import date
from datetime import timedelta
import logging
import streamlit as st
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.pyfunc import PyFuncModel
import altair as alt


from feature_engineering import climate_clean_transform

st.set_page_config(page_title='PulmoPulse', page_icon='🫁')

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
    "MLFLOW_TRACKING_URI"
)

# Setup
def setup_tracking():
    """Set up MLflow tracking URI.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)

@st.cache_resource
def load_model(model_uri: str) -> PyFuncModel:
    """Load a model from the specified URI.
    This function uses the MLflow library to load a model from a given URI.

    Args:
        model_uri (str): URI of the model to load.

    Returns:
        PyFuncModel: Loaded model instance.
    """
    return mlflow.pyfunc.load_model(model_uri)


def validate_schema(df: pd.DataFrame, model: PyFuncModel) -> Tuple[List[str], Set[str], Set[str]]:
    """Validate the input dataframe against the model's expected schema.
    This function checks for missing and extra columns in the dataframe compared to the model's input signature.

    Args:
        df (pd.DataFrame): Input dataframe to validate.
        model (PyFuncModel): The model to validate against.

    Returns:
        Tuple[List[str], Set[str], Set[str]]: A tuple containing the expected columns, missing columns, and extra columns.
    """
    sig = model.metadata.signature
    expected = [inp.name for inp in sig.inputs]
    missing = set(expected) - set(df.columns)
    extra = set(df.columns) - set(expected)
    return expected, missing, extra


def paginate_df(df: pd.DataFrame, rows_key: str, page_key: str) -> pd.DataFrame:
    """Paginate a dataframe for display in Streamlit.
    This function allows the user to select the number of rows per page and the page number.
    Args:
        df (pd.DataFrame): Dataframe to paginate
        rows_key (str): Number of rows to display per page
        page_key (str): Key for the page number input

    Returns:
        pd.DataFrame: Paginated dataframe
    """
    rows = st.number_input("Rows per page", min_value=5, max_value=50, value=10, key=rows_key)
    total = (len(df) + rows - 1) // rows
    page = st.number_input("Page", min_value=1, max_value=total, value=1, key=page_key)
    start, end = (page - 1) * rows, (page - 1) * rows + rows
    return df.iloc[start:end]

# Sidebar
def get_climate_data() -> pd.DataFrame:
    """Get climate data from user input.
    This function allows the user to either upload a CSV file or fetch data from an API.

    Returns:
        pd.DataFrame: Dataframe containing the climate data.
    """
    st.sidebar.header("Climate Data Source")
    source = st.sidebar.radio(
        "Choose data input method:",
        ["Upload CSV", "Fetch data from API"],
        index=0,
        key="climate_data_source"
    )
    df = None
    if source == "Fetch data from API":
        url = st.sidebar.text_input("Enter GET URL:", key="api_url_input")
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

# Display
def show_climate_section(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Show climate data section.

    Args:
        df (pd.DataFrame): Dataframe containing climate data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the paginated dataframe and the full dataframe.
    """
    st.subheader("Climate Data")
    # df['date'] = pd.to_datetime(df['date'])
    return paginate_df(df, "climate_rows", "climate_pages"), df


def show_aq_section(climate_df: pd.DataFrame, aq_model: PyFuncModel) -> pd.DataFrame:
    """Show air quality predictions based on climate data.
    This function validates the input data against the model's expected schema and
    generates predictions for air quality pollutants.

    Args:
        climate_df (pd.DataFrame): climate dataframe
        aq_model (PyFuncModel): air quality prediction model

    Returns:
        pd.DataFrame: dataframe containing air quality predictions
    """
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
    st.dataframe(paginate_df(df_out, "aq_rows", "aq_pages"))
    return df_out


def show_resp_section(climate_df: pd.DataFrame, df_preds_aq: pd.DataFrame, resp_model: PyFuncModel) -> pd.DataFrame:
    """Show respiratory disease predictions based on climate data and air quality predictions.

    Args:
        climate_df (pd.DataFrame): climate dataframe
        df_preds_aq (pd.DataFrame): dataframe containing air quality predictions
        resp_model (PyFuncModel): respiratory disease prediction model

    Returns:
        pd.DataFrame: dataframe containing respiratory disease predictions
    """
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
    st.dataframe(paginate_df(df_out, "resp_rows", "resp_pages"))
    return df_out


def plot_time_series(df: pd.DataFrame, id_var: str, value_vars: list, title: str):
    """Plot time series data using Altair.
    Args:
        df (pd.DataFrame): _description_
        id_var (str): _description_
        value_vars (list): _description_
        title (str): _description_
    """
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
    legend = alt.selection_point(fields=['Category'], bind='legend')
    chart = alt.Chart(filtered).mark_line().encode(
        x=alt.X(f'{id_var}:T', axis=alt.Axis(format='%b %d', labelAngle=-45)),
        y=alt.Y('Value:Q', scale=alt.Scale(type=scale)),
        color=alt.Color('Category:N'),
        opacity=alt.condition(legend, alt.value(1), alt.value(0.2))
    ).add_params(legend).properties(width=800, height=400)
    st.altair_chart(chart, use_container_width=True)

# Main

# def main():
#     st.title("Accra Air Quality and Respiratory Disease Forecasting")
#     setup_tracking()
#     # Show models in sidebar
#     st.sidebar.subheader("Models")
#     st.sidebar.write(f"**{AQ_MODEL_NAME}** v{AQ_MODEL_VERSION}")
#     st.sidebar.write(f"**{RESP_MODEL_NAME}** v{RESP_MODEL_VERSION}")
#     # Retrieve climate data
#     df_raw = get_climate_data()
#     if df_raw is None:
#         st.info("Please upload or fetch climate data to begin.")
#         st.info("Check the sidebar to upload or fetch climate data")
#         return
#     # Retrieve climate data
#     raw_page, df_full = show_climate_section(df_raw)

#     # Prepare data
#     climate_df = climate_clean_transform(df_full.copy())

#     # Load models
#     aq_model = load_model("runs:/e81a7b1389ab485d8b4de63607008f3d/model_artifact")
#     resp_model = load_model("runs:/9b84e0378ccf42379b208c11b8116b6e/model_artifact")

#     # Air Quality
#     df_preds_aq = show_aq_section(climate_df, aq_model)
#     plot_time_series(df_preds_aq, 'date', POLLUTANT_COLS, "Air Quality Forecast Time Series")

#     # Respiratory Disease
#     df_preds_resp = show_resp_section(climate_df, df_preds_aq, resp_model)
#     plot_time_series(df_preds_resp, 'date', RESP_DISEASE_COLS, "Respiratory Disease Forecast Time Series")

def get_today_metrics(df: pd.DataFrame) -> pd.Series:
    """Get today's metrics from the dataframe.
    This function filters the dataframe to get the metrics for today.

    Args:
        df (pd.DataFrame): DataFrame containing date and metrics.

    Returns:
        pd.Series: Series containing today's metrics or the last available metrics if today is not present.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    today = pd.Timestamp.now().normalize()
    today_df = df[df['date'] == today]
    if not today_df.empty:
        return today_df.iloc[0]
    return df.iloc[-1]

def compute_deltas_next_day(df: pd.DataFrame) -> pd.Series:
    """
    For each pollutant in POLLUTANT_COLS, compute the % change from 'today'
    to 'today + 1 day' as present in df.  
    If tomorrow’s row isn't in df, or there's no prior row, returns "N/A".
    """
    # 1) Copy, normalize and sort by date
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date']).dt.normalize()
    df2 = df2.sort_values('date').reset_index(drop=True)

    # 2) Define today and tomorrow
    today_ts  = pd.Timestamp(date.today())
    next_ts   = today_ts + pd.Timedelta(days=1)

    # 3) Check that tomorrow exists in your data
    if next_ts not in set(df2['date']):
        return pd.Series({col: "N/A" for col in POLLUTANT_COLS})

    # 4) Locate positions
    tomorrow_idx = df2.index[df2['date'] == next_ts][0]
    prev_idx     = tomorrow_idx - 1
    if prev_idx < 0:
        return pd.Series({col: "N/A" for col in POLLUTANT_COLS})

    # 5) Compute deltas
    deltas = {}
    for col in POLLUTANT_COLS:
        prev = df2.at[prev_idx, col]
        curr = df2.at[tomorrow_idx, col]
        # Guard against zero‐division
        if pd.notna(prev) and prev != 0:
            pct = (curr - prev) / prev * 100
            sign = "+" if pct >= 0 else ""
            deltas[col] = f"{sign}{pct:.1f}%"
        else:
            deltas[col] = "N/A"

    # 6) Return a Series so you can do things like deltas["pm2_5"]
    return pd.Series(deltas)


# MAIN Function
def main():
    st.title("Accra Air Quality and Respiratory Disease Forecasting")
    setup_tracking()

    tabs = st.tabs(["📊 Climate Data", "🌫️ Air Quality Forecast", "🫁 Respiratory Forecast"])
    climate_tab, aq_tab, resp_tab = tabs

    # climate tab
    with climate_tab:
        df_raw = get_climate_data()
        if df_raw is None:
            st.info("Please upload or fetch climate data to begin.")
            return
        raw_page, df_full = show_climate_section(df_raw)

    # clean & load models once
    climate_df = climate_clean_transform(df_full.copy())
    aq_model = load_model("runs:/e81a7b1389ab485d8b4de63607008f3d/model_artifact")
    resp_model = load_model("runs:/9b84e0378ccf42379b208c11b8116b6e/model_artifact")

    # AQ tab
    with aq_tab:
        df_preds_aq = show_aq_section(climate_df, aq_model)
        deltas = compute_deltas_next_day(df_preds_aq)
        # scorecard metrics
        metrics = (
            df_preds_aq
            .assign(date=pd.to_datetime(df_preds_aq['date']).dt.normalize())
            .set_index('date')
            .reindex([pd.Timestamp(date.today() + timedelta(days=1))], method='ffill')
            .iloc[0]
        )
        # Import logger to log metrics
        logger = logging.getLogger(__name__)
        # Log deltas
        for col in POLLUTANT_COLS:
            logger.info(f"Delta for {col}: {deltas[col]}")
        
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PM₂.₅", f"{metrics['pm2_5']:.1f}", deltas['pm2_5'])
        col2.metric("PM₁₀", f"{metrics['pm10']:.1f}", deltas['pm10'])
        col3.metric("O₃",  f"{metrics['o3']:.1f}", deltas['o3'])
        col4.metric("CO",   f"{metrics['co']:.1f}", deltas['co'])
        # Compute tomorrow's timestamp
        tomorrow_ts = pd.Timestamp(date.today()) + pd.Timedelta(days=1)
        # Display tomorrow's date
        st.subheader(f"Forecast for {tomorrow_ts.strftime('%B %d, %Y')}")
        # plot
        plot_time_series(df_preds_aq, 'date', POLLUTANT_COLS, 
                         "Air Quality Forecast Time Series")

    # Resp tab
    with resp_tab:
        df_preds_resp = show_resp_section(climate_df, df_preds_aq, resp_model)
        plot_time_series(df_preds_resp, 'date', RESP_DISEASE_COLS, 
                         "Respiratory Disease Forecast Time Series")


if __name__ == '__main__':
    main()