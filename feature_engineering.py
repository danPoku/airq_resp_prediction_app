import pandas as pd
import numpy as np


def climate_clean_transform(dataframe):
    """
    Cleans and transforms a climate-related dataset by performing various preprocessing 
    and feature engineering steps.
    Parameters:
    ----------
    dataframe : pandas.DataFrame
        Expected columns include:
        'datetime', 'sunrise', 'sunset', 'severerisk', 'preciptype', 'visibility', 
        'solarenergy', 'uvindex', 'snow', 'snowdepth', 'tempmax', 'tempmin', 'temp', 
        'dew', 'humidity', 'precip', 'windgust', 'windspeed', 'winddir', 'sealevelpressure', 
        'cloudcover', among others.
    Returns:
    -------
    pandas.DataFrame
        The transformed DataFrame with cleaned data and additional derived features.
    Processing Steps:
    -----------------
    1. Renames the 'datetime' column to 'date'.
    2. Converts 'date', 'sunrise', and 'sunset' columns to datetime format.
    3. Adds a new column 'month' extracted from the 'date' column.
    4. Fills missing values in 'severerisk' with 0 and in 'preciptype' with 'none'.
    5. Interpolates missing values for 'visibility', 'solarenergy', and 'uvindex'.
    6. Drops irrelevant columns: 'snow' and 'snowdepth'.
    7. Removes unnamed columns using a helper function `remove_unnamed_col`.
    8. Creates derived features such as:
        - Diurnal temperature range.
        - Temperature anomaly (deviation from mean temperature).
        - Rate of temperature change (daily difference).
        - Dew point depression.
        - Humidity-Temperature Index (HTI).
        - Cumulative precipitation (7-day rolling sum).
        - Precipitation event count (7-day rolling count of non-zero precipitation days).
        - Precipitation intensity (mean precipitation per precipitating day over 7 days).
        - Wind variability (difference between wind gust and average wind speed).
        - Rolling standard deviation of wind speed (7-day window).
        - Simple stagnation index (inverse of wind speed).
        - Wind rose analysis (decomposition into u and v components).
        - 7-day rolling means of u and v components.
        - Mean wind speed and direction over 7 days.
        - Pressure tendency (daily difference in sea level pressure).
        - Cloud-insolation ratio.
        - Daylight duration (difference between sunset and sunrise in hours).
        - Solar energy to UV index interaction.
        - 1-day lag of temperature.
        - 7-day rolling average and standard deviation of temperature.
        - Temperature-humidity interaction.
        - Wind-precipitation interaction.
        - Solar-cloud interaction.
    9. Drops rows with any remaining missing values.
    Notes:
    ------
    - Ensure the input DataFrame contains all required columns before calling this function.
    - The function assumes the presence of numeric and datetime columns for feature engineering.
    - Rolling window operations may introduce NaN values at the beginning of the dataset.
    """
    
    # Rename datetime to date
    dataframe.rename(columns={"datetime": "date"}, inplace=True)

    # Reformat sunrise and sunset date columns
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe["sunrise"] = pd.to_datetime(dataframe["sunrise"])
    dataframe["sunset"] = pd.to_datetime(dataframe["sunset"])

    # Add a new column 'month'
    dataframe["month"] = dataframe["date"].dt.month

    # Replace missing values for severerisk with 0
    dataframe["severerisk"] = dataframe["severerisk"].fillna(0)

    # Replace missing values for preciptype with 'none'
    dataframe["preciptype"] = dataframe["preciptype"].fillna("none")

    # Interpolate for missing values for the following features:
    # visibility, solarradiation, solarenergy and uvindex
    dataframe["visibility"] = dataframe["visibility"].interpolate(method="linear")
    dataframe["solarenergy"] = dataframe["solarenergy"].interpolate(method="linear")
    dataframe["uvindex"] = dataframe["uvindex"].interpolate(method="linear")

    # Drop irrelevant features
    dataframe.drop(["snow", "snowdepth"], axis=1, inplace=True)
    dataframe = remove_unnamed_col(dataframe)

    # DERIVED FEATURES
    # Duirnal Temperature Range
    dataframe["diurnal_temp_range"] = dataframe["tempmax"] - dataframe["tempmin"]

    # Temperature Anomaly: Deviation from the overall mean temperature
    dataframe["temperature_anomaly"] = dataframe["temp"] - dataframe["temp"].mean()

    # Rate of Temperature Change (daily difference from previous day)
    dataframe["temp_rate_of_change"] = dataframe["temp"].diff()

    # Dew Point Depression: Difference between actual temperature and dew point
    dataframe["dew_point_depression"] = dataframe["temp"] - dataframe["dew"]

    # Humidity-Temperature Index (simple composite)
    dataframe["HTI"] = dataframe["temp"] * (1 + (dataframe["humidity"] / 100))

    # Cumulative Precipitation: Rolling 7-day sum
    dataframe["cumulative_precip_7d"] = dataframe["precip"].rolling(window=7).sum()

    # Precipitation Event Count: Rolling 7-day count of days with non-zero precip
    dataframe["precip_event_count_7d"] = (
        dataframe["precip"].rolling(window=7).apply(lambda x: np.sum(x > 0), raw=True)
    )

    # Precipitation Intensity: Mean precip per precipitating day over last 7 days
    def precip_intensity(series):
        events = series[series > 0]
        return events.mean() if len(events) > 0 else 0

    dataframe["precip_intensity_7d"] = (
        dataframe["precip"].rolling(window=7).apply(precip_intensity, raw=True)
    )

    # Wind Variability (Difference between wind gust and average wind speed)
    dataframe["wind_diff"] = dataframe["windgust"] - dataframe["windspeed"]

    # Rolling Standard Deviation of Wind Speed (7-day window as an example)
    dataframe["wind_speed_std_7d"] = dataframe["windspeed"].rolling(window=7).std()

    # Simple Stagnation Index (inverse of windspeed + small epsilon to avoid division by zero)
    epsilon = 0.1
    dataframe["stagnation_index"] = 1 / (dataframe["windspeed"] + epsilon)

    # Wind Rose Analysis: Decompose wind speed into u, v components
    dataframe["u"] = dataframe["windspeed"] * np.cos(np.deg2rad(dataframe["winddir"]))
    dataframe["v"] = dataframe["windspeed"] * np.sin(np.deg2rad(dataframe["winddir"]))

    # 7-day rolling means of u and v
    dataframe["u_mean_7d"] = dataframe["u"].rolling(window=7).mean()
    dataframe["v_mean_7d"] = dataframe["v"].rolling(window=7).mean()

    # Mean wind speed and direction (over 7 days)
    dataframe["wind_mean_speed_7d"] = np.sqrt(
        dataframe["u_mean_7d"] ** 2 + dataframe["v_mean_7d"] ** 2
    )
    dataframe["wind_mean_dir_7d"] = np.rad2deg(
        np.arctan2(dataframe["v_mean_7d"], dataframe["u_mean_7d"])
    )

    # Pressure Tendency: Daily difference
    dataframe["pressure_tendency"] = dataframe["sealevelpressure"].diff()

    # Cloud-Insolation Ratio: solarenergy * (1 - cloudcover/100)
    dataframe["cloud_insolation_ratio"] = dataframe["solarenergy"] * (
        1 - (dataframe["cloudcover"] / 100)
    )

    # Daylight Duration: difference between sunset and sunrise in hours
    dataframe["sunrise"] = pd.to_datetime(dataframe["sunrise"], errors="coerce")
    dataframe["sunset"] = pd.to_datetime(dataframe["sunset"], errors="coerce")
    dataframe["daylight_duration"] = (
        dataframe["sunset"] - dataframe["sunrise"]
    ).dt.total_seconds() / 3600.0

    # Solar Energy to UV Index Interaction
    dataframe["solar_uv_interaction"] = dataframe["solarenergy"] * dataframe["uvindex"]

    # 1-day lag of temperature
    dataframe["temp_lag1"] = dataframe["temp"].shift(1)

    # 7-day rolling average & std of temperature
    dataframe["temp_rolling_mean_7d"] = dataframe["temp"].rolling(window=7).mean()
    dataframe["temp_rolling_std_7d"] = dataframe["temp"].rolling(window=7).std()

    # Temperature-Humidity Interaction
    dataframe["temp_humidity_interaction"] = dataframe["temp"] * (
        dataframe["humidity"] / 100
    )

    # Wind-Precipitation Interaction
    dataframe["wind_precip_interaction"] = dataframe["windspeed"] * dataframe["precip"]

    # Solar-Cloud Interaction
    dataframe["solar_cloud_interaction"] = dataframe["solarradiation"] * (
        1 - (dataframe["cloudcover"] / 100)
    )

    # Drop rows with missing dataframe
    # dataframe = dataframe.dropna()
    
    # Identify numeric coloumns only
    numeric_cols_data = dataframe.select_dtypes(include=['number', 'int64', 'float64']).columns
    dataframe[numeric_cols_data] = dataframe[numeric_cols_data].astype('float64')
    return dataframe


def air_clean_transform(dataframe):
    """
    Cleans and transforms an air quality dataset.
    This function performs the following operations on the input dataframe:
    1. Renames the "dt" column to "date".
    2. Converts the "date" column from a Unix timestamp to a datetime object.
    3. Reformats the "date" column to display only the date part.
    4. Aggregates the hourly air quality data into daily averages.
    5. Drops the "aqi" column as it is deemed irrelevant.
    6. Removes unnamed columns from the dataframe.
    7. Identifies numeric columns in the dataframe.
    8. Interpolates missing values in numeric columns using linear interpolation.
    9. Replaces negative values in numeric columns with the minimum non-negative value of the respective column.
    Args:
        dataframe (pd.DataFrame): The input dataframe containing air quality data.
    Returns:
        pd.DataFrame: The cleaned and transformed dataframe.
    """
    
    # Rename dt to date
    dataframe.rename(columns={"dt": "date"}, inplace=True)

    # Convert unix timestamp to datetime
    dataframe["date"] = pd.to_datetime(dataframe["date"], unit="s")

    # Air quality raw dataframe is recorded per hour.
    # Aggregated dataframe into daily averages
    dataframe = dataframe.groupby("date").mean().reset_index()

    # Drop irrelevant feature
    dataframe.drop("aqi", axis=1, inplace=True)
    dataframe = remove_unnamed_col(dataframe)

    # Identify numeric coloumns only
    numeric_cols_data = dataframe.select_dtypes(include=["number"])

    # Interpolate for missing values
    for col in numeric_cols_data.columns:
        dataframe[col] = dataframe[col].interpolate(method="linear")

    # Replace negative values in each numeric column with minimum non-negative value of the column
    for col in numeric_cols_data.columns:
        dataframe[col] = dataframe[col].apply(
            lambda x, col=col: dataframe[col][dataframe[col] > 0].min() if x < 0 else x
        )
    return dataframe


def resp_clean_transform(dataframe):
    """
    Cleans and transforms a respiratory disease dataset.
    This function processes a dataframe containing respiratory disease data by:
    1. Converting the "date" column to datetime format.
    2. Pivoting the dataframe to create a table where:
       - Rows are indexed by "date".
       - Columns represent unique values of "primary_diagnosis".
       - Values are the sum of "number_of_cases" for each combination of date and diagnosis.
    3. Filling any missing values with 0.
    4. Resetting the index of the resulting dataframe.
    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing the following columns:
            - "date" (str or datetime): The date of the record.
            - "primary_diagnosis" (str): The primary diagnosis category.
            - "number_of_cases" (int or float): The number of cases for the diagnosis.
    Returns:
        pd.DataFrame: A transformed DataFrame with:
            - A "date" column.
            - Columns for each unique "primary_diagnosis".
            - Values representing the sum of "number_of_cases" for each date and diagnosis.
    """
    
    dataframe["date"] = pd.to_datetime(dataframe["date"])

    # Transpose respiratory disease dataset
    dataframe = (
        dataframe.pivot_table(
            index="date",
            columns="primary_diagnosis",
            values="number_of_cases",
            aggfunc="sum",
        )
        .fillna(0)
        .reset_index()
    )
    return dataframe


def merge_all_datasets(df_one, df_two, df_three, on_column):
    """
    Merges three datasets into a single DataFrame based on a common column.
    This function performs two inner joins:
    1. Merges the first and second DataFrames (`df_one` and `df_two`) on the specified column.
    2. Merges the resulting DataFrame with the third DataFrame (`df_three`) on the same column.
    Parameters:
        df_one (pd.DataFrame): The first DataFrame to merge.
        df_two (pd.DataFrame): The second DataFrame to merge.
        df_three (pd.DataFrame): The third DataFrame to merge.
        on_column (str): The name of the column to merge the DataFrames on.
    Returns:
        pd.DataFrame: A DataFrame resulting from the inner joins of the three input DataFrames.
    """
    
    merged = pd.merge(
        df_one, df_two, on=on_column, how="inner"
    )  # Climate + Air Quality datasets
    merged = pd.merge(
        merged, df_three, on=on_column, how="inner"
    )  # Climate + Air Quality + Respiratory Disease datasets
    return merged


def merge_two_datasets(df_one, df_two, on_column):
    """
    Merges two pandas DataFrames on a specified column using an inner join.
    Args:
        df_one (pd.DataFrame): The first DataFrame to merge.
        df_two (pd.DataFrame): The second DataFrame to merge.
        on_column (str): The name of the column to merge the DataFrames on.
    Returns:
        pd.DataFrame: A new DataFrame resulting from the inner join of the two input DataFrames.
    """
    
    merged = pd.merge(df_one, df_two, on=on_column, how="inner")
    return merged

def filter_features_by_corr(dataframe, threshold):
    """
    Filters features in a DataFrame based on a correlation threshold.
    This function identifies and removes features that have a correlation
    higher than the specified threshold with any other feature in the DataFrame.
    It returns a list of features with low correlation.
    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the features to be filtered.
        threshold (float): The correlation threshold. Features with an absolute correlation
                           value greater than this threshold will be removed.
    Returns:
        list: A list of feature names that have a correlation below the specified threshold.
    """
    
    all_cols = set(dataframe.columns)
    col_corr = set()
    corr_matrix = dataframe.corr()
    for col in range(len(corr_matrix.columns)):
        for value in range(col):
            if abs(corr_matrix.iloc[col, value]) > threshold:
                col_name = corr_matrix.columns[col]
                col_corr.add(col_name)
    low_corr = all_cols - col_corr
    return list(low_corr)

def remove_unnamed_col(dataframe):
    """
    Removes the column named "Unnamed: 0" from the given DataFrame if it exists.
    This function checks if the column "Unnamed: 0" is present in the DataFrame's columns.
    If the column is found, it is dropped in place. This is typically used to clean up
    DataFrames where an unnecessary index column has been added during data import.
    Args:
        dataframe (pandas.DataFrame): The input DataFrame to process.
    Returns:
        pandas.DataFrame: The DataFrame with the "Unnamed: 0" column removed, if it existed.
    """

    # Extra data wrangling
    if "Unnamed: 0" in dataframe.columns:
        dataframe.drop(columns="Unnamed: 0", inplace=True)
    return dataframe
