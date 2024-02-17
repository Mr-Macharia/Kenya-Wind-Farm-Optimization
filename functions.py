import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import geopandas as gpd
from tqdm import tqdm  # For progress bars


def calculate_power_output(df, cut_in_speed, rated_speed, cut_off_speed, rated_power, time_period_hours, lon_grid, lat_grid,wind_speed_grid,  county_shapefile_path):
    # Function to calculate power output based on turbine power curve
    def industrial_turbine_power_curve(wind_speed, cut_in_speed, rated_speed, cut_off_speed, rated_power, time_period_hours):
        power_output = np.zeros_like(wind_speed)
        mask = (wind_speed >= cut_in_speed) & (wind_speed <= cut_off_speed)
        power_output[mask] = (wind_speed[mask] - cut_in_speed) / (rated_speed - cut_in_speed) * rated_power * time_period_hours / 1000  # Convert to MWh
        mask = wind_speed > cut_off_speed
        power_output[mask] = rated_power * time_period_hours / 1000  # Convert to MWh
        return power_output

    # Group by time period and calculate the mean wind speed
    if time_period_hours == 1:
        grouped_data = df.groupby(df['time'].dt.to_period("H")).agg({'wind_speed_daily_100': 'mean'})
    elif time_period_hours == 24:
        grouped_data = df.groupby(df['time'].dt.to_period("D")).agg({'wind_speed_daily_100': 'mean'})
    elif time_period_hours == 168:
        grouped_data = df.groupby(df['time'].dt.to_period("W")).agg({'wind_speed_daily_100': 'mean'})
    elif time_period_hours == 730:  # Assuming 30 days in a month
        grouped_data = df.groupby(df['time'].dt.to_period("M")).agg({'wind_speed_daily_100': 'mean'})
    elif time_period_hours == 8760:  # Assuming 365 days in a year
        grouped_data = df.groupby(df['time'].dt.to_period("Y")).agg({'wind_speed_daily_100': 'mean'})
    else:
        raise ValueError("Unsupported time period")

    # Apply turbine power curve to estimate power output for each time period
    grouped_data['estimated_power_output_MWh'] = industrial_turbine_power_curve(grouped_data['wind_speed_daily_100'], 
                                                                                   cut_in_speed, rated_speed, cut_off_speed, rated_power, time_period_hours)

    # Load Kenyan county boundaries from the shapefile
    counties_gdf = gpd.read_file(county_shapefile_path)

    # Create a GeoDataFrame from the wind speed data
    geometry = [Point(lon, lat) for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten())]
    wind_speed_gdf = gpd.GeoDataFrame({'geometry': geometry, 'wind_speed': wind_speed_grid.flatten()})

    # Add county information to the wind speed GeoDataFrame
    for index, county in counties_gdf.iterrows():
        mask = wind_speed_gdf.within(county['geometry'])
        wind_speed_gdf.loc[mask, 'county'] = county['COUNTY']

    # Create a list to store results
    result_data = []

    # Calculate power output and aggregate for each county on the specified time period
    for county_name, county_data in tqdm(wind_speed_gdf.groupby('county')):
        county_data['power_output'] = industrial_turbine_power_curve(county_data['wind_speed'], cut_in_speed, rated_speed, cut_off_speed, rated_power, time_period_hours)
        total_power_output = county_data['power_output'].sum()
        average_power_output = total_power_output / len(county_data)
        average_wind_speed = county_data['wind_speed'].mean()  # Calculate average wind speed

        # Append results to the list
        result_data.append({
            'county': county_name,
            'average_wind_speed': average_wind_speed,
            f'total_power_output_{time_period_hours}h_MWh': total_power_output,
            f'average_power_output_{time_period_hours}h_MWh': average_power_output,
            # 'average_wind_speed': average_wind_speed
        })

    # Create a DataFrame from the list of results
    result_df = pd.DataFrame(result_data)

    # Replace 'nan' with 'Offshore' in the 'county' column
    result_df.loc[result_df['county'] == 'nan', 'county'] = 'Offshore'

    return result_df



def create_model_and_combine_df(df, county_name, start_date='2023-12-01', prediction_duration=365, lag_steps=3, test_size=0.2):
    # Create lag features for the specific county
    df_county = df[[county_name]].copy()

    # Create lag features for the entire DataFrame
    for i in range(1, lag_steps + 1):
        df_county[f'lag_{i}'] = df_county[county_name].shift(i)

    # Drop rows with NaN values introduced by lag features
    df_county = df_county.dropna()

    # Split the data into training and testing sets
    train_size = int(len(df_county) * (1 - test_size))
    train_data, test_data = df_county.iloc[:train_size], df_county.iloc[train_size:]

    # Define target variable and features
    y_train = train_data[county_name]
    X_train = train_data.drop(columns=[county_name])
    y_test = test_data[county_name]
    X_test = test_data.drop(columns=[county_name])

    # Initialize XGBoost model
    model = XGBRegressor()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions for the next year
    future_dates = pd.date_range(start=start_date, periods=prediction_duration, freq='D')
    all_predictions = model.predict(df_county.drop(columns=[county_name]))

    # Create a DataFrame with dates and predictions
    predictions_df = pd.DataFrame({'time': future_dates, 'Predicted_Wind_Speed': all_predictions[:prediction_duration]})

    # Set 'time' as the index for predictions_df
    predictions_df.set_index('time', inplace=True)

    # Combine the DataFrames
    combined_df = pd.concat([df_county, predictions_df], axis=0)

    # Return predictions, model, and combined DataFrame
    return all_predictions, model, combined_df



# Update the plot_combined_data function
def plot_combined_data(combined_df, county_name):
    # Plotting the combined data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(combined_df.index, combined_df[county_name], label='Historical Wind Speed', color='blue')
    ax.plot(combined_df.index, combined_df['Predicted_Wind_Speed'], label='Predicted Wind Speed', color='red')

    # Setting labels and title
    ax.set_title(f'Combined Historical and Predicted Wind Speed for {county_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Wind Speed')
    ax.legend()
    ax.grid(True)

    return fig, ax



def get_turbine_parameters(turbine_type):
    if turbine_type == 'Siemens SWT 3.6-120T Model':
        return {
            # 'wind_speed_range': np.arange(0, 40, 0.1),
            'cut_in_speed': 3.0,
            'rated_speed': 12.0,
            'cut_off_speed': 25.0,
            'rated_power': 3600.0
        }
    elif turbine_type == 'Vesta V52 850k Model':
        return {
            # 'wind_speed_range': np.arange(0, 40, 0.1),
            'cut_in_speed': 4.0,
            'rated_speed': 14.0,
            'cut_off_speed': 25.0,
            'rated_power': 850.0
        }
    else:
        raise ValueError(f"Invalid turbine type: {turbine_type}")
