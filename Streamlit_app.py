import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
from functions import create_model_and_combine_df, plot_combined_data, calculate_power_output 



def load_data():
    # Load your data into the df DataFrame
    df = pd.read_csv(r'C:\Users\HomePC\Documents\School\Moringa\Capstone\county_wind_data_daily.csv')
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

def main():

    st.title('Wind Speed Predictions Per County')

    # Load the data
    df = load_data()

    # Sidebar - User Input
    st.sidebar.title('Wind Speed Predictions per County')

    county_to_forecast = st.sidebar.text_input('Enter county name:', 'Offshore')
    prediction_duration = st.sidebar.slider('Select prediction duration (days):', min_value=1, max_value=3650, value=365)

    # Run the prediction and create combined DataFrame
    predictions, model, combined_df = create_model_and_combine_df(df, county_to_forecast, prediction_duration=prediction_duration)

    # Display the combined DataFrame
    st.dataframe(combined_df)

    # Plot the combined data
    st.subheader(f'Combined Historical and Predicted Wind Speed for {county_to_forecast}')
    fig, ax = plot_combined_data(combined_df, county_to_forecast)
    st.pyplot(fig)

    # st.title("Turbine Power Output Prediction App")

    x_array = "C:/Users/HomePC/Documents/School/Moringa/Capstone/era5.nc"  # Use forward slashes or double backslashes
# Create Xarray Dataset
    wind_xarray = xr.open_dataset(x_array)


    county_shapefile_path = r"C:\Users\HomePC\Documents\School\Moringa\Capstone\data_files\County.shp"

    counties_gdf = gpd.read_file(county_shapefile_path)
    wind_xarray = wind_xarray.where(wind_xarray.expver == 1, drop=True)
    wind_xarray = wind_xarray.sel(time=slice(None, '2023-11-30'))
    df1 = wind_xarray.to_dataframe().reset_index()
    df1['wind_speed_daily_100'] = np.sqrt(df1['u100']**2 + df1['v100']**2)

    # Extract longitude and latitude arrays
    longitude = wind_xarray.longitude.values
    latitude = wind_xarray.latitude.values
 
    # Specify the area covered by the 1287 squares
    north = 5.0
    west = 33.9
    south = -4.5
    east = 41.9

    # Filter data within the specified area
    mask = (latitude >= south) & (latitude <= north)
    filtered_longitude = longitude[(longitude >= west) & (longitude <= east)]
    filtered_latitude = latitude[mask]


    wind_speed_list = []

    u_component = 'u100'  # Assuming 'u100' corresponds to 10-meter components
    v_component = 'v100'  # Assuming 'v100' corresponds to 10-meter components

    for lon in longitude:
        for lat in latitude:
        # Subset the data for the current square
            square_data = wind_xarray.sel(latitude=lat, longitude=lon)

            # Calculate daily averages for the chosen components directly without resampling
            u_daily_avg = square_data[u_component].mean(dim='time').item()
            v_daily_avg = square_data[v_component].mean(dim='time').item()

            # Calculate daily wind speed
            wind_speed_daily = np.sqrt(u_daily_avg**2 + v_daily_avg**2)

            # Append the daily wind speed to the list
            wind_speed_list.append(wind_speed_daily)

    # Create a meshgrid for the quiver plot
    lon_grid, lat_grid = np.meshgrid(filtered_longitude, filtered_latitude, indexing='ij')
    wind_speed_grid = np.array(wind_speed_list).reshape(lon_grid.shape)

    # Create a meshgrid for longitude and latitude
    lon_grid, lat_grid = np.meshgrid(longitude, latitude, indexing='ij')
    # Sidebar with turbine parameters
    st.sidebar.header("Turbine Parameters")
    cut_in_speed = st.sidebar.number_input("Cut-in Speed (m/s)", value=3.0, step=0.1)
    rated_speed = st.sidebar.number_input("Rated Speed (m/s)", value=12.5, step=0.1)
    cut_off_speed = st.sidebar.number_input("Cut-off Speed (m/s)", value=25.0, step=0.1)
    rated_power = st.sidebar.number_input("Rated Power (kW)", value=3600.0, step=100.0)

    # Time period selection
    time_period_hours = st.sidebar.selectbox("Select Time Period (hours)", [1, 24, 168, 730, 8760])

    # Calculate power output
    result_df = calculate_power_output(df1, cut_in_speed, rated_speed, cut_off_speed, rated_power, time_period_hours, lon_grid, lat_grid, wind_speed_grid, county_shapefile_path)

    # Display results
    st.header("Power Output Prediction Results")
    st.write(result_df)

    # You can also add additional visualizations or charts here based on the results if desired.


if __name__ == '__main__':
    main()
