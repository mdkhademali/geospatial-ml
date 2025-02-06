import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load CSV data and convert to GeoDataFrame."""
    df = pd.read_csv(filepath)
    df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    return gdf

def preprocess_data(gdf):
    """Split data into train and test sets, then scale features."""
    X = gdf[['Latitude', 'Longitude']]
    y = gdf['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    gdf = load_data("../data/sample_data.csv")  # Adjust the path as needed
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(gdf)
    print("Data Processing Completed!")