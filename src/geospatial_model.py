import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import data_processing  # Importing our data processing script

def train_model(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print accuracy."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.2f}')
    return y_pred

def plot_geospatial_data(gdf):
    """Plot the geospatial data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf[gdf['Label'] == 0].plot(ax=ax, color='blue', label='Class 0', alpha=0.6)
    gdf[gdf['Label'] == 1].plot(ax=ax, color='red', label='Class 1', alpha=0.6)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.title("Geospatial Data Classification")
    plt.show()

if __name__ == "__main__":
    gdf = data_processing.load_data("../data/sample_data.csv")
    X_train_scaled, X_test_scaled, y_train, y_test = data_processing.preprocess_data(gdf)

    model = train_model(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test)

    plot_geospatial_data(gdf)