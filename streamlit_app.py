import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# Set the Streamlit app title
st.title("Formula 1 Data Analysis")
st.subheader("Press the side arrow to attach files for visualization")
# Create a sidebar for options
st.sidebar.title("Select a Visualization")
options = st.sidebar.selectbox("Choose the analysis", 
                               ['Pit Stop Analysis', 'Qualifying vs Race Performance', 
                                'Constructor Performance', 'Driver Rankings', 
                                'Driver Fastest Laps', 'Pair Plot Analysis', 'Histograms','Regression Analysis', 'Constructor Ranking Regression','driver Ranking Regression'])

# Upload CSV files
st.sidebar.header("Upload CSV Files")
pit_stop_file = st.sidebar.file_uploader("Upload Pit Stop Records", type='csv')
qualifying_file = st.sidebar.file_uploader("Upload Qualifying Results", type='csv')
race_results_file = st.sidebar.file_uploader("Upload Race Results", type='csv')
constructor_performance_file = st.sidebar.file_uploader("Upload Constructor Performance", type='csv')
constructor_ranking_file = st.sidebar.file_uploader("Upload Constructor Rankings", type='csv')
driver_rankings_file = st.sidebar.file_uploader("Upload Driver Rankings", type='csv')
driver_details_file = st.sidebar.file_uploader("Upload Driver Details", type='csv')

# Define a function to read the CSV files
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    return None

# Load the data files
pit_stop_records = load_data(pit_stop_file)
qualifying_results = load_data(qualifying_file)
race_results = load_data(race_results_file)
constructor_performance = load_data(constructor_performance_file)
constructor_ranking = load_data(constructor_ranking_file)
driver_rankings = load_data(driver_rankings_file)
driver_details = load_data(driver_details_file)

# Analysis and Visualizations based on selection
if options == 'Pit Stop Analysis' and pit_stop_records is not None and race_results is not None:
    st.subheader("Pit Stop Duration vs Race Position")
    
    # Merge pit stop and race results data
    merged_data = pit_stop_records.merge(race_results, on=["raceId", "driverId"])
    merged_data["duration"] = pd.to_numeric(merged_data["duration"], errors='coerce')
    
    average_pit_stop_duration = merged_data.groupby("driverId")["duration"].mean().reset_index()
    merged_data = merged_data.merge(average_pit_stop_duration, on="driverId", suffixes=("_original", "_average"))
    merged_data["position"] = pd.to_numeric(merged_data["position"], errors='coerce')
    
    top_50_data = merged_data[merged_data["position"] <= 50]
    
    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(top_50_data["position"], top_50_data["duration_average"], 
                          c=top_50_data["duration_average"], cmap='viridis', alpha=0.7, 
                          edgecolors='w', linewidth=0.5)
    plt.colorbar(scatter, label='Average Pit Stop Duration')
    plt.xlabel("Race Position")
    plt.ylabel("Average Pit Stop Duration")
    plt.title("Pit Stop Duration vs. Race Position (Top 50)")
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(plt)

elif options == 'Qualifying vs Race Performance' and qualifying_results is not None and race_results is not None:
    st.subheader("Qualifying vs Race Performance for Top 3 Drivers")
    
    # Merge qualifying and race results data
    merged_data = qualifying_results.merge(race_results, on=["raceId", "driverId"])
    merged_data = merged_data.sort_values(by=["raceId", "driverId"])
    
    top_3_drivers = merged_data["driverId"].value_counts().head(3).index
    top_3_data = merged_data[merged_data["driverId"].isin(top_3_drivers)]
    
    top_3_data["rank"] = top_3_data.groupby("raceId")["rank"].rank(method='dense').astype(int) - 1
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap('tab10')
    
    for i, driver_id in enumerate(top_3_data["driverId"].unique()):
        driver_data = top_3_data[top_3_data["driverId"] == driver_id]
        plt.plot(driver_data["raceId"], driver_data["rank"], label=f"Driver {driver_id} - Qualifying", color=colors(i))
        plt.plot(driver_data["raceId"], driver_data["positionOrder"], linestyle='--', label=f"Driver {driver_id} - Race", color=colors(i))
    
    plt.xlabel("Race ID")
    plt.ylabel("Rank")
    plt.title("Qualifying Rank vs. Position (Top 3 Drivers)")
    plt.legend()
    st.pyplot(plt)

elif options == 'Constructor Performance' and constructor_performance is not None and constructor_ranking is not None:
    st.subheader("Constructor Performance vs Constructor Rankings")
    
    # Merge constructor performance and ranking data
    merged_data = constructor_performance.merge(constructor_ranking, on=['constructorId', 'raceId'])
    
    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_data, x='positionText', y='position', hue='constructorId', palette='viridis')
    plt.title('Constructor Performance vs. Constructor Rankings')
    plt.xlabel('Ranking Position')
    plt.ylabel('Points')
    st.pyplot(plt)

    # Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=merged_data, x='positionText', y='position', hue='constructorId', palette='plasma')
    plt.title('Constructor Performance vs. Constructor Rankings')
    plt.xlabel('Ranking Position')
    plt.ylabel('Points')
    st.pyplot(plt)

elif options == 'Driver Rankings' and driver_rankings is not None and race_results is not None:
    st.subheader("Driver Rankings vs Race Results")
    
    # Merge driver rankings and race results data
    merged_data = driver_rankings.merge(race_results, on=["raceId", "driverId"])
    
    top_5_drivers = merged_data.groupby("driverId")["wins"].sum().nlargest(5).index
    top_5_data = merged_data[merged_data["driverId"].isin(top_5_drivers)]
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.get_cmap('tab10')
    
    for i, driver_id in enumerate(top_5_data["driverId"].unique()):
        driver_data = top_5_data[top_5_data["driverId"] == driver_id]
        plt.bar(driver_data["raceId"], driver_data["fastestLap"], label=f"Driver {driver_id} - Fastest Lap", color=colors(i))
    
    plt.xlabel("Race ID")
    plt.ylabel("Fastest Lap")
    plt.title("Driver Rankings vs. Race Results Over the Season (Top 5 Drivers)")
    plt.legend()
    st.pyplot(plt)

elif options == 'Driver Fastest Laps' and driver_details is not None and race_results is not None:
    st.subheader("Driver Reference vs. Fastest Lap Speed (Top 10 Drivers)")
    
    # Merge driver details and race results data
    merged_data = driver_details.merge(race_results, on="driverId")
    merged_data["fastestLapSpeed"] = pd.to_numeric(merged_data["fastestLapSpeed"], errors='coerce')
    merged_data = merged_data.dropna(subset=["fastestLapSpeed"])
    
    top_10_drivers = merged_data.groupby("driverId")["fastestLapSpeed"].mean().nlargest(10).index
    top_10_data = merged_data[merged_data["driverId"].isin(top_10_drivers)]
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab10', len(top_10_data["driverId"].unique()))
    
    for i, driver_id in enumerate(top_10_data["driverId"].unique()):
        driver_data = top_10_data[top_10_data["driverId"] == driver_id]
        plt.plot(driver_data["driverRef"], driver_data["fastestLapSpeed"], marker='o', label=f"Driver {driver_data['driverRef'].iloc[0]}", color=colors(i))
    
    plt.xlabel("Driver Reference")
    plt.ylabel("Fastest Lap Speed (km/h)")
    plt.title("Driver Reference vs. Fastest Lap Speed (Top 10 Drivers)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

elif options == 'Pair Plot Analysis' and pit_stop_records is not None and qualifying_results is not None and race_results is not None:
    st.subheader("Comprehensive Pair Plot of F1 Data")
    
    # Merge pit stop, qualifying, and race results data
    merged_data = (pit_stop_records
                   .merge(qualifying_results, on=["raceId", "driverId"])
                   .merge(race_results, on=["raceId", "driverId"]))
    
    selected_columns = ["stop", "lap", "duration", "grid", "positionText", "positionOrder", "points", "laps"]
    
    for col in selected_columns:
        merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')
    
    merged_data = merged_data.dropna(subset=selected_columns)
    
    # Pair Plot
    sns.pairplot(merged_data[selected_columns].sample(100), diag_kind="kde", palette="husl")
    st.pyplot(plt)

elif options == 'Histograms' and pit_stop_records is not None and qualifying_results is not None and race_results is not None:
    st.subheader("Histograms of Selected Columns")

    # Merge pit stop, qualifying, and race results data
    merged_data = (pit_stop_records
                   .merge(qualifying_results, on=["raceId", "driverId"])
                   .merge(race_results, on=["raceId", "driverId"]))

    # Define the columns to select from
    selected_columns = ["stop", "lap", "duration", "grid", "rank", "positionOrder", "points", "laps", "fastestLap"]

    # Convert the selected columns to numeric and handle missing values
    for col in selected_columns:
        merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

    # Drop rows with missing values in the selected columns
    merged_data = merged_data.dropna(subset=selected_columns)

    # Create a multi-select box for the user to choose columns
    selected_hist_columns = st.multiselect("Select columns for histograms", selected_columns, default=selected_columns)

    if selected_hist_columns:
        # Set up the plotting area with a dynamic number of subplots based on the number of selected columns
        num_cols = len(selected_hist_columns)
        num_rows = (num_cols // 3) + (1 if num_cols % 3 != 0 else 0)

        plt.figure(figsize=(18, num_rows * 4))
        
        # Loop through the selected columns and create a histogram for each
        for i, col in enumerate(selected_hist_columns):
            plt.subplot(num_rows, 3, i + 1)
            sns.histplot(merged_data[col].dropna(), bins=20, kde=True, color=sns.color_palette('husl', num_cols)[i])
            plt.title(f'Histogram of {col.capitalize()}')
            plt.xlabel(col.capitalize())
            plt.ylabel('Frequency')

        plt.tight_layout(pad=3.0)
        st.pyplot(plt)
    else:
        st.warning("Please select at least one column for visualization.")



elif options == 'Regression Analysis' and pit_stop_records is not None and qualifying_results is not None and race_results is not None:
    st.subheader("Regression Analysis on Points vs. Laps")

    # Upload the dataset
    race_results_file = st.file_uploader("Upload a CSV file with race results", type=["csv"])
    
    if race_results_file is not None:
        data = pd.read_csv(race_results_file)

        st.write("Displaying the first few rows of the dataset:")
        st.write(data.head())

        data = data.head(20)

        # Prepare the data
        points = data['points'].values.reshape(-1, 1)
        laps = data['laps'].values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(points, laps, test_size=0.2, random_state=42)

        # Initialize models
        linear_model = LinearRegression()
        tree_model = DecisionTreeRegressor(random_state=42)
        forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train models
        linear_model.fit(X_train, y_train)
        tree_model.fit(X_train, y_train)
        forest_model.fit(X_train, y_train)

        # Predict on the test set
        y_pred_linear = linear_model.predict(X_test)
        y_pred_tree = tree_model.predict(X_test)
        y_pred_forest = forest_model.predict(X_test)

        # Calculate MSE
        mse_linear = mean_squared_error(y_test, y_pred_linear)
        mse_tree = mean_squared_error(y_test, y_pred_tree)
        mse_forest = mean_squared_error(y_test, y_pred_forest)

        st.write(f"Linear Regression MSE: {mse_linear}")
        st.write(f"Decision Tree Regression MSE: {mse_tree}")
        st.write(f"Random Forest Regression MSE: {mse_forest}")
        

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(14, 8))

        # Linear Regression Plot
        axes[0].scatter(points, laps, color='blue', label='Data points')
        axes[0].plot(points, linear_model.predict(points), color='red', label='Linear Regression Fit')
        axes[0].set_xlabel('Points')
        axes[0].set_ylabel('Laps')
        axes[0].set_title('Linear Regression')
        axes[0].legend()

        # Decision Tree Regression Plot
        axes[1].scatter(points, laps, color='blue', label='Data points')
        axes[1].plot(points, tree_model.predict(points), color='green', label='Decision Tree Fit')
        axes[1].set_xlabel('Points')
        axes[1].set_ylabel('Laps')
        axes[1].set_title('Decision Tree Regression')
        axes[1].legend()

        # Random Forest Regression Plot
        axes[2].scatter(points, laps, color='blue', label='Data points')
        axes[2].plot(points, forest_model.predict(points), color='purple', label='Random Forest Fit')
        axes[2].set_xlabel('Points')
        axes[2].set_ylabel('Laps')
        axes[2].set_title('Random Forest Regression')
        axes[2].legend()

        st.pyplot(fig)

        # Residual Plots
        st.subheader("Residual Plots")

        fig_residual, axes_residual = plt.subplots(1, 3, figsize=(14, 8))

        sns.residplot(x=y_test, y=y_pred_linear, color='red', ax=axes_residual[0])
        axes_residual[0].set_xlabel('Actual Laps')
        axes_residual[0].set_ylabel('Residuals')
        axes_residual[0].set_title('Linear Regression Residuals')

        sns.residplot(x=y_test, y=y_pred_tree, color='green', ax=axes_residual[1])
        axes_residual[1].set_xlabel('Actual Laps')
        axes_residual[1].set_ylabel('Residuals')
        axes_residual[1].set_title('Decision Tree Regression Residuals')

        sns.residplot(x=y_test, y=y_pred_forest, color='purple', ax=axes_residual[2])
        axes_residual[2].set_xlabel('Actual Laps')
        axes_residual[2].set_ylabel('Residuals')
        axes_residual[2].set_title('Random Forest Regression Residuals')

        st.pyplot(fig_residual)

        # Prediction vs Actual Plots
        st.subheader("Prediction vs Actual")

        fig_pred, axes_pred = plt.subplots(1, 3, figsize=(14, 8))

        axes_pred[0].scatter(y_test, y_pred_linear, color='red')
        axes_pred[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes_pred[0].set_xlabel('Actual Laps')
        axes_pred[0].set_ylabel('Predicted Laps')
        axes_pred[0].set_title('Linear Regression: Prediction vs Actual')

        axes_pred[1].scatter(y_test, y_pred_tree, color='green')
        axes_pred[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes_pred[1].set_xlabel('Actual Laps')
        axes_pred[1].set_ylabel('Predicted Laps')
        axes_pred[1].set_title('Decision Tree Regression: Prediction vs Actual')

        axes_pred[2].scatter(y_test, y_pred_forest, color='purple')
        axes_pred[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes_pred[2].set_xlabel('Actual Laps')
        axes_pred[2].set_ylabel('Predicted Laps')
        axes_pred[2].set_title('Random Forest Regression: Prediction vs Actual')

        st.pyplot(fig_pred)

    else:
        st.warning("Please upload a valid CSV file for analysis.")

elif options == 'Constructor Ranking Regression' and  pit_stop_records is not None and qualifying_results is not None and race_results is not None:
    st.subheader("Regression Analysis on Points vs. wins from Constructor Rankings")
    
    constructor_ranking_file = st.file_uploader("Upload a CSV file with constructor ranking file", type=["csv"])

    if constructor_ranking_file is not None:
        # Load the data
        data = pd.read_csv(constructor_ranking_file)
        st.write("Displaying the first few rows of the dataset:")
        st.write(data.head())

        # Use a subset of the data
        data = data.head(100)

        # Prepare the data
        points = data['points'].values.reshape(-1, 1)
        wins = data['wins'].values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(points, wins, test_size=0.2, random_state=42)

        # Initialize models
        linear_model = LinearRegression()
        tree_model = DecisionTreeRegressor(random_state=42)
        forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train models
        linear_model.fit(X_train, y_train)
        tree_model.fit(X_train, y_train)
        forest_model.fit(X_train, y_train)

        # Make predictions
        y_pred_linear = linear_model.predict(X_test)
        y_pred_tree = tree_model.predict(X_test)
        y_pred_forest = forest_model.predict(X_test)

        # Calculate MSE
        mse_linear = mean_squared_error(y_test, y_pred_linear)
        mse_tree = mean_squared_error(y_test, y_pred_tree)
        mse_forest = mean_squared_error(y_test, y_pred_forest)

        st.write(f"Linear Regression MSE: {mse_linear}")
        st.write(f"Decision Tree Regression MSE: {mse_tree}")
        st.write(f"Random Forest Regression MSE: {mse_forest}")

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(14, 8))

        # Linear Regression Plot
        axes[0].scatter(points, wins, color='blue', label='Data points')
        axes[0].plot(points, linear_model.predict(points), color='red', label='Linear Regression Fit')
        axes[0].set_xlabel('Points')
        axes[0].set_ylabel('Wins')
        axes[0].set_title('Linear Regression')
        axes[0].legend()

        # Decision Tree Regression Plot
        axes[1].scatter(points, wins, color='blue', label='Data points')
        axes[1].plot(points, tree_model.predict(points), color='green', label='Decision Tree Fit')
        axes[1].set_xlabel('Points')
        axes[1].set_ylabel('Wins')
        axes[1].set_title('Decision Tree Regression')
        axes[1].legend()

        # Random Forest Regression Plot
        axes[2].scatter(points, wins, color='blue', label='Data points')
        axes[2].plot(points, forest_model.predict(points), color='purple', label='Random Forest Fit')
        axes[2].set_xlabel('Points')
        axes[2].set_ylabel('Wins')
        axes[2].set_title('Random Forest Regression')
        axes[2].legend()

        st.pyplot(fig)

        # Residual Plots
        st.subheader("Residual Plots")

        fig_residual, axes_residual = plt.subplots(1, 3, figsize=(14, 8))

        sns.residplot(x=y_test, y=y_pred_linear, color='red', ax=axes_residual[0])
        axes_residual[0].set_xlabel('Actual Wins')
        axes_residual[0].set_ylabel('Residuals')
        axes_residual[0].set_title('Linear Regression Residuals')

        sns.residplot(x=y_test, y=y_pred_tree, color='green', ax=axes_residual[1])
        axes_residual[1].set_xlabel('Actual Wins')
        axes_residual[1].set_ylabel('Residuals')
        axes_residual[1].set_title('Decision Tree Regression Residuals')

        sns.residplot(x=y_test, y=y_pred_forest, color='purple', ax=axes_residual[2])
        axes_residual[2].set_xlabel('Actual Wins')
        axes_residual[2].set_ylabel('Residuals')
        axes_residual[2].set_title('Random Forest Regression Residuals')

        st.pyplot(fig_residual)

        # Prediction vs Actual Plots
        st.subheader("Prediction vs Actual")

        fig_pred, axes_pred = plt.subplots(1, 3, figsize=(14, 8))

        axes_pred[0].scatter(y_test, y_pred_linear, color='red')
        axes_pred[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes_pred[0].set_xlabel('Actual Wins')
        axes_pred[0].set_ylabel('Predicted Wins')
        axes_pred[0].set_title('Linear Regression: Prediction vs Actual')

        axes_pred[1].scatter(y_test, y_pred_tree, color='green')
        axes_pred[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes_pred[1].set_xlabel('Actual Wins')
        axes_pred[1].set_ylabel('Predicted Wins')
        axes_pred[1].set_title('Decision Tree Regression: Prediction vs Actual')

        axes_pred[2].scatter(y_test, y_pred_forest, color='purple')
        axes_pred[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes_pred[2].set_xlabel('Actual Wins')
        axes_pred[2].set_ylabel('Predicted Wins')
        axes_pred[2].set_title('Random Forest Regression: Prediction vs Actual')

        st.pyplot(fig_pred)
    else:
        st.warning("Please upload a valid CSV file for analysis.")

elif options == 'driver Ranking Regression' and pit_stop_records is not None and qualifying_results is not None and race_results is not None:
    st.subheader("Regression Analysis on Points vs. Wins from Driver Rankings")
    
    # File uploader for the driver rankings CSV
    driver_ranking_file = st.file_uploader("Upload a CSV file with driver rankings", type=["csv"])

    if driver_ranking_file is not None:
        # Load the data
        data = pd.read_csv(driver_ranking_file)
        st.write("Displaying the first few rows of the dataset:")
        st.write(data.head())

        # Use a subset of the data
        data = data.head(100)

        # Prepare the data
        points = data['points'].values.reshape(-1, 1)
        wins = data['wins'].values

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(points, wins, test_size=0.2, random_state=42)

        # Initialize models
        linear_model = LinearRegression()
        tree_model = DecisionTreeRegressor(random_state=42)
        forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train models
        linear_model.fit(X_train, y_train)
        tree_model.fit(X_train, y_train)
        forest_model.fit(X_train, y_train)

        # Make predictions
        y_pred_linear = linear_model.predict(X_test)
        y_pred_tree = tree_model.predict(X_test)
        y_pred_forest = forest_model.predict(X_test)

        # Calculate MSE
        mse_linear = mean_squared_error(y_test, y_pred_linear)
        mse_tree = mean_squared_error(y_test, y_pred_tree)
        mse_forest = mean_squared_error(y_test, y_pred_forest)

        st.write(f"Linear Regression MSE: {mse_linear}")
        st.write(f"Decision Tree Regression MSE: {mse_tree}")
        st.write(f"Random Forest Regression MSE: {mse_forest}")

        # Plotting the Regression Fits
        st.subheader("Regression Fits")
        fig, ax = plt.subplots(1, 3, figsize=(14, 8))

        # Linear Regression
        ax[0].scatter(points, wins, color='blue', label='Data points')
        ax[0].plot(points, linear_model.predict(points), color='red', label='Linear Regression Fit')
        ax[0].set_xlabel('Points')
        ax[0].set_ylabel('Wins')
        ax[0].set_title('Linear Regression')
        ax[0].legend()

        # Decision Tree Regression
        ax[1].scatter(points, wins, color='blue', label='Data points')
        ax[1].plot(points, tree_model.predict(points), color='green', label='Decision Tree Fit')
        ax[1].set_xlabel('Points')
        ax[1].set_ylabel('Wins')
        ax[1].set_title('Decision Tree Regression')
        ax[1].legend()

        # Random Forest Regression
        ax[2].scatter(points, wins, color='blue', label='Data points')
        ax[2].plot(points, forest_model.predict(points), color='purple', label='Random Forest Fit')
        ax[2].set_xlabel('Points')
        ax[2].set_ylabel('Wins')
        ax[2].set_title('Random Forest Regression')
        ax[2].legend()

        st.pyplot(fig)

        # Residual Plots
        st.subheader("Residual Plots")
        fig, ax = plt.subplots(1, 3, figsize=(14, 8))

        sns.residplot(x=y_test, y=y_pred_linear, color='red', ax=ax[0])
        ax[0].set_xlabel('Actual Wins')
        ax[0].set_ylabel('Residuals')
        ax[0].set_title('Linear Regression Residuals')

        sns.residplot(x=y_test, y=y_pred_tree, color='green', ax=ax[1])
        ax[1].set_xlabel('Actual Wins')
        ax[1].set_ylabel('Residuals')
        ax[1].set_title('Decision Tree Residuals')

        sns.residplot(x=y_test, y=y_pred_forest, color='purple', ax=ax[2])
        ax[2].set_xlabel('Actual Wins')
        ax[2].set_ylabel('Residuals')
        ax[2].set_title('Random Forest Residuals')

        st.pyplot(fig)

        # Prediction vs Actual Plots
        st.subheader("Prediction vs Actual Plots")
        fig, ax = plt.subplots(1, 3, figsize=(14, 8))

        ax[0].scatter(y_test, y_pred_linear, color='red')
        ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax[0].set_xlabel('Actual Wins')
        ax[0].set_ylabel('Predicted Wins')
        ax[0].set_title('Linear Regression')

        ax[1].scatter(y_test, y_pred_tree, color='green')
        ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax[1].set_xlabel('Actual Wins')
        ax[1].set_ylabel('Predicted Wins')
        ax[1].set_title('Decision Tree Regression')

        ax[2].scatter(y_test, y_pred_forest, color='purple')
        ax[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax[2].set_xlabel('Actual Wins')
        ax[2].set_ylabel('Predicted Wins')
        ax[2].set_title('Random Forest Regression')

        st.pyplot(fig)

    else:
        st.warning("Please upload a valid CSV file for analysis.")
