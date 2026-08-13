from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split


# ============================================================
# NASA C-MAPSS FD001
# Remaining Useful Life Prediction
#
# Goal:
# Estimate how many operating cycles an aircraft engine
# has remaining before failure.
# ============================================================


# ------------------------------------------------------------
# Project setup
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT_DIR / "train_FD001.txt"
TEST_FILE = PROJECT_DIR / "test_FD001.txt"
RUL_FILE = PROJECT_DIR / "RUL_FD001.txt"

OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# NASA column names
# ------------------------------------------------------------

columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]


# ------------------------------------------------------------
# Load the NASA dataset
# ------------------------------------------------------------

train = pd.read_csv(
    TRAIN_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)

test = pd.read_csv(
    TEST_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)

actual_test_rul = pd.read_csv(
    RUL_FILE,
    header=None,
    names=["actual_rul"]
)


# ------------------------------------------------------------
# Quick dataset overview
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("NASA C-MAPSS FD001")
print("=" * 60)

print(f"Training records : {len(train):,}")
print(f"Test records     : {len(test):,}")
print(f"Training engines : {train['engine_id'].nunique()}")
print(f"Test engines     : {test['engine_id'].nunique()}")


# ------------------------------------------------------------
# Data quality
# ------------------------------------------------------------

print("\nData quality")
print("-" * 40)

print(f"Missing values : {train.isna().sum().sum():,}")
print(f"Duplicate rows : {train.duplicated().sum():,}")


# ------------------------------------------------------------
# Engine lifetime analysis
# ------------------------------------------------------------

engine_life = (
    train
    .groupby("engine_id")["cycle"]
    .max()
)

print("\nEngine lifetime")
print("-" * 40)

print(f"Minimum : {engine_life.min()} cycles")
print(f"Maximum : {engine_life.max()} cycles")
print(f"Average : {engine_life.mean():.1f} cycles")


plt.figure(figsize=(10, 5))

plt.hist(
    engine_life,
    bins=15,
    edgecolor="black"
)

plt.title("Engine Lifetime Distribution")
plt.xlabel("Operating Cycles")
plt.ylabel("Number of Engines")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_engine_lifetime_distribution.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# Identify useful sensors
# ------------------------------------------------------------

sensor_columns = [
    column
    for column in train.columns
    if column.startswith("sensor_")
]


sensor_variation = (
    train[sensor_columns]
    .std()
    .sort_values(ascending=False)
)


useful_sensors = (
    sensor_variation[
        sensor_variation > 0.01
    ]
    .index
    .tolist()
)


print("\nSensor analysis")
print("-" * 40)

print(f"Total sensors    : {len(sensor_columns)}")
print(f"Useful sensors   : {len(useful_sensors)}")
print(f"Low-variation    : {len(sensor_columns) - len(useful_sensors)}")

print("\nSelected sensors:")

print(
    ", ".join(useful_sensors)
)


# ------------------------------------------------------------
# Sensor relationship with engine age
# ------------------------------------------------------------

cycle_relationship = (
    train[useful_sensors]
    .corrwith(train["cycle"])
    .dropna()
    .abs()
    .sort_values(ascending=False)
)


print("\nSensors most related to engine age")
print("-" * 40)

print(
    cycle_relationship.head(10)
)


# ------------------------------------------------------------
# Create the training RUL target
# ------------------------------------------------------------

final_cycle = (
    train
    .groupby("engine_id")["cycle"]
    .max()
    .rename("final_cycle")
)


train = train.merge(
    final_cycle,
    on="engine_id"
)


train["rul"] = (
    train["final_cycle"]
    - train["cycle"]
)


train.drop(
    columns="final_cycle",
    inplace=True
)


print("\nRUL target")
print("-" * 40)

print(
    f"Minimum RUL : {train['rul'].min()} cycles"
)

print(
    f"Maximum RUL : {train['rul'].max()} cycles"
)

print(
    f"Average RUL : {train['rul'].mean():.1f} cycles"
)


plt.figure(figsize=(10, 5))

plt.hist(
    train["rul"],
    bins=30,
    edgecolor="black"
)

plt.title("Training RUL Distribution")
plt.xlabel("Remaining Useful Life (Cycles)")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_rul_distribution.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

def create_features(data):

    result = data.copy()

    for sensor in useful_sensors:

        grouped = result.groupby("engine_id")[sensor]

        result[f"{sensor}_mean_5"] = (
            grouped
            .transform(
                lambda x: x.rolling(
                    5,
                    min_periods=1
                ).mean()
            )
        )

        result[f"{sensor}_std_5"] = (
            grouped
            .transform(
                lambda x: x.rolling(
                    5,
                    min_periods=1
                ).std()
            )
        )

        result[f"{sensor}_trend_5"] = (
            grouped
            .transform(
                lambda x: x.diff(5)
            )
        )

        result[f"{sensor}_trend_20"] = (
            grouped
            .transform(
                lambda x: x.diff(20)
            )
        )

        result[f"{sensor}_change"] = (
            result[sensor]
            - grouped.transform("first")
        )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result = result.fillna(0)

    return result


train_features = create_features(train)

test_features = create_features(test)


# ------------------------------------------------------------
# Build the model feature list
# ------------------------------------------------------------

feature_columns = [
    "cycle"
]


for sensor in useful_sensors:

    feature_columns.extend([
        sensor,
        f"{sensor}_mean_5",
        f"{sensor}_std_5",
        f"{sensor}_trend_5",
        f"{sensor}_trend_20",
        f"{sensor}_change"
    ])


print("\nFeature engineering")
print("-" * 40)

print(
    f"Model features : {len(feature_columns)}"
)


# ------------------------------------------------------------
# Split training engines
#
# We split by engine rather than by individual rows.
# This prevents the same engine appearing in both
# training and validation.
# ------------------------------------------------------------

engine_ids = train_features[
    "engine_id"
].unique()


training_engines, validation_engines = train_test_split(
    engine_ids,
    test_size=0.20,
    random_state=42
)


training_data = train_features[
    train_features["engine_id"].isin(
        training_engines
    )
].copy()


validation_data = train_features[
    train_features["engine_id"].isin(
        validation_engines
    )
].copy()


X_training = training_data[
    feature_columns
]

y_training = training_data[
    "rul"
]


X_validation = validation_data[
    feature_columns
]

y_validation = validation_data[
    "rul"
]


print("\nValidation setup")
print("-" * 40)

print(
    f"Training engines   : {len(training_engines)}"
)

print(
    f"Validation engines : {len(validation_engines)}"
)

print(
    f"Training records   : {len(X_training):,}"
)

print(
    f"Validation records : {len(X_validation):,}"
)


# ------------------------------------------------------------
# Baseline
# ------------------------------------------------------------

baseline_prediction = np.full(
    len(y_validation),
    y_training.mean()
)


baseline_mae = mean_absolute_error(
    y_validation,
    baseline_prediction
)


baseline_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        baseline_prediction
    )
)


print("\nBaseline")
print("-" * 40)

print(
    f"MAE  : {baseline_mae:.2f} cycles"
)

print(
    f"RMSE : {baseline_rmse:.2f} cycles"
)


# ------------------------------------------------------------
# Candidate models
# ------------------------------------------------------------

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=350,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=350,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )
}


# ------------------------------------------------------------
# Train and validate the candidate models
# ------------------------------------------------------------

validation_results = []


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    model.fit(
        X_training,
        y_training
    )

    predictions = model.predict(
        X_validation
    )

    predictions = np.maximum(
        predictions,
        0
    )

    mae = mean_absolute_error(
        y_validation,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions
        )
    )

    r2 = r2_score(
        y_validation,
        predictions
    )

    validation_results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


validation_comparison = pd.DataFrame(
    validation_results
).sort_values(
    "MAE"
).reset_index(
    drop=True
)


print("\n" + "=" * 60)
print("VALIDATION MODEL COMPARISON")
print("=" * 60)

print(
    validation_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ------------------------------------------------------------
# Select the best model using validation MAE
# ------------------------------------------------------------

best_model_name = validation_comparison.loc[
    0,
    "Model"
]


print("\nSelected model")
print("-" * 40)

print(
    best_model_name
)


# ------------------------------------------------------------
# Train the selected model on ALL training engines
# ------------------------------------------------------------

final_model = models[
    best_model_name
]


X_full_train = train_features[
    feature_columns
]

y_full_train = train_features[
    "rul"
]


print("\nTraining final model")
print("-" * 40)

print(
    f"Training records : {len(X_full_train):,}"
)

print(
    f"Training engines : {train['engine_id'].nunique()}"
)


final_model.fit(
    X_full_train,
    y_full_train
)


# ------------------------------------------------------------
# Prepare the final observation from every unseen test engine
# ------------------------------------------------------------

last_test_records = (
    test_features
    .sort_values(
        ["engine_id", "cycle"]
    )
    .groupby("engine_id")
    .tail(1)
    .copy()
)


X_test = last_test_records[
    feature_columns
]


predicted_rul = final_model.predict(
    X_test
)


predicted_rul = np.maximum(
    predicted_rul,
    0
)


actual_rul = actual_test_rul[
    "actual_rul"
].values


# ------------------------------------------------------------
# Final NASA test results
# ------------------------------------------------------------

results = pd.DataFrame({

    "engine_id":
        last_test_records[
            "engine_id"
        ].values,

    "last_cycle":
        last_test_records[
            "cycle"
        ].values,

    "actual_rul":
        actual_rul,

    "predicted_rul":
        predicted_rul
})


results["error"] = (
    results["actual_rul"]
    - results["predicted_rul"]
)


results["absolute_error"] = (
    results["error"]
    .abs()
)


final_mae = mean_absolute_error(
    actual_rul,
    predicted_rul
)


final_rmse = np.sqrt(
    mean_squared_error(
        actual_rul,
        predicted_rul
    )
)


final_r2 = r2_score(
    actual_rul,
    predicted_rul
)


print("\n" + "=" * 60)
print("FINAL NASA FD001 TEST PERFORMANCE")
print("=" * 60)

print(
    f"Model : {best_model_name}"
)

print(
    f"MAE   : {final_mae:.2f} cycles"
)

print(
    f"RMSE  : {final_rmse:.2f} cycles"
)

print(
    f"R²    : {final_r2:.3f}"
)


# ------------------------------------------------------------
# Prediction examples
# ------------------------------------------------------------

print("\nSample predictions")
print("-" * 60)

print(
    results.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ------------------------------------------------------------
# Largest prediction errors
# ------------------------------------------------------------

print("\nLargest prediction errors")
print("-" * 60)

worst_predictions = (
    results
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(10)
)


print(
    worst_predictions.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

if hasattr(
    final_model,
    "feature_importances_"
):

    feature_importance = (
        pd.Series(
            final_model.feature_importances_,
            index=feature_columns
        )
        .sort_values(
            ascending=False
        )
    )

    print("\nMost important features")
    print("-" * 50)

    print(
        feature_importance.head(15)
    )

    plt.figure(
        figsize=(10, 6)
    )

    feature_importance.head(
        15
    ).sort_values().plot(
        kind="barh"
    )

    plt.title(
        f"Top Features - {best_model_name}"
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "03_feature_importance.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ------------------------------------------------------------
# Actual vs predicted RUL
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 8)
)


plt.scatter(
    results["actual_rul"],
    results["predicted_rul"],
    alpha=0.75
)


minimum = min(
    results["actual_rul"].min(),
    results["predicted_rul"].min()
)


maximum = max(
    results["actual_rul"].max(),
    results["predicted_rul"].max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)


plt.title(
    f"NASA FD001 - Actual vs Predicted RUL\n{best_model_name}"
)

plt.xlabel(
    "Actual RUL (Cycles)"
)

plt.ylabel(
    "Predicted RUL (Cycles)"
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR / "04_actual_vs_predicted_rul.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# Prediction error distribution
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)


plt.hist(
    results["error"],
    bins=20,
    edgecolor="black"
)


plt.axvline(
    0,
    linestyle="--"
)


plt.title(
    "RUL Prediction Error Distribution"
)

plt.xlabel(
    "Error (Actual - Predicted)"
)

plt.ylabel(
    "Number of Engines"
)


plt.tight_layout()


plt.savefig(
    FIGURE_DIR / "05_prediction_error_distribution.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# Actual vs predicted RUL for the engine with the
# largest error
# ------------------------------------------------------------

worst_engine_id = int(
    results.loc[
        results["absolute_error"].idxmax(),
        "engine_id"
    ]
)


worst_engine_data = train[
    train["engine_id"] == worst_engine_id
]


if len(worst_engine_data) > 0:

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        worst_engine_data["cycle"],
        worst_engine_data["rul"]
    )

    plt.title(
        f"Training Engine {worst_engine_id} - RUL Degradation"
    )

    plt.xlabel(
        "Operating Cycle"
    )

    plt.ylabel(
        "RUL (Cycles)"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "06_engine_rul_degradation.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ------------------------------------------------------------
# Save final outputs
# ------------------------------------------------------------

results_file = (
    OUTPUT_DIR /
    "NASA_FD001_final_predictions.csv"
)


comparison_file = (
    OUTPUT_DIR /
    "NASA_FD001_model_comparison.csv"
)


summary_file = (
    OUTPUT_DIR /
    "NASA_FD001_project_summary.csv"
)


results.to_csv(
    results_file,
    index=False
)


validation_comparison.to_csv(
    comparison_file,
    index=False
)


summary = pd.DataFrame({
    "Metric": [
        "Dataset",
        "Training Engines",
        "Test Engines",
        "Training Records",
        "Model Features",
        "Selected Model",
        "Validation MAE",
        "Final Test MAE",
        "Final Test RMSE",
        "Final Test R2"
    ],

    "Value": [
        "NASA C-MAPSS FD001",
        train["engine_id"].nunique(),
        test["engine_id"].nunique(),
        len(train),
        len(feature_columns),
        best_model_name,
        validation_comparison.loc[0, "MAE"],
        final_mae,
        final_rmse,
        final_r2
    ]
})


summary.to_csv(
    summary_file,
    index=False
)


# ------------------------------------------------------------
# Final project summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)

print(
    f"Selected model : {best_model_name}"
)

print(
    f"Final MAE      : {final_mae:.2f} cycles"
)

print(
    f"Final RMSE     : {final_rmse:.2f} cycles"
)

print(
    f"Final R²       : {final_r2:.3f}"
)

print("\nOutputs created:")

print(
    f"Predictions : {results_file}"
)

print(
    f"Comparison  : {comparison_file}"
)

print(
    f"Summary     : {summary_file}"
)

print(
    f"Figures     : {FIGURE_DIR}"
)

print("\nNASA C-MAPSS RUL project finished successfully.")
