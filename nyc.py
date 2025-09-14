import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

file_path = "/content/NYC TLC Trip Record.csv"
df = pd.read_csv(file_path, low_memory=False)

print("Original Shape:", df.shape)

df.drop_duplicates(inplace=True)
df.dropna(subset=["lpep_pickup_datetime", "lpep_dropoff_datetime", "trip_distance", "fare_amount", "payment_type"], inplace=True)

df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"], errors="coerce")
df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"], errors="coerce")
df = df.dropna(subset=["lpep_pickup_datetime", "lpep_dropoff_datetime"])

df = df[(df["fare_amount"] > 0) & (df["trip_distance"] > 0)]

df["trip_duration_min"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
df = df[df["trip_duration_min"] > 0]

df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour
df["pickup_weekday"] = df["lpep_pickup_datetime"].dt.dayofweek  # 0=Monday

df["ride_type"] = df["trip_distance"].apply(lambda x: 0 if x <= 2 else 1)  # 0=Short, 1=Long

df.columns = [col.lower().replace(" ", "_") for col in df.columns]

df.to_csv("cleaned_nyc_taxi.csv", index=False)
df.to_excel("cleaned_nyc_taxi.xlsx", index=False)

print("Cleaned dataset saved as CSV and Excel!")

features = ["trip_distance", "fare_amount", "pickup_hour", "trip_duration_min", "ride_type", "pickup_weekday"]
X = df[features]
y = df["payment_type"]

payment_type_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
y = y.map(payment_type_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n XGBoost Classification Report:\n")
print(classification_report(y_test, y_pred))

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("feature_importance.csv", index=False)

print("\n Feature importance saved as 'feature_importance.csv'")
print("Top Features:\n", importance_df)