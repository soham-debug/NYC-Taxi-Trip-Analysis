# NYC-Taxi-Trip-Analysis

This project analyzes the NYC Green Taxi Trip dataset to uncover patterns in customer demand, ride behavior, and revenue.

🔧 Tech Stack

1. Python (Pandas, XGBoost) - Data cleaning, preprocessing, feature engineering, and classification.
2. Microsoft Excel - To store data in csv format and use this data in looker and will use at backend.
3. Looker Studio - Interactive dashboard for visualization and insights.

Key Steps: -

1. Data Cleaning & Preprocessing
2. Removed duplicates, handled missing values, and converted timestamps.
3. Engineered new features like trip_duration_min, pickup_hour, pickup_weekday, and ride_type.
4. Filtered invalid trips (negative fare, zero distance).
5. Exploratory Insights & Dashboard
6. Demand analysis by hour of day and day of week.
7. Revenue and fare breakdown (base fare, tips, surcharges, tolls).
8. Ride distribution by distance categories (short/medium/long).
9. Payment method trends (Cash vs Card).
10. Interactive dashboard built in Looker Studio.
11. Machine Learning (XGBoost)

1. Built a classifier to predict payment type using trip features.
2. Extracted feature importance to understand what drives payment choice.

Key Insights: -

1. Peak demand occurs during rush hours (8–9 AM, 5–8 PM).
S2. hort trips form the majority but generate lower revenue per trip.

Tips contribute significantly during late-night rides.

XGBoost showed that fare amount, trip distance, and trip duration are the strongest predictors of payment type.
