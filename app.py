import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Demand prediction", layout="wide")
st.title("Food Sales Dashboard")

uploaded_file = st.file_uploader("Upload your food sales CSV", type=["csv"])


def prediction(df):
    st.subheader("Quick Demand Prediction")
    mode = st.selectbox(
        "Choose a prediction type",
        ["Select...", "Dish + Location", "Dish", "Location", "Weather"]
    )

    prediction_input = {}
    if mode != "Select...":
        if mode == "Dish + Location":
            if {'dish_name', 'location'} <= set(df.columns):
                dish = st.selectbox("Select Dish", df['dish_name'].dropna().unique())
                loc = st.selectbox("Select Location", df['location'].dropna().unique())
                prediction_input = {"dish_name": dish, "location": loc}
        elif mode == "Dish":
            dish = st.selectbox("Select Dish", df['dish_name'].dropna().unique())
            prediction_input = {"dish_name": dish}
        elif mode == "Location":
            loc = st.selectbox("Select Location", df['location'].dropna().unique())
            prediction_input = {"location": loc}
        elif mode == "Weather":
            weather = st.selectbox("Select Weather", df['weather_condition'].dropna().unique())
            prediction_input = {"weather_condition": weather}

        if prediction_input:
            input_row = df.copy()
            for key in prediction_input:
                input_row = input_row[input_row[key] == prediction_input[key]]

            if len(input_row) == 0:
                st.warning("No matching data found for selected inputs.")
                return
            else:
                input_row = input_row.iloc[[0]].copy()

                df_proc = df.copy()
                if 'date' in df_proc.columns:
                    df_proc['date'] = pd.to_datetime(df_proc['date'], errors='coerce')
                    df_proc['day'] = df_proc['date'].dt.day
                    df_proc['month'] = df_proc['date'].dt.month
                    df_proc['weekday'] = df_proc['date'].dt.weekday
                    df_proc['is_weekend'] = df_proc['weekday'].isin([5, 6]).astype(int)

                    input_row['date'] = pd.to_datetime(input_row['date'], errors='coerce')
                    input_row['day'] = input_row['date'].dt.day
                    input_row['month'] = input_row['date'].dt.month
                    input_row['weekday'] = input_row['date'].dt.weekday
                    input_row['is_weekend'] = input_row['weekday'].isin([5, 6]).astype(int)

                char_columns = df_proc.select_dtypes(include='object').columns.tolist()
                encoder = OrdinalEncoder()
                df_proc[char_columns] = encoder.fit_transform(df_proc[char_columns])
                input_row[char_columns] = encoder.transform(input_row[char_columns])

                def classify_demand(units):
                    if units < 50:
                        return '0'
                    elif units < 150:
                        return '1'
                    else:
                        return '2'

                df_proc['demand_level'] = df_proc['units_sold'].apply(classify_demand)

                le = LabelEncoder()
                y = le.fit_transform(df_proc['demand_level'])

                X = df_proc.drop(columns=['demand_level', 'date'])
                input_X = input_row[X.columns]

                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)

                prediction = model.predict(input_X)[0]
                demand_label = {0: "Low", 1: "Medium", 2: "High"}
                st.success(f"📈 Predicted Demand: **{demand_label[prediction]}**")


def run_ml_model(df):
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    columns = df.columns.tolist()
    columns_to_remove = numeric_columns + ['date']
    char_columns = [col for col in columns if col not in columns_to_remove] 

    encoder = OrdinalEncoder()
    df[char_columns] = encoder.fit_transform(df[char_columns])

    def classify_demand(units):
        if units < 50:
            return '0'
        elif units < 150:
            return '1'
        else:
            return '2'

    df['demand_level'] = df['units_sold'].apply(classify_demand)

    le = LabelEncoder()
    y = le.fit_transform(df['demand_level'])

    train_columns_to_remove = ['demand_level', 'date']
    X = df.drop(columns=train_columns_to_remove)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown("**Model Evaluation:**")
    st.code(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']), language="text")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)



if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

 
    prediction(df)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['weekday'] = df['date'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    
    st.sidebar.header("Filters")
    for col in ['city', 'location', 'restaurant', 'dish_name']:
        if col in df.columns:
            options = sorted(df[col].dropna().unique())
            selection = st.sidebar.multiselect(f"Filter by {col}", options, default=options)
            df = df[df[col].isin(selection)]

    
    st.subheader("Overview Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Orders", int(df['units_sold'].sum()))
    with col2:
        st.metric("Unique Dishes", df['dish_name'].nunique())
    with col3:
        st.metric("Unique Locations", df['location'].nunique())

   
    if 'date' in df.columns:
        st.subheader("time Series: Units Sold")
        df_ts = df.groupby('date')['units_sold'].sum().reset_index()
        st.line_chart(df_ts.rename(columns={'units_sold': 'Units Sold'}).set_index('date'))

   
    if 'dish_name' in df.columns:
        st.subheader("Top Dishes")
        top_dishes = df.groupby('dish_name')['units_sold'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_dishes)


    if 'location' in df.columns:
        st.subheader("Top Locations")
        top_locs = df.groupby('location')['units_sold'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_locs)

    
 
    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes(include=['number'])
    if len(num_df.columns) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)

   
    st.subheader(" Demand Prediction")
    run_ml_model(df)
else:
    st.warning("Please upload a CSV file to begin.")
