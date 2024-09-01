# %%
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# %%
df = pd.read_csv("prep2.csv")


learn_col = [
    # "rent",
    # "administration",
    "total_rent",
    # "total_rent_per_m2",
    # "deposit",
    # "deposit_per_rent",
    # "gratuity",
    # "gratuity_per_rent",
    "madori",
    "menseki",
    "city",
    # "town",
    # "line",
    "nearest_station",
    "time",
    "building_age",
    "stories"
    ]

x_col = [
    # "rent",
    # "administration",
    # "total_rent",
    # "total_rent_per_m2",
    # "deposit",
    # "deposit_per_rent",
    # "gratuity",
    # "gratuity_per_rent",
    "madori",
    "menseki",
    "city",
    # "town",
    # "line",
    "nearest_station",
    "time",
    "building_age",
    "stories"
    ]

y_col = ["total_rent"]

learning_df = df[learn_col]

# %%
# encoding
objest_vars_list = learning_df.select_dtypes(include="object").columns.tolist()

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for cat in objest_vars_list:
    le = LabelEncoder()    
    learning_df[cat] = le.fit_transform(learning_df[cat])
    label_encoders[cat] = le

# %%
# TT分割
X_df = learning_df[x_col]
y_df =learning_df[y_col]
X_train, X_valid, y_train, y_valid = train_test_split(X_df, y_df, test_size=0.2)

# %%
# 学習
lgbm_params = {
    "objective" : "regression",
    "random_seed" : 1234
}

objest_vars_list_train = X_df.select_dtypes(include="object").columns.tolist()
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=objest_vars_list_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, categorical_feature=objest_vars_list_train)

model_lgb = lgb.train(
    lgbm_params,
    lgb_train,
    valid_sets= lgb_eval,
    num_boost_round=100,
    # early_stopping_rounds=20,
    # verbose_eval=10
    )

# %%
st.write("hello streamlit")

st.title("家賃の推定")
st.sidebar.title("設定")

# %%
city_unique_list = df["city"].unique().tolist()
input_cities = st.sidebar.selectbox("区を選択", city_unique_list)

city_station_dict = {}
for city in city_unique_list:
    station_list = df.query("city==@city")["nearest_station"].unique().tolist()
    city_station_dict[city] = station_list

if input_cities is not None:
    input_station = st.sidebar.selectbox("最寄り駅を選択", city_station_dict[input_cities])

input_minutes = st.sidebar.number_input('最寄り駅からの時間(分)', min_value=1)

# %%
input_menseki = st.sidebar.number_input('部屋の広さ(m2)', min_value=1)
input_building_age = st.sidebar.number_input('築年数(年)', min_value=1)
input_stories = st.sidebar.number_input('建物の高さ(階建)', min_value=1)

# %%
madori_unique_list = df["madori"].unique().tolist()
madori_unique_list.sort()
input_madori = st.sidebar.selectbox("間取りを選択", madori_unique_list)

# %%
test_data = pd.DataFrame(data={
    'madori': [input_madori],
    'menseki': [input_menseki],
    "city" : [input_cities],
    "nearest_station" : [input_station],
    "time" : [input_minutes],
    "building_age" : [input_building_age],
    "stories" : [input_stories]
})
# %%
# テストデータのencoding
for cat in objest_vars_list:
    le = label_encoders[cat]
    # テストデータに対してエンコーディングを適用
    test_data[cat] = le.transform(test_data[cat])

y_pred = model_lgb.predict(test_data, num_iteration=model_lgb.best_iteration)

st.write(y_pred)