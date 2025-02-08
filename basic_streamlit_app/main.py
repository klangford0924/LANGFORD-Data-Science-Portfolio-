import streamlit as st
import pandas as pd

#loading the data set
@st.cache_data
def load_data():
    df = pd.read_csv("wnbadraft.csv") 
    return df

df = load_data()

#create filters
st.sidebar.header("Filter Draft Picks")
team = st.sidebar.selectbox("Select Team", ["All"] + sorted(df["team"].unique()))
year = st.sidebar.selectbox("Select Year", ["All"] + sorted(df["year"].unique(), reverse=True))
draft_pick = st.sidebar.selectbox("Select Draft Pick", ["All"] + sorted(df["overall_pick"].unique()))

#make the filters work
filtered_df = df.copy()
if team != "All":
    filtered_df = filtered_df[filtered_df["team"] == team]
if year != "All":
    filtered_df = filtered_df[filtered_df["year"] == year]
if draft_pick != "All":
    filtered_df = filtered_df[filtered_df["overall_pick"] == draft_pick]

#display the filters
st.title("WNBA Draft Picks Explorer")
st.write("This app shows you all the WNBA Draft picks from 1997 till 2022. You can filter this list based on team, year, and overall draft pick")
st.write(f"Showing results for **{team if team != 'All' else 'All Teams'}**, **{year if year != 'All' else 'All Years'}**, **{draft_pick if draft_pick != 'All' else 'All Picks'}**")
st.dataframe(filtered_df)

