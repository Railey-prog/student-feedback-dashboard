# import packages
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client  
client = openai.OpenAI()    

# Helper function to get dataset path
def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "data", "feedback student.csv")

# Function to get sentiment using GenAI
@st.cache_data
def get_sentiment(text):
    if not text or pd.isna(text):
        return "Neutral"
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify the sentiment of the following feedback as exactly one word: Positive, Negative, or Neutral."},
                {"role": "user", "content": f"What's the sentiment of this feedback? {text}"}
            ],
            temperature=0,
            max_tokens=10
        )
        # ✅ fixed way to access content
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"API error: {e}")
        return "Neutral"


st.title("🎓 Student Feedback Sentiment Dashboard")
st.write("Analyze student feedback using GenAI-powered sentiment classification.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Load Dataset"):
        try:
            csv_path = get_dataset_path()
            df = pd.read_csv(csv_path, encoding="latin1")
            st.session_state["df"] = df.head(50)  # load first 50 for demo
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🔍 Analyze Sentiment"):
        if "df" in st.session_state:
            try:
                with st.spinner("Analyzing sentiment..."):
                    st.session_state["df"].loc[:4, "Sentiment"] = st.session_state["df"]["comment"].head(5).apply(get_sentiment)
                    st.success("Sentiment analysis completed!")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please ingest the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    df = st.session_state["df"]

    # 🔍 Filter section (like "Filter by Product")
    if "Id" in df.columns:
        st.subheader("🔍 Filter by Student ID")
        selected_id = st.selectbox("Choose a Student ID", ["All IDs"] + list(df["Id"].unique()))
        st.subheader(f"📁 Feedback for {selected_id}")

        if selected_id != "All IDs":
            filtered_df = df[df["Id"] == selected_id]
        else:
            filtered_df = df
    else:
        st.warning("⚠️ No 'Id' column found in the dataset. Showing all data.")
        filtered_df = df

    # Show table of feedback
    st.dataframe(filtered_df[["Id", "comment"]])

    # 📊 Sentiment Breakdown
    if "Sentiment" in df.columns:
        st.subheader(f"📊 Sentiment Breakdown for {selected_id}")

        sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        # Define custom order and colors
        sentiment_order = ['Negative', 'Neutral', 'Positive']
        sentiment_colors = {'Negative': 'red', 'Neutral': 'lightgray', 'Positive': 'green'}

        # Only include categories that exist
        existing_sentiments = sentiment_counts['Sentiment'].unique()
        filtered_order = [s for s in sentiment_order if s in existing_sentiments]
        filtered_colors = {s: sentiment_colors[s] for s in existing_sentiments if s in sentiment_colors}

        # Reorder for nice plotting
        sentiment_counts['Sentiment'] = pd.Categorical(
            sentiment_counts['Sentiment'],
            categories=filtered_order,
            ordered=True
        )
        sentiment_counts = sentiment_counts.sort_values('Sentiment')

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title=f"Distribution of Sentiment Classifications - {selected_id}",
            labels={"Sentiment": "Sentiment Category", "Count": "Number of Feedback"},
            color="Sentiment",
            color_discrete_map=filtered_colors
        )
        fig.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Number of Feedback",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
