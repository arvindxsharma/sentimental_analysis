import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

sentiment_pipeline = pipeline("sentiment-analysis")


st.title("Sentiment Analysis App")


uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    
    df = pd.read_csv(uploaded_file)

    
    st.write("Uploaded Data:")
    st.write(df)

    
    if 'review' in df.columns:
        
        if st.button("Analyze Sentiment"):
            
            df['predicted_sentiment'] = df['review'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
            
            
            st.write("Data with Sentiment Analysis:")
            st.write(df)
            
            
            sentiment_counts = df['predicted_sentiment'].value_counts()
            
            
            st.write("Sentiment Distribution:")
            fig, ax = plt.subplots()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
            
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(df)
            st.download_button(
                label="Download updated CSV",
                data=csv,
                file_name="sentiment_analysis_output.csv",
                mime="text/csv"
            )
    else:
        st.error("The uploaded CSV must contain a 'review' column for sentiment analysis.")
else:
    
    input_text = st.text_area("Or, enter text to analyze", placeholder="Type your text here...")

    
    if st.button("Analyze"):
        if input_text:
            
            results = sentiment_pipeline(input_text)
            
        
            st.write("Sentiment Analysis Results:")
            for result in results:
                label = result['label']
                score = result['score']
                st.write(f"Sentiment: {label}, Confidence: {score:.2f}")
        else:
            st.write("Please enter some text to analyze.")
