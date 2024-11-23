import streamlit as st
import google.generativeai as genai
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langdetect import detect
from googletrans import Translator
from io import BytesIO
from fpdf import FPDF
import concurrent.futures
import json
import time
from textblob import TextBlob
import re

# Configure API Key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# App Configuration
st.set_page_config(page_title="Fast Email Storytelling AI", page_icon="⚡", layout="wide")
st.title("⚡ Lightning-Fast Email Storytelling AI")
st.write("Rapidly extract insights and generate professional responses from emails.")

# Sidebar for Features
st.sidebar.header("Settings")
enable_wordcloud = st.sidebar.checkbox("Generate Word Cloud")
enable_sentiment = st.sidebar.checkbox("Perform Sentiment Analysis")
enable_highlights = st.sidebar.checkbox("Highlight Key Phrases")
enable_response = st.sidebar.checkbox("Generate Suggested Response")
enable_export = st.sidebar.checkbox("Export Options (Text, JSON, PDF)")
enable_multilingual_translation = st.sidebar.checkbox("Enable Multilingual Translation")
enable_ner = st.sidebar.checkbox("Named Entity Recognition (NER)")
enable_priority_detection = st.sidebar.checkbox("Detect Email Priority Level")
enable_tone_detection = st.sidebar.checkbox("Detect Tone of Email")
enable_readability_score = st.sidebar.checkbox("Calculate Readability Score")
enable_grammar_check = st.sidebar.checkbox("Grammar and Style Check")

# Input Email Section
st.subheader("Input Email Content")
email_content = st.text_area("Paste your email content here:", height=200)

# Limit the length of the input to optimize performance (e.g., 1000 characters)
MAX_EMAIL_LENGTH = 1000

# Cache the AI responses to improve performance (to avoid repeated API calls)
@st.cache_data
def get_summary_from_api(email_content):
    try:
        start_time = time.time()
        model = genai.GenerativeModel("gemini-1.5-flash")
        summary_prompt = f"Summarize the email in a concise, actionable format:\n\n{email_content[:MAX_EMAIL_LENGTH]}"
        response = model.generate_content(summary_prompt)
        st.write(f"Summary generation took {time.time() - start_time:.2f} seconds.")
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return ""

@st.cache_data
def get_response_from_api(email_content):
    try:
        start_time = time.time()
        model = genai.GenerativeModel("gemini-1.5-flash")
        response_prompt = f"Draft a professional response to this email:\n\n{email_content[:MAX_EMAIL_LENGTH]}"
        response = model.generate_content(response_prompt)
        st.write(f"Response generation took {time.time() - start_time:.2f} seconds.")
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""

@st.cache_data
def get_highlights_from_api(email_content):
    try:
        start_time = time.time()
        model = genai.GenerativeModel("gemini-1.5-flash")
        highlight_prompt = f"Highlight key points and actions in this email:\n\n{email_content[:MAX_EMAIL_LENGTH]}"
        response = model.generate_content(highlight_prompt)
        st.write(f"Highlight generation took {time.time() - start_time:.2f} seconds.")
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating highlights: {e}")
        return ""

def generate_export_files(export_text, export_json):
    try:
        pdf_content = export_pdf(export_text)
        pdf_buffer = BytesIO(pdf_content)
        return pdf_buffer, export_text, export_json
    except Exception as e:
        st.error(f"Error generating export files: {e}")
        return None, export_text, export_json

def export_pdf(export_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    clean_text = export_text.replace('’', "'").replace('“', '"').replace('”', '"')  # Add more replacements as needed
    pdf.multi_cell(0, 10, clean_text)
    return pdf.output(dest='S').encode('latin1')

def get_sentiment_analysis(email_content):
    sentiment = TextBlob(email_content).sentiment
    polarity = sentiment.polarity
    sentiment_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    return sentiment_label, polarity

def get_ner(email_content):
    model = genai.GenerativeModel("gemini-1.5-flash")
    ner_prompt = f"Extract named entities such as names, dates, locations, and organizations from the following email:\n\n{email_content[:MAX_EMAIL_LENGTH]}"
    ner_response = model.generate_content(ner_prompt)
    return ner_response.text.strip()

def get_readability_score(email_content):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', email_content)  # Remove special characters for readability analysis
    words = text.split()
    sentence_count = len(re.findall(r'[.!?]', email_content))
    word_count = len(words)
    if sentence_count == 0:
        return 0
    avg_words_per_sentence = word_count / sentence_count
    return avg_words_per_sentence

def detect_email_priority(email_content):
    if 'urgent' in email_content.lower() or 'asap' in email_content.lower():
        return "High Priority"
    elif 'please respond by' in email_content.lower():
        return "Medium Priority"
    else:
        return "Low Priority"

def detect_email_tone(email_content):
    model = genai.GenerativeModel("gemini-1.5-flash")
    tone_prompt = f"Analyze the tone of the following email:\n\n{email_content[:MAX_EMAIL_LENGTH]}"
    tone_response = model.generate_content(tone_prompt)
    return tone_response.text.strip()

def grammar_and_style_check(email_content):
    model = genai.GenerativeModel("gemini-1.5-flash")
    grammar_prompt = f"Check the grammar and style of the following email and suggest improvements:\n\n{email_content[:MAX_EMAIL_LENGTH]}"
    grammar_response = model.generate_content(grammar_prompt)
    return grammar_response.text.strip()

if email_content and st.button("Generate Insights"):
    try:
        # Step 1: Detect and Translate Language (if necessary)
        detected_lang = "en"
        if enable_multilingual_translation:
            detected_lang = detect(email_content)
            if detected_lang != "en":
                st.info(f"Detected Language: {detected_lang.upper()} - Translating...")
                translator = Translator()
                email_content = translator.translate(email_content, src=detected_lang, dest="en").text

        # Step 2: Use concurrent futures to parallelize tasks
        with st.spinner("Generating insights..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_summary = executor.submit(get_summary_from_api, email_content)
                future_response = executor.submit(get_response_from_api, email_content) if enable_response else None
                future_highlights = executor.submit(get_highlights_from_api, email_content) if enable_highlights else None
                future_sentiment = executor.submit(get_sentiment_analysis, email_content) if enable_sentiment else None
                future_ner = executor.submit(get_ner, email_content) if enable_ner else None
                future_priority = executor.submit(detect_email_priority, email_content) if enable_priority_detection else None
                future_tone = executor.submit(detect_email_tone, email_content) if enable_tone_detection else None
                future_readability = executor.submit(get_readability_score, email_content) if enable_readability_score else None
                future_grammar_check = executor.submit(grammar_and_style_check, email_content) if enable_grammar_check else None

                summary = future_summary.result()
                response = future_response.result() if future_response else ""
                highlights = future_highlights.result() if future_highlights else ""
                sentiment_label, polarity = future_sentiment.result() if future_sentiment else ("", 0)
                ner = future_ner.result() if future_ner else ""
                priority = future_priority.result() if future_priority else ""
                tone = future_tone.result() if future_tone else ""
                readability = future_readability.result() if future_readability else 0
                grammar = future_grammar_check.result() if future_grammar_check else ""

        # Step 3: Display Results in a mobile-friendly layout
        st.subheader("AI Summary")
        st.write(summary)

        if enable_response:
            st.subheader("Suggested Response")
            st.write(response)

        if enable_highlights:
            st.subheader("Key Highlights")
            st.write(highlights)

        if enable_sentiment:
            st.subheader("Sentiment Analysis")
            st.write(f"**Sentiment:** {sentiment_label} (Polarity: {polarity:.2f})")

        if enable_ner:
            st.subheader("Named Entity Recognition (NER)")
            st.write(ner)

        if enable_priority_detection:
            st.subheader("Priority Level")
            st.write(priority)

        if enable_tone_detection:
            st.subheader("Tone Analysis")
            st.write(tone)

        if enable_readability_score:
            st.subheader("Readability Score")
            st.write(f"Average words per sentence: {readability:.2f}")

        if enable_grammar_check:
            st.subheader("Grammar and Style Check")
            st.write(grammar)

        # Step 4: Word Cloud (Optional)
        if enable_wordcloud:
            st.subheader("Word Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(email_content)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # Step 5: Export Options
        if enable_export:
            st.subheader("Export Results")
            export_text = f"Summary:\n{summary}\n\nResponse:\n{response}\n\nHighlights:\n{highlights}\n\nSentiment:\n{sentiment_label}\n\nPriority Level:\n{priority}\n\nTone:\n{tone}\n\nNER:\n{ner}\n\nReadability Score:\n{readability:.2f}\n\nGrammar & Style Check:\n{grammar}"
            export_json = {
                "summary": summary,
                "response": response,
                "highlights": highlights,
                "sentiment": sentiment_label,
                "priority": priority,
                "tone": tone,
                "ner": ner,
                "readability_score": readability,
                "grammar_check": grammar,
            }

            # Generate exportable formats and handle in threads
            pdf_buffer, export_text, export_json = generate_export_files(export_text, export_json)

            # Provide download buttons
            buffer_txt = BytesIO(export_text.encode("utf-8"))
            buffer_json = BytesIO(json.dumps(export_json, indent=4).encode("utf-8"))
            st.download_button("Download as Text", data=buffer_txt, file_name="analysis.txt", mime="text/plain")
            st.download_button("Download as JSON", data=buffer_json, file_name="analysis.json", mime="application/json")
            st.download_button("Download as PDF", data=pdf_buffer, file_name="analysis.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Paste email content and click 'Generate Insights' to start.")

# Footer
st.markdown("---")
st.write("⚡ Built for Speed | Powered by Generative AI | Streamlit")
