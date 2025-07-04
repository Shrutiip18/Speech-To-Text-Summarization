import streamlit as st
import speech_recognition as sr
import requests
import time
from deep_translator import GoogleTranslator
import graphviz
from dotenv import load_dotenv
import os

import PyPDF2
import docx
from io import StringIO

import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import defaultdict

SPEAKER_COLORS = ["#FFB6C1", "#ADD8E6", "#90EE90", "#FFFF99", "#FFA07A", "#D8BFD8"]
st.set_page_config(layout="wide")

load_dotenv()

API_TOKEN = os.getenv("ASSEMBLY_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

#Session state variables for File Tab
for var in ["file_transcribed_text", "file_summary", "file_translated_text", "file_translated_summary", "file_accuracy", "extracted_data","pdf_raw_text","pdf_summary","pdf_translated_summary","pdf_translated_text"]:
    if var not in st.session_state:
        st.session_state[var] = ""

#Session state variables for Real-Time Tab
for var in ["real_transcribed_text", "real_summary", "real_translated_text", "real_translated_summary"]:
    if var not in st.session_state:
        st.session_state[var] = ""

#Upload file to AssemblyAI
def upload_file(api_token, path):
    headers = {'authorization': api_token}
    response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=open(path, 'rb'))
    
    if response.status_code == 200:
        return response.json()["upload_url"]
    else:
        st.error(f"Upload Error: {response.status_code} - {response.text}")
        return None

# Function to transcribe uploaded audio
def create_transcript(api_token, audio_url):
    url = "https://api.assemblyai.com/v2/transcript"
    headers = {"authorization": api_token, "content-type": "application/json"}
    data = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "summarization": True,
        "summary_model": "informative",
        "summary_type": "bullets"
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        st.error(f"Transcript creation failed: {response.status_code} - {response.text}")
        return None
    
    transcript_id = response.json().get('id')

    if not transcript_id:
        st.error("Error retrieving transcript ID")
        st.json(response.json()) 
        return None

    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    with st.spinner("Transcribing..."):
        while True:
            transcription_result = requests.get(polling_endpoint, headers=headers).json()
            if transcription_result['status'] == 'completed':
                return transcription_result
            elif transcription_result['status'] == 'error':
                st.error(f"Transcription failed: {transcription_result['error']}")
                return None
            time.sleep(3)

# Function to calculate speech recognition accuracy
def calculate_recognition_accuracy(transcript):
    word_list = transcript.get("words", [])
    if not word_list:
        return 0.0

    total_confidence = sum(word["confidence"] for word in word_list)
    num_words = len(word_list)
    accuracy = (total_confidence / num_words) * 100 if num_words > 0 else 0
    return round(accuracy, 2)

# Function to summarize real time text
def get_summary_huggingface(text):
    if len(text) > 1024:
        text = text[:1024]
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"authorization": f"Bearer {HUGGINGFACE_API_KEY}", "content-type": "application/json"}
    payload = {
        "inputs": text,
        "parameters": {
            "min_length": 100,     # Increase this as needed
            "max_length": 300,     # Set a higher max length
            "do_sample": False     # Optional: deterministic output
        }
    }

    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 200:
        summary = response.json()[0]['summary_text']
        return summary
    else:
        st.error(f"Hugging Face Summarization Error: {response.status_code} - {response.text}")
        return None

# Function to translate text
def translate_text(text, target_lang="hi"):
    if not text:
        return "No text available for translation."
    
    translator = GoogleTranslator(source='auto', target=target_lang)
    chunk_size = 300
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translations = [translator.translate(chunk) for chunk in chunks]
    return " ".join(translations)

# Extract text from uploaded pdf/text file
def extract_text(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".txt":
        return StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    elif ext == ".pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif ext == ".docx":
        doc = docx.Document(uploaded_file)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format. Please use .txt, .pdf, or .docx")


def generate_flowchart(summary_text):
    if not summary_text:
        st.warning("No summary available to generate a flowchart!")
        return None

    graph = graphviz.Digraph()
    steps = summary_text.split(". ")  # Split summary into steps

    for i, step in enumerate(steps):
        graph.node(f"{i}", step[:40])  # Limit node text to 40 characters for readability
        if i > 0:
            graph.edge(f"{i-1}", f"{i}")  # Connect nodes sequentially

    return graph

def extract_statistical_data(text):
    if not text:
        return {}
    
    # Dictionary to store our extracted data
    extracted_data = {}
    
    # Extract percentages (e.g., "45%", "12.5 percent")
    percentage_pattern = r'(\d+(?:\.\d+)?)\s*%|\b(\d+(?:\.\d+)?)\s*percent\b'
    percentages = re.findall(percentage_pattern, text, re.IGNORECASE)
    if percentages:
        extracted_data["percentages"] = []
        for p in percentages:
            # Check which group matched and use that value
            value = p[0] if p[0] else p[1]
            # Look for context (what the percentage is about)
            context_pattern = r'([^.!?\n]*(?:\d+(?:\.\d+)?)\s*%|\b(?:\d+(?:\.\d+)?)\s*percent\b[^.!?\n]*)'
            context_matches = re.findall(context_pattern, text)
            for match in context_matches:
                if value in match or value + "%" in match:
                    # Extract potential category/label (words before the percentage)
                    label_pattern = r'(\b[A-Za-z\s]+)\s+(?:\w+\s+){0,3}' + re.escape(value) + r'\s*%|\b' + re.escape(value) + r'\s*percent\b'
                    label_match = re.search(label_pattern, match)
                    label = label_match.group(1).strip() if label_match else "Item " + str(len(extracted_data["percentages"]) + 1)
                    
                    extracted_data["percentages"].append({
                        "label": label,
                        "value": float(value),
                        "context": match.strip()
                    })
                    break
    
    # Extract numerical comparisons (e.g., "increased by 25", "decreased from 10 to 5")
    comparison_pattern = r'(increased|decreased|grew|declined|reduced|rose|fell)(?:\s+by|\s+to|\s+from)?\s+(\d+(?:\.\d+)?)'
    comparisons = re.findall(comparison_pattern, text, re.IGNORECASE)
    if comparisons:
        extracted_data["comparisons"] = []
        for direction, value in comparisons:
            # Look for context
            context_pattern = r'([^.!?\n]*(?:increased|decreased|grew|declined|reduced|rose|fell)(?:\s+by|\s+to|\s+from)?\s+\d+(?:\.\d+)?[^.!?\n]*)'
            context_matches = re.findall(context_pattern, text, re.IGNORECASE)
            for match in context_matches:
                if direction in match.lower() and value in match:
                    # Extract potential subject (what changed)
                    label_pattern = r'(\b[A-Za-z\s]+)\s+(?:\w+\s+){0,3}(?:increased|decreased|grew|declined|reduced|rose|fell)'
                    label_match = re.search(label_pattern, match, re.IGNORECASE)
                    label = label_match.group(1).strip() if label_match else "Item " + str(len(extracted_data["comparisons"]) + 1)
                    
                    # Try to find "from X to Y" pattern for more complete data
                    from_to_pattern = r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)'
                    from_to_match = re.search(from_to_pattern, match)
                    
                    if from_to_match:
                        extracted_data["comparisons"].append({
                            "label": label,
                            "from_value": float(from_to_match.group(1)),
                            "to_value": float(from_to_match.group(2)),
                            "direction": direction.lower(),
                            "context": match.strip()
                        })
                    else:
                        extracted_data["comparisons"].append({
                            "label": label,
                            "change_value": float(value),
                            "direction": direction.lower(),
                            "context": match.strip()
                        })
                    break
    
    # Extract numerical lists (e.g., "The scores were 85, 90, and 78")
    list_pattern = r'(\b[A-Za-z\s]+(?:is|are|were|was)(?:\s+\w+){0,3}[:]\s+(?:\d+(?:\.\d+)?(?:\s*,\s*|\s+and\s+)){2,}\d+(?:\.\d+)?)'
    list_matches = re.findall(list_pattern, text, re.IGNORECASE)
    
    if list_matches:
        extracted_data["lists"] = []
        for match in list_matches:
            # Extract the category/subject
            category_pattern = r'(\b[A-Za-z\s]+)(?:is|are|were|was)'
            category_match = re.search(category_pattern, match, re.IGNORECASE)
            category = category_match.group(1).strip() if category_match else "List " + str(len(extracted_data["lists"]) + 1)
            
            # Extract the values
            values_pattern = r'\d+(?:\.\d+)?'
            values = [float(x) for x in re.findall(values_pattern, match)]
            
            if values:
                extracted_data["lists"].append({
                    "category": category,
                    "values": values,
                    "context": match.strip()
                })
    
    return extracted_data

# Function to generate visualizations based on extracted data
def generate_visualizations(extracted_data):
    visualizations = []
    
    if "percentages" in extracted_data and extracted_data["percentages"]:
        # Create pie chart for percentages
        df = pd.DataFrame(extracted_data["percentages"])
        fig = px.pie(df, values='value', names='label', title='Percentage Distribution')
        visualizations.append(("Percentage Distribution (Pie Chart)", fig))
        
        # Create bar chart for percentages
        fig2 = px.bar(df, x='label', y='value', title='Percentage Values',
                     labels={'value': 'Percentage (%)', 'label': 'Category'})
        visualizations.append(("Percentage Distribution (Bar Chart)", fig2))
    
    if "comparisons" in extracted_data and extracted_data["comparisons"]:
        # Check if we have from/to values
        from_to_data = [item for item in extracted_data["comparisons"] if "from_value" in item and "to_value" in item]
        
        if from_to_data:
            df = pd.DataFrame(from_to_data)
            fig = go.Figure()
            
            for i, row in df.iterrows():
                fig.add_trace(go.Bar(
                    name=row['label'],
                    x=['Before', 'After'],
                    y=[row['from_value'], row['to_value']],
                    text=[f"{row['from_value']}", f"{row['to_value']}"],
                    textposition='auto'
                ))
            
            fig.update_layout(title='Before vs After Comparison', barmode='group')
            visualizations.append(("Before vs After Comparison", fig))
        
        # For items with just change values
        change_data = [item for item in extracted_data["comparisons"] if "change_value" in item]
        if change_data:
            df = pd.DataFrame(change_data)
            # Apply direction to make increases positive and decreases negative
            df['adjusted_value'] = df.apply(
                lambda x: x['change_value'] if x['direction'] in ['increased', 'grew', 'rose'] else -x['change_value'], 
                axis=1
            )
            
            fig = px.bar(df, x='label', y='adjusted_value', 
                        title='Changes in Values',
                        labels={'adjusted_value': 'Change Amount', 'label': 'Category'},
                        color='direction')
            visualizations.append(("Value Changes", fig))
    
    if "lists" in extracted_data and extracted_data["lists"]:
        for i, list_item in enumerate(extracted_data["lists"]):
            df = pd.DataFrame({
                'Index': range(1, len(list_item['values']) + 1),
                'Value': list_item['values']
            })
            
            fig = px.line(df, x='Index', y='Value', 
                         title=f"{list_item['category']} Values",
                         markers=True)
            
            visualizations.append((f"{list_item['category']} Trend", fig))
            
            # Also create a bar chart
            fig2 = px.bar(df, x='Index', y='Value',
                         title=f"{list_item['category']} Values")
            visualizations.append((f"{list_item['category']} Distribution", fig2))
    
    return visualizations

def display_speaker_transcript(utterances):
    speaker_colors_map = {}
    scrollable_content = "" 
    for utt in utterances:
        speaker_id = utt['speaker']
        speaker_name = f"Speaker {speaker_id}"
        if speaker_id not in speaker_colors_map:
            color = SPEAKER_COLORS[len(speaker_colors_map) % len(SPEAKER_COLORS)]
            speaker_colors_map[speaker_id] = color
        else:
            color = speaker_colors_map[speaker_id]

        scrollable_content += (
            f"<div style='background-color:{color}; padding:10px; border-radius:10px; margin-bottom:5px;'>"
            f"<strong>{speaker_name}:</strong> {utt['text']}</div>"
        )
        st.markdown(
        f"<div style='height: 300px; overflow-y: scroll; border: 1px solid #000000; padding: 10px; border-radius: 10px; background-color: #ffffff;'>"
        f"{scrollable_content}"
        f"</div>",
        unsafe_allow_html=True
    )
        
# Show pie chart for speaking time
def speaker_time_distribution(utterances):
    speaker_durations = defaultdict(float)
    for utt in utterances:
        duration = utt['end'] - utt['start']
        speaker_durations[utt['speaker']] += duration
    labels = [f"Speaker {s}" for s in speaker_durations]
    values = list(speaker_durations.values())
    fig, ax = plt.subplots(figsize=(2,2))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, textprops={'fontsize': 8})
    ax.axis("equal")
    plt.tight_layout()
    st.subheader("🗣️ Talk Time Analysis")
    st.pyplot(fig, use_container_width=False)

# Main function to render diarization features

def render_diarized_transcript(transcript):
    if 'utterances' not in transcript:
        return
    utterances = transcript['utterances']
    # st.subheader("📃 Speaker-wise Transcript")
    # display_speaker_transcript(utterances)
    speaker_time_distribution(utterances)




def scrollable_textbox(content, height=300):
    st.markdown(f"""
        <div style="border:1px solid #ffffff; padding:10px;border-radius: 10px; height:{height}px; overflow-y:scroll; background-color:#000000">
            {content}
        </div>
    """, unsafe_allow_html=True)

# Streamlit UI
st.title("🎙 Speech Transcription & Summarization App")

tab1, tab2, tab3 = st.tabs(["📂 Upload File", "🎤 Real-Time Transcription","📊 Data Visualization"])

# ====== File Upload Section ======
with tab1:
    uploaded_file = st.file_uploader("Upload an Audio or Video File", type=["mp3", "wav", "mp4", "m4a"])
    st.subheader("OR")
    uploaded_doc = st.file_uploader("Upload a Text, PDF, or DOCX file", type=["txt", "pdf", "docx"])

    if uploaded_file is not None:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success("File Uploaded Successfully!")

        if st.button("Start File Transcription"):
            upload_url = upload_file(API_TOKEN, file_path)

            if upload_url:
                transcript = create_transcript(API_TOKEN, upload_url)

                if transcript:
                    st.session_state.file_transcribed_text = transcript["text"]
                    st.session_state.file_summary = transcript["summary"]
                    st.session_state.file_accuracy = calculate_recognition_accuracy(transcript)

                    # Extract statistical data
                    st.session_state.extracted_data = extract_statistical_data(transcript["text"])

                    if "utterances" in transcript:
                        render_diarized_transcript(transcript)
                        

    # Display File Transcription
        if st.session_state.file_transcribed_text:

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📝 Transcribed Text:")
                scrollable_textbox(st.session_state.file_transcribed_text, height=400)

            with col2:
                st.subheader("📌 Summary:")
                st.write(st.session_state.file_summary)

            #Recognition Accuracy:
            st.subheader("🎯 Speech Recognition Accuracy:")
            st.write(f"{st.session_state.file_accuracy}%")

            #Translation Section for file transcription
            if st.session_state.file_summary or st.session_state.file_transcribed_text:
                st.subheader("🌍 Translation Options")

                lang_options = {"Hindi": "hi", "French": "fr", "Spanish": "es", "German": "de"}
                selected_lang = st.selectbox("Choose Translation Language:", list(lang_options.keys()))

                if st.button("Translate Text & Summary"):
                    col1, col2 = st.columns(2)
                    #Display translated file text
                    with col1:
                        if st.session_state.file_transcribed_text:
                            st.session_state.file_translated_text = translate_text(st.session_state.file_transcribed_text, lang_options[selected_lang])
                            if st.session_state.file_translated_text:
                                st.subheader("📝 Translated File Transcription:")
                                scrollable_textbox(st.session_state.file_translated_text, height=350)

                    #Display translated file summary
                    with col2:
                        if st.session_state.file_summary:
                            st.session_state.file_translated_summary = translate_text(st.session_state.file_summary, lang_options[selected_lang])
                            if st.session_state.file_translated_summary:
                                st.subheader("📌 Translated File Summary:")
                                st.write(st.session_state.file_translated_summary)

            #Flowchart Section for file summary                 
            if st.session_state.file_summary:
                st.title("📌 Summary Flowchart Generator")
                st.session_state.file_summary_text = st.text_area("Enter or Paste Summary:", "")
                if st.button("Generate Flowchart"):
                    flowchart = generate_flowchart(st.session_state.file_summary_text)
                    if flowchart:
                        st.graphviz_chart(flowchart) 

    if uploaded_doc:
    # Extract button outside any conditionals
        extract_button = st.button("Extract and Summarize Text")
        
        # This section runs on extract button or if we already have data
        if extract_button:
            try:
                st.session_state.pdf_raw_text = extract_text(uploaded_doc)
                st.session_state.pdf_summary = get_summary_huggingface(st.session_state.pdf_raw_text)
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Always show the extracted text and summary if they exist in session state
        if st.session_state.pdf_raw_text or st.session_state.pdf_summary:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📄 Extracted Text")
                scrollable_textbox(st.session_state.pdf_raw_text, height=250)
            with col2:
                st.subheader("📌 Summary:")
                st.write(st.session_state.pdf_summary)
            
            # Translation section
            st.subheader("🌍 Translation Options")
            lang_options = {"Hindi": "hi", "French": "fr", "Spanish": "es", "German": "de"}
            selected_lang = st.selectbox("Choose Translation Language:", list(lang_options.keys()), key="pdf_lang_select")
            
            translate_button = st.button("Translate PDF Text & Summary", key="pdf_translate_btn")
            
            # This will run on translate button OR if we already have translations
            if translate_button:
                if st.session_state.pdf_raw_text:
                    st.session_state.pdf_translated_text = translate_text(
                        st.session_state.pdf_raw_text, lang_options[selected_lang]
                    )
                if st.session_state.pdf_summary:
                    st.session_state.pdf_translated_summary = translate_text(
                        st.session_state.pdf_summary, lang_options[selected_lang]
                    )
            
            # Always show translations if they exist in session state
            if st.session_state.pdf_translated_text or st.session_state.pdf_translated_summary:
                col1, col2 = st.columns(2)
                with col1:
                    if st.session_state.pdf_translated_text:
                        st.subheader("📝 Translated File Transcription:")
                        scrollable_textbox(st.session_state.pdf_translated_text, height=350)
                with col2:
                    if st.session_state.pdf_translated_summary:
                        st.subheader("📌 Translated File Summary:")
                        st.write(st.session_state.pdf_translated_summary)
            
            # Flowchart Section
            if st.session_state.pdf_summary:
                st.subheader("📌 Summary Flowchart Generator")
                summary_text_input = st.text_area(
                    "Enter or Paste Summary:", 
                    value=st.session_state.pdf_summary,
                    key="pdf_flowchart_summary"
                )
                
                if st.button("Generate PDF Flowchart", key="pdf_flowchart_btn"):
                    flowchart = generate_flowchart(summary_text_input)
                    if flowchart:
                        st.graphviz_chart(flowchart)

        



# ====== Real-Time Transcription Section ======
with tab2:
    st.write("Click the button and start speaking...")

    recognizer = sr.Recognizer()

    if st.button("Start Listening"):
        with sr.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=1.0)
            st.write("Listening...")

            try:
                audio = recognizer.listen(mic)
                text = recognizer.recognize_google(audio)
                st.session_state.real_transcribed_text += " " + text  # Append new text
                st.success("Transcription Successful!")
            except sr.UnknownValueError:
                st.error("Could not understand the audio. Please try again.")
            except sr.RequestError:
                st.error("Google API is not reachable. Check your internet connection.")

    # Display real-time transcribed text
    if st.session_state.real_transcribed_text:
        st.subheader("📝 Transcribed Text:")
        st.write(st.session_state.real_transcribed_text)

    # #Translate Recognized Text
    # if st.session_state.real_transcribed_text:
    #     st.subheader("🌍 Translation Options")
    #     lang_options = {"Hindi": "hi", "French": "fr", "Spanish": "es", "German": "de"}
    #     selected_lang = st.selectbox("Choose Translation Language:", list(lang_options.keys()))

    #     if st.button("Translate Text"):
    #         st.session_state.real_translated_text = translate_text(st.session_state.real_transcribed_text, lang_options[selected_lang])

    #         if st.session_state.real_translated_text:
    #             st.subheader("📝 Translated Real-Time Transcription:")
    #             st.write(st.session_state.real_translated_text)

    # Summarization for real-time transcription
    if st.session_state.real_transcribed_text:
        if st.button("Summarize Speech"):
            summary = get_summary_huggingface(st.session_state.real_transcribed_text)
            if summary:
                st.session_state.real_summary = summary


    # Display Summary
    if st.session_state.real_summary:
        st.subheader("📌 Summary:")
        st.write(st.session_state.real_summary)

        #Translate Summarized Text
        st.subheader("🌍 Translation Options")
        lang_options = {"Hindi": "hi", "French": "fr", "Spanish": "es", "German": "de"}
        selected_lang = st.selectbox("Choose Translation Language:", list(lang_options.keys()))

        if st.button("Translate Summary"):
            st.session_state.real_translated_summary = translate_text(st.session_state.real_summary, lang_options[selected_lang])

            if st.session_state.real_translated_summary:
                st.subheader("📌 Translated Real-Time Summary:")
                st.write(st.session_state.real_translated_summary)

    if st.session_state.real_summary:
        st.title("📌 Summary Flowchart Generator")
        st.session_state.realtime_summary_text = st.text_area("Enter or Paste Summary:", "")
        if st.button("Generate Flowchart"):
            flowchart = generate_flowchart(st.session_state.realtime_summary_text)
            if flowchart:
                st.graphviz_chart(flowchart)

with tab3:
    st.subheader("📊 Statistical Data Visualization")
    
    # Button to manually trigger data extraction
    if st.button("Extract Statistical Data"):
        text_to_analyze = st.session_state.file_transcribed_text or st.session_state.transcribed_text
        if text_to_analyze:
            st.session_state.extracted_data = extract_statistical_data(text_to_analyze)
            # st.success("Statistical data extracted successfully!")
        else:
            st.warning("No transcript available for analysis.")
    
    # Display extracted data details
    if st.session_state.extracted_data:
        # Generate visualizations
        visualizations = generate_visualizations(st.session_state.extracted_data)
        
        if visualizations:
            for title, fig in visualizations:
                st.subheader(title)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No statistical data found that can be visualized.")
            
        # Display raw extracted data
        if st.checkbox("Show Raw Extracted Data"):
            st.json(st.session_state.extracted_data)
    else:
        st.info("No statistical data has been extracted yet. Try transcribing some content with numerical information first.")


