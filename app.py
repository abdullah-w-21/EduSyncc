import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import speedtest
from pathlib import Path
import requests
import json
from datetime import datetime, timedelta
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from groq import Groq
import PyPDF2
import io
import requests
import re
from urllib.parse import urlparse
import tempfile
import docx2txt
from bs4 import BeautifulSoup
import urllib.request
from io import BytesIO

# pages config
st.set_page_config(
    page_title="EduSyncc",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# init groq
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# init sessions
if 'analyzed_content' not in st.session_state:
    st.session_state.analyzed_content = {}
if 'curriculum_data' not in st.session_state:
    st.session_state.curriculum_data = None
if 'download_queue' not in st.session_state:
    st.session_state.download_queue = []


# db
def init_db():
    conn = sqlite3.connect('edusyncc.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS speed_tests
                 (timestamp TEXT, download REAL, upload REAL, time_of_day TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS content_analysis
                 (file_path TEXT, url TEXT, summary TEXT, learning_objectives TEXT,
                  grade_level TEXT, subjects TEXT, curriculum_match REAL,
                  prerequisites TEXT, estimated_duration TEXT,
                  size_mb REAL, priority REAL, timestamp TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS download_schedule
                 (file_path TEXT, url TEXT, scheduled_time TEXT, priority REAL, 
                  status TEXT, size_mb REAL, retry_count INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS curriculum
                 (grade_level TEXT, subject TEXT, topic TEXT, 
                  learning_objectives TEXT, prerequisites TEXT)''')
    conn.commit()
    conn.close()


init_db()


def extract_google_drive_id(url):
    """Extract file ID from Google Drive URL."""
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/d/([a-zA-Z0-9_-]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_from_google_drive(file_id):
    """Download file from Google Drive."""

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    return response.content


def extract_text_from_pdf(pdf_content):
    """Extract text from PDF content."""
    try:
        pdf_file = io.BytesIO(pdf_content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")


def extract_text_from_docx(docx_content):
    """Extract text from DOCX content."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(docx_content)
            temp_file.flush()
            text = docx2txt.process(temp_file.name)
        return text
    except Exception as e:
        raise Exception(f"DOCX extraction failed: {str(e)}")


def extract_text_from_webpage(url):
    """Extract text content from webpage."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove scripts, styles, and navigation elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        return soup.get_text(separator='\n')
    except Exception as e:
        raise Exception(f"Webpage extraction failed: {str(e)}")


def process_url_content(url):
    """Process content from various URL types."""
    try:
        parsed_url = urlparse(url)

        # Handle Google Drive links
        if 'drive.google.com' in parsed_url.netloc:
            file_id = extract_google_drive_id(url)
            if not file_id:
                raise Exception("Invalid Google Drive URL")
            content = download_from_google_drive(file_id)
        else:
            # direct download links
            response = requests.get(url, stream=True)
            content = response.content

        # content type and extract text
        content_type = response.headers.get('content-type', '').lower()

        if 'pdf' in content_type:
            return extract_text_from_pdf(content)
        elif 'word' in content_type or 'docx' in content_type:
            return extract_text_from_docx(content)
        elif 'text' in content_type:
            return content.decode('utf-8')
        elif 'html' in content_type:
            return extract_text_from_webpage(url)
        else:
            # guess format from URL if content-type is not helpful
            if url.lower().endswith('.pdf'):
                return extract_text_from_pdf(content)
            elif url.lower().endswith(('.doc', '.docx')):
                return extract_text_from_docx(content)
            elif url.lower().endswith(('.txt', '.text')):
                return content.decode('utf-8')
            else:
                # PDF first, then fall back to text
                try:
                    return extract_text_from_pdf(content)
                except:
                    return content.decode('utf-8', errors='ignore')

    except Exception as e:
        raise Exception(f"URL processing failed: {str(e)}")


def process_uploaded_file(file):
    """Process uploaded file content."""
    try:
        content = file.read()
        file_type = file.type if hasattr(file, 'type') else ''
        file_name = file.name.lower()

        if 'pdf' in file_type.lower() or file_name.endswith('.pdf'):
            # handle PDF
            if isinstance(content, str):
                content = content.encode('utf-8')
            text_content = extract_text_from_pdf(content)
        elif ('word' in file_type.lower() or
              file_name.endswith(('.doc', '.docx'))):
            # handle Word documents
            if isinstance(content, str):
                content = content.encode('utf-8')
            text_content = extract_text_from_docx(content)
        elif 'text' in file_type.lower() or file_name.endswith(('.txt', '.text')):
            # handle text files
            if isinstance(content, bytes):
                text_content = content.decode('utf-8', errors='ignore')
            else:
                text_content = content
        else:
            # try to decode as text if unknown type
            if isinstance(content, bytes):
                text_content = content.decode('utf-8', errors='ignore')
            else:
                text_content = content

        # clean the text content
        text_content = ' '.join(text_content.split())  # Remove extra whitespace
        if not text_content.strip():
            raise Exception("No text content could be extracted from the file")

        return text_content

    except Exception as e:
        raise Exception(f"File processing failed: {str(e)}")


def analyze_with_llm(content, curriculum_context):
    """Analyze content using Groq's Llama 3."""
    try:
        # ensure content is string and clean it
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')

        # clean and prepare content
        cleaned_content = ' '.join(content.split())  # rm whitespace
        if len(cleaned_content) > 4000:
            cleaned_content = cleaned_content[:4000] + "..."

        # check if content is empty
        if not cleaned_content.strip():
            raise Exception("No valid content to analyze")

        prompt = f"""As an educational content analyzer, analyze the following content in the context of the curriculum requirements. 

Curriculum Context:
{curriculum_context}

Content to Analyze:
{cleaned_content}

Analyze this educational content and provide a JSON response in the following format:
{{
    "summary": "Brief but comprehensive overview of the content",
    "learning_objectives": ["objective 1", "objective 2", ...],
    "grade_level": "Recommended grade level",
    "subjects": ["subject1", "subject2", ...],
    "prerequisites": ["prerequisite1", "prerequisite2", ...],
    "estimated_duration": "Estimated time to complete",
    "priority_score": number between 1-5,
    "curriculum_match_details": {{
        "alignment_score": number between 0-1,
        "gaps_identified": ["gap1", "gap2", ...],
        "strengths": ["strength1", "strength2", ...]
    }}
}}

Ensure the response is properly formatted JSON."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system",
                 "content": "You are an expert educational content analyzer. Provide analysis in valid JSON format only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        # etr and val json response
        result = response.choices[0].message.content.strip()

        # preamble handle
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = result[json_start:json_end]

        try:
            parsed_result = json.loads(result)
            # validate required fields
            required_fields = ['summary', 'learning_objectives', 'grade_level',
                               'subjects', 'prerequisites', 'estimated_duration',
                               'priority_score', 'curriculum_match_details']
            for field in required_fields:
                if field not in parsed_result:
                    raise Exception(f"Missing required field: {field}")
            return parsed_result
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from LLM: {str(e)}")

    except Exception as e:
        raise Exception(f"LLM analysis failed: {str(e)}")



def is_good_network_condition():
    """
    Check if current network conditions are suitable for downloading.
    Returns True if conditions are good, False otherwise.
    """
    try:
        # latest speed test result
        conn = sqlite3.connect('edusyncc.db')
        latest_test = pd.read_sql_query("""
            SELECT download, upload 
            FROM speed_tests 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, conn)
        conn.close()

        if latest_test.empty:
            # if no speed test data, run a quick test
            speed_test = speedtest.Speedtest()
            speed_test.get_best_server()
            download_speed = speed_test.download() / 1_000_000  # Convert to Mbps
            upload_speed = speed_test.upload() / 1_000_000
        else:
            download_speed = latest_test.iloc[0]['download']
            upload_speed = latest_test.iloc[0]['upload']

        # current hour for time-based decisions
        current_hour = datetime.now().hour

        # def thresholds
        MIN_DOWNLOAD_SPEED = 5.0  # Minimum 5 Mbps
        PEAK_HOURS = range(9, 18)  # 9 AM to 6 PM
        PEAK_MIN_SPEED = 10.0  # Higher threshold during peak hours

        # check conds
        if current_hour in PEAK_HOURS:
            return download_speed >= PEAK_MIN_SPEED
        else:
            return download_speed >= MIN_DOWNLOAD_SPEED

    except Exception as e:
        st.error(f"Error checking network conditions: {str(e)}")
        return False  # Default to False if there's an error


# net predictor
def train_network_predictor():
    conn = sqlite3.connect('edusyncc.db')
    df = pd.read_sql_query("SELECT * FROM speed_tests", conn)
    conn.close()

    if len(df) < 48:
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.weekday

    # Feature engineering
    df['peak_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)
    df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    X = df[['hour', 'day_of_week', 'peak_hours', 'weekend', 'download', 'upload']].values
    y = df['download'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump((model, scaler), 'network_predictor.joblib')
    return model, scaler


def predict_network_conditions(hours_ahead=24):
    if not os.path.exists('network_predictor.joblib'):
        return None

    model, scaler = joblib.load('network_predictor.joblib')

    # gen prediction data
    current_time = datetime.now()
    prediction_times = [current_time + timedelta(hours=i) for i in range(hours_ahead)]

    prediction_data = []
    for dt in prediction_times:
        prediction_data.append([
            dt.hour,
            dt.weekday(),
            1 if 9 <= dt.hour <= 17 else 0,
            1 if dt.weekday() >= 5 else 0,
            0,  # placeholder for current download
            0  # placeholder for current upload
        ])

    X_pred = scaler.transform(np.array(prediction_data))
    predictions = model.predict(X_pred)

    return pd.DataFrame({
        'timestamp': prediction_times,
        'predicted_speed': predictions
    })


# downloading manager
def download_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        file_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return True, file_size
    except Exception as e:
        return False, str(e)


def process_download_queue():
    conn = sqlite3.connect('edusyncc.db')
    queue_df = pd.read_sql_query("""
        SELECT * FROM download_schedule 
        WHERE status = 'pending' 
        ORDER BY priority DESC, scheduled_time ASC
    """, conn)

    for _, item in queue_df.iterrows():
        if is_good_network_condition():
            success, result = download_file(item['url'], item['file_path'])

            if success:
                c = conn.cursor()
                c.execute("""
                    UPDATE download_schedule 
                    SET status = 'completed', size_mb = ? 
                    WHERE file_path = ?
                """, (result / (1024 * 1024), item['file_path']))
            else:
                c = conn.cursor()
                c.execute("""
                    UPDATE download_schedule 
                    SET retry_count = retry_count + 1,
                        status = CASE WHEN retry_count >= 3 THEN 'failed' ELSE 'pending' END
                    WHERE file_path = ?
                """, (item['file_path'],))

            conn.commit()

    conn.close()


# curriculum savingt
def save_curriculum(curriculum_data):
    conn = sqlite3.connect('edusyncc.db')
    c = conn.cursor()

    c.execute("DELETE FROM curriculum")

    for grade in curriculum_data:
        for subject in curriculum_data[grade]:
            for topic, details in curriculum_data[grade][subject].items():
                c.execute("""
                    INSERT INTO curriculum 
                    VALUES (?, ?, ?, ?, ?)
                """, (grade, subject, topic,
                      json.dumps(details['objectives']),
                      json.dumps(details['prerequisites'])))

    conn.commit()
    conn.close()


def load_curriculum():
    conn = sqlite3.connect('edusyncc.db')
    df = pd.read_sql_query("SELECT * FROM curriculum", conn)
    conn.close()

    if df.empty:
        return None

    curriculum_data = {}
    for _, row in df.iterrows():
        if row['grade_level'] not in curriculum_data:
            curriculum_data[row['grade_level']] = {}
        if row['subject'] not in curriculum_data[row['grade_level']]:
            curriculum_data[row['grade_level']][row['subject']] = {}

        curriculum_data[row['grade_level']][row['subject']][row['topic']] = {
            'objectives': json.loads(row['learning_objectives']),
            'prerequisites': json.loads(row['prerequisites'])
        }

    return curriculum_data


# pg funcs
def curriculum_page():
    st.title("📚 Curriculum Management")

    if st.session_state.curriculum_data is None:
        st.session_state.curriculum_data = load_curriculum()

    with st.expander("Add New Curriculum Content"):
        col1, col2 = st.columns(2)
        with col1:
            grade_level = st.selectbox(
                "Grade Level",
                ["Elementary", "Middle School", "High School", "College"]
            )
            subject = st.text_input("Subject")

        with col2:
            topic = st.text_input("Topic")

        objectives = st.text_area("Learning Objectives (one per line)")
        prerequisites = st.text_area("Prerequisites (one per line)")

        if st.button("Add to Curriculum"):
            objectives_list = [obj.strip() for obj in objectives.split('\n') if obj.strip()]
            prerequisites_list = [pre.strip() for pre in prerequisites.split('\n') if pre.strip()]

            if st.session_state.curriculum_data is None:
                st.session_state.curriculum_data = {}

            if grade_level not in st.session_state.curriculum_data:
                st.session_state.curriculum_data[grade_level] = {}
            if subject not in st.session_state.curriculum_data[grade_level]:
                st.session_state.curriculum_data[grade_level][subject] = {}

            st.session_state.curriculum_data[grade_level][subject][topic] = {
                'objectives': objectives_list,
                'prerequisites': prerequisites_list
            }

            save_curriculum(st.session_state.curriculum_data)
            st.success("Curriculum updated!")

    if st.session_state.curriculum_data:
        st.subheader("Current Curriculum")
        for grade in st.session_state.curriculum_data:
            with st.expander(f"{grade}"):
                for subject in st.session_state.curriculum_data[grade]:
                    st.write(f"**{subject}**")
                    for topic, details in st.session_state.curriculum_data[grade][subject].items():
                        st.write(f"*{topic}*")
                        st.write("Learning Objectives:")
                        for obj in details['objectives']:
                            st.write(f"- {obj}")
                        st.write("Prerequisites:")
                        for pre in details['prerequisites']:
                            st.write(f"- {pre}")
                        st.write("---")


def content_analysis_page():
    st.title("📚 Content Analysis")

    if st.session_state.curriculum_data is None:
        st.warning("Please set up curriculum first in the Curriculum Management page.")
        return

    tab1, tab2 = st.tabs(["Upload Content", "Add URL"])

    with tab1:
        uploaded_file = st.file_uploader("Upload educational content", type=['txt', 'pdf', 'docx'])
        if uploaded_file:
            try:
                # process uploaded file (based on file types)
                content = process_uploaded_file(uploaded_file)
                analyze_content(content, uploaded_file.name, None)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    with tab2:
        url = st.text_input("Enter content URL")
        if url and st.button("Analyze URL"):
            try:
                # process URL content
                content = process_url_content(url)
                analyze_content(content, url.split('/')[-1], url)
            except Exception as e:
                st.error(f"Error processing URL: {str(e)}")


def analyze_content(content, filename, url=None):
    try:
        # ensure is string
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        content = ' '.join(content.split())

        if not content.strip():
            raise ValueError("No valid content to analyze")

        curriculum_context = json.dumps(st.session_state.curriculum_data, indent=2)

        with st.spinner("Analyzing content..."):
            analysis = analyze_with_llm(content, curriculum_context)

            if analysis:
                # save analysis results
                conn = sqlite3.connect('edusyncc.db')
                c = conn.cursor()
                try:
                    c.execute("""INSERT INTO content_analysis 
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                              (filename, url, analysis['summary'],
                               json.dumps(analysis['learning_objectives']),
                               analysis['grade_level'],
                               json.dumps(analysis['subjects']),
                               analysis['curriculum_match_details']['alignment_score'],
                               json.dumps(analysis['prerequisites']),
                               analysis['estimated_duration'],
                               0,  # size_mb will be updated when downloaded
                               analysis['priority_score'],
                               datetime.now().isoformat()))

                    # schedule download if URL provided
                    if url:
                        c.execute("""INSERT INTO download_schedule 
                                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                  (filename, url,
                                   (datetime.now() + timedelta(hours=1)).isoformat(),
                                   analysis['priority_score'],
                                   'pending', 0, 0))
                    conn.commit()
                except Exception as e:
                    st.error(f"Error saving analysis: {str(e)}")
                finally:
                    conn.close()

                # display results
                display_analysis_results(analysis)

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        raise

def display_analysis_results(analysis):
    st.success("Analysis complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Content Summary")
        st.write(analysis['summary'])

        st.markdown("### Learning Objectives")
        for obj in analysis['learning_objectives']:
            st.write(f"- {obj}")

    with col2:
        st.markdown("### Analysis Results")
        st.write(f"Grade Level: {analysis['grade_level']}")
        st.write(f"Subjects: {', '.join(analysis['subjects'])}")
        st.write(f"Estimated Duration: {analysis['estimated_duration']}")
        st.write(f"Priority Score: {analysis['priority_score']}/5")

        st.markdown("### Curriculum Alignment")
        st.write(f"Match Score: {analysis['curriculum_match_details']['alignment_score']:.2%}")

        st.markdown("#### Strengths")
        for strength in analysis['curriculum_match_details']['strengths']:
            st.write(f"- {strength}")

        st.markdown("#### Gaps Identified")
        for gap in analysis['curriculum_match_details']['gaps_identified']:
            st.write(f"- {gap}")


def network_monitor_page():
    st.title("🌐 Network Monitor & Download Manager")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Network Performance")
        if st.button("Run Speed Test"):
            with st.spinner("Testing network speed..."):
                try:
                    speed_test = speedtest.Speedtest()
                    download_speed = speed_test.download() / 1_000_000  # convert to Mbps
                    upload_speed = speed_test.upload() / 1_000_000

                    # save test results
                    conn = sqlite3.connect('edusyncc.db')
                    c = conn.cursor()
                    c.execute("""INSERT INTO speed_tests 
                               VALUES (?, ?, ?, ?)""",
                              (datetime.now().isoformat(), download_speed, upload_speed,
                               datetime.now().strftime('%H:%M')))
                    conn.commit()
                    conn.close()

                    # retrain the model
                    train_network_predictor()

                    st.success("Speed test completed!")
                    st.metric("Download Speed", f"{download_speed:.1f} Mbps")
                    st.metric("Upload Speed", f"{upload_speed:.1f} Mbps")
                except Exception as e:
                    st.error(f"Speed test failed: {str(e)}")

        # show speed history
        conn = sqlite3.connect('edusyncc.db')
        history_df = pd.read_sql_query("""
            SELECT timestamp, download, upload 
            FROM speed_tests 
            ORDER BY timestamp DESC 
            LIMIT 100
        """, conn)
        conn.close()

        if not history_df.empty:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            fig = px.line(history_df, x='timestamp', y=['download', 'upload'],
                          title='Network Speed History',
                          labels={'value': 'Speed (Mbps)', 'timestamp': 'Time'})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Download Queue")
        conn = sqlite3.connect('edusyncc.db')
        queue_df = pd.read_sql_query("""
            SELECT file_path, scheduled_time, priority, status, retry_count 
            FROM download_schedule 
            WHERE status != 'completed'
            ORDER BY priority DESC, scheduled_time ASC
        """, conn)
        conn.close()

        if not queue_df.empty:
            st.dataframe(queue_df)

            if st.button("Process Download Queue"):
                process_download_queue()
                st.success("Queue processed!")
        else:
            st.info("No pending downloads")

        # Show network predictions
        st.subheader("Network Predictions")
        predictions = predict_network_conditions(24)
        if predictions is not None:
            fig = px.line(predictions, x='timestamp', y='predicted_speed',
                          title='Predicted Network Speed (Next 24 Hours)')
            st.plotly_chart(fig, use_container_width=True)


def dashboard_page():
    st.title("📊 EduSyncc Dashboard")

    # summary metrics
    col1, col2, col3 = st.columns(3)

    conn = sqlite3.connect('edusyncc.db')

    with col1:
        content_count = pd.read_sql_query("""
            SELECT COUNT(*) as count FROM content_analysis
        """, conn).iloc[0]['count']
        st.metric("Total Content Items", content_count)

    with col2:
        pending_downloads = pd.read_sql_query("""
            SELECT COUNT(*) as count FROM download_schedule
            WHERE status = 'pending'
        """, conn).iloc[0]['count']
        st.metric("Pending Downloads", pending_downloads)

    with col3:
        avg_speed = pd.read_sql_query("""
            SELECT AVG(download) as avg_speed FROM speed_tests
            WHERE timestamp >= datetime('now', '-1 day')
        """, conn).iloc[0]['avg_speed']
        st.metric("Avg Download Speed (24h)", f"{avg_speed:.1f} Mbps")

    # content analysis summary
    st.subheader("Content Analysis Overview")
    content_df = pd.read_sql_query("""
        SELECT grade_level, subjects, curriculum_match, priority
        FROM content_analysis
        ORDER BY timestamp DESC
        LIMIT 10
    """, conn)

    if not content_df.empty:
        fig = px.scatter(content_df, x='curriculum_match', y='priority',
                         color='grade_level', title='Content Analysis Distribution',
                         labels={'curriculum_match': 'Curriculum Match Score',
                                 'priority': 'Priority Score'})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Network Optimization Insights")
    col1, col2 = st.columns(2)

    with col1:
        # best download times
        predictions = predict_network_conditions(24)
        if predictions is not None:
            best_times = predictions.nlargest(3, 'predicted_speed')
            st.write("Recommended Download Times:")
            for _, row in best_times.iterrows():
                st.write(f"- {row['timestamp'].strftime('%H:%M')} ({row['predicted_speed']:.1f} Mbps)")

    with col2:
        # download success rate
        download_stats = pd.read_sql_query("""
            SELECT status, COUNT(*) as count
            FROM download_schedule
            GROUP BY status
        """, conn)
        if not download_stats.empty:
            fig = px.pie(download_stats, values='count', names='status',
                         title='Download Status Distribution')
            st.plotly_chart(fig, use_container_width=True)

    conn.close()


# sidebar nav
st.sidebar.title("EduSyncc")
st.sidebar.markdown("---")

navigation = st.sidebar.radio(
    "Navigate to",
    ["Dashboard", "Curriculum Management", "Content Analysis", "Network Monitor"]
)

# main content
if navigation == "Dashboard":
    dashboard_page()
elif navigation == "Curriculum Management":
    curriculum_page()
elif navigation == "Content Analysis":
    content_analysis_page()
elif navigation == "Network Monitor":
    network_monitor_page()

# configuration and help sections
with st.sidebar.expander("Settings"):
    st.write("API Configuration")
    if not st.secrets.get("GROQ_API_KEY"):
        st.warning("Groq API key not configured")
        st.write("Add to ..streamlit/secrets.toml:")
        st.code("GROQ_API_KEY = 'your-key-here'")

    st.write("Storage Directory")
    if st.button("Initialize Database"):
        init_db()
        st.success("Database initialized!")

with st.sidebar.expander("Help"):
    st.write("""
    **Quick Start Guide:**
    1. Set up curriculum in Curriculum Management
    2. Upload or provide URLs for educational content
    3. Let the system analyze and optimize downloads
    4. Monitor network conditions and download progress

    For support, visit: [EduSyncc](https://github.com/abdullah-w-21/EduSyncc)
    """)


