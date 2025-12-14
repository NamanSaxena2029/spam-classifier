import streamlit as st
import pickle
import re
import pandas as pd
from datetime import datetime
import json
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Spam Classifier Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {'spam': 0, 'ham': 0, 'total': 0}
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Dark mode colors
if st.session_state.dark_mode:
    bg_color = "#1e1e1e"
    text_color = "#ffffff"
    card_bg = "#2d2d2d"
    card_text = "#ffffff"
    input_bg = "#2d2d2d"
    input_text = "#ffffff"
    border_color = "#667eea"
else:
    bg_color = "#f5f7fa"
    text_color = "#1a1a1a"
    card_bg = "#ffffff"
    card_text = "#1a1a1a"
    input_bg = "#ffffff"
    input_text = "#1a1a1a"
    border_color = "#667eea"

st.markdown(f"""
<style>
/* App Background */
.stApp {{
    background-color: {bg_color} !important;
}}

/* ============================= */
/* FORCE MAIN CONTENT VISIBILITY */
/* ============================= */
.main h1, .main h2, .main h3, .main h4,
.main p, .main span, .main li, .main a,
.main div, .main ul, .main ol, .main code,
.main label {{
    color: {text_color} !important;
    opacity: 1 !important;
    visibility: visible !important;
    -webkit-text-fill-color: {text_color} !important;
}}

/* FIX Streamlit TITLE FADE BUG (IMPORTANT) */
[data-testid="stMarkdownContainer"] h1 {{
    color: {text_color} !important;
    opacity: 1 !important;
    -webkit-text-fill-color: {text_color} !important;
}}

/* Force label colors */
label, .stMarkdown label {{
    color: {text_color} !important;
    -webkit-text-fill-color: {text_color} !important;
}}

/* ================= */
/* SIDEBAR       */
/* ================= */
[data-testid="stSidebar"] {{
    background-color: {card_bg} !important;
}}

[data-testid="stSidebar"] *,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: {card_text} !important;
    -webkit-text-fill-color: {card_text} !important;
}}

/* ================= */
/* STAT CARDS        */
/* ================= */
.stat-card {{
    background: {card_bg} !important;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
    margin: 10px 0;
    border: 2px solid #667eea;
}}

.stat-card h2,
.stat-card h3,
.stat-card p {{
    color: {card_text} !important;
    opacity: 1 !important;
}}

/* ================= */
/* HISTORY ITEMS     */
/* ================= */
.history-item {{
    background: {card_bg} !important;
    color: {card_text} !important;
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
    border-left: 4px solid;
    font-size: 14px;
}}

.history-item strong,
.history-item small {{
    color: {card_text} !important;
}}

.history-spam {{ border-left-color: #f44336; }}
.history-ham {{ border-left-color: #4CAF50; }}

/* ================= */
/* INPUTS / TEXTAREA */
/* ================= */
textarea,
.stTextArea textarea,
.stTextInput input,
input {{
    color: {input_text} !important;
    background-color: {input_bg} !important;
    caret-color: {input_text} !important;
    border: 2px solid {border_color} !important;
    border-radius: 10px !important;
    -webkit-text-fill-color: {input_text} !important;
    opacity: 1 !important;
}}

textarea::placeholder,
input::placeholder {{
    color: #6b7280 !important;
    opacity: 0.6 !important;
    -webkit-text-fill-color: #6b7280 !important;
}}

/* ================= */
/* FILE UPLOADER     */
/* ================= */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div {{
    color: {text_color} !important;
    -webkit-text-fill-color: {text_color} !important;
}}

/* ================= */
/* BUTTONS           */
/* ================= */
.stButton > button {{
    width: 100% !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    height: 3em !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    border: none !important;
    transition: all 0.3s ease !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(102,126,234,0.4) !important;
}}

/* ================= */
/* SHARE BUTTONS     */
/* ================= */
.share-btn {{
    display: inline-block;
    padding: 10px 20px;
    margin: 5px;
    border-radius: 8px;
    color: white !important;
    text-decoration: none;
    font-weight: bold;
    transition: transform 0.2s;
}}

.share-btn:hover {{ transform: scale(1.05); }}

.whatsapp {{ background: #25D366; }}
.email {{ background: #EA4335; }}
.copy {{ background: #1DA1F2; }}

/* ================= */
/* TABS          */
/* ================= */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    color: {text_color} !important;
    opacity: 1 !important;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: #667eea !important;
}}

/* ================= */
/* METRICS           */
/* ================= */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {{
    color: {text_color} !important;
}}

/* ================= */
/* EXPANDER          */
/* ================= */
.streamlit-expanderHeader,
.streamlit-expanderHeader p {{
    color: {text_color} !important;
}}

/* ================= */
/* ALERTS            */
/* ================= */
.stAlert,
.stAlert p,
.stAlert div,
.stAlert span {{
    color: {text_color} !important;
    -webkit-text-fill-color: {text_color} !important;
}}

/* ================= */
/* DATAFRAME         */
/* ================= */
.dataframe,
.dataframe * {{
    color: {text_color} !important;
}}

/* ================= */
/* ANIMATION         */
/* ================= */
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.result-animation {{
    animation: fadeIn 0.5s ease-in;
}}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def predict_spam(text, model, vectorizer):
    clean_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([clean_text])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    label = "SPAM" if prediction == 1 else "HAM"
    confidence = float(probability[prediction] * 100)
    return label, confidence

def add_to_history(message, label, confidence):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.insert(0, {
        'time': timestamp,
        'message': message[:50] + "..." if len(message) > 50 else message,
        'label': label,
        'confidence': confidence
    })
    if len(st.session_state.history) > 10:
        st.session_state.history.pop()

    if label == "SPAM":
        st.session_state.stats['spam'] += 1
    else:
        st.session_state.stats['ham'] += 1
    st.session_state.stats['total'] += 1

def create_gauge_chart(confidence, label):
    color = "#f44336" if label == "SPAM" else "#4CAF50"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{label} Confidence", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#e8f5e9'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def get_download_link(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}" class="share-btn copy">📥 Download Report</a>'

def create_stats_chart():
    if st.session_state.stats['total'] == 0:
        return None

    labels = ['HAM', 'SPAM']
    values = [st.session_state.stats['ham'], st.session_state.stats['spam']]
    colors = ['#4CAF50', '#f44336']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker_colors=colors
    )])

    fig.update_layout(
        title_text="Today's Classification Stats",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# Main App
def main():
    global text_color, card_text 

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f'<h1 style="text-align: center; color: {text_color} !important;">🛡️ Spam Classifier Pro</h1>', unsafe_allow_html=True)
    with col2:
        if st.button("🌓 Dark Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown("---")

    model, vectorizer = load_model()
    if model is None:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown(f'<h2 style="color: {card_text};">📊 Dashboard</h2>', unsafe_allow_html=True)

        # Stats
        st.markdown(f"""
        <div class="stat-card">
            <h2 style="color: #667eea; margin: 0;">{st.session_state.stats['total']}</h2>
            <p style="margin: 5px 0;">Total Checks</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="color: #4CAF50; margin: 0;">{st.session_state.stats['ham']}</h3>
                <p style="margin: 5px 0; font-size: 12px;">HAM</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="color: #f44336; margin: 0;">{st.session_state.stats['spam']}</h3>
                <p style="margin: 5px 0; font-size: 12px;">SPAM</p>
            </div>
            """, unsafe_allow_html=True)

        # Stats Chart
        if st.session_state.stats['total'] > 0:
            st.markdown("### 📈 Distribution")
            fig = create_stats_chart()
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.header("📜 Recent History")
        if st.session_state.history:
            for item in st.session_state.history[:5]:
                emoji = "🚫" if item['label'] == "SPAM" else "✅"
                css_class = "history-spam" if item['label'] == "SPAM" else "history-ham"
                st.markdown(f"""
                <div class="history-item {css_class}">
                    <strong>{emoji} {item['time']}</strong><br>
                    {item['message']}<br>
                    <small>{item['confidence']:.1f}% confidence</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No checks yet!")

        st.markdown("---")

        # Model Info
        with st.expander("ℹ️ Model Info"):
            st.write("**Algorithm:** Naive Bayes")
            st.write("**Accuracy:** 97.76%")
            st.write("**Dataset:** 5,572+ messages")

    # Main Content - Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Check", "📁 Bulk Check", "📊 Analytics"])

    # TAB 1: Single Check
    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f'<h3 style="color: {text_color};">✉️ Enter Message</h3>', unsafe_allow_html=True)

            if 'default_message' not in st.session_state:
                st.session_state.default_message = ""

            user_input = st.text_area(
                "Type or paste your message:",
                value=st.session_state.default_message,
                height=200,
                placeholder="Example: WINNER! You've won $1000! Click here now...",
            )

            # Classify button with progress
            if st.button("🔍 Classify Message", type="primary"):
                if user_input.strip():
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)

                    label, confidence = predict_spam(user_input, model, vectorizer)
                    add_to_history(user_input, label, confidence)

                    st.session_state.result = {
                        'label': label,
                        'confidence': confidence,
                        'message': user_input
                    }
                    progress_bar.empty()
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter a message!")

        with col2:
            # *** MODIFIED CODE: Used 'text_color' for tab header visibility ***
            st.markdown(f'<h3 style="color: {text_color};">🧪 Quick Examples</h3>', unsafe_allow_html=True)

            examples = {
                "✅ Legitimate": "Hey, are we still meeting for lunch tomorrow?",
                "🚫 Spam": "WINNER!! You won $1000! Call now!",
                "💼 Work": "Please review the attached document and send feedback.",
                "🎁 Promo Spam": "FREE iPhone! Click here NOW to win!"
            }

            for title, example in examples.items():
                if st.button(title, use_container_width=True, key=f"example_{title}"):
                    st.session_state.default_message = example
                    st.rerun()

        # Display Results
        if 'result' in st.session_state:
            st.markdown("---")
            st.markdown('<div class="result-animation">', unsafe_allow_html=True)

            # *** MODIFIED CODE: Used 'text_color' for result header visibility ***
            st.markdown(f'<h3 style="color: {text_color};">🎯 Classification Result</h3>', unsafe_allow_html=True)

            result = st.session_state.result

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Your Message:**")

                if st.session_state.dark_mode:
                    message_box_bg = "#1a237e"
                    message_box_border = "#667eea"
                    message_box_text = "#ffffff"
                else:
                    message_box_bg = "#e3f2fd"
                    message_box_border = "#2196F3"
                    message_box_text = "#0d47a1"

                st.markdown(f"""
                <div style="background-color: {message_box_bg}; padding: 15px; border-radius: 10px; border-left: 4px solid {message_box_border};">
                    <p style="color: {message_box_text}; margin: 0; font-size: 16px; line-height: 1.5; font-weight: 500;">{result['message']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Share buttons
                st.markdown("**📤 Share Result:**")
                message_encoded = result['message'].replace(' ', '%20')
                whatsapp_link = f"https://wa.me/?text=Message: {message_encoded}%0AResult: {result['label']} ({result['confidence']:.1f}%)"
                email_link = f"mailto:?subject=Spam Check Result&body=Message: {result['message']}%0AResult: {result['label']} ({result['confidence']:.1f}%)"

                st.markdown(f"""
                <a href="{whatsapp_link}" target="_blank" class="share-btn whatsapp">📱 WhatsApp</a>
                <a href="{email_link}" class="share-btn email">📧 Email</a>
                """, unsafe_allow_html=True)

                if st.button("📋 Copy to Clipboard"):
                    st.success("✅ Copied! (Use Ctrl+C manually)")

                # Download report
                report_text = f"""SPAM CLASSIFICATION REPORT
{'='*50}
Message: {result['message']}
Classification: {result['label']}
Confidence: {result['confidence']:.2f}%
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*50}
"""
                st.markdown(get_download_link(report_text, f"spam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"), unsafe_allow_html=True)

            with col2:
                # Gauge chart
                fig = create_gauge_chart(result['confidence'], result['label'])
                st.plotly_chart(fig, use_container_width=True)

                if result['label'] == "SPAM":
                    st.error("🚫 **SPAM DETECTED**")
                else:
                    st.success("✅ **LEGITIMATE**")

            # Expandable info
            if st.button("💡 What does this mean? (Click to view)", use_container_width=True):
                st.info("""
                **SPAM**: Suspicious message - likely unwanted advertising, scam, or phishing

                **LEGITIMATE (HAM)**: Normal message - appears to be genuine communication

                **Confidence**: How certain the model is (higher = more confident)
                """)

            st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: Bulk Check
    with tab2:
        # *** MODIFIED CODE: Used 'text_color' for tab header visibility ***
        st.markdown(f'<h3 style="color: {text_color};">📁 Bulk Message Classification</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Option 1: Upload File**")
            uploaded_file = st.file_uploader("Upload CSV or TXT file", type=['csv', 'txt'])

            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if 'message' in df.columns:
                        messages = df['message'].tolist()
                    else:
                        messages = df.iloc[:, 0].tolist()
                else:
                    content = uploaded_file.read().decode('utf-8')
                    messages = content.split('\n')

                if st.button("🚀 Classify All"):
                    results = []
                    progress = st.progress(0)

                    for i, msg in enumerate(messages):
                        if msg.strip():
                            label, conf = predict_spam(msg, model, vectorizer)
                            results.append({'Message': msg, 'Classification': label, 'Confidence': f"{conf:.2f}%"})
                        progress.progress((i + 1) / len(messages))

                    result_df = pd.DataFrame(results)
                    st.dataframe(result_df, use_container_width=True)

                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "bulk_classification_results.csv",
                        "text/csv"
                    )

        with col2:
            st.write("**Option 2: Paste Multiple Messages**")
            bulk_text = st.text_area("One message per line:", height=200, key="bulk_text_area")

            if st.button("🔍 Check All", key="bulk_check_btn"):
                if bulk_text.strip():
                    messages = [m.strip() for m in bulk_text.split('\n') if m.strip()]
                    results = []

                    for msg in messages:
                        label, conf = predict_spam(msg, model, vectorizer)
                        results.append({'Message': msg[:50]+"...", 'Result': label, 'Confidence': f"{conf:.1f}%"})

                    st.dataframe(pd.DataFrame(results), use_container_width=True)

    # TAB 3: Analytics
    with tab3:
        # *** MODIFIED CODE: Used 'text_color' for tab header visibility ***
        st.markdown(f'<h3 style="color: {text_color};">📊 Advanced Analytics</h3>', unsafe_allow_html=True)

        if st.session_state.stats['total'] > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Classifications", st.session_state.stats['total'])
                st.metric("Spam Detected", st.session_state.stats['spam'],
                         delta=f"{(st.session_state.stats['spam']/st.session_state.stats['total']*100):.1f}%")

            with col2:
                st.metric("Legitimate Messages", st.session_state.stats['ham'],
                         delta=f"{(st.session_state.stats['ham']/st.session_state.stats['total']*100):.1f}%")

            # Full history table
            if st.session_state.history:
                st.markdown("### 📜 Full Classification History")
                history_df = pd.DataFrame(st.session_state.history)
                st.dataframe(history_df, use_container_width=True)

                if st.button("🗑️ Clear History"):
                    st.session_state.history = []
                    st.session_state.stats = {'spam': 0, 'ham': 0, 'total': 0}
                    st.rerun()
        else:
            st.info("📭 No data yet! Start classifying messages to see analytics.")

    # Footer
    st.markdown("---")

    if st.session_state.dark_mode:
        footer_text_color = "#999"
    else:
        footer_text_color = "#666"

    st.markdown(f"""
    <div style="text-align: center; color: {footer_text_color}; padding: 20px;">
        <p style="color: {footer_text_color};">🔒 Your messages are processed locally and not stored.</p>
        <p style="font-size: 14px; color: {footer_text_color};">Made with ❤️ using Python • Scikit-learn • Streamlit • Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()