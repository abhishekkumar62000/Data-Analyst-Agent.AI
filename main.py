
import json
import tempfile
import csv
import streamlit as st
import pandas as pd
import openai
import duckdb
import re
import os
from dotenv import load_dotenv
import traceback
import plotly.express as px

# Inject custom dark theme and animation CSS
try:
    with open("custom_dark_theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Streamlit app

# Animated App Title & Subtitle with unique colors, selection fix, and unique animations
st.markdown("""
<style>
.animated-title {
    font-size: 2.4em;
    font-weight: 900;
    background: linear-gradient(90deg, #00fff7, #ff00cc 30%, #fffd00 60%, #00ff99 90%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: titleGradientMove 3.5s ease-in-out infinite, titleGlow 2.2s alternate infinite, titleBounce 2.8s cubic-bezier(.4,2,.6,1) infinite;
    letter-spacing: 2px;
    text-shadow: 0 0 18px #00fff7cc, 0 2px 8px #ff00cc99;
    margin-bottom: 0.2em;
    text-align: center;
    user-select: text;
    -webkit-user-select: text;
    position: relative;
}
/* Fix selection: dark bg, light text, emoji visible */
.animated-title::selection, .animated-title *::selection {
    background: #181c24 !important;
    color: #fff !important;
    -webkit-text-fill-color: unset !important;
    text-shadow: none !important;
    -webkit-background-clip: unset !important;
    filter: none !important;
}
.animated-title span, .animated-title em, .animated-title strong {
    -webkit-text-fill-color: unset !important;
    color: #fff !important;
    filter: none !important;
}
.animated-title .emoji, .animated-title svg, .animated-title img {
    filter: none !important;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    background: none !important;
    text-shadow: none !important;
}
@keyframes titleGradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
@keyframes titleGlow {
    0% {text-shadow: 0 0 18px #00fff7cc, 0 2px 8px #ff00cc99;}
    100% {text-shadow: 0 0 32px #fffd00cc, 0 2px 16px #00fff7cc;}
}
@keyframes titleBounce {
    0%, 100% {transform: translateY(0);}
    20% {transform: translateY(-6px) scale(1.04);}
    40% {transform: translateY(2px) scale(0.98);}
    60% {transform: translateY(-2px) scale(1.01);}
    80% {transform: translateY(1px) scale(1.02);}
}
.animated-subtitle {
    font-size: 1.18em;
    font-weight: 700;
    background: linear-gradient(90deg, #fffd00, #00fff7 40%, #ff00cc 80%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: #fffd00;
    letter-spacing: 1.2px;
    margin-bottom: 18px;
    text-align: center;
    text-shadow: 0 2px 8px #fffd00cc, 0 1px 2px #000a;
    animation: subtitleFadeIn 2.2s cubic-bezier(.4,2,.6,1) 0.2s both, subtitlePulse 2.5s infinite alternate, subtitleWave 4s linear infinite;
    user-select: text;
    -webkit-user-select: text;
    position: relative;
}
.animated-subtitle::selection, .animated-subtitle *::selection {
    background: #181c24 !important;
    color: #fff !important;
    -webkit-text-fill-color: unset !important;
    text-shadow: none !important;
    -webkit-background-clip: unset !important;
    filter: none !important;
}
.animated-subtitle span, .animated-subtitle em, .animated-subtitle strong {
    -webkit-text-fill-color: unset !important;
    color: #fff !important;
    filter: none !important;
}
.animated-subtitle .emoji, .animated-subtitle svg, .animated-subtitle img {
    filter: none !important;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    background: none !important;
    text-shadow: none !important;
}
@keyframes subtitleFadeIn {
    from {opacity:0;transform:translateY(-24px) scale(0.95);}
    to {opacity:1;transform:translateY(0) scale(1);}
}
@keyframes subtitlePulse {
    0% {filter: brightness(1) drop-shadow(0 0 8px #fffd00cc);}
    100% {filter: brightness(1.18) drop-shadow(0 0 18px #00fff7cc);}
}
@keyframes subtitleWave {
    0% {background-position: 0% 50%;}
    100% {background-position: 100% 50%;}
}
</style>
<div class='animated-title'>📊 Data Analyst Agent🤖</div>
<div class='animated-subtitle'>Unlock the Power of Data with AI — Instantly Analyze, Visualize, and Discover Insights!</div>
""", unsafe_allow_html=True)

# Sidebar for API keys
# Sidebar for API keys

# Load .env file
load_dotenv()
default_openai_key = os.getenv("OPENAI_API_KEY", "")


with st.sidebar:
    # --- App Logo ---
    st.image("Logo.png", width=80)
    # --- User Profile ---
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:10px;'>
        <span style='color:#00fff7;font-weight:700;font-size:1.1em;'>Guest User</span><br>
        <span style='color:#fffd00;font-size:0.9em;'>AI Data Analyst</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Theme Switcher ---
    theme = st.radio("Theme", ["🌙 Dark", "☀️ Light"], horizontal=True, key="theme_switch")
    
    # --- Quick Navigation ---
    st.markdown("---")
    st.subheader("Quick Navigation")
    nav = st.radio("Go to", [
        "AI Data Analyst", "SQL Analysis", "Data Visualization Studio", "PowerBI Analyst", "Advanced BI Tools", "Machine Learning Analyst", "AI Chatbot"
    ], key="sidebar_nav")
    # Set session state for navigation (simulate tab switch)
    if 'tab_index' not in st.session_state:
        st.session_state.tab_index = 0
    tab_map = {
        "AI Data Analyst": 0,
        "SQL Analysis": 1,
        "Data Visualization Studio": 2,
        "PowerBI Analyst": 3,
        "Advanced BI Tools": 4,
        "Machine Learning Analyst": 5,
        "AI Chatbot": 6
    }
    st.session_state.tab_index = tab_map[nav]

    # --- Recent Files ---
    st.markdown("---")
    st.subheader("Recent Files")
    if 'recent_files' not in st.session_state:
        st.session_state.recent_files = []
    # Add current file to recent if uploaded
    if 'uploaded_file_name' in st.session_state:
        fname = st.session_state['uploaded_file_name']
        if fname not in st.session_state.recent_files:
            st.session_state.recent_files = [fname] + st.session_state.recent_files[:4]
    if st.session_state.recent_files:
        for f in st.session_state.recent_files:
            st.markdown(f"<span style='color:#00fff7;'>{f}</span>", unsafe_allow_html=True)
    else:
        st.caption("No recent files yet.")

    # --- API Key Section (existing) ---
    st.markdown("---")
    st.subheader("API Keys")
    openai_key = st.text_input("Enter your OpenAI API key:", type="password", value=default_openai_key)
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("API key saved! (Will only be used for requests)")
    else:
        st.warning("Please enter your OpenAI API key to proceed.")


    # --- App Settings ---
    st.markdown("---")
    st.subheader("App Settings")
    lang = st.selectbox("Language", ["English", "हिंदी", "Español", "Français", "中文"], key="sidebar_lang")
    font_size = st.slider("Font Size", 12, 24, 16, key="sidebar_fontsize")
    st.caption(f"Current language: {lang}, Font size: {font_size}px")

    # --- App Version & Updates ---
    st.markdown("---")
    st.subheader("App Version & Updates")
    st.markdown("<span style='color:#00fff7;font-weight:600;'>v1.0.0</span>", unsafe_allow_html=True)
    st.caption("Last updated: Aug 2025")
    with st.expander("What's New?"):
        st.markdown("- Neon dark theme and animated UI\n- Sidebar navigation and feedback\n- More features coming soon!")

    # --- Social Links ---
    st.markdown("---")
    st.subheader("Connect with Us")
    st.markdown("""
    <div style='display:flex;gap:12px;'>
        <a href='https://github.com/abhishekkumar62000' target='_blank'><img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg' width='28' style='filter:invert(1);'></a>
        <a href='https://twitter.com/' target='_blank'><img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/twitter.svg' width='28' style='filter:invert(0.7);'></a>
        <a href='https://linkedin.com/' target='_blank'><img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg' width='28' style='filter:invert(0.6);'></a>
    </div>
    """, unsafe_allow_html=True)

    # --- Contact/Support ---
    st.markdown("---")
    st.subheader("Contact & Support")
    st.markdown("For support, email: <a href='mailto:abhiydv23096@gmail.com' style='color:#00fff7;'>abhiydv23096@gmail.com</a>", unsafe_allow_html=True)
    st.caption("We usually respond within 24 hours.")

    # --- Help & Feedback ---
    st.markdown("---")
    st.subheader("Help & Feedback")
    with st.expander("FAQ / Help"):
        st.markdown("- **How to use?** Upload a CSV/Excel file and explore the tabs.\n- **Need more help?** Contact support below.")
    feedback = st.text_area("Your feedback", key="sidebar_feedback")
    if st.button("Submit Feedback", key="sidebar_feedback_btn"):
        st.success("Thank you for your feedback!")

    # --- Developer Credit ---
    st.markdown("""
    <div style='margin-top:32px;text-align:center;font-size:1.1em;color:#00fff7;font-weight:700;'>
        Developer🧑‍💻: <span style='color:#fffd00;'>ABHI❤️Yadav</span>
    </div>
    """, unsafe_allow_html=True)

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# --- Ensure chat_history is always initialized ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None and "openai_key" in st.session_state:
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    # Always store df and file name in session_state for chatbot and other tabs
    st.session_state['df'] = df
    st.session_state['uploaded_file_name'] = uploaded_file.name
    
    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)  # Use st.dataframe for an interactive table

        # --- Data Cleaning (now in main app, not sidebar) ---
        st.subheader("🧹 Data Cleaning & Transformation")
        clean_df = df.copy()
        columns = list(clean_df.columns)
        # Filter rows
        col_to_filter = st.selectbox("Filter column", [None] + columns, key="filter_col_main")
        filter_value = st.text_input("Value to filter by", key="filter_val_main")
        if st.button("Apply Filter", key="apply_filter_main") and col_to_filter and filter_value:
            clean_df = clean_df[clean_df[col_to_filter].astype(str) == filter_value]
        # Fill missing values
        fill_col = st.selectbox("Fill missing in column", [None] + columns, key="fill_col_main")
        fill_method = st.selectbox("Fill method", [None, "mean", "median", "zero", "custom"], key="fill_method_main")
        fill_custom = st.text_input("Custom fill value", key="fill_custom_main")
        if st.button("Fill Missing", key="fill_missing_main") and fill_col:
            if fill_method == "mean":
                clean_df[fill_col] = clean_df[fill_col].fillna(clean_df[fill_col].mean())
            elif fill_method == "median":
                clean_df[fill_col] = clean_df[fill_col].fillna(clean_df[fill_col].median())
            elif fill_method == "zero":
                clean_df[fill_col] = clean_df[fill_col].fillna(0)
            elif fill_method == "custom" and fill_custom:
                clean_df[fill_col] = clean_df[fill_col].fillna(type(clean_df[fill_col].dropna().iloc[0])(fill_custom))
        # Drop columns
        drop_col = st.multiselect("Drop columns", columns, key="drop_col_main")
        if st.button("Drop Selected Columns", key="drop_selected_main") and drop_col:
            clean_df = clean_df.drop(columns=drop_col)
        # Change type
        type_col = st.selectbox("Change type column", [None] + columns, key="type_col_main")
        type_target = st.selectbox("To type", [None, "int", "float", "str"], key="type_target_main")
        if st.button("Change Type", key="change_type_main") and type_col and type_target:
            try:
                clean_df[type_col] = clean_df[type_col].astype(type_target)
            except Exception as e:
                st.error(f"Type conversion error: {e}")
        df = clean_df
        # --- Ensure numeric_cols is always defined for suggestions
        if 'numeric_cols' not in locals():
            numeric_cols = df.select_dtypes(include='number').columns.tolist() if 'df' in locals() else []

        # --- Automated Insights & Anomaly Detection ---
        st.subheader("🔎 Automated Insights & Anomaly Detection")
        try:
            desc = df.describe(include='all').T
            outlier_info = []
            for col in df.select_dtypes(include='number').columns:
                mean = df[col].mean()
                std = df[col].std()
                outliers = df[(df[col] > mean + 3*std) | (df[col] < mean - 3*std)]
                if not outliers.empty:
                    outlier_info.append(f"Column '{col}' has {len(outliers)} outlier(s) (>|3 std| from mean). Example: {outliers.iloc[0].to_dict()}")
            st.write("**Outlier Info:**")
            if outlier_info:
                for o in outlier_info:
                    st.warning(o)
            else:
                st.success("No strong outliers detected in numeric columns.")
            # LLM-powered summary
            import io
            summary_prompt = f"Here is a summary of the data: {desc.to_string()[:4000]}\nFind any interesting patterns, outliers, or anomalies."
            client = openai.OpenAI(api_key=st.session_state.openai_key)
            summary = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a data scientist."}, {"role": "user", "content": summary_prompt}],
                temperature=0.1
            ).choices[0].message.content
            st.info(summary)
        except Exception as e:
            st.error(f"Insight/Anomaly detection error: {e}")
        # ...existing code...

        # --- Conversational Data Exploration (Contextual Memory) ---
        # When building the prompt for OpenAI, include last 3 queries/SQL
        context = ""
        if len(st.session_state.chat_history) > 0:
            for entry in st.session_state.chat_history[-3:]:
                context += f"Previous question: {entry['query']}\nPrevious SQL: {entry['sql']}\n"
        # In your OpenAI prompt for SQL generation, use:
        # prompt = f"{context}\nNow, answer this new question: {user_query}\n(Remember, the table name is 'uploaded_data')"
        # ...existing code...

        # --- Chat History Display with Download Options ---
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if st.session_state.chat_history:
            st.markdown("### Chat History")
            for i, entry in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{i+1}: {entry['query']}"):
                    st.code(entry["sql"], language="sql")
                    # Download buttons for SQL and result
                    st.download_button("Download SQL", entry["sql"], file_name=f"query_{i+1}.sql")
                    if not entry["result"].empty:
                        st.download_button("Download Result (CSV)", entry["result"].to_csv(index=False), file_name=f"result_{i+1}.csv")
                        try:
                            import io
                            import openpyxl
                            output = io.BytesIO()
                            entry["result"].to_excel(output, index=False, engine='openpyxl')
                            st.download_button("Download Result (Excel)", output.getvalue(), file_name=f"result_{i+1}.xlsx")
                        except Exception:
                            pass
                    st.dataframe(entry["result"])
                    if entry.get("viz") is not None:
                        st.plotly_chart(entry["viz"])

        # --- Smart Query Suggestions & Templates ---
        st.markdown("---")
        st.subheader("💡 Example Questions & Templates")
        # Dynamically generate 20+ example questions based on columns
        example_questions = []
        if columns:
            for col in columns:
                example_questions.append(f"Show unique values in {col}.")
                example_questions.append(f"Count of missing values in {col}.")
                example_questions.append(f"Show top 5 most frequent values in {col}.")
                if str(df[col].dtype).startswith("float") or str(df[col].dtype).startswith("int"):
                    example_questions.append(f"What is the average of {col}?")
                    example_questions.append(f"What is the minimum and maximum of {col}?")
                    example_questions.append(f"Show the distribution of {col}.")
                if str(df[col].dtype).startswith("datetime"):
                    example_questions.append(f"Show trends over time for {col}.")
            if len(columns) > 1:
                for i in range(len(columns)):
                    for j in range(i+1, len(columns)):
                        example_questions.append(f"Show the correlation between {columns[i]} and {columns[j]}.")
                        example_questions.append(f"Show the distribution of {columns[i]} by {columns[j]}.")
                        example_questions.append(f"Group by {columns[j]} and show average of {columns[i]}.")
        # Add some generic questions
        example_questions += [
            "Show the first 10 rows of the data.",
            "Count total rows in the dataset.",
            "Show summary statistics for all columns.",
            "Find rows with missing values.",
            "Show duplicate rows.",
            "Which columns have the most missing data?",
            "Show a random sample of 5 rows.",
            "Show the number of unique values in each column.",
            "Show the most common value in each column."
        ]
        # Remove duplicates and keep only 20-25
        example_questions = list(dict.fromkeys(example_questions))[:25]
        selected_example = st.selectbox("Select an example question to use:", ["-- Select --"] + example_questions)
        if selected_example and selected_example != "-- Select --":
            user_query = st.text_area("Ask a query about the data:", value=selected_example)
        else:
            user_query = st.text_area("Ask a query about the data:")

        # --- Visualization Controls ---
        st.markdown("---")
        st.markdown("#### Optional: Create a chart from your data")
        chart_type = st.selectbox("Chart type", ["None", "Bar", "Line", "Scatter", "Pie"])
        x_col = st.selectbox("X axis", [None] + columns)
        y_col = st.selectbox("Y axis", [None] + columns)
        color_col = st.selectbox("Color (optional)", [None] + columns)

        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            elif not st.session_state.openai_key or st.session_state.openai_key.strip() == "":
                st.error("OpenAI API key is missing or invalid. Please check your key in the sidebar.")
            else:
                try:
                    with st.spinner('Generating SQL with OpenAI...'):
                        prompt = f"""
You are an expert data analyst. Given the following table columns: {columns}
The table name is 'uploaded_data'.
Generate a DuckDB-compatible SQL query to answer the user's question below. Always use 'uploaded_data' as the table name. Return only the SQL query, enclosed in triple backticks (```sql ... ```):
User question: {user_query}
"""
                        client = openai.OpenAI(api_key=st.session_state.openai_key)
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful data analyst that writes SQL for DuckDB."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1
                        )
                        sql_code = response.choices[0].message.content
                        match = re.search(r"```sql(.*?)```", sql_code, re.DOTALL)
                        if match:
                            sql_query = match.group(1).strip()
                        else:
                            sql_query = sql_code.strip()
                        st.code(sql_query, language="sql")

                    with st.spinner('Running SQL on your data...'):
                        con = duckdb.connect()
                        con.register("uploaded_data", df)
                        try:
                            result = con.execute(sql_query).fetchdf()
                            st.dataframe(result)
                        except Exception as sql_e:
                            st.error(f"SQL execution error: {sql_e}")
                            result = None
                        finally:
                            con.close()

                    # --- Visualization ---
                    viz = None
                    if result is not None and chart_type != "None" and x_col and y_col and x_col in result.columns and y_col in result.columns:
                        try:
                            if chart_type == "Bar":
                                viz = px.bar(result, x=x_col, y=y_col, color=color_col if color_col else None)
                            elif chart_type == "Line":
                                viz = px.line(result, x=x_col, y=y_col, color=color_col if color_col else None)
                            elif chart_type == "Scatter":
                                viz = px.scatter(result, x=x_col, y=y_col, color=color_col if color_col else None)
                            elif chart_type == "Pie":
                                viz = px.pie(result, names=x_col, values=y_col, color=color_col if color_col else None)
                            if viz is not None:
                                st.plotly_chart(viz)
                        except Exception as viz_e:
                            st.error(f"Visualization error: {viz_e}")

                    # --- Update Chat History ---
                    st.session_state.chat_history.append({
                        "query": user_query,
                        "sql": sql_query,
                        "result": result if result is not None else pd.DataFrame(),
                        "viz": viz
                    })
                except Exception as e:
                    st.error(f"Error generating SQL or running query: {e}")
                    st.error(traceback.format_exc())
                    st.info("Tip: Check your API key, try rephrasing your query, or verify your data format. See details above.")

# --- Main App Tabs ---
tabs = st.tabs([
    "AI Data Analyst",
    "SQL Analysis",
    "📈 Data Visualization Studio",
    "📈PowerBI Analyst",
    "🛠️ Advanced BI Tools",
    "🤖 Machine Learning Analyst",
    "🤖 AI Chatbot"
])
with tabs[6]:
    st.header("🤖 AI Chatbot (One-to-One Conversation)")
    st.info("Chat with your data! Ask any question about your uploaded CSV. The AI will remember the conversation context, just like ChatGPT or Copilot.")
    if 'chatbot_history' not in st.session_state:
        st.session_state.chatbot_history = []  # [(role, message, timestamp)]
    import datetime
    def render_chat():
        for idx, (role, msg, ts) in enumerate(st.session_state.chatbot_history):
            if role == 'user':
                st.markdown(f"""
<div style='background:linear-gradient(90deg,#232526,#414345);color:#fff;padding:10px;border-radius:10px;margin-bottom:5px;max-width:80%;margin-left:auto;text-align:right;box-shadow:0 2px 8px #0002;'>
<b>You</b> <span style='font-size:10px;color:#bbb;'>{ts}</span><br>{msg}
</div>""", unsafe_allow_html=True)
            elif role == 'assistant':
                st.markdown(f"""
<div style='background:linear-gradient(90deg,#141e30,#243b55);color:#fff;padding:10px;border-radius:10px;margin-bottom:5px;max-width:80%;margin-right:auto;text-align:left;box-shadow:0 2px 8px #0002;'>
<b>AI</b> <span style='font-size:10px;color:#bbb;'>{ts}</span><br>{msg}
</div>""", unsafe_allow_html=True)
    # Robustly get df for chatbot tab
    # Always use st.session_state['df'] for chatbot
    df_chatbot = st.session_state.get('df', None)
    dataset_path = f"./{st.session_state['uploaded_file_name']}" if 'uploaded_file_name' in st.session_state and df_chatbot is not None else None
    if df_chatbot is not None and dataset_path is not None:
        # --- Chatbot UI ---
        st.markdown("""
<style>
div[data-testid='stTextInput'] textarea {background:#232526;color:#fff;border-radius:8px;border:1.5px solid #ffd700;}
.stButton>button {background:#ffd700;color:#222;border-radius:8px;}
.stButton>button:hover {background:#ffe066;}
</style>
""", unsafe_allow_html=True)
        col1, col2 = st.columns([4,1])
        with col2:
            if st.button("🧹 Clear Chat", key="clear_chatbot"):
                st.session_state.chatbot_history = []
        render_chat()
        user_input = st.text_input("Type your question for the AI:", key="chatbot_input")
        regenerate = False
        if st.button("Send", key="chatbot_send") and user_input.strip():
            regenerate = False
            last_user_input = user_input
        elif st.button("🔄 Regenerate Response", key="regenerate_chatbot") and st.session_state.chatbot_history:
            # Find last user message
            for i in range(len(st.session_state.chatbot_history)-1, -1, -1):
                if st.session_state.chatbot_history[i][0] == 'user':
                    last_user_input = st.session_state.chatbot_history[i][1]
                    # Remove last AI response
                    if i+1 < len(st.session_state.chatbot_history) and st.session_state.chatbot_history[i+1][0]=='assistant':
                        st.session_state.chatbot_history.pop(i+1)
                    break
            regenerate = True
        else:
            last_user_input = None
        if last_user_input:
            if dataset_path is None:
                st.error("No dataset path available. Please upload a CSV file first.")
            else:
                # --- Enhanced Prompt Engineering with Data Knowledge ---
                import io
                buf = io.StringIO()
                df_chatbot.info(buf=buf)
                info_str = buf.getvalue()
                head_str = df_chatbot.head(5).to_string()
                try:
                    stats_str = df_chatbot.describe(include='all').to_string()
                except Exception:
                    stats_str = ''
                system_prompt = f"""You are a world-class, highly accurate, step-by-step AI Data Analyst with expert knowledge of all data science, analytics, and business intelligence skills. The user has uploaded a CSV at '{dataset_path}'.

Here is a summary of the uploaded dataset:
COLUMNS and TYPES:
{info_str}
HEAD (first 5 rows):
{head_str}
STATS:
{stats_str}

Always:
- Use the above dataset summary to answer questions directly, as if you have already loaded the data.
- Think step by step and explain your reasoning clearly.
- If code is needed, provide a Python code block and explain the code before and after.
- If the question is ambiguous, ask clarifying questions.
- If you don't know, say so honestly.
- Always provide the most accurate, up-to-date, and relevant answer possible.
- Format your answer with clear sections, bullet points, and tables if helpful.
- Respond in a friendly, conversational, and professional tone.
"""
                messages = [{"role": "system", "content": system_prompt}]
                for i, (role, msg, ts) in enumerate(st.session_state.chatbot_history):
                    if role in ("user", "assistant"):
                        messages.append({"role": role, "content": msg})
                messages.append({"role": "user", "content": last_user_input})
                with st.spinner('AI is thinking...'):
                    try:
                        client = openai.OpenAI(api_key=st.session_state.openai_key)
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            temperature=0.2
                        )
                        ai_msg = response.choices[0].message.content
                        now = datetime.datetime.now().strftime('%H:%M:%S')
                        if not regenerate:
                            st.session_state.chatbot_history.append(("user", last_user_input, now))
                        st.session_state.chatbot_history.append(("assistant", ai_msg, now))
                        st.success("AI responded!")
                        render_chat()
                    except Exception as e:
                        st.error(f"AI error: {e}")
    else:
        st.warning("Please upload a CSV file with data in the Data Analysis tab first.")
        dataset_path = None
with tabs[5]:
    st.header("🤖 Machine Learning Analyst")
    st.info("Build, explain, and deploy machine learning models with AutoML, feature engineering, explainability, and more!")
    import numpy as np
    import joblib
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, classification_report, r2_score
    import shap
    import tempfile
    # --- 1. AutoML Model Builder ---
    if uploaded_file is not None and temp_path and columns and df is not None:
        st.subheader("1️⃣ AutoML Model Builder")
        target_col = st.selectbox("Select target column (label)", [None] + columns, key="ml_target_col")
        task_type_widget = st.selectbox("Task type", ["Auto", "Classification", "Regression"], key="ml_task_type")
        test_size = st.slider("Test size (%)", 10, 50, 20, key="ml_test_size")
        run_automl = st.button("Run AutoML", key="ml_run_automl")
        if run_automl and target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            # Simple preprocessing: drop NA, encode categoricals
            X = X.select_dtypes(include=[np.number]).fillna(0)
            y = y.fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            results = []
            leaderboard = []
            # Determine task (use local variable, don't overwrite widget key)
            task_type = task_type_widget
            if task_type == "Auto":
                if y.nunique() <= 10 and y.dtype in [np.int64, np.int32, np.int16]:
                    task_type = "Classification"
                else:
                    task_type = "Regression"
            if task_type == "Classification":
                models = {
                    "RandomForestClassifier": RandomForestClassifier(n_estimators=50, random_state=42),
                    "LogisticRegression": LogisticRegression(max_iter=500)
                }
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        acc = accuracy_score(y_test, preds)
                        f1 = f1_score(y_test, preds, average='weighted')
                        leaderboard.append({"Model": name, "Accuracy": acc, "F1": f1})
                    except Exception as e:
                        leaderboard.append({"Model": name, "Error": str(e)})
            else:
                models = {
                    "RandomForestRegressor": RandomForestRegressor(n_estimators=50, random_state=42),
                    "LinearRegression": LinearRegression()
                }
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        mse = mean_squared_error(y_test, preds)
                        r2 = r2_score(y_test, preds)
                        leaderboard.append({"Model": name, "RMSE": np.sqrt(mse), "R2": r2})
                    except Exception as e:
                        leaderboard.append({"Model": name, "Error": str(e)})
            st.markdown("#### Model Leaderboard")
            st.dataframe(leaderboard)
            st.session_state["ml_leaderboard"] = leaderboard
            st.session_state["ml_X_train"] = X_train
            st.session_state["ml_X_test"] = X_test
            st.session_state["ml_y_train"] = y_train
            st.session_state["ml_y_test"] = y_test
            st.session_state["ml_models"] = models
            # Do NOT assign to st.session_state["ml_task_type"] here
            st.session_state["ml_target_col"] = target_col
    # --- 2️⃣ Interactive Feature Engineering ---
    st.subheader("2️⃣ Interactive Feature Engineering")
    if "ml_X_train" in st.session_state and "ml_X_test" in st.session_state and "ml_y_train" in st.session_state and "ml_y_test" in st.session_state:
        X_train = st.session_state["ml_X_train"]
        X_test = st.session_state["ml_X_test"]
        y_train = st.session_state["ml_y_train"]
        y_test = st.session_state["ml_y_test"]
        feature_cols = list(X_train.columns)
        selected_features = st.multiselect("Select features to use", feature_cols, default=feature_cols, key="ml_selected_features")
        retrain = st.button("Apply Feature Selection & Retrain Models", key="ml_apply_features")
        if retrain:
            # Retrain models with selected features
            models = {}
            leaderboard = []
            task_type = st.session_state.get("ml_task_type", "Classification")
            if task_type == "Regression":
                models = {
                    "RandomForestRegressor": RandomForestRegressor(n_estimators=50, random_state=42),
                    "LinearRegression": LinearRegression()
                }
                for name, model in models.items():
                    try:
                        model.fit(X_train[selected_features], y_train)
                        preds = model.predict(X_test[selected_features])
                        mse = mean_squared_error(y_test, preds)
                        r2 = r2_score(y_test, preds)
                        leaderboard.append({"Model": name, "RMSE": np.sqrt(mse), "R2": r2})
                    except Exception as e:
                        leaderboard.append({"Model": name, "Error": str(e)})
            else:
                models = {
                    "RandomForestClassifier": RandomForestClassifier(n_estimators=50, random_state=42),
                    "LogisticRegression": LogisticRegression(max_iter=500)
                }
                for name, model in models.items():
                    try:
                        model.fit(X_train[selected_features], y_train)
                        preds = model.predict(X_test[selected_features])
                        acc = accuracy_score(y_test, preds)
                        f1 = f1_score(y_test, preds, average='weighted')
                        leaderboard.append({"Model": name, "Accuracy": acc, "F1": f1})
                    except Exception as e:
                        leaderboard.append({"Model": name, "Error": str(e)})
            st.session_state["ml_models"] = models
            st.session_state["ml_leaderboard"] = leaderboard
            st.success("Models retrained with selected features!")
            st.dataframe(leaderboard)
        # Feature importances (if model trained and supports it)
        if "ml_models" in st.session_state:
            for name, model in st.session_state["ml_models"].items():
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    st.markdown(f"**{name} Feature Importances:**")
                    st.bar_chart(dict(zip(selected_features, importances)))
    else:
        st.info("Run AutoML first to enable feature engineering.")

    # --- 3️⃣ Model Explainability & Interpretation ---
    st.subheader("3️⃣ Model Explainability & Interpretation")
    if "ml_models" in st.session_state and "ml_X_test" in st.session_state:
        import matplotlib.pyplot as plt
        for name, model in st.session_state["ml_models"].items():
            try:
                # Only use SHAP for tree-based models
                if hasattr(model, "feature_importances_"):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(st.session_state["ml_X_test"])
                    st.markdown(f"**{name} Feature Importance (SHAP):**")
                    shap.summary_plot(shap_values, st.session_state["ml_X_test"], show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                else:
                    st.info(f"{name} does not support SHAP explainability.")
            except Exception as e:
                st.info(f"{name} SHAP explainability not available: {e}")
    else:
        st.info("Train a model to enable explainability.")

    # --- 4️⃣ Prediction Playground ---
    st.subheader("4️⃣ Prediction Playground")
    if "ml_models" in st.session_state and "ml_X_test" in st.session_state:
        model_names = list(st.session_state["ml_models"].keys())
        selected_model_name = st.selectbox("Select trained model for prediction", model_names, key="ml_pred_model")
        model = st.session_state["ml_models"][selected_model_name]
        input_data = {}
        for col in st.session_state["ml_X_test"].columns:
            try:
                default_val = float(st.session_state["ml_X_test"][col].iloc[0])
            except Exception:
                default_val = 0.0
            val = st.number_input(f"{col}", value=default_val, key=f"ml_pred_{col}")
            input_data[col] = val
        if st.button("Predict", key="ml_predict_btn"):
            input_df = pd.DataFrame([input_data])
            try:
                pred = model.predict(input_df)[0]
                st.success(f"Prediction: {pred}")
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)
                    st.info(f"Probabilities: {proba}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.info("Train a model to enable prediction playground.")

    # --- 5️⃣ Model Export & Download ---
    st.subheader("5️⃣ Model Export & Download")
    if "ml_models" in st.session_state and len(st.session_state["ml_models"]) > 0:
        for name, model in st.session_state["ml_models"].items():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                joblib.dump(model, tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button(f"Download {name} Model", f.read(), file_name=f"{name}.pkl")
        if "ml_X_test" in st.session_state:
            pred_df = st.session_state["ml_X_test"].copy()
            for name, model in st.session_state["ml_models"].items():
                try:
                    pred_df[f"{name}_pred"] = model.predict(st.session_state["ml_X_test"])
                except Exception:
                    continue
            st.download_button("Download Predictions (CSV)", pred_df.to_csv(index=False), file_name="predictions.csv")
    else:
        st.info("Train a model to enable export and download.")
    # --- 6. AI-Powered ML Assistant ---
    st.subheader("6️⃣ AI-Powered ML Assistant")
    ml_query = st.text_area("Ask an ML question or request code (e.g., 'How to tune RandomForest?')", key="ml_ai_query")
    if st.button("Ask ML Assistant", key="ml_ai_btn") and ml_query and st.session_state.openai_key:
        try:
            client = openai.OpenAI(api_key=st.session_state.openai_key)
            ml_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a machine learning expert."}, {"role": "user", "content": ml_query}],
                temperature=0.1
            ).choices[0].message.content
            st.info(ml_response)
        except Exception as e:
            st.error(f"ML Assistant error: {e}")
with tabs[4]:
    st.header("🛠️ Advanced BI Tools")
    if uploaded_file is not None and temp_path and columns and df is not None:
        st.info("Enterprise-grade BI features for analysts and business users!")
        # --- 1. Time Intelligence & Period Analysis ---
        st.subheader("📅 Time Intelligence & Period Analysis")
        date_cols = [col for col in columns if str(df[col].dtype).startswith("datetime")]
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if date_cols and num_cols:
            time_col = st.selectbox("Select date column", date_cols, key="bi_time_col")
            metric_col = st.selectbox("Select metric column", num_cols, key="bi_metric_col")
            period = st.selectbox("Period", ["Day", "Month", "Year"], key="bi_period")
            if st.button("Show Period Analysis", key="bi_period_btn"):
                df_period = df.copy()
                if period == "Month":
                    df_period['period'] = df_period[time_col].dt.to_period('M')
                elif period == "Year":
                    df_period['period'] = df_period[time_col].dt.to_period('Y')
                else:
                    df_period['period'] = df_period[time_col].dt.to_period('D')
                agg = df_period.groupby('period')[metric_col].sum().reset_index()
                st.line_chart(agg.set_index('period'))
        # --- 2. Advanced Drill-Through & Data Exploration ---
        st.subheader("🏆 Advanced Drill-Through & Data Exploration")
        st.markdown("Click a value in the table below to drill through to detailed data.")
        selected_col = st.selectbox("Drill-through column", columns, key="bi_drill_col")
        if st.button("Show Drill Table", key="bi_drill_btn"):
            unique_vals = df[selected_col].dropna().unique()
            selected_val = st.selectbox("Select value to drill into", unique_vals, key="bi_drill_val")
            drill_df = df[df[selected_col] == selected_val]
            st.dataframe(drill_df)
        # --- 3. Custom Visual Marketplace ---
        st.subheader("🧩 Custom Visual Marketplace")
        st.markdown(":bulb: _Demo: Add a radar/spider chart (advanced visual)._ ")
        import plotly.graph_objects as go
        if st.button("Show Radar Chart", key="bi_radar_btn") and len(num_cols) >= 3:
            radar_df = df[num_cols[:3]].mean()
            fig = go.Figure(data=go.Scatterpolar(r=radar_df.values, theta=radar_df.index, fill='toself'))
            st.plotly_chart(fig)
        # --- 4. Row-Level Security & User Roles ---
        st.subheader("🔒 Row-Level Security & User Roles")
        st.markdown("Simulate user roles and restrict data access.")
        role_col = st.selectbox("Role column (e.g., Region, Department)", [None] + columns, key="bi_role_col")
        user_role = st.text_input("Enter user role to simulate", key="bi_user_role")
        if st.button("Apply Row-Level Security", key="bi_rls_btn") and role_col and user_role:
            rls_df = df[df[role_col].astype(str) == user_role]
            st.dataframe(rls_df)
        # --- 5. Automated Data Refresh & Scheduling ---
        st.subheader("🤖 Automated Data Refresh & Scheduling")
        st.markdown(":bulb: _Demo: Click to simulate a data refresh._ ")
        if st.button("Refresh Data Now", key="bi_refresh_btn"):
            st.success("Data refreshed from source!")
        # --- 6. Publish & Share Interactive Reports ---
        st.subheader("📤 Publish & Share Interactive Reports")
        st.markdown("Download your dashboard as HTML or share a link.")
        import io
        import streamlit.components.v1 as components
        if st.button("Download Dashboard as HTML", key="bi_html_btn"):
            st.info("Feature coming soon: Export full dashboard as HTML.")
        st.markdown(":bulb: _For sharing, deploy this app and share the URL with your team!_ ")
with tabs[3]:
    st.header("📈 PowerBI Analyst")
    if uploaded_file is not None and temp_path and columns and df is not None:
        st.info("Advanced Power BI-style analytics, modeling, and integration features for data analysts!")
        # --- 1. Data Modeling & Relationships ---
        st.subheader("🏗️ Data Modeling & Relationships")
        st.markdown("Define relationships between tables, join datasets, and create calculated columns.")
        # For demo: allow user to upload a second dataset and join
        other_file = st.file_uploader("Upload another dataset to join (optional)", type=["csv", "xlsx"], key="pbi_other_file")
        if other_file:
            if other_file.name.endswith('.csv'):
                df2 = pd.read_csv(other_file)
            elif other_file.name.endswith('.xlsx'):
                df2 = pd.read_excel(other_file)
            else:
                df2 = None
            if df2 is not None:
                st.write("Second dataset preview:")
                st.dataframe(df2.head())
                join_col1 = st.selectbox("Join column in main dataset", columns, key="pbi_join_col1")
                join_col2 = st.selectbox("Join column in second dataset", df2.columns, key="pbi_join_col2")
                join_type = st.selectbox("Join type", ["inner", "left", "right", "outer"], key="pbi_join_type")
                if st.button("Join Datasets", key="pbi_join_btn"):
                    joined_df = pd.merge(df, df2, left_on=join_col1, right_on=join_col2, how=join_type)
                    st.dataframe(joined_df.head())
        # Calculated column
        st.markdown("Add a calculated column:")
        calc_col_name = st.text_input("New column name", key="pbi_calc_col_name")
        calc_expr = st.text_input("Pandas expression (e.g., col1 + col2)", key="pbi_calc_expr")
        if st.button("Add Calculated Column", key="pbi_calc_btn") and calc_col_name and calc_expr:
            try:
                df[calc_col_name] = df.eval(calc_expr)
                st.success(f"Added column '{calc_col_name}'!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Calculation error: {e}")
        # --- 2. KPI & Metrics Dashboard ---
        st.subheader("📊 KPI & Metrics Dashboard")
        st.markdown("Define and visualize custom KPIs and metrics.")
        kpi_col = st.selectbox("Select column for KPI", [None] + columns, key="pbi_kpi_col")
        kpi_type = st.selectbox("KPI Type", ["Sum", "Average", "Min", "Max", "Count"], key="pbi_kpi_type")
        if st.button("Show KPI", key="pbi_kpi_btn") and kpi_col:
            val = None
            if kpi_type == "Sum":
                val = df[kpi_col].sum()
            elif kpi_type == "Average":
                val = df[kpi_col].mean()
            elif kpi_type == "Min":
                val = df[kpi_col].min()
            elif kpi_type == "Max":
                val = df[kpi_col].max()
            elif kpi_type == "Count":
                val = df[kpi_col].count()
            st.metric(f"{kpi_type} of {kpi_col}", val)
        # --- 3. Custom DAX/Expression Editor ---
        st.subheader("🧩 Custom DAX/Expression Editor")
        st.markdown("Write custom calculations using Python expressions (like DAX). Preview results instantly.")
        dax_expr = st.text_input("Enter calculation (e.g., df['col1'] / df['col2'])", key="pbi_dax_expr")
        if st.button("Evaluate Expression", key="pbi_dax_eval_btn") and dax_expr:
            try:
                result = eval(dax_expr, {"df": df, "pd": pd, "np": __import__('numpy')})
                st.write(result)
            except Exception as e:
                st.error(f"Expression error: {e}")
        # --- 4. Data Source Connector Hub ---
        st.subheader("🔗 Data Source Connector Hub")
        st.markdown("Connect to external data sources and import data directly.")
        st.markdown(":bulb: _Demo: Paste a public CSV URL to import data._")
        csv_url = st.text_input("CSV URL", key="pbi_csv_url")
        if st.button("Import CSV from URL", key="pbi_csv_import_btn") and csv_url:
            try:
                ext_df = pd.read_csv(csv_url)
                st.dataframe(ext_df.head())
            except Exception as e:
                st.error(f"Import error: {e}")
        # --- Export Data for Power BI ---
        st.subheader("Export Data for Power BI")
        st.download_button("Download as CSV for Power BI", df.to_csv(index=False), file_name="powerbi_data.csv")
        import io
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button("Download as Excel for Power BI", excel_buffer, file_name="powerbi_data.xlsx")
        # --- Auto-generate Power BI DAX/Power Query Scripts ---
        st.subheader("Auto-generate Power BI Scripts")
        if st.button("Generate DAX Measures for Numeric Columns", key="pbi_dax_btn"):
            dax_measures = []
            for col in df.select_dtypes(include='number').columns:
                dax_measures.append(f"Total {col} = SUM('{os.path.splitext(uploaded_file.name)[0]}'[{col}])")
                dax_measures.append(f"Average {col} = AVERAGE('{os.path.splitext(uploaded_file.name)[0]}'[{col}])")
            st.code("\n".join(dax_measures), language="DAX")
        if st.button("Generate Power Query Script", key="pbi_m_btn"):
            m_script = f"let\n    Source = Csv.Document(File.Contents(\"powerbi_data.csv\"),[Delimiter=\",\", Columns={len(columns)}, Encoding=65001, QuoteStyle=QuoteStyle.None]),\n    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true])\nin\n    PromotedHeaders"
            st.code(m_script, language="M")
        # --- Power BI Integration Tips ---
        st.subheader("Power BI Integration Tips")
        st.markdown("- **Step 1:** Download your data as CSV or Excel using the buttons above.")
        st.markdown("- **Step 2:** Open Power BI Desktop and import the file.")
        st.markdown("- **Step 3:** Copy-paste the generated DAX or Power Query scripts into Power BI for instant measures and queries.")
        st.markdown("- **Step 4:** Build your dashboard visually in Power BI!")
        st.markdown(":bulb: _For advanced users: Use Power BI REST API to push data directly from Python (requires Azure setup)._ ")

with tabs[0]:
    # ...existing AI Data Analyst code...
    pass
with tabs[2]:
    st.header("📈 Data Visualization Studio")
    if uploaded_file is not None and temp_path and columns and df is not None:
        st.info("Create custom charts, dashboards, and interactive visualizations from your data. Drag, drop, and explore!")
        # --- Auto Chart Suggestions ---
        st.subheader("Auto Chart Suggestions & Recommended Visualizations")
        import plotly.express as px
        import numpy as np
        chart_suggestions = []
        # Suggest chart types based on column types
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        date_cols = [col for col in columns if str(df[col].dtype).startswith("datetime")]
        if num_cols:
            chart_suggestions.append(("Histogram", px.histogram(df, x=num_cols[0])))
            if len(num_cols) > 1:
                chart_suggestions.append(("Scatter Plot", px.scatter(df, x=num_cols[0], y=num_cols[1], color=cat_cols[0] if cat_cols else None)))
            chart_suggestions.append(("Box Plot", px.box(df, y=num_cols[0], color=cat_cols[0] if cat_cols else None)))
        if cat_cols:
            chart_suggestions.append(("Bar Chart", px.bar(df, x=cat_cols[0], y=num_cols[0] if num_cols else None)))
            chart_suggestions.append(("Pie Chart", px.pie(df, names=cat_cols[0], values=num_cols[0] if num_cols else None)))
        if date_cols and num_cols:
            chart_suggestions.append(("Line Chart (Time Series)", px.line(df, x=date_cols[0], y=num_cols[0], color=cat_cols[0] if cat_cols else None)))
        st.markdown("**Recommended Visualizations:**")
        for name, fig in chart_suggestions[:3]:
            st.markdown(f"**{name}:**")
            st.plotly_chart(fig, use_container_width=True)
        # --- Drag-and-Drop Chart Builder ---
        st.subheader("Drag-and-Drop Chart Builder")
        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie", "Box", "Histogram", "Heatmap"])
        x_col = st.selectbox("X Axis", [None] + columns, key="viz_x_col")
        y_col = st.selectbox("Y Axis", [None] + columns, key="viz_y_col")
        color_col = st.selectbox("Color", [None] + columns, key="viz_color_col")
        size_col = st.selectbox("Size (for scatter)", [None] + columns, key="viz_size_col")
        filter_col = st.selectbox("Filter Column", [None] + columns, key="viz_filter_col")
        filter_val = st.text_input("Filter Value", key="viz_filter_val") if filter_col else None
        filtered_df = df.copy()
        if filter_col and filter_val:
            filtered_df = filtered_df[filtered_df[filter_col].astype(str) == filter_val]
        chart = None
        if st.button("Build Chart", key="viz_build_btn"):
            if chart_type == "Bar" and x_col and y_col:
                chart = px.bar(filtered_df, x=x_col, y=y_col, color=color_col if color_col else None)
            elif chart_type == "Line" and x_col and y_col:
                chart = px.line(filtered_df, x=x_col, y=y_col, color=color_col if color_col else None)
            elif chart_type == "Scatter" and x_col and y_col:
                chart = px.scatter(filtered_df, x=x_col, y=y_col, color=color_col if color_col else None, size=size_col if size_col else None)
            elif chart_type == "Pie" and x_col and y_col:
                chart = px.pie(filtered_df, names=x_col, values=y_col, color=color_col if color_col else None)
            elif chart_type == "Box" and y_col:
                chart = px.box(filtered_df, y=y_col, color=color_col if color_col else None)
            elif chart_type == "Histogram" and x_col:
                chart = px.histogram(filtered_df, x=x_col, color=color_col if color_col else None)
            elif chart_type == "Heatmap" and x_col and y_col:
                chart = px.density_heatmap(filtered_df, x=x_col, y=y_col, color_continuous_scale="Viridis")
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        # --- Chart Templates & One-Click Dashboards ---
        st.subheader("Chart Templates & One-Click Dashboards")
        if st.button("Generate Dashboard", key="viz_dashboard_btn"):
            st.markdown("### Dashboard Preview")
            if num_cols and cat_cols:
                st.plotly_chart(px.bar(df, x=cat_cols[0], y=num_cols[0]), use_container_width=True)
            if num_cols and date_cols:
                st.plotly_chart(px.line(df, x=date_cols[0], y=num_cols[0]), use_container_width=True)
            if len(num_cols) > 1:
                st.plotly_chart(px.scatter(df, x=num_cols[0], y=num_cols[1]), use_container_width=True)
        # --- Interactive Filtering & Drill-Down ---
        st.subheader("Interactive Filtering & Drill-Down")
        for col in columns:
            if str(df[col].dtype).startswith("int") or str(df[col].dtype).startswith("float"):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                val_range = st.slider(f"Filter {col} range", min_val, max_val, (min_val, max_val), key=f"viz_slider_{col}")
                filtered_df = filtered_df[(filtered_df[col] >= val_range[0]) & (filtered_df[col] <= val_range[1])]
        st.dataframe(filtered_df)
        # --- Export & Sharing Options ---
        st.subheader("Export & Sharing Options")
        if chart:
            st.download_button("Download Chart as PNG", chart.to_image(format="png"), file_name="chart.png")
            st.download_button("Download Chart as HTML", chart.to_html(), file_name="chart.html")
        st.download_button("Download Dashboard Data (CSV)", filtered_df.to_csv(index=False), file_name="dashboard_data.csv")
        # --- AI-Powered Insight Annotations ---
        st.subheader("AI-Powered Insight Annotations")
        if chart and st.button("Generate AI Insights", key="viz_ai_insights_btn"):
            import openai
            ai_prompt = f"Analyze this chart and data. Give 2-3 key insights, trends, or anomalies.\nData columns: {list(filtered_df.columns)}\nSample data: {filtered_df.head(10).to_dict()}"
            try:
                client = openai.OpenAI(api_key=st.session_state.openai_key)
                ai_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "You are a data visualization expert."}, {"role": "user", "content": ai_prompt}],
                    temperature=0.1
                ).choices[0].message.content
                st.info(ai_response)
            except Exception as e:
                st.error(f"AI insight error: {e}")
        # --- Custom Calculations & Aggregations ---
        st.subheader("Custom Calculations & Aggregations")
        calc_col = st.selectbox("Select column for calculation", [None] + num_cols, key="viz_calc_col")
        calc_type = st.selectbox("Calculation type", ["None", "Moving Average", "Growth Rate"], key="viz_calc_type")
        window = st.number_input("Window (for moving average)", min_value=1, max_value=100, value=3, key="viz_calc_window") if calc_type == "Moving Average" else None
        if st.button("Apply Calculation", key="viz_calc_btn") and calc_col and calc_type != "None":
            calc_df = filtered_df.copy()
            if calc_type == "Moving Average":
                calc_df[f"{calc_col}_ma{window}"] = calc_df[calc_col].rolling(window=window).mean()
                st.dataframe(calc_df[[calc_col, f"{calc_col}_ma{window}"]])
            elif calc_type == "Growth Rate":
                calc_df[f"{calc_col}_growth"] = calc_df[calc_col].pct_change()
                st.dataframe(calc_df[[calc_col, f"{calc_col}_growth"]])
    pass  # (your main app logic remains here)

with tabs[1]:
    st.header("📝 SQL Analysis")
    if uploaded_file is not None and temp_path and columns and df is not None:
        st.info("Write or edit SQL queries below. Table name is 'uploaded_data'.")
        # --- 1. AI-Powered SQL Auto-Completion & Error Correction ---
        st.subheader("SQL Query Editor with AI Assistance")
        default_sql = f"SELECT * FROM uploaded_data LIMIT 10;"
        # Use session_state to persist editor value
        if "sql_editor" not in st.session_state:
            st.session_state["sql_editor"] = default_sql
        sql_query = st.session_state["sql_editor"]
        col1, col2 = st.columns([1,1])
        with col1:
            run_query_clicked = st.button("Run Query")
        with col2:
            ai_suggest_clicked = st.button("AI: Suggest Completion / Fix Errors")
        # Render the text area after handling button logic
        sql_query = st.text_area("SQL Query", value=sql_query, height=150, key="sql_editor")
        # Visual Query Builder (update session_state and rerun)
        with st.expander("Visual Query Builder (No-code)"):
            selected_cols = st.multiselect("Select columns", columns, default=columns[:1], key="sql_vqb_select_cols")
            filter_col = st.selectbox("Filter column", [None] + columns, key="sql_vqb_filter_col")
            filter_val = st.text_input("Filter value") if filter_col else None
            group_by = st.multiselect("Group by", columns, key="sql_vqb_group_by")
            order_by = st.selectbox("Order by", [None] + columns, key="sql_vqb_order_by")
            order_dir = st.radio("Order direction", ["ASC", "DESC"], key="sql_vqb_order_dir")
            limit = st.number_input("Limit", min_value=1, max_value=1000, value=10, key="sql_vqb_limit")
            if st.button("Build SQL Query", key="sql_vqb_build_btn"):
                visual_sql = f"SELECT {', '.join(selected_cols) if selected_cols else '*'} FROM uploaded_data"
                if filter_col and filter_val:
                    visual_sql += f" WHERE {filter_col} = '{filter_val}'"
                if group_by:
                    visual_sql += f" GROUP BY {', '.join(group_by)}"
                if order_by:
                    visual_sql += f" ORDER BY {order_by} {order_dir}"
                if limit:
                    visual_sql += f" LIMIT {limit}"
                st.session_state["sql_editor"] = visual_sql
                st.experimental_rerun()
        # Handle Run Query and AI Suggestion buttons
        if run_query_clicked:
            import duckdb
            con = duckdb.connect()
            con.register("uploaded_data", df)
            try:
                result = con.execute(st.session_state["sql_editor"]).fetchdf()
                st.dataframe(result)
                st.success("Query executed successfully.")
            except Exception as e:
                st.error(f"SQL Error: {e}")
            finally:
                con.close()
        if ai_suggest_clicked:
            ai_prompt = f"You are a SQL expert. Review and improve this SQL for DuckDB. Suggest completions, fix errors, and explain any changes.\nSQL:\n{st.session_state['sql_editor']}"
            try:
                client = openai.OpenAI(api_key=st.session_state.openai_key)
                ai_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "You are a SQL expert."}, {"role": "user", "content": ai_prompt}],
                    temperature=0.1
                ).choices[0].message.content
                st.info(ai_response)
            except Exception as e:
                st.error(f"AI suggestion error: {e}")
        # --- 2. Interactive Visual Query Builder ---
        with st.expander("Visual Query Builder (No-code)"):
            selected_cols = st.multiselect("Select columns", columns, default=columns[:1])
            filter_col = st.selectbox("Filter column", [None] + columns)
            filter_val = st.text_input("Filter value") if filter_col else None
            group_by = st.multiselect("Group by", columns)
            order_by = st.selectbox("Order by", [None] + columns)
            order_dir = st.radio("Order direction", ["ASC", "DESC"])
            limit = st.number_input("Limit", min_value=1, max_value=1000, value=10)
            if st.button("Build SQL Query"):
                visual_sql = f"SELECT {', '.join(selected_cols) if selected_cols else '*'} FROM uploaded_data"
                if filter_col and filter_val:
                    visual_sql += f" WHERE {filter_col} = '{filter_val}'"
                if run_query_clicked:
                    import duckdb
                    con = duckdb.connect()
                    con.register("uploaded_data", df)
                    try:
                        result = con.execute(st.session_state["sql_editor"]).fetchdf()
                        st.dataframe(result)
                        st.success("Query executed successfully.")
                    except Exception as e:
                        st.error(f"SQL Error: {e}")
                    finally:
                        con.close()
                if ai_suggest_clicked:
                    ai_prompt = f"You are a SQL expert. Review and improve this SQL for DuckDB. Suggest completions, fix errors, and explain any changes.\nSQL:\n{st.session_state['sql_editor']}"
                    try:
                        client = openai.OpenAI(api_key=st.session_state.openai_key)
                        ai_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "system", "content": "You are a SQL expert."}, {"role": "user", "content": ai_prompt}],
                            temperature=0.1
                        ).choices[0].message.content
                        st.info(ai_response)
                    except Exception as e:
                        st.error(f"AI suggestion error: {e}")
        try:
            result = con.execute(sql_query).fetchdf()
            exec_time = time.time() - start_time
            st.dataframe(result)
            st.success(f"Query executed successfully in {exec_time:.3f} seconds.")
            # Query plan
            plan = con.execute(f"EXPLAIN {sql_query}").fetchall()
            st.code("\n".join(str(row[0]) for row in plan), language="text")
            # LLM-powered optimization
            if st.button("Suggest Query Optimization"):
                opt_prompt = f"Analyze and optimize this DuckDB SQL query for performance. Suggest improvements, indexes, or rewrites.\nSQL:\n{sql_query}\nDuckDB EXPLAIN output:\n{chr(10).join(str(row[0]) for row in plan)}"
                try:
                    client = openai.OpenAI(api_key=st.session_state.openai_key)
                    opt_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system", "content": "You are a SQL performance expert."}, {"role": "user", "content": opt_prompt}],
                        temperature=0.1
                    ).choices[0].message.content
                    st.info(opt_response)
                except Exception as e:
                    st.error(f"Optimization suggestion error: {e}")
        except Exception as e:
            st.error(f"SQL Error: {e}")
        finally:
            if 'con' in locals():
                con.close()
        # --- 5. Collaborative SQL & Data Analysis Workspace ---
        st.subheader("Collaborative Workspace")
        # Share Query (generate shareable code)
        if st.button("Generate Shareable Query Code"):
            import base64
            code = base64.urlsafe_b64encode(sql_query.encode()).decode()
            st.code(code, language="text")
            st.info("Share this code with a teammate. They can paste it below to load your query.")
        shared_code = st.text_input("Paste shared query code here to load:")
        if st.button("Load Shared Query") and shared_code:
            try:
                import base64
                loaded_sql = base64.urlsafe_b64decode(shared_code.encode()).decode()
                st.session_state["sql_editor"] = loaded_sql
                st.success("Loaded shared query!")
            except Exception as e:
                st.error(f"Failed to load shared query: {e}")
        # Query versioning and comments (simple session-based)
        st.subheader("Query Versioning & Comments")
        comment = st.text_input("Comment on this query:")
        if st.button("Save Query Version"):
            if "query_versions" not in st.session_state:
                st.session_state.query_versions = []
            st.session_state.query_versions.append({"sql": sql_query, "comment": comment})
            st.success("Query version saved!")
        if "query_versions" in st.session_state and st.session_state.query_versions:
            st.markdown("**Query History:**")
            for i, v in enumerate(st.session_state.query_versions[::-1]):
                st.code(v["sql"], language="sql")
                if v["comment"]:
                    st.markdown(f"_Comment: {v['comment']}_")