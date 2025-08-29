<img width="908" height="895" alt="Logo" src="https://github.com/user-attachments/assets/3b653d5a-03a4-4f30-9a4b-27010c3d8656" />

<img width="1916" height="1056" alt="page" src="https://github.com/user-attachments/assets/a0e2cdd1-c6a8-495e-91b2-c5d5a707dfc4" />

🔗 **Live App:** [data-analyst-ai-agent.streamlit.app](https://data-analyst-ai-agent.streamlit.app/)

<img width="1916" height="1018" alt="page2" src="https://github.com/user-attachments/assets/e79cb4e2-5cd2-49fd-9e4e-1cbe0887474a" />



---

# 📊 Data Analyst Agent 🤖

### Unlock the Power of Data with AI — Instantly Analyze, Visualize, and Discover Insights!

🔗 **Live App:** [data-analyst-ai-agent.streamlit.app](https://data-analyst-ai-agent.streamlit.app/)

---

## 🚀 Project Overview

**Data Analyst Agent** is a next-generation AI-powered analytics platform built with **Streamlit**. It empowers users—whether beginners, business professionals, or data scientists—to **upload their data, clean it, analyze it, visualize it, and even build machine learning models instantly**.

The app integrates **modern UI/UX design** with **cutting-edge AI models**, providing both **automation** and **interactivity** in a seamless way. Users can chat with their data, generate SQL queries, build dashboards, and even perform **AutoML**—all in one place.

This is not just a tool—it’s your **virtual AI Data Analyst**.

---

## ✨ Key Features

### 🎨 1. Modern, Animated UI/UX

* Dark neon-inspired design with glassmorphism.
* Animated gradients and interactive transitions.
* Sidebar with profile, settings, and navigation.
* Responsive and elegant design for all devices.

---

### 📂 2. Sidebar Features

* App logo + user profile section.
* Quick navigation across tabs (Upload, Insights, SQL, Visualization, ML, etc.).
* Theme switcher (light/dark/glass).
* Recent file history for quick access.
* API key management (`.env` or manual entry).
* App settings (language, font size, preferences).
* Social links (GitHub, Twitter, LinkedIn).
* Support, feedback, and developer credit section.

---

### 📑 3. Data Upload & Preprocessing

* Upload **CSV/Excel** files with drag-and-drop.
* Automatic type detection, date parsing, and missing value handling.
* Preview dataset instantly.

---

### 🧹 4. Data Cleaning & Transformation

* Fill, drop, or filter missing values.
* Convert/transform data types interactively.
* Apply filters and transformations directly from UI.

---

### 📊 5. Automated Insights & Anomaly Detection

* AI-powered anomaly detection.
* Summary statistics at your fingertips.
* OpenAI-powered insights for **business storytelling with data**.

---

### 💬 6. Conversational Data Exploration

* Chat with your data like a human analyst.
* Multi-turn memory for context-aware conversations.
* Export chat history, SQL queries, and responses.

---

### 🧑‍💻 7. SQL Analysis

* AI-generated SQL queries from plain English.
* Query correction with error explanations.
* Visual query builder for no-code SQL.
* Version control and collaborative query sharing.

---

### 📈 8. Data Visualization Studio

* Automatic chart recommendations.
* Drag-and-drop chart builder.
* Build complete dashboards with filtering & drill-down.
* Export charts as images, data, or dashboards.
* AI-powered annotations & insights.

---

### 📊 9. Power BI Analyst

* Data modeling & KPI dashboards.
* Auto-generate **DAX** and **Power Query** scripts.
* Data source connectors and refresh scheduling.
* Export reports for Power BI.

---

### ⚡ 10. Advanced BI Tools

* Time intelligence (period over period analysis).
* Drill-through navigation.
* Custom visual marketplace (e.g., radar charts).
* Row-level security simulation.
* Publish & share dashboards securely.

---

### 🤖 11. Machine Learning Analyst

* AutoML model builder (classification & regression).
* Feature engineering & selection tools.
* Explainability via **SHAP** values.
* Prediction playground for testing models.
* Export models for reuse.

---

## 🛠️ Tech Stack & Integrations

* **Frontend/UI:** Streamlit (dark neon theme with animations).
* **Data Processing:** Pandas, NumPy, DuckDB.
* **LLM Features:** OpenAI API.
* **Visualization:** Plotly, PyDeck.
* **Machine Learning:** Scikit-learn, SHAP, Joblib.
* **Environment Management:** `.env` + Streamlit secrets.

---

## 📌 Why This Project?

Data analysis is often time-consuming and technical. With **Data Analyst Agent**, you get the power of a **data analyst, BI engineer, and ML engineer** all in one **friendly AI-powered tool**. Whether you’re analyzing sales data, visualizing trends, or building predictive models—this app does it all, instantly.

---

## 🧩 App Workflow (LangGraph + Decision Tree)

Below is the **end-to-end workflow** of how the app works, represented as a **graph of nodes (states) and edges (decisions/actions):**

```mermaid
flowchart TD

%% START
A[🚀 Start App\nUser Opens Streamlit App] --> B[📂 Upload Data\n(CSV/Excel)]

%% DATA UPLOAD
B --> C{✅ File Uploaded?}
C -->|No| B
C -->|Yes| D[🔍 Preprocessing\nType inference, missing values, date parsing]

%% CLEANING & TRANSFORMATION
D --> E[🧹 Data Cleaning\nFilter, fill, drop, type change]
E --> F[📊 Automated Insights\nStats + Anomaly Detection + LLM insights]

%% BRANCH OPTIONS
F --> G{📌 Next Step?}

%% Conversational Exploration
G -->|💬 Chat with Data| H[🤖 Conversational Q&A\nMulti-turn memory, export chat]

%% SQL
G -->|🧑‍💻 SQL Analysis| I[🗃️ SQL Query Generator\nAI query + visual builder + versioning]

%% Visualization
G -->|📈 Visualization| J[🎨 Visualization Studio\nAuto charts + dashboard builder]

%% Power BI
G -->|📊 Power BI Analyst| K[⚡ Power BI Tools\nDAX, metrics, connector hub]

%% Machine Learning
G -->|🤖 ML Analyst| L[🧠 AutoML Model Builder\nTrain, explain, predict, export]

%% Endpoints
H --> Z[✅ Download Results]
I --> Z
J --> Z
K --> Z
L --> Z

%% END
Z --> X[🏁 End Session / Export Data & Reports]
```

---

### 🛠 How It Works (Step by Step)

1. **🚀 Launch App** → User opens the Streamlit app (live demo).
2. **📂 Upload Data** → CSV/Excel file uploaded.
3. **🔍 Preprocessing** → Automatic type detection, missing value handling, date parsing.
4. **🧹 Cleaning** → User interacts to clean/transform data.
5. **📊 Insights** → LLM + stats engine provides anomalies & insights.
6. **Decision Node** → User chooses next action:

   * **💬 Chat** → Natural language exploration.
   * **🗃️ SQL** → AI-powered SQL queries & visual builder.
   * **🎨 Visualization** → Build charts/dashboards.
   * **⚡ Power BI** → Advanced BI & DAX scripts.
   * **🧠 ML Analyst** → Train AutoML models.
7. **✅ Export** → Download insights, dashboards, queries, models.
8. **🏁 End** → Session complete, reports ready to share.

---


---

## 👨‍💻 Developer Notes
Abhi Yadav
This project is actively evolving. Future updates will bring:

* More visualization templates.
* Deeper integrations with BI tools (Tableau, Power BI cloud).
* Natural language → dashboard automation.
* Real-time data streaming support.

---

## 🌐 Live Demo

👉 Try it here: [data-analyst-ai-agent.streamlit.app](https://data-analyst-ai-agent.streamlit.app/)

---

Would you like me to **format this into a polished `README.md` file (with emojis, badges, and installation guide)** so you can directly upload to GitHub?
