# ☀️ Solar Industry AI Assistant

An AI-powered rooftop analysis tool that uses satellite imagery to assess the potential for solar panel installations. Built for solar professionals and homeowners, this system offers installation recommendations and ROI estimates using vision-based AI and intelligent analysis.

---

## 🚀 Project Overview

This project was developed as part of an internship assessment to demonstrate the integration of large language models (LLMs), computer vision, and web-based delivery for solving a real-world solar industry challenge.

**Core Goal:**  
Analyze rooftops from satellite images and generate accurate solar potential assessments, panel layout suggestions, and ROI predictions.

---

## 🧠 Features

- 🔍 **Rooftop Analysis via Vision AI**  
  Identifies usable rooftop space for solar installation using satellite images.

- 📊 **Solar Potential & ROI Estimation**  
  Calculates panel fit, energy generation, installation costs, and estimated payback periods.

- 📦 **Installation Recommendations**  
  Provides ideal panel types, quantity, mounting angles, and cost insights.

- 🧾 **Structured AI Output**  
  Uses prompt engineering and context management to produce accurate, formatted reports.

---

## 🛠️ Technologies Used

| Area                | Tools/Tech                            |
|---------------------|----------------------------------------|
| AI Vision           | OpenRouter API / Hugging Face Vision |
| Backend / Frontend  | Python, Streamlit or Gradio           |
| Deployment          | Hugging Face Spaces / Local ZIP       |
| Data Handling       | Prompt Engineering, Context Wrangling |
| Documentation       | Markdown, GitHub                      |

---

## ⚙️ Project Setup

### 🔧 Environment Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/solar-ai-assistant.git
   cd solar-ai-assistant
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set environment variables (API keys etc.) in a `.env` file.

---

## ▶️ Run the Application

### Option 1: Streamlit (Default)
```bash
streamlit run app.py
```

### Option 2: Gradio
```bash
python gradio_app.py
```

---

## 💼 Example Use Cases

- A solar sales rep uploads an image of a customer’s rooftop and receives a detailed solar panel layout with ROI projections.
- A homeowner wants to evaluate the solar potential of their property before contacting providers.
- An engineer explores various installation angles and configurations for cost-performance tradeoffs.

---

## 📈 Potential Improvements

- Integrate dynamic weather and regional solar data for real-time accuracy.
- Add 3D rooftop modeling and shadow detection using LiDAR or Google Earth API.
- Connect to solar provider APIs for automated cost estimation.
- Expand support for regulatory codes and permit estimation.

---

## 📂 Deliverables

- ✅ Full Codebase
- ✅ Local Setup Instructions
- ✅ Sample Output & Use Cases
- ✅ Deployment-Ready Interface (Streamlit or Gradio)

---

## 📜 License

This project is developed for educational and evaluation purposes. Contact for further usage rights or contributions.
