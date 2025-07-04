# 🗣️ Improving AI Voice Assistants: A Danish NLP Evaluation and Enhancement Framework

## 🔍 Overview

This project presents a comprehensive framework to evaluate, improve, and visualize the performance of AI voice assistants in Danish. It combines synthetic data simulation, advanced NLP preprocessing, fine-tuned transformer models, and interactive visualizations. Designed for underrepresented languages, the pipeline identifies critical linguistic and contextual errors and enhances the performance and interpretability of intent recognition models.

> ⚠️ **Note**: This project uses a synthetic dataset for demonstration. It reflects common linguistic structures but does not represent real user behavior. Results should be interpreted accordingly.

---

## 🎯 Objective

### 📌 Business Context

Voice assistants are integral to digital ecosystems. However, underrepresented languages like Danish lack robust NLP support. Misunderstandings in native language interactions reduce user trust and satisfaction.

### 🎯 Goal

To build an end-to-end system for:
- Preprocessing and validating Danish conversational datasets.
- Enhancing intent classification using BERT-based models.
- Evaluating paraphrase similarity and user satisfaction.
- Visualizing model insights via EDA and Streamlit apps.
- Guiding future improvements with actionable metrics.

---

## 🧱 Project Components

### 1. 🧹 Data Cleaning & Preprocessing

- Deduplication of over 1,100 redundant rows.
- Context enrichment using entity parsing.
- Advanced text normalization (Danish-specific).
- Context-aware tokenization using spaCy pipelines.
- EDA readiness and quality certification (intent balance, context coverage, feedback metrics).

### 2. 📊 Exploratory Data Analysis (EDA)

EDA modules include:
- **Intent Analysis**: Balanced across 6 classes (e.g., `påmindelse`, `vejrudsigten`, `nyheder`)
- **User Satisfaction**:
  - Helpfulness: 73.7%
  - Average Rating: 3.95/5
  - Satisfaction impacted by `needs_clarification` (−0.62 correlation)
- **Entity Analysis**:
  - 30%+ of interactions enriched with entities (city, time)
  - Entities improve satisfaction by ~4%
- **Paraphrase Similarity**:
  - Mean Jaccard score: 0.993
  - Semantic similarity (MiniLM): 0.96 cosine avg
- **Contextual Impact**:
  - Satisfaction varies across city and time contexts

### 3. 🤖 Model Training & Evaluation

#### Intent Classification

| Model             | Accuracy | Precision | Recall | F1-score |
|------------------|----------|-----------|--------|----------|
| Danish BERT      | 0.976    | 0.976     | 0.976  | 0.976    |
| XLM-RoBERTa       | 0.973    | 0.973     | 0.973  | 0.973    |

- Input: Cleaned Danish utterances
- Label: One of 6 intents
- Features: Context-aware text, embedded with spaCy and transformers

#### Paraphrase Similarity

- Model: `paraphrase-multilingual-MiniLM-L12-v2`
- Mean cosine similarity: 0.96
- Strong alignment across all intents

### 4. 📈 Comparative Analysis

- Paraphrase consistency does not harm satisfaction.
- Clarifications significantly reduce user satisfaction.
- Facebook-like short queries (`påmindelse`) yield higher accuracy and satisfaction.
- Recommendation: Strengthen out-of-scope and question-answering logic.

---

## 💻 Technologies Used

| Tool/Library       | Purpose                                      |
|--------------------|----------------------------------------------|
| **Python**         | Data processing, modeling                    |
| **pandas, NumPy**  | Data handling and transformation             |
| **spaCy**          | NLP preprocessing with Danish pipelines      |
| **Transformers**   | BERT & XLM-RoBERTa model training            |
| **SentenceTransformers** | Semantic similarity modeling         |
| **Matplotlib/Seaborn** | Data visualization                     |
| **Streamlit**      | Interactive dashboard (planned)              |
| **Parquet/JSON**   | Efficient data storage and reporting         |


---

## 📌 Key Takeaways

- Robust NLP pipelines can greatly improve underrepresented languages in AI.
- Context, entities, and paraphrase variation are critical to satisfaction.
- BERT-based multilingual models perform exceptionally well for Danish intents.
- Paraphrase diversity is not detrimental—diverse phrasing increases model robustness.

---

## 🔮 Future Improvements

1. Add a **Streamlit app** to demo model predictions and satisfaction analysis.
2. Integrate **real Danish datasets** to validate findings in production.
3. Add **voice-to-text** preprocessing to simulate full assistant workflows.
4. Improve anomaly detection for out-of-scope intents and unusual phrasing.

---

## 📬 Contact

Built with ❤️ by a data scientist passionate about multilingual NLP and human-centered AI.  
For inquiries or collaboration ideas, feel free to [connect via LinkedIn](#) or raise an issue in the repo.

---

