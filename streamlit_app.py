import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import seaborn as sns
import re
import os
import torch
from pathlib import Path

# =============================================================================
# APP CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="🇩🇰 Danish Voice Assistant NLP",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-card {
        border-radius: 10px;
        padding: 15px;
        background-color: #f0f2f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .section-title {
        font-size: 24px;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .intent-display {
        font-size: 22px;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        text-align: center;
        margin: 15px 0;
    }
    /* Rule example styling */
    .rule-example {
        padding: 12px 15px;
        border-radius: 10px;
        margin: 12px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .rule-example:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .news-rule {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        color: #0d47a1;
    }
    .question-rule {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
        color: #1b5e20;
    }
    .courtesy-rule {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 5px solid #9c27b0;
        color: #4a148c;
    }
    .rule-title {
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    .rule-title i {
        margin-right: 8px;
        font-size: 20px;
    }
    .rule-example-item {
        padding: 6px 0;
        border-bottom: 1px dashed rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
    }
    .rule-example-item:last-child {
        border-bottom: none;
    }
    .arrow {
        margin: 0 10px;
        font-size: 18px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data():
    try:
        # Use Path for cross-platform compatibility
        data_path = Path("data") / "da_voice_assistant_enriched.csv"
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        df = pd.DataFrame()
    
    json_files = {}
    for file in ['improvement_plan', 'baseline_results', 'sentence_similarity_results']:
        try:
            json_path = Path("results") / f"{file}.json"
            with open(json_path) as f:
                json_files[file] = json.load(f)
        except Exception as e:
            st.error(f"Failed to load {file}.json: {str(e)}")
            json_files[file] = {}
    
    # Load label encoder for intent mapping
    label_encoder = {}
    try:
        label_encoder_path = Path("data") / "label_encoder.json"
        with open(label_encoder_path) as f:
            label_encoder = json.load(f)
    except Exception as e:
        st.error(f"Failed to load label encoder: {str(e)}")
    
    return df, json_files.get('improvement_plan', {}), json_files.get('baseline_results', {}), json_files.get('sentence_similarity_results', {}), label_encoder

# =============================================================================
# INTENT MAPPING
# =============================================================================
def get_intent_mapping():
    """Map encoded labels to Danish and English intent names"""
    return {
        "LABEL_0": {"danish": "alarm", "english": "set alarm"},
        "LABEL_1": {"danish": "nyheder", "english": "news"},
        "LABEL_2": {"danish": "out_of_scope", "english": "out of scope"},
        "LABEL_3": {"danish": "påmindelse", "english": "set reminder"},
        "LABEL_4": {"danish": "spørgsmål", "english": "question"},
        "LABEL_5": {"danish": "vejrudsigten", "english": "weather forecast"}
    }

# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_model():
    try:
        model_dir = Path("models") / "danish_bert_intent"
        
        # Check if model directory exists
        if not model_dir.exists():
            st.error(f"Model directory not found: {model_dir}")
            return None
            
        # Load configuration
        config = AutoConfig.from_pretrained(model_dir)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model with safetensors
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            config=config,
            use_safetensors=True
        )
        
        # Apply quantization for efficient CPU inference
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        quantized_model.eval()
        
        # Create pipeline
        return pipeline(
            "text-classification",
            model=quantized_model,
            tokenizer=tokenizer,
            device=-1  # Force CPU usage
        )
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================
def get_recommendations(intent):
    """Generate recommendations based on predicted intent"""
    recommendations = {
        "nyheder": [
            "Add news source selection capability",
            "Include trending news categories",
            "Implement 'more details' option for news items"
        ],
        "spørgsmål": [
            "Provide multiple answer options for questions",
            "Suggest follow-up questions",
            "Offer to save question for later response"
        ],
        "out_of_scope": [
            "Politely suggest alternative actions",
            "Offer to transfer to human agent",
            "Provide help documentation links"
        ],
        "alarm": [
            "Confirm alarm details before setting",
            "Suggest smart alarm options",
            "Enable multiple alarm functionality"
        ],
        "påmindelse": [
            "Implement location-based reminders",
            "Enable reminder sharing with others",
            "Provide detailed confirmation"
        ],
        "vejrudsigten": [
            "Include weather alerts system",
            "Show 7-day forecast option",
            "Suggest clothing based on weather"
        ]
    }
    return recommendations.get(intent, ["No specific recommendations available"])

# =============================================================================
# RULE-BASED CORRECTION
# =============================================================================
def apply_rule_based_corrections(text, current_prediction):
    """Apply rule-based corrections to model predictions"""
    correction_rules = [
        {
            "pattern": r"\b(?:nyhed(?:er)?|avisen|dagblad(?:et)?|avis(?:en)?|overskrift(?:er)?|seneste nyt)\b",
            "correct_intent": "nyheder",
            "examples": ["Hvad er seneste nyt?", "Vis mig avisen", "Hvad sker der i dagbladet?"]
        },
        {
            "pattern": r"\b(?:hvem|hvad|hvor|hvorfor|hvornår|forklar|betyder)\b",
            "correct_intent": "spørgsmål",
            "examples": ["Hvem er statsminister?", "Hvorfor er himlen blå?", "Hvad betyder dette ord?"]
        },
        {
            "pattern": r"\b(?:tak|farvel|hej|undskyld|hav en god)\b",
            "correct_intent": "out_of_scope",
            "examples": ["Tak for hjælpen", "Hej, hvordan går det?", "Farvel for nu"]
        }
    ]
    
    for rule in correction_rules:
        try:
            if re.search(rule["pattern"], text, re.IGNORECASE):
                return rule["correct_intent"], rule["examples"]
        except re.error as e:
            st.error(f"Regex error in rule: {str(e)}")
    return current_prediction, []

# =============================================================================
# APP SECTIONS
# =============================================================================
def project_overview():
    st.markdown('<div class="section-title">🗣️ Danish Voice Assistant NLP Project</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Project Overview
        This end-to-end NLP pipeline demonstrates the development of a Danish-language voice assistant, 
        addressing challenges specific to low-resource languages. The project includes:
        
        - **Data Validation**: Quality checks for Danish text data
        - **Intent Classification**: Using Danish BERT and XLM-Roberta
        - **Error Analysis**: Simulated confusion patterns
        - **Improvement Plan**: Rule-based corrections and augmentation
        
        ### Dataset Information
        - **Synthetic dataset** created for demonstration purposes
        - 5,473 Danish voice assistant interactions
        - 6 intents: alarm, nyheder, out_of_scope, påmindelse, spørgsmål, vejrudsigten
        """)
        
        st.markdown("""
        <div class="warning-box">
        <strong>Portfolio Note</strong>:  
        This project uses a synthetic dataset. Real-world performance may vary due to:
        - Artificial patterns in generated data
        - Limited domain coverage
        - Potential bias in data generation
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Use placeholder if image not available
        image_path = Path("results") / "simulated_confusion_matrix.png"
        if image_path.exists():
            st.image(str(image_path), caption="Simulated Confusion Matrix")
        else:
            st.warning(f"Confusion matrix image not found: {image_path}")
        
        # Use placeholder if image not available
        image_path = Path("results") / "intent_mean_similarity.png"
        if image_path.exists():
            st.image(str(image_path), caption="Intent Similarity Analysis")
        else:
            st.warning(f"Similarity analysis image not found: {image_path}")

def data_exploration(df):
    st.markdown('<div class="section-title">🔍 Data Exploration</div>', unsafe_allow_html=True)
    
    if df.empty:
        st.error("No data available for exploration")
        return
    
    # Intent distribution
    st.subheader("Intent Distribution")
    intent_counts = df['intent'].value_counts().reset_index()
    intent_counts.columns = ['Intent', 'Count']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=intent_counts, x='Intent', y='Count', palette='viridis')
    plt.title('Intent Distribution')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Satisfaction metrics
    st.subheader("User Satisfaction Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Rating", f"{df['user_rating'].mean():.2f}/5.0")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Helpfulness Rate", f"{df['was_helpful'].mean()*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Satisfaction Score", f"{df['user_satisfaction'].mean():.2f}/10.0")
        st.markdown('</div>', unsafe_allow_html=True)

def model_performance(baseline_results, similarity_results):
    st.markdown('<div class="section-title">🤖 Model Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Metrics")
        
        if 'classifiers' in baseline_results and 'danish-bert-botxo' in baseline_results['classifiers']:
            bert_results = baseline_results['classifiers']['danish-bert-botxo']
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{bert_results.get('accuracy', 0)*100:.2f}%")
            st.metric("F1-Score", f"{bert_results.get('f1', 0)*100:.2f}%")
            st.metric("Training Time", f"{bert_results.get('training_time', 0)/60:.1f} min")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Baseline results not available")
        
        st.subheader("Recommendations")
        st.info("""
        - **Production Implementation**: Add rule-based correction layer
        - **Data Enhancement**: Focus on nyheder/spørgsmål confusion pairs
        - **Monitoring**: Track low-confidence predictions in real-time
        """)
    
    with col2:
        st.subheader("Performance Visualization")
        
        # Use placeholder if image not available
        image_path = Path("results") / "intent_classification_comparison.png"
        if image_path.exists():
            st.image(str(image_path), caption="Model Comparison")
        else:
            st.warning(f"Model comparison image not found: {image_path}")
        
        # Use placeholder if image not available
        image_path = Path("results") / "sentence_similarity_distribution.png"
        if image_path.exists():
            st.image(str(image_path), caption="Sentence Similarity Distribution")
        else:
            st.warning(f"Similarity distribution image not found: {image_path}")
        
        st.subheader("Critical Findings")
        st.warning("""
        - High confusion between nyheder ↔ spørgsmål
        - out_of_scope frequently misclassified
        - Low similarity scores for nyheder paraphrases
        """)

def interactive_demo(classifier, improvement_plan, label_encoder):
    st.markdown('<div class="section-title">🎮 Interactive Demo</div>', unsafe_allow_html=True)
    
    if classifier is None:
        st.error("⚠️ Model failed to load. Demo functionality disabled.")
        return
    
    # Create intent mapping
    intent_mapping = get_intent_mapping()
    
    user_input = st.text_area("Test Danish voice input:", 
                              "Hvad er vejret i København i morgen?",
                              height=100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Predict Intent", use_container_width=True):
            try:
                # Get prediction
                prediction = classifier(user_input)[0]
                label = prediction['label']
                confidence = prediction['score']
                
                # Map label to intent
                if label in intent_mapping:
                    danish_intent = intent_mapping[label]["danish"]
                    english_intent = intent_mapping[label]["english"]
                else:
                    # Fallback to label encoder if available
                    danish_intent = label_encoder.get(label, label)
                    english_intent = danish_intent  # Default to same if no translation
                
                # Display results with styling
                st.markdown(f'<div class="intent-display">🇩🇰 Danish: {danish_intent}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="intent-display">🇬🇧 English: {english_intent}</div>', unsafe_allow_html=True)
                
                st.metric("Confidence", f"{confidence:.2%}")
                st.progress(float(confidence))
                
                # Apply rule-based correction
                corrected_danish, rule_examples = apply_rule_based_corrections(user_input, danish_intent)
                if corrected_danish != danish_intent:
                    # Find English equivalent for corrected intent
                    corrected_english = next(
                        (v["english"] for k, v in intent_mapping.items() if v["danish"] == corrected_danish),
                        corrected_danish
                    )
                    
                    st.warning(f"Rule-based correction applied: **{danish_intent} → {corrected_danish}**")
                    st.info(f"English equivalent: **{english_intent} → {corrected_english}**")
                    
                    danish_intent = corrected_danish
                    english_intent = corrected_english
                
                # Show recommendations
                st.subheader("Recommendations")
                for rec in get_recommendations(danish_intent):
                    st.markdown(f"- {rec}")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    with col2:
        st.subheader("Rule-Based Corrections")
        st.markdown("""
        The system applies these rules to correct model predictions based on keyword matching:
        """)
        
        # Show rule examples with new design
        st.subheader("Rule Examples")
        
        # News Rule
        st.markdown("""
        <div class="rule-example news-rule">
            <div class="rule-title"><i>📰</i> News Rule</div>
            <div class="rule-example-item">"Hvad er seneste nyt?" <span class="arrow">→</span> nyheder</div>
            <div class="rule-example-item">"Vis mig avisen" <span class="arrow">→</span> nyheder</div>
            <div class="rule-example-item">"Hvad sker der i dagbladet?" <span class="arrow">→</span> nyheder</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Question Rule
        st.markdown("""
        <div class="rule-example question-rule">
            <div class="rule-title"><i>❓</i> Question Rule</div>
            <div class="rule-example-item">"Hvem er statsminister?" <span class="arrow">→</span> spørgsmål</div>
            <div class="rule-example-item">"Hvorfor er himlen blå?" <span class="arrow">→</span> spørgsmål</div>
            <div class="rule-example-item">"Hvad betyder dette ord?" <span class="arrow">→</span> spørgsmål</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Courtesy Rule
        st.markdown("""
        <div class="rule-example courtesy-rule">
            <div class="rule-title"><i>👋</i> Courtesy Rule</div>
            <div class="rule-example-item">"Tak for hjælpen" <span class="arrow">→</span> out_of_scope</div>
            <div class="rule-example-item">"Hej, hvordan går det?" <span class="arrow">→</span> out_of_scope</div>
            <div class="rule-example-item">"Farvel for nu" <span class="arrow">→</span> out_of_scope</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Intent mapping reference
        st.subheader("Intent Mapping Reference")
        mapping_data = {
            "Label": list(intent_mapping.keys()),
            "Danish": [v["danish"] for v in intent_mapping.values()],
            "English": [v["english"] for v in intent_mapping.values()]
        }
        st.dataframe(pd.DataFrame(mapping_data), hide_index=True)

def improvement_roadmap(improvement_plan, similarity_results):
    st.markdown('<div class="section-title">🚀 Improvement Roadmap</div>', unsafe_allow_html=True)
    
    if not improvement_plan:
        st.error("Improvement plan data not available")
        return
    
    st.subheader("Key Findings")
    findings = improvement_plan.get("key_findings", {})
    col1, col2 = st.columns(2)
    
    with col1:
        if 'most_confused_intents' in findings:
            st.markdown("**Most Confused Intents**")
            for intent_pair in findings["most_confused_intents"]:
                st.markdown(f"- {intent_pair}")
        
        if 'rule_based_corrections' in findings:
            st.markdown("**Rule-Based Corrections**")
            for rule in findings["rule_based_corrections"]:
                st.markdown(f"- {rule}")
    
    with col2:
        if 'low_similarity_intents' in findings:
            st.markdown("**Low Similarity Intents**")
            for intent in findings["low_similarity_intents"]:
                st.markdown(f"- {intent}")
            
            # Generate similarity scores if available
            similarity_scores = []
            if 'intent_similarities' in similarity_results:
                for intent in findings["low_similarity_intents"]:
                    if intent in similarity_results['intent_similarities']:
                        score = similarity_results['intent_similarities'][intent]['mean']
                        similarity_scores.append(f"{intent}: {score:.4f}")
            
            if similarity_scores:
                st.markdown("**Similarity Scores**")
                for score in similarity_scores:
                    st.markdown(f"- {score}")
    
    st.subheader("Action Plan")
    if 'action_steps' in improvement_plan:
        for step in improvement_plan["action_steps"]:
            st.markdown(f"📌 {step}")
    else:
        st.warning("No action steps defined")
    
    st.subheader("Implementation Timeline")
    timeline_data = {
        "Task": [
            "Data Augmentation", 
            "Model Fine-tuning", 
            "Rule Implementation",
            "Adversarial Testing",
            "Monitoring Setup"
        ],
        "Start": ["2023-11-01", "2023-11-08", "2023-11-15", "2023-11-22", "2023-11-29"],
        "End": ["2023-11-07", "2023-11-14", "2023-11-21", "2023-11-28", "2023-12-05"]
    }
    st.dataframe(pd.DataFrame(timeline_data), hide_index=True)

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Check required directories
    required_dirs = ["models", "results", "data"]
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]
    
    if missing_dirs:
        st.error(f"⚠️ Critical: Missing required directories: {', '.join(missing_dirs)}")
        st.stop()
    
    # Load data and model
    df, improvement_plan, baseline_results, similarity_results, label_encoder = load_data()
    classifier = load_model()
    
    # Sidebar configuration
    st.sidebar.title("🇩🇰 Danish Voice Assistant")
    
    # Use flag emoji if image not available
    flag_path = Path("assets") / "danish_flag.png"
    if flag_path.exists():
        st.sidebar.image(str(flag_path), width=100)
    else:
        st.sidebar.subheader("🇩🇰")
    
    st.sidebar.markdown("### Navigation")
    
    sections = {
        "Project Overview": lambda: project_overview(),
        "Data Exploration": lambda: data_exploration(df),
        "Model Performance": lambda: model_performance(baseline_results, similarity_results),
        "Interactive Demo": lambda: interactive_demo(classifier, improvement_plan, label_encoder),
        "Improvement Roadmap": lambda: improvement_roadmap(improvement_plan, similarity_results)
    }
    
    selection = st.sidebar.radio("", list(sections.keys()))
    sections[selection]()
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Portfolio Project Notes**:
    - Uses synthetic Danish dataset
    - Focuses on workflow demonstration
    - 100% accuracy reflects dataset characteristics
    - Real-world deployment would require:
        - Real user data collection
        - Comprehensive error analysis
        - Continuous monitoring
    """)

if __name__ == "__main__":
    main()