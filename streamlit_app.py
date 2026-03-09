import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
from core.nlu_engine import NLUEngine
from evaluation.evaluator import NLUEvaluator
from core.llm_client import transcribe_audio
from streamlit_mic_recorder import mic_recorder

# Set page config for a wider layout
st.set_page_config(page_title="BotTrainer NLU", layout="wide")

# Load custom CSS
def load_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🤖 BotTrainer Enterprise")
st.markdown("---")

@st.cache_resource
def load_engine():
    return NLUEngine()

engine = load_engine()

# --- Advanced Sidebar Navigation ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Navigation")
    st.markdown("Select a module below:")
    page = st.radio(
        "Select a module",
        ["💬 Interactive Chat", "📊 Dataset Intelligence", "📈 Performance Evaluation"],
        index=0,
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("⚙️ Powered by Hugging Face (Llama-3-8B)")
    st.caption("🔧 v2.0 Enterprise Build")

# ================================
# PAGE 1: Chat Interface
# ================================
if page == "💬 Interactive Chat":
    st.header("Interactive Chat Interface")
    st.markdown("Test the Natural Language Understanding pipeline in real-time. Type a complex sentence!")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a greeting message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I am your NLU assistant. Try asking me to book a flight, order food, check weather, track an order, or cancel an order."
        })
        
    # Display historical chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "results" in msg:
                res = msg["results"]
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(label="Predicted Intent", value=res.get("intent", "unknown"))
                    st.metric(label="Confidence", value=f"{res.get('confidence', 0.0) * 100}%")
                with c2:
                    st.caption("Extracted Entities:")
                    st.json(res.get("entities", {}))

    # Input Area: Typing and Voice
    st.write("")
    col_mic, col_chat = st.columns([1, 10])
    
    with col_mic:
        st.write("") # Padding
        audio = mic_recorder(
            start_prompt="🎙️", 
            stop_prompt="🛑", 
            just_once=True,
            use_container_width=True,
            key='mic'
        )
        
    prompt = st.chat_input("E.g. Book a flight to Delhi tomorrow")
    
    # Process Audio Input
    if audio and "bytes" in audio:
        with st.spinner("Transcribing audio..."):
            transcribed_text = transcribe_audio(audio["bytes"])
            if not transcribed_text.startswith("Error"):
                prompt = transcribed_text
            else:
                st.error(transcribed_text)

    # React to user input (from either text or voice target)
    if prompt:
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing message..."):
                result = engine.predict(prompt)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {result['error']}"})
                else:
                    response_text = f"I matched your message to the **{result.get('intent')}** intent."
                    st.markdown(response_text)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(label="Predicted Intent", value=result.get("intent", "unknown"))
                        st.metric(label="Confidence", value=f"{result.get('confidence', 0.0) * 100}%")
                    with c2:
                        st.caption("Extracted Entities:")
                        st.json(result.get("entities", {}))
                        
                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "results": result
                    })

# ================================
# PAGE 2: Dataset Insights
# ================================
elif page == "📊 Dataset Intelligence":
    st.header("Dataset Intelligence Dashboard")
    st.markdown("Explore the synthetic data driving the zero-shot NLU model.")
    
    with open("dataset/intents.json", "r") as f:
        data = json.load(f)
        
    intents = data.get("intents", [])
    entities = data.get("entities", {})
    
    # Generate Interactive Distribution Chart
    st.subheader("Dataset Distribution")
    intent_names = [i['name'] for i in intents]
    intent_counts = [len(i['examples']) for i in intents]
    
    df_dist = pd.DataFrame({"Intent": intent_names, "Examples": intent_counts})
    fig_dist = px.bar(
        df_dist, x="Intent", y="Examples", color="Intent", 
        title="Number of Synthetic Examples per Intent",
        text="Examples", template="plotly_dark"
    )
    fig_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.divider()
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Supported Intents")
        for i in intents:
            with st.expander(f"Intent 🏷️: {i['name']}"):
                st.caption(f"Extracts entities: {', '.join(i.get('entities', []))}")
                for ex in i['examples'][:5]: # Show preview of first 5
                    st.write(f"- {ex}")
                st.write(f"*(...and {len(i['examples']) - 5} more)*")
                
    with col_b:
        st.subheader("Recognized Entity Types")
        for entity_name, entity_values in entities.items():
            with st.expander(f"Entity 📦: {entity_name}"):
                st.write(", ".join(entity_values))

# ================================
# PAGE 3: Model Evaluation
# ================================
elif page == "📈 Performance Evaluation":
    st.header("Automated Model Evaluation")
    st.markdown("Run the LLM against a random subset from `intents.json` to calculate live metrics.")
    
    eval_limit = st.slider("Number of test records PER INTENT to evaluate:", min_value=1, max_value=10, value=3)
    
    if st.button("Run Evaluation"):
        evaluator = NLUEvaluator()
        
        with st.spinner("Evaluating subset through Mistral-7B... This may take a moment."):
            try:
                metrics = evaluator.evaluate(limit=eval_limit)
                
                st.subheader("Evaluation Metrics")
                m0, m1, m2, m3, m4 = st.columns(5)
                m0.metric("Total Tests", metrics.get('total_tested', 0))
                m1.metric("Accuracy", f"{metrics['accuracy']*100}%")
                m2.metric("Precision", f"{metrics['precision']*100}%")
                m3.metric("Recall", f"{metrics['recall']*100}%")
                m4.metric("F1 Score", f"{metrics['f1_score']*100}%")
                
                st.divider()
                st.subheader("Interactive Confusion Matrix")
                if "confusion_matrix_fig" in metrics:
                    st.plotly_chart(metrics["confusion_matrix_fig"], use_container_width=True)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
