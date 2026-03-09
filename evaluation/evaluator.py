import os
import random
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import plotly.express as px
from core.nlu_engine import NLUEngine
from core.validator import validate_response

class NLUEvaluator:
    def __init__(self):
        self.engine = NLUEngine()
        
    def evaluate(self, limit: int = 5) -> dict:
        """
        Runs the LLM over a random sample from the loaded intents dataset 
        and calculates metrics.
        """
        
        # 1. Build a dynamic test set from the intents file
        test_data = []
        for intent in self.engine.intents_data.get("intents", []):
            intent_name = intent["name"]
            # Sample up to 'limit' examples per intent
            examples = intent["examples"]
            sampled = random.sample(examples, min(limit, len(examples)))
            for ex in sampled:
                test_data.append({"text": ex, "label": intent_name})
                
        # Shuffle to mix intents during evaluation
        random.shuffle(test_data)
        
        print(f"Evaluating {len(test_data)} total records across all intents. This will make API calls...")
            
        actual_intents = [item["label"] for item in test_data]
        predicted_intents = []
        
        for item in test_data:
            text = item["text"]
            # 1. Fetch LLM response
            raw_response = self.engine.predict(text)
            
            # 2. Validate format
            validated = validate_response(raw_response)
            
            if "error" in validated:
                predicted_intents.append("unknown")
            else:
                predicted_intents.append(validated.get("intent", "unknown"))
                
            # Respect Hugging Face Free-Tier rate limits
            time.sleep(1)
                
        # 3. Calculate Metrics
        accuracy = accuracy_score(actual_intents, predicted_intents)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_intents, predicted_intents, average="weighted", zero_division=0
        )
        
        metrics = {
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2),
            "total_tested": len(test_data)
        }
        
        # 4. Generate Interactive Confusion Matrix
        labels = list(set(actual_intents + predicted_intents))
        cm = confusion_matrix(actual_intents, predicted_intents, labels=labels)
        
        fig = self._generate_confusion_matrix(cm, labels)
        metrics["confusion_matrix_fig"] = fig
        
        return metrics
        
    def _generate_confusion_matrix(self, cm, labels):
        """Generates an interactive Plotly confusion matrix."""
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted Intent", y="Actual Intent", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Interactive NLU Confusion Matrix"
        )
        
        fig.update_layout(
            xaxis_title="Predicted Intent",
            yaxis_title="Actual Intent",
            font=dict(family="sans serif", size=14)
        )
        return fig
