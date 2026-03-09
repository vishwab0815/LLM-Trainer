from pydantic import BaseModel, Field
from typing import Dict, Any

class NLUResponse(BaseModel):
    intent: str = Field(..., description="The classified intent of the message.")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0.")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities as a key-value pair.")

def validate_response(llm_output: dict) -> dict:
    """
    Validates the structure of the LLM JSON response. 
    If valid, returns the normalized dict. Otherwise, flags an error.
    """
    try:
        if "error" in llm_output:
            return llm_output
            
        validated = NLUResponse(**llm_output)
        return validated.model_dump()
    except Exception as e:
        print(f"Validation Error: {e}")
        return {
            "error": "Failed to validate LLM output schema.",
            "raw_output": llm_output
        }
