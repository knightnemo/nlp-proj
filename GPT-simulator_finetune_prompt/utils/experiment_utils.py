import json
from collections import Counter

def extract_confidence(response):
    """Extract confidence score from response"""
    try:
        if "confidence:" in response.lower():
            confidence_text = response.lower().split("confidence:")[1].split("%")[0]
            return float(confidence_text.strip())
        return 50.0
    except:
        return 50.0

def select_majority_prediction(predictions):
    """Select the most common prediction"""
    pred_strings = [json.dumps(p, sort_keys=True) for p in predictions if p is not None]
    
    if not pred_strings:
        return None
    
    counts = Counter(pred_strings)
    most_common = counts.most_common(1)[0][0] 
    return json.loads(most_common)  