from fastapi import APIRouter, HTTPException, Depends
from app.schema.sentiment import SentimentRequest, SentimentResponse, FeedBack
from app.core.service import get_model, get_vocab, get_preprocessor
import sys
import os
# Add parent directory to path


from app.api.src.model.textcnn import predict_sentiment
import os
router = APIRouter()
DATAPATH = 'ntc-scv/data_train/train'


@router.post("/predict", response_model=SentimentResponse)
async def predict(
    text: SentimentRequest,
    model = Depends(get_model),
    vocab = Depends(get_vocab),
    preprocessor = Depends(get_preprocessor)
): 
    inp = text.text
    probs, sentiment = predict_sentiment(inp, model, vocab=vocab, preprocessor=preprocessor, max_length=50)
    
    return SentimentResponse(sentiment=sentiment, probability=probs)


@router.post("/addsample")
async def add_data(feedback: FeedBack):
    label = feedback.label
    text = feedback.text
    
    if label == 0:
        target_dir = os.path.join(DATAPATH, "neg")
    elif label == 1:
        target_dir = os.path.join(DATAPATH, "pos")
    else:
        raise HTTPException(status_code=400, detail="Invalid label. Must be 0 (negative) or 1 (positive)")
    
    # Ensure directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    # Find the next available file number
    existing_files = [f for f in os.listdir(target_dir) if f.endswith('.txt')]
    if existing_files:
        # Extract numbers from filenames and find the max
        file_numbers = [int(os.path.splitext(f)[0]) for f in existing_files if os.path.splitext(f)[0].isdigit()]
        next_num = max(file_numbers) + 1 if file_numbers else 1
    else:
        next_num = 1

    new_filename = os.path.join(target_dir, f"{next_num}.txt")
    
    try:
        # Write the feedback text to the file
        with open(new_filename, "w", encoding="utf-8") as f:
            f.write(text)
        
        return {"success": True, "message": f"Feedback saved to {new_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
    
    
    
    
    
        