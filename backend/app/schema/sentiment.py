from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str
    
class SentimentResponse(BaseModel):
    sentiment: str
    probability: float
    
class FeedBack(BaseModel):
    text: str
    label: int
