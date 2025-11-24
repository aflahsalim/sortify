from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your Outlook add-in frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your Render domain later
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    text: str
    attachment: str = "Unknown"  # Optional field from frontend

@app.post("/classify")
async def classify_email(request: EmailRequest):
    email_text = request.text.lower()

    # 🔍 Simple rule-based scoring
    phishing_keywords = ["verify", "urgent", "click", "login", "update", "reset", "confirm"]
    matches = sum(1 for word in phishing_keywords if word in email_text)
    score = round(matches / len(phishing_keywords), 2)

    label = "phishing" if score > 0.5 else "safe"

    # 🧠 Risk factor analysis
    sender_reputation = "Low / Unverified" if "outlook.com" in email_text else "Trusted"
    link_analysis = "Suspicious Redirects" if "http" in email_text else "No Links Found"
    content_check = "Urgency Patterns" if any(w in email_text for w in ["urgent", "verify"]) else "Normal Language"

    return {
        "score": score,                  # float between 0 and 1
        "label": label,                  # "phishing" or "safe"
        "sender": sender_reputation,
        "links": link_analysis,
        "content": content_check,
        "attachment": request.attachment
    }
