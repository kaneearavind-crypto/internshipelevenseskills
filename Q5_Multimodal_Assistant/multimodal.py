"""Image captioning, OCR evidence extraction and conservative response validation."""
from PIL import Image
import io, re

def analyze_image(uploaded_file):
    image=Image.open(uploaded_file).convert("RGB")
    evidence=[]
    try:
        from transformers import pipeline
        captioner=pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")
        caption=captioner(image,max_new_tokens=60)[0].get("generated_text","")
        if caption: evidence.append("Visual caption: "+caption)
    except Exception as exc:
        evidence.append("Image caption model unavailable: "+str(exc))
    try:
        import pytesseract
        text=pytesseract.image_to_string(image).strip()
        if text: evidence.append("OCR text: "+text[:3000])
    except Exception:
        pass
    if not evidence:
        return "No reliable visual evidence could be extracted. Ask the user to describe the image or upload a clearer image."
    return "\n".join(evidence)

def validate_evidence(question, evidence):
    if not evidence or "No reliable visual evidence" in evidence:
        return {"answer":"I cannot confidently interpret this image from the available evidence. Please upload a clearer image or describe the relevant details.", "validated":False}
    uncertain=("unavailable" in evidence.lower())
    answer=("Based on the image evidence I could extract:\n\n"+evidence+
            "\n\nThis is an automated interpretation and may be incomplete. "
            "Please verify important details against the original image.")
    if uncertain:
        answer += "\n\nNote: One or more analysis components were unavailable."
    return {"answer":answer,"validated":not uncertain}

def detect_image_type(image):
    return Image.open(image).format
