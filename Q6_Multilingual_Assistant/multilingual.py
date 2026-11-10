"""Cross-lingual conversation using language detection and open-source NLLB-200."""
import re
import streamlit as st

LANGS={"en":"eng_Latn","hi":"hin_Deva","ta":"tam_Taml","es":"spa_Latn"}
NAMES={"en":"English","hi":"Hindi","ta":"Tamil","es":"Spanish"}

def detect_language(text):
    try:
        from langdetect import detect_langs
        detected=detect_langs(text)
        if not detected: return "unknown",0.0,False
        lang=detected[0].lang
        confidence=float(detected[0].prob)
    except Exception:
        lang,confidence="unknown",0.0
    # Script cues help detect code-switching and short inputs.
    has_deva=bool(re.search(r"[\u0900-\u097F]",text))
    has_tamil=bool(re.search(r"[\u0B80-\u0BFF]",text))
    has_latin=bool(re.search(r"[A-Za-z]",text))
    mixed=sum([has_deva,has_tamil,has_latin])>1
    if has_tamil: lang="ta"
    elif has_deva: lang="hi"
    if lang not in LANGS or confidence < 0.45 and not (has_tamil or has_deva):
        return "unknown",confidence,mixed
    return lang,confidence,mixed

@st.cache_resource
def _translator():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name="facebook/nllb-200-distilled-600M"
    tokenizer=AutoTokenizer.from_pretrained(name)
    model=AutoModelForSeq2SeqLM.from_pretrained(name)
    return tokenizer,model

def _translate(text, source_code, target_code):
    if source_code==target_code: return text
    import torch
    tokenizer,model=_translator()
    tokenizer.src_lang=source_code
    inputs=tokenizer(text,return_tensors="pt",truncation=True,max_length=512)
    output=model.generate(**inputs,forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),max_length=512)
    return tokenizer.batch_decode(output,skip_special_tokens=True)[0]

def translate_to_english(text, lang):
    return _translate(text,LANGS[lang],LANGS["en"])

def translate_from_english(text, lang):
    return _translate(text,LANGS["en"],LANGS[lang])

def generate_contextual_reply(message_english, history):
    lower=message_english.lower()
    if any(word in lower for word in ["hello","hi","good morning","good evening"]):
        return "Hello! I can help answer questions and keep track of our conversation. What would you like to explore?"
    if any(word in lower for word in ["thank","thanks"]):
        return "You're welcome. Would you like me to explain anything further?"
    previous="\n".join(history[-6:])
    if previous:
        return ("I understand your latest question: "+message_english+
                "\n\nI will keep the earlier conversation in mind: "+previous[-500:]+
                "\n\nCould you clarify the specific detail you want me to focus on? I can then give a more precise answer.")
    return ("I understand your question: "+message_english+
            "\n\nI can help with this. Please share one more detail about the outcome or explanation you need.")
