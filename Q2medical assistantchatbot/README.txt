# MedAssist AI — MedQuAD starter
1. Upload `MedAssist_AI_MedQuAD.ipynb` to Google Colab.
2. Run cells in order.
3. The notebook clones the MedQuAD GitHub repository, parses XML Q&A pairs, creates a Sentence Transformers embedding index with FAISS, tests retrieval, writes `app.py`, and downloads a ZIP.
4. The app is a retrieval prototype; it is not a medical advice system.
5. The dynamic update UI validates an uploaded CSV but does not automatically activate it. For actual activation, merge new rows into the knowledge base and rebuild embeddings/index.
Dataset: https://github.com/abachaa/MedQuAD
Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
