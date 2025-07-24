#whenever you start this up, first check what version of python you're running. BERTeley only supports
#python versions 3.8 to 3.10. Check the bottom right corner for your version.

from berteley.preprocessing import (
    preprocess,
    preprocess_parallel,
    alive_bar_joblib,
    combine_hyphens,
    expand_contractions,
    lemmatize,
    remove_extraspace,
    remove_html,
    remove_punctuation,
    remove_stopwords
)

from berteley.models import (
    create_barcharts,
    fit,
    initialize_model
)

import os
import fitz  # PyMuPDF

folder_path = r"C:\Users\despa\VCTopicModeling\online_articles"
papers = []

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        papers.append(text)

print(f"✅ Loaded {len(papers)} PDF articles.")

from berteley.preprocessing import preprocess
preprocessed_papers = preprocess(papers)

# Initialize
from berteley.models import initialize_model
model = initialize_model()

from berteley.models import fit
model,docs_with_topics=fit(model, preprocessed_papers)

# Print top 10 words per topic
for idx, topic in enumerate(model.get_topics()):
    print(f"\n🧵 Topic {idx}:")
    print(", ".join([term for term, weight in topic[:10]]))

from berteley.models import create_barcharts
create_barcharts(model)