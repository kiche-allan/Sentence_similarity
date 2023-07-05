from flask import Flask, request
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from transformers import pipeline


app = Flask(__name__)

model = SentenceTransformer('sentence-transformers/stsb-roberta-large')

class Query(BaseModel):
    question_one: str

@app.route('/extract', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        query = request.get_json()
        text = query['question_one']
        ner = pipeline("ner", model="samrawal/bert-base-uncased_clinical-ner", tokenizer="samrawal/bert-base-uncased_clinical-ner")
        output = ner(text)
        filtered_data = [d for d in output if not d['word'].startswith('#')]
        disease_list = [d['word'] for d in filtered_data if d['entity'] == 'B-problem' or d['entity'] == 'I-problem']
        return str(disease_list) + "           " + str(output)

if __name__ == '__main__':
    app.run()
