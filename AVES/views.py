import warnings
import string
import random
import numpy as np
import pandas as pd
import nltk
import torch
from django.http import JsonResponse
from django.shortcuts import render
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords, wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
from flashtext import KeywordProcessor
from nltk.tokenize import sent_tokenize
from sense2vec import Sense2Vec
from textdistance import levenshtein
import pke

# Download NLTK data
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')

warnings.filterwarnings("ignore", message="This sequence already has </s>.")

# File paths
DATA_PATH = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_datasets\generated_qa.csv"
PASSAGE_PATH = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_datasets\news.csv"
T5QG_MODEL_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\qg_model"
T5QG_TOKENIZER_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\qg_tokenizer"
T5AG_MODEL_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\t5_model"
T5AG_TOKENIZER_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\t5_tokenizer"
S2V_MODEL_PATH = 's2v_old'

# Preload models and dataset
data = pd.read_csv(DATA_PATH)
passage = pd.read_csv(PASSAGE_PATH)
t5ag_model = T5ForConditionalGeneration.from_pretrained(T5AG_MODEL_DIR)
t5ag_tokenizer = T5Tokenizer.from_pretrained(T5AG_TOKENIZER_DIR)
t5qg_model = T5ForConditionalGeneration.from_pretrained(T5QG_MODEL_DIR)
t5qg_tokenizer = T5Tokenizer.from_pretrained(T5QG_TOKENIZER_DIR)
summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
s2v = Sense2Vec().from_disk(S2V_MODEL_PATH)
sentence_transformer_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v2")

def answer_question(question, context):
    """Generate an answer for a given question and context."""
    input_text = f"question: {question} context: {context}"
    input_ids = t5ag_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        output = t5ag_model.generate(input_ids, max_length=512, num_return_sequences=1, max_new_tokens=200)

    return t5ag_tokenizer.decode(output[0], skip_special_tokens=True)


def get_nouns_multipartite(content):
    """Extract keywords from content using MultipartiteRank algorithm."""
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content, language='en')
        pos_tags = {'PROPN', 'NOUN', 'ADJ', 'VERB', 'ADP', 'ADV', 'DET', 'CONJ', 'NUM', 'PRON', 'X'}

        stoplist = list(string.punctuation) + ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')

        extractor.candidate_selection(pos=pos_tags)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extractor.get_n_best(n=15)
        
        return [val[0] for val in keyphrases]
    except:
        return []

def postprocesstext(content):
    """Post-process the text by capitalizing the first letter of each sentence."""
    sentences = sent_tokenize(content)
    return " ".join([sentence.capitalize() for sentence in sentences])

def summarizer(text, model, tokenizer):
    """Generate a summary of the given text."""
    text = "summarize: " + text.strip().replace("\n", " ")
    encoding = tokenizer.encode_plus(text, max_length=512, truncation=True, return_tensors="pt").to('cpu')

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    
    outs = model.generate(input_ids=input_ids, attention_mask=attention_mask, early_stopping=True, num_beams=3,
                          num_return_sequences=1, no_repeat_ngram_size=2, min_length=75, max_length=300)

    summary = tokenizer.decode(outs[0], skip_special_tokens=True)
    return postprocesstext(summary).strip()

def get_keywords(original_text):
    """Get exactly 5 important keywords from the original text."""
    while True:
        keywords = get_nouns_multipartite(original_text)
        keyword_processor = KeywordProcessor()
        summary_text = summarizer(original_text, summary_model, summary_tokenizer)

        for keyword in keywords:
            keyword_processor.add_keyword(keyword)

        keywords_found = keyword_processor.extract_keywords(summary_text)
        keywords_found = list(set(keywords_found))

        sorted_keywords = sorted(keywords, key=lambda x: len(x))

        if len(sorted_keywords) >= 5:
            return sorted_keywords[:5]


def get_question(context, answer, model, tokenizer):
    """Generate a question for the given answer and context."""
    answer_span = context.replace(answer, f"<hl>{answer}<hl>") + "</s>"
    inputs = tokenizer(answer_span, return_tensors="pt")
    question = model.generate(input_ids=inputs.input_ids, max_length=50)[0]

    return tokenizer.decode(question, skip_special_tokens=True)


def filter_same_sense_words(original, wordlist):
    """Filter words that have the same sense as the original word."""
    base_sense = original.split('|')[1]
    return [word[0].split('|')[0].replace("_", " ").title().strip() for word in wordlist if word[0].split('|')[1] == base_sense]

def get_max_similarity_score(wordlist, word):
    """Get the maximum similarity score between the word and a list of words."""
    return max(levenshtein.normalized_similarity(word.lower(), each.lower()) for each in wordlist)

def sense2vec_get_words(word, s2v, topn, question):
    """Get similar words using Sense2Vec."""
    try:
        sense = s2v.get_best_sense(word, senses=["NOUN", "PERSON", "PRODUCT", "LOC", "ORG", "EVENT", "NORP", "WORK OF ART", "FAC", "GPE", "NUM", "FACILITY"])
        most_similar = s2v.most_similar(sense, n=topn)
        output = filter_same_sense_words(sense, most_similar)
    except:
        output = []
    
    threshold = 0.6
    final = [word]
    checklist = question.split()

    for similar_word in output:
        if get_max_similarity_score(final, similar_word) < threshold and similar_word not in final and similar_word not in checklist:
            final.append(similar_word)
    
    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    """Maximal Marginal Relevance (MMR) for keyword extraction."""
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        mmr = (lambda_param * candidate_similarities) - ((1 - lambda_param) * target_similarities.reshape(-1, 1))
        mmr_idx = candidates_idx[np.argmax(mmr)]

        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def get_distractors_wordnet(word):
    """Get distractors using WordNet."""
    distractors = []
    try:
        synset = wn.synsets(word, 'n')[0]
        hypernym = synset.hypernyms()
        if not hypernym:
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name().replace("_", " ").title()
            if name.lower() != word.lower() and name not in distractors:
                distractors.append(name)
    except:
        pass

    return distractors

def get_distractors(word, original_sentence, sense2vec_model, sentence_model, top_n, lambda_val):
    """Get distractors for a given word using various methods."""
    distractors = sense2vec_get_words(word, sense2vec_model, top_n, original_sentence)
    if not distractors:
        return []

    distractors_new = [word.capitalize()] + distractors
    embedding_sentence = f"{original_sentence} {word.capitalize()}"
    keyword_embedding = sentence_model.encode([embedding_sentence])
    distractor_embeddings = sentence_model.encode(distractors_new)

    max_keywords = min(len(distractors_new), 5)
    filtered_keywords = mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambda_val)
    return [kw.capitalize() for kw in filtered_keywords if kw.lower() != word.lower()][1:]


def get_mca_questions(context, qg_model, qg_tokenizer, s2v, sentence_transformer_model, num_questions=5, max_attempts=2):
    """
    Generate multiple-choice questions for a given context.
    
    Parameters:
        context (str): The context from which questions are generated.
        qg_model (T5ForConditionalGeneration): The question generation model.
        qg_tokenizer (T5Tokenizer): The tokenizer for the question generation model.
        s2v (Sense2Vec): The Sense2Vec model for finding similar words.
        sentence_transformer_model (SentenceTransformer): The sentence transformer model for embeddings.
        num_questions (int): The number of questions to generate.
        max_attempts (int): The maximum number of attempts to generate questions.
    
    Returns:
        list: A list of dictionaries with questions and their corresponding distractors.
    """
    output_list = []

    imp_keywords = get_keywords(context)
    print(f"Extracted keywords: {imp_keywords}")

    generated_questions = set()
    attempts = 0

    while len(output_list) < num_questions and attempts < max_attempts:
        attempts += 1

        for keyword in imp_keywords:
            if len(output_list) >= num_questions:
                break
            
            question = get_question(context, keyword, qg_model, qg_tokenizer)
            print(f"Generated question: {question} for keyword: {keyword}")
            
            if question in generated_questions:
                print(f"Question '{question}' already generated, skipping.")
                continue
            
            generated_questions.add(question)

            distractors = get_distractors(keyword.capitalize(), question, s2v, sentence_transformer_model, 40, 0.2)
            print(f"Generated distractors: {distractors} for question: {question}")

            t5_answer = answer_question(question, context)
            print(f"Generated answer: {t5_answer} for question: {question}")

            if len(distractors) == 0:
                print("No distractors found, using important keywords as distractors.")
                distractors = imp_keywords

            distractors = [d.capitalize() for d in distractors if d.lower() != keyword.lower()]

            if len(distractors) < 3:
                additional_distractors = [kw.capitalize() for kw in imp_keywords if kw.capitalize() not in distractors and kw.lower() != keyword.lower()]
                distractors.extend(additional_distractors[:3-len(distractors)])
            else:
                distractors = distractors[:3]

            print(f"Final distractors: {distractors} for question: {question}")

            options = distractors + [t5_answer]
            random.shuffle(options)
            print(f"Options: {options} for question: {question}")

            output_list.append({
                'question': question,
                'options': options
            })

        print(f"Generated {len(output_list)} questions so far after {attempts} attempts")

    while len(output_list) < num_questions:
        keyword = random.choice(imp_keywords)
        dummy_question = f"What is {keyword}?"
        distractors = [kw.capitalize() for kw in imp_keywords if kw.lower() != keyword.lower()][:3]
        dummy_answer = keyword.capitalize()

        options = distractors + [dummy_answer]
        random.shuffle(options)
        
        output_list.append({
            'question': dummy_question,
            'options': options
        })
    
    return output_list[:num_questions]



def gen(passage):
    """Generate a random context from the dataset."""
    return passage.sample(n=1)['text'].values[0]

def homepage(request):
    """Render the homepage with generated questions."""
    original_context = gen(passage)
    questions_and_distractors = get_mca_questions(original_context, t5qg_model, t5qg_tokenizer, s2v, sentence_transformer_model, num_questions=5)
    
    context = {
        'passage': original_context,
        'questions_and_distractors': questions_and_distractors
    }

    print(questions_and_distractors)

    return render(request, 'homepage.html', context)
