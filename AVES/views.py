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
from sense2vec import Sense2Vec
from textdistance import levenshtein
import pke

from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')

warnings.filterwarnings("ignore", message="This sequence already has </s>.")

# File paths
RANDOM_PASSAGE_PATH = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_datasets\generated_qa.csv"
NEWS_PASSAGE_PATH = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_datasets\news.csv"
T5QG_MODEL_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\qg_model"
T5QG_TOKENIZER_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\qg_tokenizer"
T5AG_MODEL_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\t5_model"
T5AG_TOKENIZER_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\t5_tokenizer"
S2V_MODEL_PATH = 's2v_old'

# Preload models and dataset
random_passage = pd.read_csv(RANDOM_PASSAGE_PATH)
news_passage = pd.read_csv(NEWS_PASSAGE_PATH)
t5ag_model = T5ForConditionalGeneration.from_pretrained(T5AG_MODEL_DIR)
t5ag_tokenizer = T5Tokenizer.from_pretrained(T5AG_TOKENIZER_DIR)
t5qg_model = T5ForConditionalGeneration.from_pretrained(T5QG_MODEL_DIR)
t5qg_tokenizer = T5Tokenizer.from_pretrained(T5QG_TOKENIZER_DIR)
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
    except Exception as e:
        print(f"Error extracting nouns: {e}")
        return []

    
def get_keywords(passage):

    vectorizer = TfidfVectorizer(stop_words='english')
    
    tfidf_matrix = vectorizer.fit_transform([passage])
    
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_scores = tfidf_matrix.toarray().flatten()
    
    word_scores = dict(zip(feature_names, tfidf_scores))

    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    keywords = [word for word, score in sorted_words]
    
    return keywords



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
        
        if sense is None:
            print(f"[DEBUG] No suitable sense found for word: '{word}'")
            return []

        most_similar = s2v.most_similar(sense, n=topn)
        output = filter_same_sense_words(sense, most_similar)
    except Exception as e:
        print(f"Error in Sense2Vec: {e}")
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
    except Exception as e:
        print(f"Error in WordNet distractors: {e}")
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

    imp_keywords = get_keywords(context)  # Extract keywords only once
    print(f"[DEBUG] Extracted keywords: {imp_keywords}, length: {len(imp_keywords)}")

    generated_questions = set()
    attempts = 0

    while len(output_list) < num_questions and attempts < max_attempts:
        attempts += 1

        for keyword in imp_keywords:
            if len(output_list) >= num_questions:
                break
            
            question = get_question(context, keyword, qg_model, qg_tokenizer)
            print(f"[DEBUG] Generated question: '{question}' for keyword: '{keyword}'")
            
            # Encode the new question
            new_question_embedding = sentence_transformer_model.encode(question, convert_to_tensor=True)
            is_similar = False

            # Check similarity with existing questions
            for generated_q in generated_questions:
                existing_question_embedding = sentence_transformer_model.encode(generated_q, convert_to_tensor=True)
                similarity = cosine_similarity(new_question_embedding.unsqueeze(0), existing_question_embedding.unsqueeze(0))[0][0]

                if similarity > 0.8:
                    is_similar = True
                    print(f"[DEBUG] Question '{question}' is too similar to an existing question, skipping.")
                    break

            if is_similar:
                continue

            generated_questions.add(question)

            t5_answer = answer_question(question, context)
            print(f"[DEBUG] Generated answer: '{t5_answer}' for question: '{question}'")

            distractors = get_distractors(t5_answer.capitalize(), question, s2v, sentence_transformer_model, 40, 0.2)
            print(f"[DEBUG] Generated distractors: {distractors} for question: '{question}'")

            if len(distractors) == 0:
                print("[DEBUG] No distractors found, using important keywords as distractors.")
                distractors = imp_keywords

            distractors = [d.capitalize() for d in distractors if d.lower() != keyword.lower()]

            if len(distractors) < 3:
                additional_distractors = [kw.capitalize() for kw in imp_keywords if kw.capitalize() not in distractors and kw.lower() != keyword.lower()]
                distractors.extend(additional_distractors[:3 - len(distractors)])
            else:
                distractors = distractors[:3]

            print(f"[DEBUG] Final distractors: {distractors} for question: '{question}'")

            options = distractors + [t5_answer]
            random.shuffle(options)
            print(f"[DEBUG] Options: {options} for question: '{question}'")

            output_list.append({
                'question': question,
                'options': options
            })

        print(f"[DEBUG] Generated {len(output_list)} questions so far after {attempts} attempts")

    return output_list[:num_questions]


def gen(passage):
    """Generate a random context from the dataset."""
    return passage.sample(n=1)['context'].values[0]

def homepage(request):
    """Render the homepage with generated questions."""
    original_context = gen(random_passage)
    questions_and_distractors = get_mca_questions(original_context, t5qg_model, t5qg_tokenizer, s2v, sentence_transformer_model, num_questions=5)
    
    context = {
        'passage': original_context,
        'questions_and_distractors': questions_and_distractors
    }

    print("[DEBUG] Final questions and distractors:")
    print(questions_and_distractors)

    return render(request, 'homepage.html', context)
