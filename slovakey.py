from sentence_transformers import SentenceTransformer
import numpy as np
import io
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from operator import itemgetter
import stanza
from sklearn.feature_extraction.text import TfidfVectorizer


class SlovaKey:
    def __init__(self, model_name='kinit/slovakbert-sts-stsb'):
        self.model = SentenceTransformer(model_name)

    def r_file(self, file):
        with io.open(file, encoding='utf-8-sig', errors='ignore') as f:
            return f.read()

    def w_file(self, filename, text):
        with io.open(filename, 'w', encoding='utf-8-sig') as f:
            f.write(text)

    def pos_patterns(self, document, start, end):
        stanza.download('sk')
        nlp = stanza.Pipeline('sk', processors='tokenize,pos')
        doc = nlp(" ".join(document))
        upos = [word.upos for sent in doc.sentences for word in sent.words]
        words = [word.text for sent in doc.sentences for word in sent.words]
        selected_patterns = []

        if start == end == 1:
            for i in range(0, len(upos)):
                if upos[i] == 'NOUN':
                    selected_patterns.append(words[i])
        elif start == end == 2:
            for i in range(0, len(upos) - 1):
                if upos[i] == 'ADJ' and upos[i + 1] == 'NOUN':
                    selected_patterns.append(f"{words[i]} {words[i + 1]}")
                if upos[i] == 'NOUN' and upos[i + 1] == 'NOUN':
                    selected_patterns.append(f"{words[i]} {words[i + 1]}")
        elif start == end == 3:
            for i in range(0, len(upos) - 2):
                if upos[i] == 'ADJ' and upos[i + 1] == 'NOUN' and upos[i + 2] == 'NOUN':
                    selected_patterns.append(f"{words[i]} {words[i + 1]} {words[i + 2]}")
                if upos[i] == 'NOUN' and upos[i + 1] == 'PREP' and upos[i + 1] == 'NOUN':
                    selected_patterns.append(f"{words[i]} {words[i + 1]} {words[i + 2]}")
        return selected_patterns

    def statistical_representation(self, input_text, selected_patterns, start, end):
        vectorizer = TfidfVectorizer(vocabulary=set(selected_patterns), ngram_range=(start, end))
        X = vectorizer.fit_transform(input_text)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = np.asarray(X.mean(axis=0)).ravel()
        tfidf_scores = tfidf_scores.astype(float)
        sorted_feature_names = feature_names[np.argsort(tfidf_scores)[::-1]]
        feature_tfidf_pairs = list(zip(sorted_feature_names, tfidf_scores))
        sorted_feature_tfidf_pairs = sorted(feature_tfidf_pairs, key=lambda x: x[1], reverse=True)
        candidates = []
        for feature, tfidf in sorted_feature_tfidf_pairs:
            if tfidf > 0.0:
                candidates.append(feature)
        return candidates

    def mmr(self, doc_embedding, word_embeddings, words, top_n, diversity):
        doc_embedding = np.nan_to_num(doc_embedding, nan=0.0)
        word_embeddings = np.nan_to_num(word_embeddings, nan=0.0)
        word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding.reshape(1, -1))
        word_similarity = cosine_similarity(word_embeddings)
        keywords_idx = [np.argmax(word_doc_similarity)]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]
        for _ in range(min(top_n - 1, len(words) - 1)):
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(
                word_similarity[candidates_idx][:, keywords_idx], axis=1
            )
            mmr = (
                1 - diversity
            ) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)
        keywords = [
            (words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4))
            for idx in keywords_idx
        ]
        keywords = sorted(keywords, key=itemgetter(1), reverse=True)
        return keywords

    def extract_keywords(self, input_text, start=2, end=2, top_n_keywords=10, diversity=0.8):
        document = ' '.join(input_text).split('\n')
        document = [sentence.lower() for sentence in document]
        sentence_embeddings = self.model.encode(document, convert_to_tensor=True)
        document_embedding = sentence_embeddings.mean(dim=0)
        selected_patterns = self.pos_patterns(document, start, end)
        tfidf_candidates = self.statistical_representation(input_text, selected_patterns, start, end)
        candidate_embeddings = self.model.encode(tfidf_candidates, convert_to_tensor=True)
        mmr_candidates = self.mmr(document_embedding, candidate_embeddings, tfidf_candidates, top_n_keywords,
                                  diversity)
        keywords = [x[0] for x in mmr_candidates]
        return keywords
