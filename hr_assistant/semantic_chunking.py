import re
import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

class SemanticChunking:
    def __init__(self, breakpoint_percentile=95, buffer_size=1):
        self.embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_KEY)
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size

    def _process_sentences(self, text):
      
        sentences = [
            {"sentence": s, "index": i} for i, s in enumerate(re.split(r"(?<=[.?!])\s+", text))
        ]

        for i, current in enumerate(sentences):
            context_range = range(
                max(0, i - self.buffer_size),
                min(len(sentences), i + self.buffer_size + 1),
            )
            current["combined_sentence"] = " ".join(
                sentences[j]["sentence"] for j in context_range
            )

        return sentences

    def _calculate_distances(self, sentences):
        embeddings = self.embeddings.embed_documents(
            [s["combined_sentence"] for s in sentences]
        )

        distances = []
        for i in range(len(sentences) - 1):
            distance = 1 - cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distances.append(distance)

        return distances

    def chunk_text(self, text):

        single_sentences_list = re.split(r"(?<=[.?!])\s+", txt)
        sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = SemanticChunking.combine_sentences(sentences)
        oaiembeds = OpenAIEmbeddings(
            openai_api_key= Config.OPENAI_KEY
        )

        embeddings = oaiembeds.embed_documents(
            [x["combined_sentence"] for x in sentences]
        )

        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]

        distances, sentences = SemanticChunking.calculate_cosine_distances(sentences)

        breakpoint_percentile_threshold = 95
        breakpoint_distance_threshold = np.percentile(
            distances, breakpoint_percentile_threshold
        )

        num_distances_above_theshold = len(
            [x for x in distances if x > breakpoint_distance_threshold]
        )

        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        ]

        start_index = 0

        chunks = []

        for index in indices_above_thresh:

            end_index = index

            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            chunks.append(combined_text)

            start_index = index + 1

        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks
