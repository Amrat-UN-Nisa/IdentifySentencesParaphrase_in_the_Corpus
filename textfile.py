import os
import sys
sys.path.append('D:/Profit_pro/ResearchPaper/paper_output')
from sentence_transformers import SentenceTransformer, util

# Define the function for mining paraphrases
def mine_paraphrases(sentences1, sentences2):
    all_sentences = sentences1 + sentences2
    model = SentenceTransformer('all-MiniLM-L6-v2')
    paraphrases = util.paraphrase_mining(model, all_sentences, show_progress_bar=True)
    return [(score, i, j) for score, i, j in paraphrases]

def main():
    subcorpus1_path = 'D:/Profit_pro/ResearchPaper/paper_output/ReadTextFile/6(1).txt'
    subcorpus2_path = 'D:/Profit_pro/ResearchPaper/paper_output/ReadTextFile/6(2).txt'
    output_dir = 'D:/Profit_pro/ResearchPaper/paper_output\output'

    sentences1 = []
    sentences2 = []

    # Read sentences from subcorpus 1
    with open(subcorpus1_path, 'r', encoding='utf-8') as file:
        sentences1.extend(file.readlines())

    # Read sentences from subcorpus 2
    with open(subcorpus2_path, 'r', encoding='utf-8') as file:
        sentences2.extend(file.readlines())

    # Mine paraphrases
    paraphrases = mine_paraphrases(sentences1, sentences2)

    # Write results to files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for score, i, j in paraphrases:
        score_category = str(int(score * 10))
        output_filename = os.path.join(output_dir, f"{score_category}.txt")
        with open(output_filename, 'a', encoding='utf-8') as output_file:
            # Adjust indices to correspond to individual lists
            index1 = i if i < len(sentences1) else i - len(sentences1)
            index2 = j if j < len(sentences2) else j - len(sentences2)
            output_file.write(f"Sentence 1 (Index {index1}): {sentences1[index1].strip()} \nSentence 2 (Index {index2}): {sentences2[index2].strip()} \nSimilarity score: {score}\n")

if __name__ == "__main__":
    main()

