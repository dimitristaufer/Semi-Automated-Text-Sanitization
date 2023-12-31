# pylint: disable=W0105
# pylint: disable=W0012
# pylint: disable=unused-import
# pylint: disable=no-name-in-module

import os
import utils
import time
from bs4 import BeautifulSoup
import pandas as pd
import datetime
from tqdm import tqdm
import spacy
import textstat
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from transformers import pipeline, AutoTokenizer, BertTokenizer
from scipy.special import kl_div
from peft import PeftConfig

'''
Start Screen:
screen -S mytraining

Detach:
Ctrl+A then Ctrl+D

List screens:
screen -ls

Reattacht:
screen -r mytraining

Enable scrolling:
Ctrl+A then Esc
'''

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L12-v2')

# Load sentiment analysis pipeline (pipeline, by default only returns the label with the highest score! But we can use top_k=None to get all)
sentiment_model = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment", device="mps")
sentiment_tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Load the same tokenizer used in utils.py
config = PeftConfig.from_pretrained(utils.default_model)
tokenizer_id = config.base_model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

# So we can have progress bars when performing heavy pandas operations
tqdm.pandas()

def automaticSanitization(text, temperature=0.5):
    """
    This function simulates a sanitization pipeline, as if it were used with the front end.

    Parameters:
    text (str): The text to be sanitized.
    temperature (float, optional): The temperature parameter for FLAN T5. Default is 0.5.

    Returns:
    tuple: Returns sanitized text and time elapsed during sanitization.
    """

    start_time = time.time()  # Start measuring time

    tokenizationHTML = utils.tokenize_return_positions(text)
    annotationList = utils.process_text(text, initialRun=True)[0]

    def annotate_html(tokenizationHTML, annotationList):
        soup = BeautifulSoup(tokenizationHTML, "html.parser")

        for annotation in annotationList:
            annotation_ids = [str(id) for id in annotation["ids"]]
            id_str = ", ".join(annotation_ids)
            marks = soup.select(f'mark[ids="[{id_str}]"]')

            for mark in marks:
                mark["class"] = mark.get("class", [])
                mark["class"] = [
                    c
                    for c in mark["class"]
                    if c
                    not in [
                        "never-sensitive",
                        "potentially-sensitive",
                        "sensitive",
                        "highly-sensitive",
                    ]
                ]
                if annotation["sensitivity"] == 2:
                    mark["class"].append("highly-sensitive")
                elif annotation["sensitivity"] == 1:
                    mark["class"].append("sensitive")
                elif annotation["sensitivity"] == 0:
                    mark["class"].append("potentially-sensitive")

        return str(soup)

    annotation = annotate_html(tokenizationHTML, annotationList)

    sentences_with_phrases_to_remove = []
    removed_strings = []

    annotation, more_removed_strings, new_sentences_with_phrases_to_remove = utils.remove_adjadv_psshs(annotation)
    removed_strings.extend(more_removed_strings)
    sentences_with_phrases_to_remove += new_sentences_with_phrases_to_remove
    soup = BeautifulSoup(annotation, "html.parser")

    annotation, more_removed_strings, new_sentences_with_phrases_to_remove = utils.replace_pronouns_psshs(annotation)
    removed_strings.extend(more_removed_strings)
    sentences_with_phrases_to_remove += new_sentences_with_phrases_to_remove
    soup = BeautifulSoup(annotation, "html.parser")

    annotation, more_removed_strings, new_sentences_with_phrases_to_remove = utils.replace_entities_psshs(annotation)
    removed_strings.extend(more_removed_strings)
    sentences_with_phrases_to_remove += new_sentences_with_phrases_to_remove
    soup = BeautifulSoup(annotation, "html.parser")

    annotation, more_removed_strings, new_sentences_with_phrases_to_remove = utils.replace_propnouns_s(annotation)
    removed_strings.extend(more_removed_strings)
    sentences_with_phrases_to_remove += new_sentences_with_phrases_to_remove
    soup = BeautifulSoup(annotation, "html.parser")

    annotation, more_removed_strings, new_sentences_with_phrases_to_remove = utils.exclude_remove_nouns_shs(annotation)
    removed_strings.extend(more_removed_strings)
    sentences_with_phrases_to_remove += new_sentences_with_phrases_to_remove
    soup = BeautifulSoup(annotation, "html.parser")

    annotation, more_removed_strings, new_sentences_with_phrases_to_remove = utils.remove_propnouns_hs(annotation)
    removed_strings.extend(more_removed_strings)
    sentences_with_phrases_to_remove += new_sentences_with_phrases_to_remove
    soup = BeautifulSoup(annotation, "html.parser")

    text_for_paraphrasing = soup.text

    for entry in sentences_with_phrases_to_remove:
        text_for_paraphrasing = text_for_paraphrasing.replace(entry['sentence'], utils.remove_dependent_phrases(entry['sentence'], entry['entity']))

    utils.cancelGeneration = False

    final_text = ""
    generated_text = utils.paraphrase_text(
        text, temperature=temperature, stream=False, excluded_strings=removed_strings
    )
    for text in generated_text:
        final_text += text

    end_time = time.time()  # Stop measuring time
    elapsed_time = int(
        (end_time - start_time) * 1000
    )  # Calculate elapsed time in milliseconds

    return final_text, elapsed_time

def save_dataframe(df, filename, cols_to_drop=[]):
    """
    This function saves the dataframe to a csv file.

    Parameters:
    df (pandas.DataFrame): The dataframe to be saved.
    filename (str): The output filename.
    cols_to_drop (list, optional): A list of columns to be dropped. Default is an empty list.

    Returns:
    None
    """

    # Drop specified columns if they exist in dataframe
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    df.to_csv(filename, index=False)

def generate_filename(prefix, outPath):
    """
    This function generates a filename with a timestamp.

    Parameters:
    prefix (str): The prefix for the filename.
    outPath (str): The output path for the file.

    Returns:
    str: Returns the filename as a string.
    """

    datestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = outPath + "/" + prefix + datestamp + ".csv"
    return filename

def sanitizeImdb62(inCSVPath, outPath, checkpointFile="evaluation_data/checkpoint.csv"):
    """
    This function sanitizes a csv file of the IMDb62 dataset

    Parameters:
    inCSVPath (str): The input path for the csv file.
    outPath (str): The output path for the sanitized csv file.
    checkpointFile (str, optional): The file path for the checkpoint csv file. Default is "evaluation_data/checkpoint.csv".

    Returns:
    None: This function does not return any value but writes the results to an output CSV file.
    """

    # Read the CSV file
    df = pd.read_csv(inCSVPath) # len = 12370

    # Exclude rows where "text" is more than 512 tokens, because T5 can only process 512 input tokens (subword units)
    df = df[df['text'].apply(lambda x: len(tokenizer.tokenize(x))) <= 511] # 511, because all longer than 512 will be cut off to 512

    # Reset the pandas dataframe index, because now we have less entries
    df = df.reset_index(drop=True)

    # Initial setup
    if not set(["textSanitized", "textLanguageQualityFRE", "textLanguageQualityFKGL", "textLength",
                "textSanitizedLength", "textWordCount", "textSanitizedWordCount", "tokenizerId",
                "model_id", "processingTime", "FREloss", "FKGLloss", "semanticSimilarity",
                "utilityLoss", "textPropnCount", "textNounmodCount", "textNounCount",
                "textNamedEntityCount"]).issubset(df.columns):
        df["textSanitized"] = ""
        df["textLanguageQualityFRE"] = df["text"].apply(textstat.flesch_reading_ease) # Flesch Reading Ease Score via textstat (human evaluation would be best)
        df["textLanguageQualityFKGL"] = df["text"].apply(textstat.flesch_kincaid_grade) # Flesch-Kincaid Grade Level via textstat (human evaluation would be best)
        df["textLength"] = df["text"].apply(len)
        df["textSanitizedLength"] = 0
        df["textWordCount"] = 0
        df["textSanitizedWordCount"] = 0
        df["tokenizerId"] = tokenizer_id
        df["model_id"] = utils.default_model
        df["processingTime"] = 0.0
        df["FREloss"] = 0.0
        df["FKGLloss"] = 0.0
        df["semanticSimilarity"] = 0.0
        df["utilityLoss"] = 0.0
        df["textPropnCount"] = 0
        df["textNounmodCount"] = 0
        df["textNounCount"] = 0
        df["textNamedEntityCount"] = 0

    if os.path.exists(checkpointFile):
        # Resume from checkpoint if it exists
        df_checkpoint = pd.read_csv(checkpointFile)
        start_idx = len(df_checkpoint)
    else:
        # Initialize an empty DataFrame with the same columns as df
        df_checkpoint = pd.DataFrame(columns=df.columns)
        # Save the initial state as checkpoint
        df_checkpoint.to_csv(checkpointFile, index=False)
        start_idx = 0

    print(f"Starting process from index {start_idx} out of {len(df)}, this might take a while...")
    for i in tqdm(range(start_idx, len(df))):
        # Apply automaticSanitization to the 'text' column
        start_time = time.time()
        df.at[i, "textSanitized"] = automaticSanitization(
            df.at[i, "text"], temperature=0.5
        )[0]
        end_time = time.time()

        df.at[i, "processingTime"] = end_time - start_time
        df.at[i, "textSanitizedLength"] = len(df.at[i, "textSanitized"])

        # Truncate sanitized text to 512 tokens if it exceeds this limit (because the BERT sentiment model cannot process more than 512)
        sentiment_tokens = sentiment_tokenizer.tokenize(df.at[i, "textSanitized"])
        if len(sentiment_tokens) > 510: # 510 becase we have to account for special tokens ([CLS] and [SEP])
            sentiment_tokens = sentiment_tokens[:510]
            df.at[i, "textSanitized"] = sentiment_tokenizer.convert_tokens_to_string(sentiment_tokens)


        doc = utils.nlp(df.at[i, "textSanitized"])
        utils.merge_phrases(doc)
        df.at[i, "textSanitizedWordCount"] = len(
            [token for token in doc if not token.is_punct]
        )

        # Compute language quality metrics
        df.at[i, "textSanitizedLanguageQualityFRE"] = textstat.flesch_reading_ease(df.at[i, "textSanitized"])
        df.at[i, "textSanitizedLanguageQualityFKGL"] = textstat.flesch_kincaid_grade(df.at[i, "textSanitized"])

        # Normalize the scores (0 -> 1) and calculate the “loss“ (negative means the sanitized text has higher quality)
        df.at[i, "FREloss"] = (df.at[i, "textLanguageQualityFRE"]/100) - (df.at[i, "textSanitizedLanguageQualityFRE"]/100)
        df.at[i, "FKGLloss"] = (df.at[i, "textSanitizedLanguageQualityFKGL"]/20) - (df.at[i, "textLanguageQualityFKGL"]/20)

        # Compute semantic similarity (SBERT semantic similarity)
        original_embedding = model.encode(df.at[i, "text"], convert_to_tensor=True)
        sanitized_embedding = model.encode(df.at[i, "textSanitized"], convert_to_tensor=True)
        df.at[i, "semanticSimilarity"] = pytorch_cos_sim(original_embedding, sanitized_embedding).item()

        # Determine sentiment scores
        original_sentiment_distribution = [result['score'] for result in sentiment_model(df.at[i, "text"], top_k=None)]
        sanitized_sentiment_distribution = [result['score'] for result in sentiment_model(df.at[i, "textSanitized"], top_k=None)]

        # Calculate the utility loss (BERT-based sentiment analysis) as the KL divergence of the two distributions
        # Kullback-Leibler (KL) quantifies how much one probability distribution differs from another one
        df.at[i, "utilityLoss"] = kl_div(original_sentiment_distribution, sanitized_sentiment_distribution).sum()

        # Calculate the word count (use the same POS tagging modificiations we did in utils.py)
        doc = utils.nlp(df.at[i, "text"])
        utils.merge_phrases(doc)
        df.at[i, "textWordCount"] = len([token for token in doc if not token.is_punct])

        # Calculate the number of proper nouns in the original text (use the same POS tagging modificiations we did in utils.py)
        df.at[i, "textPropnCount"] = len([token for token in doc if token.pos_ == 'PROPN'])

        # Calculate the number of noun modifiers (adjectives and adverbs) in the original text
        df.at[i, "textNounmodCount"] = len([token for token in doc if token.pos_ in ['ADJ', 'ADV']])

        # Calculate the number of nouns in the original text
        df.at[i, "textNounCount"] = len([token for token in doc if token.pos_ == 'NOUN'])

        # Calculate the number of named entities excluding certain types
        excluded_entity_types = ["PRODUCT", "WORK_OF_ART", "ORG", "NORP", "FAC", "CARDINAL"]
        entities = doc.ents
        df.at[i, "textNamedEntityCount"] = len([entity for entity in entities if entity.label_ not in excluded_entity_types])

        # Read the checkpoint dataframe and append the new row
        df_checkpoint = pd.read_csv(checkpointFile)
        df_checkpoint = pd.concat([df_checkpoint, df.iloc[i:i+1]], ignore_index=True) # Append the row to the dataframe

        # Update checkpoint file
        df_checkpoint.to_csv(checkpointFile, index=False)

        #if i >= 6:
        #    break

    if os.path.exists(checkpointFile):
        # Get filenames for final CSVs
        outCSVPath = generate_filename("sanitized_", outPath)
        outCSVPath_dropped = generate_filename("sanitized_dropped_", outPath)

        # Save dataframes
        save_dataframe(df_checkpoint, outCSVPath)
        save_dataframe(df_checkpoint, outCSVPath_dropped, ["text", "textSanitized"])

        print("File without 'text' and 'textSanitized' has been written to: ", outCSVPath_dropped)
        print("File has been written to: ", outCSVPath)

        # Remove the checkpoint file
        os.remove(checkpointFile)



def evaluateResults(inCSVPath, outPath):
    """
    Reads a CSV file and evaluates various metrics present in the CSV file by calculating their
    aggregate statistics including mean, median, standard deviation, minimum, and maximum values.
    The calculated statistics are then saved to a new CSV file.

    Parameters:
    inCSVPath (str): The path to the input CSV file.
    outPath (str): The path to the output directory where the results will be written to.

    Returns:
    None: This function does not return any value but writes the results to an output CSV file.
    """

    # Read the CSV file
    df = pd.read_csv(inCSVPath)

    # Calculate various aggregate statistics for each metric
    metrics = ["processingTime", "semanticSimilarity", "utilityLoss",
               "textLanguageQualityFRE", "textLanguageQualityFKGL",
               "textSanitizedLanguageQualityFRE", "textSanitizedLanguageQualityFKGL",
               "textLength", "textSanitizedLength", "textWordCount",
               "textSanitizedWordCount", "textPropnCount", "textNounmodCount", "textNounCount", "textNamedEntityCount"]

    # Create a DataFrame to store the results
    results = pd.DataFrame(index=metrics, columns=["mean", "median", "std", "min", "max"])

    # Compute the aggregate statistics
    for metric in metrics:
        results.loc[metric, "mean"] = df[metric].mean()
        results.loc[metric, "median"] = df[metric].median()
        results.loc[metric, "std"] = df[metric].std()
        results.loc[metric, "min"] = df[metric].min()
        results.loc[metric, "max"] = df[metric].max()

    # Save the results to a CSV file
    outCSVPath = outPath + "/" + "evaluation_results.csv"
    results.to_csv(outCSVPath)

    print("Results have been written to: ", outCSVPath)


#sanitizeImdb62("evaluation_data/imdb62_AA_test.csv", "evaluation_data/sanitization_results")
#evaluateResults("", "evaluation_data/sanitization_results")
