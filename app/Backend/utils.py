# pylint: disable=W0105
# pylint: disable=W0012
# pylint: disable=unused-import
# pylint: disable=no-name-in-module

import hashlib
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from nltk.corpus import wordnet as wn
import nltk
from constituent_treelib import ConstituentTree, Language
import os
import psutil
from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForSeq2SeqLM
import numpy as np
from lemminflect import getAllLemmas, getAllInflections, getLemma
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import torch
from threading import Thread
from bs4 import BeautifulSoup
import wikidata_utils
from spellchecker import SpellChecker
from peft import PeftModel, PeftConfig
import modelManager

print("Downloading Wordnet...")
# Download Wordnet
nltk.download('wordnet')

# Sets an environment variable that controls whether tokenization operations are parallelized (reduced memory usage)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ... so the transformers library will not attempt to download models or other resources from the internet.
os.environ['TRANSFORMERS_OFFLINE'] = "1"

nlp = spacy.load("en_core_web_lg") # FYI: over 1 million unique words

# Create a pipeline for ConstituentTree
ct_pipeline = ConstituentTree.create_pipeline(Language.English, ConstituentTree.SpacyModelSize.Large, quiet=True)

# Load the spell checker
spell = SpellChecker()

def custom_tokenizer(nlp):
    """
    Customizes the SpaCy tokenizer to treat NOUN-NOUN (with dashes) as one token, source: https://stackoverflow.com/questions/59993683/how-can-i-get-spacy-to-stop-splitting-both-hyphenated-numbers-and-words-into-sep
    Args:
        nlp (spacy.lang): The SpaCy language model.
        text (str): The text to be tokenized.
    Returns:
        spacy.tokens.doc.Doc: The tokenized text.
    """

    inf = list(nlp.Defaults.infixes)               # Default infixes
    # Remove the generic op between numbers or between a number and a -
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")
    inf = tuple(inf)                               # Convert inf to tuple
    # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])
    # Remove - between letters rule
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x]
    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                     suffix_search=nlp.tokenizer.suffix_search,
                     infix_finditer=infix_re.finditer,
                     token_match=nlp.tokenizer.token_match,
                     rules=nlp.Defaults.tokenizer_exceptions)

def merge_phrases(doc):
    """
    Identifies and merges phrases in the given SpaCy Doc object. This is required for nouns chunks like 'MacBook Pro'
    Args:
        doc (spacy.tokens.doc.Doc): The SpaCy Doc object representing the text.
    Returns:
        spacy.tokens.doc.Doc: The Doc object with phrases merged.
    """

    with doc.retokenize() as retokenizer:
        start = None  # Start index of the phrase
        for i, token in enumerate(doc):
            # If the token is not a noun, proper noun, or number
            if token.pos_ not in ["NOUN", "PROPN", "NUM", "X"]:
                if start is not None:  # If we're in a phrase, merge it
                    retokenizer.merge(doc[start:i])
                    start = None  # Reset the start index
            else:
                if start is None:  # If we're not in a phrase, start a new one
                    start = i

        if start is not None:  # If we're in a phrase at the end of the document, merge it
            retokenizer.merge(doc[start:])

# Replace the default tokenizer of the nlp object with our custom tokenizer function
nlp.tokenizer = custom_tokenizer(nlp)

# 'all-MiniLM-L12-v2' is a pre-trained model provided by the SentenceTransformers library
# See: https://www.sbert.net/docs/pretrained_models.html
sentence_transformer_model = SentenceTransformer('all-MiniLM-L12-v2')

def set_language_model(newPath):
    """
    Sets the language model for text processing.
    Args:
        model (str): The name of the language model.
    """

    global model
    global modelPath
    global tokenizer
    global device
    global model_available

    model_available = False
    modelPath = newPath

    print(f'Setting model to: {newPath}')
    config = PeftConfig.from_pretrained(newPath)

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        cache_dir="./cached_huggingface_models",
        local_files_only=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path,
            cache_dir="./cached_huggingface_models",
            local_files_only=True,
            device_map="auto")
        model = PeftModel.from_pretrained(model, newPath)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path,
            cache_dir="./cached_huggingface_models",
            local_files_only=True,
            device_map={"": device})
        model = PeftModel.from_pretrained(model, newPath, device_map={"": device})

    model.eval()
    print(f'Model is using: {str(model.get_memory_footprint() / (1024**3))} GB of memory.')

    # Set the flag to True when the models are ready
    model_available = True
    print("Model initialized.")

    return model

def get_model_memory_usage():
    """
    Calculates the memory usage of the given language model.
    Args:
        model (str): The name of the language model.
    Returns:
        int: The memory usage of the model in GB.
    """

    modelMemory = model.get_memory_footprint() / (1024.0 ** 3)
    return f"{modelMemory:.2f}"

def get_total_system_memory():
    """
    Retrieves the total memory available in the system.
    Returns:
        int: The total system memory in GB.
    """

    totalMemory = psutil.virtual_memory().total / (1024.0 ** 3)
    return f"{totalMemory:.2f}"

def init_models():
    """
    Initializes the language models for text processing.
    """

    global model_available
    global default_model
    modelManager.prepare_models()
    set_language_model(default_model)

tokenizer = None
model = None
modelPath = None
device = None
model_available = False
default_model = "Backend/chatgpt_paraphrases_out_100000_xl" # "Backend/chatgpt_paraphrases_out_1000000_base"

max_length = 512  # T5 max length tokens! (not characters)
cancelGeneration = False

init_models()

excluded_entity_types = ["PRODUCT", "WORK_OF_ART", "ORG", "NORP", "FAC", "CARDINAL"]


# Tokenization

def tokenize_return_positions(text):
    """
    Tokenizes the given text and returns the tokens as HTML code.
    Args:
        text (str): The text to be tokenized.
    Returns:
        list: A list of tuples, where each tuple contains a token and its position.
    """

    doc = nlp(text)
    merge_phrases(doc)

    marked_sentence = ""
    current_phrase = ""
    current_indices = []
    entity_type = "NONENTITY"
    entities = doc.ents
    filtered_entities = [entity for entity in entities if entity.label_ not in excluded_entity_types]

    is_in_entity = [any([token in ent for ent in filtered_entities]) for token in doc]

    for i, token in enumerate(doc):
        # Assign a unique id to each token based on its text and position
        u_token_id = int(hashlib.sha256(
            (token.text + str(i)).encode('utf-8')).hexdigest(), 16) % 10**8

        # Assign a second semi-unique id based on its text and the -1/+1 items's text
        previous_token = doc[i-1].text if i-1 >= 0 else ''
        next_token = doc[i+1].text if i+1 < len(doc) else ''
        su_token_id = int(hashlib.sha256(
            (doc[i].text + previous_token + next_token).encode('utf-8')).hexdigest(), 16) % 10**8

        # Check if the token is part of a named entity
        if is_in_entity[i]:
            current_phrase += token.text + " "
            current_indices.append(u_token_id)
            entity_type = token.ent_type_  # Set entity type
        # If the token is not part of a named entity
        else:
            if current_phrase:  # If previous chunk was an entity, add it
                marked_sentence += f'<mark positionICS="{[i-len(current_indices)+j for j in range(len(current_indices))]}" ids="{current_indices}" suid="{su_token_id}" class="never-sensitive" entityType="{entity_type}">{current_phrase.strip()}</mark>'
                if not token.is_punct:
                    marked_sentence += ' '
                current_phrase = ""
                current_indices = []
                entity_type = "NONENTITY"

            # For other words, add them individually
            marked_sentence += f'<mark positionICS="[{i}]" ids="[{u_token_id}]" suid="{su_token_id}" class="never-sensitive" entityType="NONENTITY {token.pos_}">{token.text}</mark>'
            if i+1 < len(doc) and (not doc[i+1].is_punct or token.is_punct):
                marked_sentence += ' '

    # In case the last chunk was an entity
    if current_phrase:
        marked_sentence += f'<mark positionICS="{[i-len(current_indices)+j+1 for j in range(len(current_indices))]}" ids="{current_indices}" suid="{su_token_id}" class="never-sensitive" entityType="{entity_type}">{current_phrase.strip()}</mark>'

    return marked_sentence.strip()

# Estimation and Annotation

def get_entities(entities):
    """
    Identifies the entities in the given text using spaCy NER.
    Args:
        text (str): The text in which to identify entities.
    Returns:
        list: A list of entities identified in the text.
    """

    values = [ent.text for ent in entities]
    labels = [ent.label_ for ent in entities]
    return {"values": values, "labels": labels}

def get_surprisedness(doc, percentile):
    """
    Calculates the surprisedness of the given text.
    Args:
        text (str): The text for which to calculate surprisedness.
    Returns:
        int: The surprisedness of the text.
    """

    # Initialize a list for the entities and their surprisedness values
    entities = []
    surprisedness_values = []

    # Add the nouns and proper nouns to a list
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:
            entities.append(token.text)
            surprisedness_values.append(0.0)

    # Remove None values from entities
    entities = [e for e in entities if e is not None]

    # Check if entities list is empty
    if not entities:
        print("No entities found.")
        return {}, {}, {}

    # Compute embeddings for each entity
    embeddings = sentence_transformer_model.encode(
        entities, convert_to_tensor=True)

    # Compute the mean vector
    mean_vector = torch.mean(embeddings, dim=0)

    # Compute the similarity of each entity to the mean vector
    for i, entity in enumerate(entities):
        sim = pytorch_cos_sim(embeddings[i], mean_vector)
        # surprisedness = 1 - similarity
        surprisedness_values[i] = 1 - sim.item()

    # Create a dictionary with the original order of entities
    surprisedness_dict = dict(zip(entities, surprisedness_values))

    # Create a dictionary with the entries sorted by descending surprisedness
    sorted_dict = dict(sorted(surprisedness_dict.items(),
                       key=lambda item: item[1], reverse=True))

    # Compute the surprisedness value at the given percentile
    cutoff = np.percentile(list(surprisedness_dict.values()), percentile)

    # Create a dictionary with only the top percentile of surprisedness values
    top_percentile_dict = {word: surprisedness for word,
                           surprisedness in sorted_dict.items() if surprisedness >= cutoff}

    return surprisedness_dict, sorted_dict, top_percentile_dict

def calculate_sensitivity(doc, entities, annotations):
    """
    Calculates the sensitivity level of the annotations based on the tokens identified in the text.
    Args:
        doc (spacy.tokens.doc.Doc): The SpaCy Doc object representing the text.
        filtered_entities (list): The list of entities identified in the text.
        annotations (list): The existing annotations.
    Returns:
        list: The updated list of annotations with sensitivity levels calculated.
    """

    annotations_copy = annotations.copy()

    # CHECK 1 - Sensitivity = 0 : Noun, noun modifiers and pronouns
    for annotation in annotations_copy:
        if annotation["tag"] in ['NOUN', 'ADJ', 'ADV', 'PRON']:
            annotation["sensitivity"] = 0
            annotation["expl"] = 1

    # CHECK 2 - Sensitivity = 1 : Proper nouns
    for annotation in annotations_copy:
        if annotation["tag"] in ['PROPN']:
            annotation["sensitivity"] = 1
            annotation["expl"] = 2

    # CHECK 3 - Sensitivity = 1 : Out of Vocabulary words
    #oovs = [token.text for token in doc if token.is_oov]
    oovs = [token.text for token in doc if token.is_oov or token.text not in spell]
    print("OOVS:" + str(oovs))
    for annotation in annotations_copy:
        if annotation["value"] in oovs:
            annotation["sensitivity"] = 1
            annotation["expl"] = 3

    # CHECK 4 - Sensitivity = 1 : Surprisedness
    surprisedness_dict, sorted_dict, top_percentile_dict = get_surprisedness(
        doc, 75)
    surprisedness_words = set(top_percentile_dict.keys())
    for annotation in annotations_copy:
        if annotation["value"] in surprisedness_words:
            annotation["sensitivity"] = 1
            annotation["expl"] = 4

    # CHECK 5 - Sensitivity = 2 : Name Entity Recognition Check
    entities_value_label = get_entities(entities)
    for annotation in annotations_copy:
        if annotation["value"] in entities_value_label["values"]:
            annotation["sensitivity"] = 2
            annotation["expl"] = 5

    return annotations_copy

def process_text(text, initialRun=False):
    """
    Analyzes an unsanitized text using SpaCy, and generates annotations for each token in the text.
    Depending on the part of speech of a token, it assigns a sensitivity level to it.
    Args:
        text (str): The text to be processed.
        initialRun (bool): Specifies whether this is the first run of the function. If True, sensitivity scores will be calculated.
    Returns:
        tuple: A tuple containing the list of annotations, the SpaCy Doc object, and a list of filtered entities.
    """
    doc = nlp(text)
    merge_phrases(doc)

    entities = doc.ents  # only calculate once and pass through
    filtered_entities = [entity for entity in entities if entity.label_ not in excluded_entity_types]

    annotations = []
    annotation_indices = []
    annotation_phrase = ""
    annotation_type = ""
    sensitivity = 0

    is_in_entity = [any([token in ent for ent in filtered_entities]) for token in doc]

    for i, token in enumerate(doc):
        # Assign a unique id to each token based on its text and position
        u_token_id = int(hashlib.sha256(
            (token.text + str(i)).encode('utf-8')).hexdigest(), 16) % 10**8

        # Assign a second semi-unique id based on its text and the -1/+1 items's text
        previous_token = doc[i-1].text if i-1 >= 0 else ''
        next_token = doc[i+1].text if i+1 < len(doc) else ''
        su_token_id = int(hashlib.sha256(
            (doc[i].text + previous_token + next_token).encode('utf-8')).hexdigest(), 16) % 10**8

        # Check if the token is part of a named entity
        if is_in_entity[i]:
            annotation_indices.append(u_token_id)
            annotation_phrase += token.text + ' '
            annotation_type = token.ent_type_  # Set entity type
            sensitivity = 2
        # If the token is not part of a named entity
        else:
            if annotation_indices:  # If previous chunk was an entity, add it
                annotations.append({"ids": annotation_indices,
                                    "suid": su_token_id,
                                    "positionICS": [i - len(annotation_indices) + j for j in
                                                                  range(len(annotation_indices))],
                                    "value": annotation_phrase.strip(),
                                    "tag": annotation_type,
                                    "sensitivity": sensitivity,
                                    "expl" : 0})
                annotation_indices = []
                annotation_phrase = ""
                annotation_type = ""
                sensitivity = 0

            # nouns, adjectives, adverbs, proper nouns -> default sensitivity = 0
            if token.pos_ in ['NOUN', 'ADJ', 'ADV', 'PRON']:
                annotations.append({"ids": [u_token_id],
                                    "suid": su_token_id,
                                    "positionICS": [token.i],
                                    "value": token.text,
                                    "tag": token.pos_,
                                    "sensitivity": 0,
                                    "expl" : 0})

            # proper nouns -> default sensitivity = 1
            if token.pos_ in ['PROPN']:
                annotations.append({"ids": [u_token_id],
                                    "suid": su_token_id,
                                    "positionICS": [token.i],
                                    "value": token.text,
                                    "tag": token.pos_,
                                    "sensitivity": 1,
                                    "expl" : 0})



    # Check for the last entity in a sentence (would otherwise be missed)
    if annotation_indices:
        annotations.append({"ids": annotation_indices,
                            "suid": su_token_id,
                            "positionICS": [i - len(annotation_indices) + j + 1 for j in
                                                          range(len(annotation_indices))],
                            "value": annotation_phrase.strip(),
                            "tag": annotation_type,
                            "sensitivity": sensitivity,
                            "expl" : 0})

    if initialRun:
        #print("IS INITIAL")
        annotations = calculate_sensitivity(doc, filtered_entities, annotations)

    return annotations, doc, filtered_entities

def update_annotation(text, annotation):
    """
    Processes a new version of the text, and updates the annotations accordingly.
    Args:
        text (str): The updated text.
        annotation (list): The existing annotations.
    Returns:
        list: The updated list of annotations.
    """

    # Process the new text
    new_annotation, doc, entities = process_text(text)

    # Contains the annotation of words that were added to the text
    # and have not yet been evaluated by calculate_sensitivity()
    unmatched_words = []

    # Preserve sensitivities based on matching values
    for new_word in new_annotation:
        matched = False
        for old_word in annotation:
            if new_word['suid'] == old_word['suid']:
                new_word['sensitivity'] = old_word['sensitivity']
                new_word['expl'] = old_word['expl']
                matched = True
                break
        if not matched:
            # Newly added noun, adv, adj that hasn't been evaluated yet
            unmatched_words.append(new_word)
            new_annotation.remove(new_word)

    #print(len(unmatched_words))
    #print(unmatched_words)
    if len(unmatched_words) > 0:
        evaluated_unmatched_words = calculate_sensitivity(
            doc, entities, unmatched_words)
        new_annotation.extend(evaluated_unmatched_words)  # add them back

    return new_annotation

# Modify Sensitivity

def modify_sensitivity(ids, annotation):
    """
    Calculates an returns the sensitivity level of the token(s) for the given IDs.
    Args:
        ids (list): The IDs of the token(s) whose sensitivity level is to be modified.
        annotation (list): The existing annotations.
    Returns:
        list: The updated list of annotations.
    """

    annotation_copy = annotation.copy()
    for data in annotation:
        if data['ids'] == ids:
            data['sensitivity'] = (data['sensitivity'] + 1) % 3
            data['expl'] = 6
            break
    return annotation_copy # not sure why we're working on a copy here!?

# Sanitization

def get_inflections(word):
    """
    Gets all inflections of the given word.
    Args:
        word (str): The word for which to get the inflections.
    Returns:
        list: A list of all inflections of the word.
    """

    lemmas = []
    inflections = []

    lemmas_data = getAllLemmas(word)
    for pos, lemma_list in lemmas_data.items():
        for lemma in lemma_list:
            lemmas.append(str(lemma))

    lemmas = list(set(lemmas))  # remove duplicates

    for lemma in lemmas:
        inflections_data = getAllInflections(lemma)
        for pos, inflection_list in inflections_data.items():
            for inflection in inflection_list:
                inflections.append(str(inflection))

    inflections = list(set(inflections))  # remove duplicates

    if len(inflections) == 0:
        inflections.append(word)

    # Add lowercase or capitalized variant of each inflection too
    for word_infl in inflections:
        if word_infl.lower() not in inflections:
            inflections.append(word_infl.lower())
        if word_infl.capitalize() not in inflections:
            inflections.append(word_infl.capitalize())

    return inflections

def get_bad_words(excluded_strings):
    """
    Gets all inflections of the words in the excluded_strings list.
    Args:
        excluded_strings (list): The list of strings to exclude.
    Returns:
        list: A list of all bad words.
    """

    excluded_strings_copy = excluded_strings.copy()
    for excluded_string in excluded_strings:
        excluded_strings_copy += get_inflections(excluded_string)
    return excluded_strings_copy

def get_useful_hypernym(word):
    """
    Gets the most useful hypernym of the given word, based on word embeddings and cosine similarity.
    Args:
        word (str): The word for which to get the hypernym.
    Returns:
        str: The most useful hypernym of the word.
    """

    # Lemmatize the word to get its base form
    try:
        word = getLemma(word, "NOUN")[0]
    except:
        word = word

    # Get the synsets (synonym sets) of the word
    synsets = wn.synsets(word)

    if not synsets:
        return word # return the original word, because it does not exist in WordNet

    # Use the first synset as the most common meaning of the word
    synset = synsets[0]

    # Get all the hypernyms of the synset along with their level
    hypernyms = []
    level = 0
    for hyper in synset.closure(lambda s: s.hypernyms()):
        level += 1
        for lemma in hyper.lemmas():
            # Replace underscores in multi-word lemmas with spaces
            name = lemma.name().replace('_', ' ')
            hypernyms.append((name, level))

    # Remove duplicates while preserving order
    hypernyms = list(dict(hypernyms).items())
    if len(hypernyms) == 0:
        return word

    # Encode the word and the hypernyms
    word_embedding = sentence_transformer_model.encode([word], convert_to_tensor=True)
    hypernym_embeddings = sentence_transformer_model.encode([hypernym[0] for hypernym in hypernyms], convert_to_tensor=True)

    # Calculate the cosine similarity of each hypernym to the word using Sentence Transformers (hence usefulness)
    similarities = pytorch_cos_sim(hypernym_embeddings, word_embedding)

    # Combine the hypernyms, levels, and similarities into one list of tuples
    hypernyms_similarities = [(hypernyms[i][0], hypernyms[i][1], similarities[i].item()) for i in range(len(hypernyms))]

    # Sort the hypernyms by similarity, descending (x[2] is the cosine_similarity)
    hypernyms_similarities.sort(key=lambda x: x[2], reverse=True)

    try:
        return hypernyms_similarities[0][0] # get the first hypernym [0] and the string value of it [0]
    except:
        return word # return the original word, because it does not have any hypernyms in WordNet

def remove_dependent_phrases(sentence, entity):
    """
    Removes phrases that are dependent on the given entity from the sentence.
    Args:
        sentence (str): The sentence from which to remove the phrases.
        entity (str): The entity on which the phrases to be removed are dependent.
    Returns:
        str: The sentence with the dependent phrases removed.
    """

    try:
        # Generating the constituent tree for the input sentence
        tree = ConstituentTree(sentence, ct_pipeline)
        # Extracting all phrases from the sentence
        all_phrases = tree.extract_all_phrases()
    except Exception as e:
        # Usually happens when the input sentence is neither of English, German, French, Polish, Hungarian, Swedish, Chinese and Korean.
        print(f"Exception occurred: {e}")
        return sentence

    # Parsing the sentence using Spacy to extract all nouns
    doc = nlp(sentence)
    merge_phrases(doc)
    # Creating a list of all common and proper nouns in the sentence
    nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]

    # If there are no nouns or there is only one common noun or proper noun in the input sentence,
    # the entire sentence should be replaced with an empty string.
    if len(nouns) <= 1:
        return ""

    sub_sentences = sorted(all_phrases.get('S', []), key=len, reverse=False)
    verb_phrases = sorted(all_phrases.get('VP', []), key=len, reverse=False)
    prepositional_phrases = sorted(all_phrases.get('PP', []), key=len, reverse=False)
    noun_phrases = sorted(all_phrases.get('NP', []), key=len, reverse=False)

    # Find the target phrase containing the entity and remove it from the sentence.
    for sub_sentence in sub_sentences:
        if entity in sub_sentence and len(sub_sentence) < len(sentence) * 0.9:
            sentence = sentence.replace(sub_sentence, '')
            break

    for phrase in verb_phrases:
        if entity in phrase:
            sentence = sentence.replace(phrase, '')
            break

    for phrase in prepositional_phrases:
        if entity in phrase:
            sentence = sentence.replace(phrase, '')
            break

    for phrase in noun_phrases:
        if entity in phrase:
            sentence = sentence.replace(phrase, '')
            break

    # Removing any double white spaces and trimming the sentence
    sentence = ' '.join(sentence.split())

    return sentence

def find_sentence(node):
    """
    Finds the sentence in which the given node (token) is present.
    Args:
        node (Node): The node for which to find the sentence.
    Returns:
        str: The sentence in which the node is present.
    """

    left_sentence = []
    right_sentence = []

    # Iterate to the left
    left_node = node.find_previous_sibling()
    while left_node and not 'PUNCT' in left_node.get('entitytype'):
        left_sentence.insert(0, left_node.text.strip())
        left_node = left_node.find_previous_sibling()

    # Handle the current node
    center_sentence = [node.text.strip()]

    # Iterate to the right
    right_node = node.find_next_sibling()
    while right_node and not 'PUNCT' in right_node.get('entitytype'):
        # no need to handle period here
        right_sentence.append(right_node.text.strip())
        right_node = right_node.find_next_sibling()

    return ' '.join(left_sentence + center_sentence + right_sentence)

def remove_adjadv_psshs(annotatedText, tags=None, sensitivities=None):
    """
    Removes adjectives and adverbs (ADJ and ADV) from the annotated text that have a sensitivity level specified in `sensitivities`.
    Args:
        annotatedText (str): The text annotated with HTML tags.
        tags (list): A list of part-of-speech tags to be removed. Defaults to ["ADJ", "ADV"].
        sensitivities (list): A list of sensitivity levels. Defaults to ["sensitive", "highly-sensitive"].
    Returns:
        tuple: A tuple containing the modified text, a list of removed strings, and an empty list.
    """

    if tags is None:
        tags = ["ADJ", "ADV"]
    if sensitivities is None:
        sensitivities = ["sensitive", "highly-sensitive"]

    # Set up the BeautifulSoup library to parse annotatedText as HTML
    soup = BeautifulSoup(annotatedText, 'html.parser')
    removed_strings = []
    for tag in soup.find_all('mark'):
        entity_type = tag.get('entitytype')
        entity_class = tag.get('class')
        if entity_class and entity_class[0] in sensitivities and len(str(tag.text)) >= 2:
            if any(t in entity_type for t in tags):
                removed_strings.append(str(tag.text)) # ZERO WEIGHT in parahrase generation
                # "sensitive" (yellow) are only excluded from generation
                if entity_class[0] == "highly-sensitive":
                    tag.string.replace_with("") # e.g. RED car -> car
    return str(soup), removed_strings, []

def replace_pronouns_psshs(annotatedText, tags=None, sensitivities=None):
    """
    Replaces pronouns (PRON) in the annotated text with "somebody" if they have a sensitivity level specified in `sensitivities`.
    Args:
        annotatedText (str): The text annotated with HTML tags.
        tags (list): A list of part-of-speech tags to be replaced. Defaults to ["PRON"].
        sensitivities (list): A list of sensitivity levels. Defaults to ["sensitive", "highly-sensitive"].
    Returns:
        tuple: A tuple containing the modified text, a list of removed strings, and an empty list.
    """

    if tags is None:
        tags = ["PRON"]
    if sensitivities is None:
        sensitivities = ["sensitive", "highly-sensitive"] # "potentially-sensitive" removed, set back if we get bad results

    soup = BeautifulSoup(annotatedText, 'html.parser')
    removed_strings = []
    for tag in soup.find_all('mark'):
        entity_type = tag.get('entitytype')
        entity_class = tag.get('class')
        if entity_class and entity_class[0] in sensitivities and len(str(tag.text)) >= 1:
            if any(t in entity_type for t in tags):
                # Excluding from generation makes no sense, because the same pronoun might be harmless elsewhere
                tag.string.replace_with("somebody")
    return str(soup), removed_strings, []

def replace_entities_psshs(annotatedText, sensitivities=None):
    """
    Replaces named entities in the annotated text with a static label if they have a sensitivity level specified in `sensitivities`.
    Args:
        annotatedText (str): The text annotated with HTML tags.
        sensitivities (list): A list of sensitivity levels. Defaults to ["sensitive", "highly-sensitive"].
    Returns:
        tuple: A tuple containing the modified text, a list of removed strings, and an empty list.
    """

    if sensitivities is None:
        sensitivities = ["sensitive", "highly-sensitive"]

    soup = BeautifulSoup(annotatedText, 'html.parser')

    entity_lookup = {
        "PERSON": "certain person",
        "GPE": "certain region",
        "LOC": "certain location",
        "EVENT": "certain event",
        "LAW": "certain law",
        "LANGUAGE": "certain language",
        "DATE": "certain date",
        "TIME": "certain time",
        "PERCENT": "certain percentage",
        "MONEY": "certain money",
        "QUANTITY": "certain quantity",
        "ORDINAL": "certain ordinal"
    }


    removed_strings = []
    for tag in soup.find_all('mark'):
        entity_type = tag.get('entitytype')
        entity_class = tag.get('class')
        if "NONENTITY" not in entity_type and entity_class and entity_class[0] in sensitivities and len(str(tag.text)) >= 2:
            for entity, replacement in entity_lookup.items():
                if entity in entity_type:
                    removed_strings.append(str(tag.string)) # ZERO WEIGHT in parahrase generation
                    # "sensitive" (yellow) are only excluded from generation
                    if entity_class[0] == "highly-sensitive":
                        tag.string.replace_with(replacement)
                    break

    return str(soup), removed_strings, []

def replace_propnouns_s(annotatedText, tags=None, sensitivities=None):
    """
    Replaces proper nouns (PROPN) in the annotated text that have a sensitivity level specified in `sensitivities` with a more general term from Wikidata.
    Args:
        annotatedText (str): The text annotated with HTML tags.
        tags (list): A list of part-of-speech tags to be replaced. Defaults to ["PROPN"].
        sensitivities (list): A list of sensitivity levels. Defaults to ["sensitive"].
    Returns:
        tuple: A tuple containing the modified text, a list of removed strings, and an empty list.
    """

    if tags is None:
        tags = ["PROPN"]
    if sensitivities is None:
        sensitivities = ["sensitive"]

    soup = BeautifulSoup(annotatedText, 'html.parser')
    removed_strings = []

    # create an empty dictionary to store the tags and their corresponding wikibase_items
    tag_wikibase_dict = {}

    for tag in soup.find_all('mark'):
        entity_type = tag.get('entitytype')
        entity_class = tag.get('class')
        if entity_class and entity_class[0] in sensitivities and len(str(tag.text)) >= 2:
            if any(t in entity_type for t in tags):
                string_to_remove = str(tag.text)
                removed_strings.append(string_to_remove) # ZERO WEIGHT in parahrase generation

                # get up to 5 previous words (tag.text) separated by spaces
                previous_siblings = tag.find_all_previous(string=True, limit=6)
                context = ''.join(sib for sib in previous_siblings[::-1])

                #print("String to remove: " + string_to_remove)
                #print("Context used for search: " + context)

                wikidata_entity = wikidata_utils.entity_for_word(string_to_remove, context)

                if wikidata_entity is None:
                    print(f"No Wikidata entity found for word: {string_to_remove}")
                    continue
                else:
                    wikibase_item = wikidata_entity["wikibase_item"]

                if wikibase_item is None:
                    print(f"No Wikidata entity found for word: {string_to_remove}")
                    continue
                else:
                    # store tag and its wikibase_item
                    tag_wikibase_dict[tag] = wikibase_item

    # Get hierarchies for all wikibase_items
    wikibase_items = list(tag_wikibase_dict.values())
    hierarchies = wikidata_utils.hierarchy_for_entities(wikibase_items)

    # Replace each tag with its corresponding title
    for tag, hierarchy in zip(tag_wikibase_dict, hierarchies):
        try:
            item = hierarchy[1][0]
            title = item.get('title')
            if title:
                tag.string.replace_with(title)
            else:
                tag.string.replace_with("") # replace with empty string
        except IndexError:
            print("Index out of bounds")

    return str(soup), removed_strings, []

def exclude_remove_nouns_shs(annotatedText, tags=None, sensitivities=None):
    """
    Excludes nouns (NOUN) from the annotated text that have a sensitivity level specified in `sensitivities`.
    Args:
        annotatedText (str): The text annotated with HTML tags.
        tags (list): A list of part-of-speech tags to be excluded. Defaults to ["NOUN"].
        sensitivities (list): A list of sensitivity levels. Defaults to ["sensitive", "highly-sensitive"].
    Returns:
        tuple: A tuple containing the modified text, a list of removed strings, and a list of sentences with highly sensitive nouns.
    """

    if tags is None:
        tags = ["NOUN"]
    if sensitivities is None:
        sensitivities = ["sensitive", "highly-sensitive"]

    sentences_with_highly_sensitive_nouns = []

    soup = BeautifulSoup(annotatedText, 'html.parser')
    removed_strings = []
    for tag in soup.find_all('mark'):
        entity_type = tag.get('entitytype')
        entity_class = tag.get('class')
        if entity_class and entity_class[0] in sensitivities and len(str(tag.text)) >= 2:
            if any(t in entity_type for t in tags):
                removed_strings.append(str(tag.text)) # ZERO WEIGHT in parahrase generation
                if entity_class[0] == "sensitive":
                    general_common_noun = get_useful_hypernym(str(tag.text))
                    tag.string.replace_with(general_common_noun)
                if entity_class[0] == "highly-sensitive":
                    sentences_with_highly_sensitive_nouns.append({"sentence" : find_sentence(tag), "entity" : str(tag.text)})

    return str(soup), removed_strings, sentences_with_highly_sensitive_nouns

def remove_propnouns_hs(annotatedText, tags=None, sensitivities=None):
    """
    Excludes proper nouns (PROPN) from the annotated text that have a sensitivity level specified in `sensitivities`.
    Args:
        annotatedText (str): The text annotated with HTML tags.
        tags (list): A list of part-of-speech tags to be excluded. Defaults to ["PROPN"].
        sensitivities (list): A list of sensitivity levels. Defaults to ["highly-sensitive"].
    Returns:
        tuple: A tuple containing the modified text, a list of removed strings, and a list of sentences with highly sensitive nouns.
    """

    if tags is None:
        tags = ["PROPN"]
    if sensitivities is None:
        sensitivities = ["highly-sensitive"]

    sentences_with_highly_sensitive_nouns = []

    soup = BeautifulSoup(annotatedText, 'html.parser')
    removed_strings = []
    for tag in soup.find_all('mark'):
        entity_type = tag.get('entitytype')
        entity_class = tag.get('class')
        if entity_class and entity_class[0] in sensitivities:
            if any(t in entity_type for t in tags):
                removed_strings.append(str(tag.text)) # ZERO WEIGHT in parahrase generation
                sentences_with_highly_sensitive_nouns.append({"sentence" : find_sentence(tag), "entity" : str(tag.text)})

    return str(soup), removed_strings, sentences_with_highly_sensitive_nouns

def paraphrase_text(text, temperature=0.5, stream=True, excluded_strings=None, no_repeat_bigram=True, extreme_length_penalty=False):
    """
    Generates a paraphrase of the given text using our fine-tuned FLAN T5 model. Excludes certain words and applies various constraints during generation.
    Args:
        text (str): The text to be paraphrased.
        temperature (float): The temperature parameter for the generation. Defaults to 0.5.
        stream (bool): If True, the function yields text as it is being generated. If False, it waits until the entire text is generated and then yields it. Defaults to True.
        excluded_strings (list): A list of strings to be excluded from the paraphrase. Defaults to None.
        no_repeat_bigram (bool): If True, the same 2-gram will not be repeated in the generated text. Defaults to True.
        extreme_length_penalty (bool): If True, applies an extreme length penalty during generation. Defaults to False.
    Returns:
        generator: A generator that yields the generated text.
    """

    bad_words = []
    bad_words_ids = []
    if excluded_strings:
        bad_words = get_bad_words(excluded_strings)
        #print("Bad words:")
        #print(bad_words)
        bad_words_ids = [tokenizer.encode(
            bad_word, add_special_tokens=False) for bad_word in bad_words]

    input = f'paraphrase: {text}' # f'Paraphrase: "{text}"'
    print(f"Input string to model: {input}")
    # Stores print-ready text in a queue, to be used by a downstream application as an iterator
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(input, return_tensors="pt", truncation=True).to(device)

    # Prepare the arguments for the generation
    generation_args = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_length": 512,
        "do_sample": True, # If false, always chooses next token with the highest probability -> "greedy" decoding (repetitive and deterministic). If true, randomly sample from the probability distribution
        "temperature": float(temperature),
        "length_penalty": -float('-inf') if extreme_length_penalty == True else 1.0,
        "no_repeat_ngram_size": 2 if no_repeat_bigram == True else 0, # a value of 2 means that the same 2-gram will not be repeated in the generated text -> helps to avoid generating repetitive phrases. Default is 0.
        "streamer": streamer
    }

    # Only add `bad_words_ids` if it's not empty
    if bad_words_ids:
        generation_args["bad_words_ids"] = bad_words_ids

    print("Starting generation thread...")
    # Run 'model.generate' on a background thread (required by TextIteratorStreamer)
    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        if cancelGeneration:
            streamer.on_finalized_text("", stream_end=True)
            return
        generated_text += new_text
        if stream:
            yield generated_text
    if not stream:
        yield generated_text


