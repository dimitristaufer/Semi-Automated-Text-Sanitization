# pylint: disable=W0105
# pylint: disable=W0012
# pylint: disable=unused-import
# pylint: disable=no-name-in-module

# FYI: MediaWiki API has a limit of 200 requests per second

import pprint
import requests
from requests.adapters import HTTPAdapter, Retry
from lxml import html
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

sentence_transformer_model = SentenceTransformer('all-mpnet-base-v2')

WIKI_API_BASE = "https://en.wikipedia.org/w/api.php?action=query&format=json"

def get_with_retry(session, url, **kwargs):
    """Make a GET request with retries.

    Args:
        session (requests.Session): The session to use.
        url (str): The URL to request.
        **kwargs: Additional keyword arguments to pass to session.get().

    Returns:
        dict: The JSON-decoded response text.
    """
    retries = Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 502, 503, 504])  # 429 = too many requests
    session.mount('https://', HTTPAdapter(max_retries=retries))

    response = session.get(url, timeout=10, **kwargs)
    response.raise_for_status()

    return response.json()

def get_wikipedia_search_results(session, word, limit=5):
    """Queries the Wikipedia API and returns a list of search results.

    Args:
        session (requests.Session): The session to use.
        word (str): The word to search for.
        limit (int): The maximum number of search results to return.

    Returns:
        list: A list of dictionaries, each representing a search result.
    """
    url = f"{WIKI_API_BASE}&list=search&formatversion=2&srlimit={limit}"\
          f"&srprop=size%7Csnippet%7Ccategorysnippet&srsearch={word}"

    response = get_with_retry(session, url)
    return response.get('query', {}).get('search', [])

def get_wikidata_item_id(session, page_id):
    """Queries the Wikipedia API for the Wikidata item ID of a page.

    Args:
        session (requests.Session): The session to use.
        page_id (int): The ID of the Wikipedia page.

    Returns:
        str: The Wikidata item ID of the page, or an empty string if not found.
    """
    url = f"{WIKI_API_BASE}&pageids={page_id}&prop=pageprops"

    response = get_with_retry(session, url)
    return response.get('query', {}).get('pages', {}).get(str(page_id), {}).get("pageprops", {}).get("wikibase_item", "")

def entity_for_word(word, context=''):
    """Finds the Wikipedia page that is most similar to a given word and context.

    Args:
        word (str): The word to search for.
        context (str): The context in which the word is used.

    Returns:
        dict: A dictionary containing the title and Wikidata item ID of the most similar page.
    """
    session = requests.Session()
    search_results = get_wikipedia_search_results(session, word)
    if not search_results:
        return None

    #pprint.pprint(search_results)

    descriptions = [
        str(html.fromstring(result.get('snippet', {})).text_content())
        for result in search_results if result.get('snippet', {})
    ]
    #print(descriptions)

    # Compute embeddings for each description
    embeddings = sentence_transformer_model.encode(
        descriptions, convert_to_tensor=True)

    # Compute embeddings for the context
    context_embedding = sentence_transformer_model.encode(context, convert_to_tensor=True)

    similarities = []
    # Compute the similarity of each description to the context
    for i, description in enumerate(descriptions):
        sim = cosine_similarity(embeddings[i].unsqueeze(0), context_embedding.unsqueeze(0))
        similarities.append(sim.item())

    #print(similarities)
    most_similar_id = similarities.index(max(similarities))

    if max(similarities) < 0.2:
        most_similar_id = 0

    most_similar_entity = search_results[most_similar_id]
    title = most_similar_entity.get('title', '')
    page_id = most_similar_entity.get('pageid', '')
    wikibase_item = get_wikidata_item_id(session, page_id)

    return {
        "title" : title,
        "wikibase_item" : wikibase_item
    }

def wikidata_query(query=None, url=None):
    """Perform a Wikidata SPARQL query.

    Args:
        query (str): The SPARQL query to perform. Ignored if url is also provided.
        url (str): A URL to fetch. If provided, query is ignored.

    Returns:
        list: A list of bindings from the query results.
    """
    session = requests.Session()

    if url:
        json_response = get_with_retry(session, url)
    elif query:
        url = "https://query.wikidata.org/sparql"
        params = {"query": query, "format": "json"}
        json_response = get_with_retry(session, url, params=params)
    else:
        return []

    return json_response.get('results', {}).get('bindings', [])

def get_direct_children(entity_ids, bindings_copy):
    next_children = []
    for entity_id in entity_ids:
        for binding in bindings_copy:
            if entity_id in binding['item']['value']:
                if 'linkTo' in binding:
                    if not binding['item']['value'] == binding['linkTo']['value']:  # no loop
                        next_children.append({
                            'title': binding['linkTo']['value'],
                            'wikibase_item': binding['linkTo']['value'],
                        })
                else:
                    break
    return next_children


def hierarchy_for_entities(entity_id_strings, depth=1):
    # instance of (P31) and subclass of (P279)
    # https://www.wikidata.org/wiki/Help:Basic_membership_properties

    hierarchies = []

    f_entity_id_strings = ""
    for entity_id_string in entity_id_strings:
        f_entity_id_strings = f_entity_id_strings + 'wd:' + entity_id_string + ' '
    query = f'SELECT DISTINCT ?item ?itemLabel ?linkTo WHERE {{ VALUES ?class {{ {f_entity_id_strings}}} {{ ?class (wdt:P31*) ?item }} UNION {{ ?class (wdt:P279*) ?item }} OPTIONAL {{ ?item (wdt:P31|wdt:P279) ?linkTo. }} FILTER(?item != ?linkTo) SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}}}'

    bindings = wikidata_query(query=query)

    for entity_id_string in entity_id_strings:

        bindings_copy = bindings.copy()

        levels = []
        root_binding = {}
        for binding in bindings_copy:
            if entity_id_string in binding.get('item', {}).get('value', ''):
                root_binding = {
                    'title': binding.get('itemLabel', {}).get('value', ''),
                    'wikibase_item': binding.get('item', {}).get('value', ''),
                }
                levels.append([root_binding])
        #print(root_binding)
        level = get_direct_children([root_binding.get('wikibase_item', '')], bindings_copy)
        max_iterations = depth
        while level and max_iterations > 0:
            level_with_labels = []
            for child in level:
                #label = next((binding['itemLabel']['value'] for binding in bindings_copy if binding['item']['value'] == child['wikibase_item']), None)
                label = next((binding.get('itemLabel', {}).get('value', '') for binding in bindings_copy if binding.get('item', {}).get('value', '') == child.get('wikibase_item', {})), None)
                if label:
                    if child is not None and 'title' in child: # added, does this work?
                        child['title'] = label
                        level_with_labels.append(child)
            levels.append(level_with_labels)
            level = get_direct_children([child.get('wikibase_item', '') for child in level], bindings_copy)
            max_iterations -= 1

        levels = [levels[x] for x in range(len(levels)) if not levels[x] in levels[:x]]  # remove duplicates
        hierarchies.append(levels)

    #pprint.pprint(hierarchies)
    return hierarchies
