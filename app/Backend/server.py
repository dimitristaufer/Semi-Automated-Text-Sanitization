# pylint: disable=W0105
# pylint: disable=W0012
# pylint: disable=unused-import
# pylint: disable=no-name-in-module

import json
from flask import Flask, request, abort, Response, Blueprint
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import utils
from bs4 import BeautifulSoup

app = Flask(__name__)
"""
Flask application object that handles incoming HTTP requests and serves HTTP responses.

Attributes:
    name (str): The name of the application module.
"""

CORS(app)  # Enable CORS (all routes)
socketio = SocketIO(app, path='/socket.io', cors_allowed_origins="*")
bp = Blueprint('backend', __name__, url_prefix='/backend')

connectedClients = 0

def get_json_data(*expected_keys):
    """
    Retrieves and validates JSON data from an HTTP request.

    Parameters:
        *expected_keys (str): Variable-length argument list of keys that should be present in the JSON data.

    Returns:
        dict: The JSON data from the HTTP request.

    Raises:
        HTTPException: If the request does not contain valid JSON or if a required key is missing from the JSON data.
    """

    data = request.get_json()
    if not data:
        abort(400, "Invalid JSON")
    for key in expected_keys:
        if key not in data:
            abort(400, f"Missing expected key: {key}")
    return data


@socketio.on('connect')
def handle_connect():
    """
    Handles a new WebSocket connection, incrementing the count of connected clients.
    Emits a message if there are more than one client connected (block UI).

    Parameters:
    None

    Returns:
    None: This function does not return any value.
    """

    global connectedClients
    connectedClients += 1
    if connectedClients > 1: # There is more than one client connected
        emit('too_many_clients', 'too_many', broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handles a client's disconnection from the WebSocket, decrementing the count of connected clients.
    Emits a message based on the remaining number of connected clients (unblock/block UI).

    Parameters:
    None

    Returns:
    None: This function does not return any value.
    """

    global connectedClients
    connectedClients -= 1
    if connectedClients <= 1: # There is only one client connected
        emit('too_many_clients', 'ok', broadcast=True)
    else: # There is still more than one client connected
        emit('too_many_clients', 'too_many', broadcast=True)

@bp.route("/online", methods=["GET"])
def online():
    """
    Checks if the server is available and sends its status.

    Parameters:
    None

    Returns:
    JSON: A dictionary containing the status of the application model.
    """

    if utils.model_available == True:
        return json.dumps({"status": "online"})
    else:
        return json.dumps({"status": "downloading-model"})


@bp.route("/tokenize", methods=["POST"])
def get_tokenization():
    """
    Gets the text to be sanitized from an HTTP request and returns its tokenization.

    Parameters:
    None

    Returns:
    JSON: A dictionary containing the tokenized form of the text.
    """

    data = get_json_data("text")
    return utils.tokenize_return_positions(data["text"])


@bp.route("/annotation", methods=["POST"])
def get_annotation():
    """
    Gets the text to be sanitized and current annotation data from an HTTP request and processes the text as new text if no annotation is provided,
    else it returns updated annotation.

    Parameters:
    None

    Returns:
    JSON: A dictionary containing the processed text or updated annotation.
    """

    data = get_json_data("text", "annotation")
    return json.dumps(
        utils.process_text(data["text"], initialRun=True)[0]
        if not data["annotation"]
        else utils.update_annotation(data["text"], data["annotation"])
    )


@bp.route("/modify_sensitivity", methods=["POST"])
def get_modify_sensitivity():
    """
    Gets ids and annotation data from an HTTP request and returns modified sensitivity.

    Parameters:
    None

    Returns:
    JSON: A dictionary containing the modified sensitivity.
    """

    data = get_json_data("ids", "annotation")
    return json.dumps(utils.modify_sensitivity(data["ids"], data["annotation"]))


@bp.route("/sanitize", methods=["POST"])
def stream_generated_text():
    """
    Gets the required data from an HTTP request, performs various sanitization steps and emits status and generated text.

    Parameters:
    None

    Returns:
    JSON: A dictionary containing the status of the sanitization process.
    """

    data = get_json_data("text", "annotation", "temperature", "noRepeatBigram", "extremeLengthPenalty")

    sentences_with_phrases_to_remove = []

    sanitization_steps = {
        "Removed Modifiers": utils.remove_adjadv_psshs,
        "Replaced Pronouns": utils.replace_pronouns_psshs,
        "Replaced Entities with Labels": utils.replace_entities_psshs,
        "Replaced Proper Nouns with Wikidata": utils.replace_propnouns_s,
        "Excluded/Removed Nouns": utils.exclude_remove_nouns_shs,
        "Removed Proper Nouns": utils.remove_propnouns_hs,
    }

    removed_strings = []

    for step_message, step_function in sanitization_steps.items():
        data["annotation"], more_removed_strings, new_sentences_with_phrases_to_remove = step_function(data["annotation"])
        removed_strings.extend(more_removed_strings)
        sentences_with_phrases_to_remove += new_sentences_with_phrases_to_remove
        soup = BeautifulSoup(data["annotation"], "html.parser")
        socketio.emit("status", {"text": f"{step_message}: {soup.text}"})

    text_for_paraphrasing = soup.text

    text_for_paraphrasing = " ".join(text_for_paraphrasing.split()) # remove duplicate spaces, because this prevented dependent phrase removal

    print(f"text after normal ops: {text_for_paraphrasing}")
    print(sentences_with_phrases_to_remove)
    for entry in sentences_with_phrases_to_remove:
        sentence = " ".join(str(entry['sentence']).split())
        entity = " ".join(str(entry['entity']).split())
        text_for_paraphrasing = text_for_paraphrasing.replace(sentence, utils.remove_dependent_phrases(sentence, entity))

    if text_for_paraphrasing == "":
        socketio.emit("generated_text", {"text": "(Text is to short to be sanitized)"})
    else:
        utils.cancelGeneration = False
        generated_text = utils.paraphrase_text(
            text_for_paraphrasing,
            temperature=data["temperature"] if data["temperature"] else 0.5,
            no_repeat_bigram=data["noRepeatBigram"] if data["noRepeatBigram"] else True,
            extreme_length_penalty=data["extremeLengthPenalty"] if data["extremeLengthPenalty"] else False,
            stream=True,
            excluded_strings=removed_strings
        )
        for partial_text in generated_text:
            socketio.emit("generated_text", {"text": partial_text})
    return json.dumps({"status": "complete"})


@bp.route("/stop_generating", methods=["POST"])
def stop_generating():
    """
    Cancels the generation process and returns its status.

    Parameters:
    None

    Returns:
    JSON: A dictionary containing the status of the generation process.
    """

    utils.cancelGeneration = True
    return json.dumps({"status": "stopped"})

@bp.route("/set_model", methods=["POST"])
def set_model():
    """
    Sets a new model path from an HTTP request.

    Parameters:
    None

    Returns:
    JSON: A dictionary containing the status of the model setting process.
    """

    data = request.get_json()
    try:
        utils.set_language_model(newPath=data["newPath"])
        return Response(json.dumps({"status": "ready"}), mimetype='application/json')
    except Exception as e:
        print(str(e))
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@bp.route("/get_model", methods=["GET"])
def get_model():
    """
    Gets the model path, memory usage of the model and total system memory.

    Parameters:
    None

    Returns:
    JSON: A dictionary containing the model path, memory usage of the model and total system memory.
    """

    return json.dumps({"model": utils.modelPath,
                       "memory_usage" : utils.get_model_memory_usage(),
                       "system_memory" : utils.get_total_system_memory()})

app.register_blueprint(bp)

socketio.run(app, host="0.0.0.0", port=3050, debug=False, allow_unsafe_werkzeug=True)
