let backendBaseUrl;
if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    backendBaseUrl = `http://${window.location.hostname}/backend`;
} else {
    backendBaseUrl = `https://${window.location.hostname}/backend`;
}

let socketBaseUrl;
if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    socketBaseUrl = `http://${window.location.hostname}`;
} else {
    socketBaseUrl = `https://${window.location.hostname}`;
}

let currentAnnotation = [];
let writeMode = true;
let previousText = ""; // This is required to check whether a completely new text was pasted in
let currentlySanitizing = false;
let sanitizationRequestController;
let sanitizationDurationInterval;
let fetch;
let socket;

var modelOptions = {
  languageModel: "Backend/chatgpt_paraphrases_out_100000_xl", // Default model
  temperature: 0.5, // Default temperature
};

initializeFetchAndSocket();
checkServerStatus();
setInterval(checkServerStatus, 3000);

socket.on("too_many_clients", function (msg) {
  if (msg == "too_many") {
    clientsOverlay.style.opacity = "1";
    clientsOverlay.style.visibility = "visible";
  } else {
    clientsOverlay.style.opacity = "0";
    clientsOverlay.style.visibility = "hidden";
  }
});

socket.on("connect", () => {
  console.log("Socket Connected!");
});

socket.on("connect_error", (error) => {
  console.log("Socket Connection Error: " + error);
});

socket.on("connect_timeout", () => {
  console.log("Socket Connection Timeout");
});

socket.on("error", (error) => {
  console.log("Socket Error: " + error);
});

socket.on("disconnect", (reason) => {
  console.log("Socket Disconnected: " + reason);
});

// This function sends a request to the backend server to check its status,
// and updates the user interface depending on the server's status.
function checkServerStatus() {
  fetch(backendBaseUrl + "/online")
    .then((response) => {
      if (response.ok) {
        response.json().then((data) => {
          if (
            data["status"] == "downloading-model" &&
            modelLoadingOverlay.style.opacity != "1"
          ) {
            offlineOverlay.innerHTML =
              '<p>Preparing language models... <span class="current-mode">(takes a while if this is the first time)</p></span></p>';
            offlineOverlay.style.opacity = "1";
            offlineOverlay.style.visibility = "visible";
            if (writeMode == false) {
              // Go back to write mode
              toggleMode();
              toggleSwitch.checked = !toggleSwitch.checked;
            }
          } else {
            offlineOverlay.style.opacity = "0";
            offlineOverlay.style.visibility = "hidden";
          }
        });
      }
    })
    .catch((error) => {
      console.log(error);
      offlineOverlay.innerHTML =
        "<p>Server offline. Please start the server using <i>python server.py</i>.</p>";
      offlineOverlay.style.opacity = "1";
      offlineOverlay.style.visibility = "visible";
      if (writeMode == false) {
        // Go back to write mode
        toggleMode();
        toggleSwitch.checked = !toggleSwitch.checked;
      }
    });
}

// This function initializes the fetch and socket functions based on the current runtime environment.
// If running in a browser, it uses the global fetch and socket.io functions.
// If running in a Node.js environment, it requires and uses the node-fetch and socket.io-client modules.
function initializeFetchAndSocket() {
  const socketOptions = {
    autoConnect: true,
  };
  if (typeof window !== "undefined") {
    // Running in a browser environment
    fetch = window.fetch;
    const io = window.io;
    socket = io(socketBaseUrl, socketOptions);
  } else {
    // Running in Node.js environment
    fetch = require("node-fetch");
    const ioLib = require("socket.io-client");
    io = ioLib.default || ioLib;
    socket = io(socketBaseUrl, socketOptions);
  }
}

// This function calculates the Jaccard similarity between two text inputs.
// The Jaccard similarity is the size of the intersection divided by the size of the union of the word sets of two texts.
// So, e.g., if the proportion of shared words is less than 50%, it considers the texts to be more than 50% different..
function jaccardSimilarity(text1, text2) {
  let set1 = new Set(text1.split(" "));
  let set2 = new Set(text2.split(" "));
  let intersection = new Set([...set1].filter((x) => set2.has(x)));
  let union = new Set([...set1, ...set2]);
  return intersection.size / union.size;
}

// This function toggles the visibility of the side menu.
// If the side menu is currently visible, it will be hidden, and vice versa.
function toggleMenu() {
  get_model()
    .then((data) => {
      console.log(`The server-side model is ${data['model']}`);
      if (data['model'] == "Backend/chatgpt_paraphrases_out_100000_xl") {
        t5BaseSwitch.checked = false;
        t5XlSwitch.checked = true;
      } else {
        t5BaseSwitch.checked = true;
        t5XlSwitch.checked = false;
      }
      memoryUsageP.innerHTML = `Memory usage: ${data['memory_usage']}/${data['system_memory']} GB`;
    })
    .catch((error) => {
      console.error("Error:", error);
    });

  const sideMenu = document.getElementsByClassName("side-menu")[0];
  const overlay = document.getElementById("overlay");
  if (sideMenu.style.left === "0px") {
    sideMenu.style.left = "-450px"; // 300px + Padding
    overlay.style.opacity = "0";
    overlay.style.visibility = "hidden";
  } else {
    sideMenu.style.left = "0px";
    overlay.style.opacity = "1";
    overlay.style.visibility = "visible";
  }
}

// This function sends a request to the server to set a new language model.
// It sends a POST request to the server with the path of the new model in the request body.
function set_model(newPath) {
  return new Promise((resolve, reject) => {
    fetch(backendBaseUrl + "/set_model", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        newPath: newPath,
      }),
    })
      .then((response) => {
        if (!response.ok) {
          // If a server-side error occurred, reject the promise
          response.json().then((errorData) => {
            console.error("Error:", errorData.error);
            reject(errorData.error);
          });
        } else {
          // Otherwise, resolve the promise with the response data
          response.json().then((data) => resolve(data));
        }
      })
      .catch((error) => {
        // Network or connection errors are caught here
        console.error("Error:", error);
        reject(error);
      });
  });
}

function get_model() {
  return new Promise((resolve, reject) => {
    fetch(backendBaseUrl + "/get_model", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((response) => response.json())
      .then((data) => resolve(data))
      .catch((error) => {
        console.error("Error:", error);
        reject([]);
      });
  });
}

// This function sends a request to the server to get the tokenization of a given text.
// i.e., splitting up text into individual words or tokens.
function get_tokenization(text) {
  return new Promise((resolve, reject) => {
    fetch(backendBaseUrl + "/tokenize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: text,
      }),
    })
      .then((response) => response.text())
      .then((data) => resolve(data))
      .catch((error) => {
        console.error("Error:", error);
        reject([]);
      });
  });
}

// This function sends a request to the server to get the annotation of a given text.
function get_annotation(text, annotation = []) {
  if (1 - jaccardSimilarity(previousText, text) > 0.5) {
    // texts are more than 50% different.
    annotation = []; // i.e. tell the Python backend to efficiently recompute the whole text
  }
  previousText = text;
  return new Promise((resolve, reject) => {
    const startTime = new Date(); // Record the start time

    fetch(backendBaseUrl + "/annotation", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: text,
        annotation: annotation,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        const endTime = new Date(); // Record the end time
        const processingTime = endTime - startTime; // Calculate the processing time in milliseconds

        console.log("Annotation processing time:", processingTime, "ms");

        currentAnnotation = data; // update local current annotation
        resolve(data);
      })
      .catch((error) => {
        console.error("Error:", error);
        reject([]);
      });
  });
}

// This function sends a request to the server to modify the sensitivity of certain IDs in the text annotation.
// It sends a POST request to the server with the IDs and the current annotation in the request body.
function modifySensitivity(ids, annotation = []) {
  return new Promise((resolve, reject) => {
    fetch(backendBaseUrl + "/modify_sensitivity", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ids: ids,
        annotation: annotation,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        resolve(data);
      })
      .catch((error) => {
        console.error("Error:", error);
        reject([]);
      });
  });
}

// This function fetches the tokenization of the text in the original input field, and updates the field's content with mark tags.
async function addMarkTags() {
  const originalInput = document.getElementById("original-input");
  try {
    const markTags = await get_tokenization(originalInput.textContent);
    originalInput.innerHTML = markTags;
  } catch (error) {
    console.error("addMarkTags() Error:", error);
  }
}

// This function fetches the annotation of the text in the original input field, and adds corresponding classes to the mark tags.
async function addAnnotationToMarks() {
  const originalInputTextContent =
    document.getElementById("original-input").textContent;
  const annotations = await get_annotation(
    originalInputTextContent,
    currentAnnotation
  );

  for (let i = 0; i < annotations.length; i++) {
    const annotation = annotations[i];
    const annotationIds = Array.from(annotation.ids);
    let mark = document.querySelector(
      `mark[ids="[${annotationIds.join(", ")}]"]`
    );
    if (mark !== null) {
      mark.classList.remove(
        ...[
          "never-sensitive",
          "potentially-sensitive",
          "sensitive",
          "highly-sensitive",
        ]
      );
      mark.classList.add(
        annotation.sensitivity === 2
          ? "highly-sensitive"
          : annotation.sensitivity === 1
          ? "sensitive"
          : annotation.sensitivity === 0
          ? "potentially-sensitive"
          : ""
      );
      // Set the sensitivity explanation number
      mark.setAttribute("expl", annotation.expl);
    }
  }
}

// This function resets the annotation of all potentially-sensitive, sensitive, or highly-sensitive mark tags to potentially-sensitive (grey)
function clearAnnotation() {
  removeTooltips();
  const markElements = document.querySelectorAll("#original-input mark");
  markElements.forEach((mark) => {
    if (
      mark !== null &&
      (mark.classList.contains("potentially-sensitive") ||
        mark.classList.contains("sensitive") ||
        mark.classList.contains("highly-sensitive"))
    ) {
      mark.classList.remove(
        ...["potentially-sensitive", "sensitive", "highly-sensitive"]
      );
      mark.classList.add("potentially-sensitive");
    }
  });
  addTooltips();
  updateWordListeners();
  currentAnnotation = [];
}

const tooltipDictionary = {
  "potentially-sensitive": {
    entity:
      "This entity will remain in the LLM's input and its embedding can be used in the paraphrasing step.",
    NOUN: "This common noun will remain in the LLM's input and its embedding can be used in the paraphrasing step.",
    PROPN:
      "This proper noun will remain in the LLM's input and its embedding can be used in the paraphrasing step.",
    PRON: "This pronoun will remain in the LLM's input and its embedding can be used in the paraphrasing step.",
    ADJ: "This adjective will remain in the LLM's input and its embedding can be used in the paraphrasing step.",
    ADV: "This adverb will remain in the LLM's input and its embedding can be used in the paraphrasing step.",
  },
  sensitive: {
    entity:
      "For context, this entity will remain in the LLM's input. However, its embedding will be given zero probability in the paraphrasing step.",
    NOUN: "This common noun will be replaced with a more general term from WordNet (if it exists). Also, its embedding will be given zero probability in the paraphrasing step.",
    PROPN:
      "This proper noun will be replaced with a more general term from Wikidata. Also, its embedding will be given zero probability in the paraphrasing step.",
    PRON: "This pronoun will be replaced with the static term “somebody“ in the LLM's input. However, its embedding can be used elsewhere in the paraphrasing step.",
    ADJ: "This adjective will remain in the LLM's input. However, its embedding cannot be used in the paraphrasing step.",
    ADV: "This adverb will remain in the LLM's input. However, its embedding cannot be used in the paraphrasing step.",
  },
  "highly-sensitive": {
    entity:
      "This entity will be replaced with the static string, “certain entity“ in the LLM's input. Also, its embedding will be given zero probability in the paraphrasing step.",
    NOUN: "The phrase that depends on this common noun will be removed from the LLM's input. Also, the noun's embedding will be given zero probability in the paraphrasing step.",
    PROPN:
      "The phrase that depends on this proper noun will be removed from the LLM's input. Also, the proper noun's embedding will be given zero probability in the paraphrasing step.",
    PRON: "This pronoun will be replaced with the static term “somebody“ in the LLM's input. However, its embedding can be used elsewhere in the paraphrasing step.",
    ADJ: "This adjective will be removed from the LLM's input. Also, its embedding cannot be used in the paraphrasing step.",
    ADV: "This adverb will be removed from the LLM's input. Also, its embedding cannot be used in the paraphrasing step.",
  },
};

const entity_lookup = {
  PERSON: "certain person",
  GPE: "certain region",
  LOC: "certain location",
  EVENT: "certain event",
  LAW: "certain law",
  LANGUAGE: "certain language",
  DATE: "certain date",
  TIME: "certain time",
  PERCENT: "certain percentage",
  MONEY: "certain money",
  QUANTITY: "certain quantity",
  ORDINAL: "certain ordinal",
};

function getSanitizationExplanation(markClass, entityType, entityKey) {
  // Prepare an entity replacement if necessary
  let entityReplacement =
    entityKey in entity_lookup ? entity_lookup[entityKey] : "certain entity";

  // Check if the markClass and entityType exist in the tooltipDictionary
  if (
    tooltipDictionary[markClass] &&
    tooltipDictionary[markClass][entityType]
  ) {
    // If they do, return the corresponding explanation
    let explanation = tooltipDictionary[markClass][entityType];

    // Replace "certain entity" with the appropriate entity replacement
    if (explanation.includes("certain entity")) {
      explanation = explanation.replace("certain entity", entityReplacement);
    }

    return explanation;
  } else {
    // If they don't, return a default explanation
    return "The provided markClass or entityType does not exist in the tooltipDictionary.";
  }
}

function getSensitivityExplanation(id, sensitivity) {
  switch (id) {
    case "0":
      return "This token has not been evaluated yet and therefor has no sensitivity value.";
    case "1":
      return `Nouns, pronouns, adjectives, adverbs are “${sensitivity}“ by default.`;
    case "2":
      return `Proper nouns are “${sensitivity}“ by default, because they describe a specific person, place, or thing.`;
    case "3":
      return `This token is either not a valid English word or highly domain specific. Using it may re-identify you, hence “${sensitivity}“.`;
    case "4":
      return `This token is unexpected based on the overal topic of your text. Therefor, it may re-identify you, hence “${sensitivity}“.`;
    case "5":
      return `This token is a common named entity and thus gives away a lot of information. Therefor, it may contribute to re-identifying you, hence “${sensitivity}“.`;
    case "6":
      return `This token's annotation “${sensitivity}“ was manually set by you.`;
    default:
      return `We have lost track why this token is “${sensitivity}“.`;
  }
}

// This function wraps sensitive mark tags in tooltip divs.
// The tooltips provide extra information about the sensitive parts of the text.
function addTooltips() {
  // Get all the mark elements
  const markElements = document.querySelectorAll("#original-input mark");

  // Loop through each mark element
  markElements.forEach((mark) => {
    // Get class of mark
    const markClass = mark.getAttribute("class");

    // Get sensitivity explanation
    const expl = mark.getAttribute("expl");

    // If class is one of the specified ones
    if (
      markClass === "potentially-sensitive" ||
      markClass === "sensitive" ||
      markClass === "highly-sensitive"
    ) {
      // Create a wrapper div with 'tooltip' class
      const wrapper = document.createElement("div");
      wrapper.classList.add("tooltip");

      // Create a tooltip text span
      const tooltipText = document.createElement("span");
      tooltipText.classList.add("tooltiptext");

      // Populate the tooltip text
      let entityType = mark.getAttribute("entitytype");
      let entityTypeTooltipString = "";

      if (entityType.includes("NONENTITY")) {
        entityTypeTooltipString =
          entityType.replace("NONENTITY ", "") + " (POS)"; // Replaces "NOENTITY " with "" and appends "(POS)"
      } else {
        entityTypeTooltipString = entityType + " (NER)"; // Appends "(NER)"
      }

      let sensitivity = markClass.replace("-", " ");

      let tooltipDictionaryLookupKey;
      if (entityType.includes("NONENTITY")) {
        tooltipDictionaryLookupKey = entityType.replace("NONENTITY ", "");
      } else {
        tooltipDictionaryLookupKey = "entity";
      }
      const sanitizationExplanation = getSanitizationExplanation(
        markClass,
        tooltipDictionaryLookupKey,
        entityType
      );
      const sensitivitiyExplanation = getSensitivityExplanation(expl, sensitivity);

      tooltipText.innerHTML = `
      <table>
        <tr>
            <th>Type:</th>
            <td><span class="type-span" data-entity="${entityTypeTooltipString}">${entityTypeTooltipString}</span></td>
        </tr>
        <tr>
            <th>Sensitivity:</th>
            <td><span class="sensitivity-span" data-entity="${sensitivity}">${sensitivity}</span></td>
        </tr>
        <tr>
            <th>Explanation:</th>
            <td>${sensitivitiyExplanation}</td>
        </tr>
        <tr>
            <th>Sanitization:</th>
            <td>${sanitizationExplanation}</td>
        </tr>
      </table>

      `;

      // Insert the mark element inside the wrapper div
      wrapper.appendChild(mark.cloneNode(true));

      // Append the tooltip to the wrapper div
      wrapper.appendChild(tooltipText);

      // Replace mark with our wrapper div
      mark.replaceWith(wrapper);
    }
  });
}

// This function removes all tooltips from mark tags.
// This can be used when the sensitivity of the text has changed, and the tooltips need to be regenerated.
function removeTooltips() {
  // Get all the tooltip elements
  const tooltips = document.querySelectorAll(".tooltip");

  // Loop through each tooltip
  tooltips.forEach((tooltip) => {
    // Get the mark element within the tooltip
    const mark = tooltip.querySelector("mark");

    // Replace tooltip with the original mark element
    if (mark) {
      tooltip.replaceWith(mark.cloneNode(true));
    }
  });
}

// This function toggles the sensitivity of a given annotation, and updates the user interface to reflect this change.
async function toggleSensitivity(annotationIds) {
  removeTooltips();

  let mark = document.querySelector(
    `mark[ids="[${annotationIds.join(", ")}]"]`
  );
  if (mark !== null) {
    mark.removeAttribute("style");

    if (mark.classList.contains("sensitive")) {
      mark.classList.replace("sensitive", "highly-sensitive");
    } else if (mark.classList.contains("highly-sensitive")) {
      mark.classList.replace("highly-sensitive", "potentially-sensitive");
    } else {
      mark.classList.replace("potentially-sensitive", "sensitive");
    }

    mark.style.cursor = "pointer";

    // Set the sensitivity explanation number
    mark.setAttribute("expl", 6);
  }

  addTooltips();
  updateWordListeners();

  let newAnnotation = await modifySensitivity(annotationIds, currentAnnotation);
  currentAnnotation = newAnnotation;
}

// This function adjusts a given color by a certain amount.
// It is used for dynamically generating lighter or darker shades of a base color.
function adjust(color, amount) {
  let newColor =
    "#" +
    color
      .replace(/^#/, "")
      .replace(/../g, (color) =>
        (
          "0" +
          Math.min(255, Math.max(0, parseInt(color, 16) + amount)).toString(16)
        ).substr(-2)
      );
  if (newColor !== "#ffffff") {
    return newColor;
  }
}

// This function converts an RGB color string to a hex color string.
function rgbToHex(rgb) {
  const rgbArray = rgb.match(/\d+/g).map(Number);
  return `#${rgbArray.map((x) => x.toString(16).padStart(2, "0")).join("")}`;
}

// This function fetches the computed style of an HTML element.
// It is used for retrieving the current color of a mark tag, so that it can be adjusted.
function getStyle(el, styleProp) {
  let styleValue;

  if (el.currentStyle) {
    styleValue = el.currentStyle[styleProp];
  } else {
    styleValue = document.defaultView.getComputedStyle(el, null)[styleProp];
  }

  if (styleValue.includes("rgb")) {
    return rgbToHex(styleValue);
  } else {
    return styleValue;
  }
}

// This function removes all mark tags from the original input field, and replaces them with plain text.
function removeMarkTags() {
  const markElements = document
    .getElementById("original-input")
    .querySelectorAll("mark");

  for (const mark of markElements) {
    const plainText = mark.textContent;
    const textNode = document.createTextNode(plainText);
    mark.parentNode.replaceChild(textNode, mark);
  }
}

// This function sets the text cursor to the end of a contentEditable element.
function setCursorToEnd(element) {
  element.focus();
  const range = document.createRange();
  const selection = window.getSelection();
  const lastChild = element.lastChild;

  if (lastChild) {
    const lastChildLength =
      lastChild.nodeType === Node.TEXT_NODE
        ? lastChild.textContent.length
        : lastChild.childNodes.length;
    range.setStart(lastChild, lastChildLength);
    range.collapse(true);
    selection.removeAllRanges();
    selection.addRange(range);
  }
}

// This function updates the event listeners of all the words in the text.
// It removes any existing event listeners, and then adds new ones based on the current sensitivity of the words.
function updateWordListeners() {
  let markElements = document.querySelectorAll("mark");
  markElements.forEach((markElement) => {
    removeAllEventListeners(markElement);
  });
  markElements = document.querySelectorAll("mark");
  markElements.forEach((markElement, index) => {
    if (
      markElement.className == "potentially-sensitive" ||
      markElement.className == "sensitive" ||
      markElement.className == "highly-sensitive"
    ) {
      markElement.addEventListener("mouseover", function (event) {
        this.setAttribute("style", "cursor:pointer !important");
        this.style.backgroundColor = adjust(
          getStyle(this, "backgroundColor"),
          -20
        );
      });
      markElement.addEventListener("mouseout", function () {
        this.style.backgroundColor = adjust(
          getStyle(this, "backgroundColor"),
          +20
        );
      });
      markElement.addEventListener("click", function () {
        toggleSensitivity(JSON.parse(markElement.getAttribute("ids")));
      });
      markElement.addEventListener("mouseover", function () {
        // On mouseover, reduce the opacity of all other mark elements
        markElements.forEach(function (otherMark) {
          if (otherMark !== markElement) {
            otherMark.style.opacity = "0.5";
          }
        });
      });

      markElement.addEventListener("mouseout", function () {
        // On mouseout, reset the opacity of all other mark elements
        markElements.forEach(function (otherMark) {
          otherMark.style.opacity = "1.0";
        });
      });
    }
  });
}

// This function removes all event listeners from an HTML element by cloning the element and replacing the original with the clone.
// Therefor, the clone will not have any of the original element's event listeners.
function removeAllEventListeners(element) {
  const clone = element.cloneNode(true);
  element.parentNode.replaceChild(clone, element);
  return clone;
}

// This function toggles the mode of the user interface between writing and highlighting.
// In write mode, the user can edit the text. In highlight mode, the user can view and modify the sensitivity of the words in the text.
async function toggleMode() {
  const originalInput = document.getElementById("original-input");
  const originalHeader = document.getElementById("originalHeader");
  const sanitizedContentOverlay = document.getElementById(
    "sanitized-content-overlay"
  );
  const sanitizedContentP = document.getElementById("sanitized-text");

  originalInput.contentEditable = !originalInput.isContentEditable;
  originalInput.style.userSelect =
    originalInput.style.userSelect === "none" ? "" : "none";
  writeMode = !writeMode;
  if (writeMode == false) {
    // highlight mode

    sanitizedStatusOverlay.style.color = "#848484";

    sanitizedContentOverlay.style.opacity = 0;
    sanitizedContentP.style.pointerEvents = "auto";

    originalHeader.innerHTML =
      'Original <span class="current-mode">(please wait...)</span>';

    animateBlurOut(sanitizedContentP);

    removeTooltips();
    removeMarkTags();
    await addMarkTags();
    await addAnnotationToMarks();
    addTooltips();

    updateWordListeners();

    originalHeader.innerHTML =
      'Original <span class="current-mode">(annotate mode)</span>';

    setCursorToEnd(originalInput);
    originalInput.focus();
  } else {
    // write mode

    sanitizedStatusOverlay.style.color = "#ffffff";

    sanitizedContentOverlay.style.opacity = 1;
    sanitizedContentP.style.pointerEvents = "none";

    originalHeader.innerHTML =
      'Original <span class="current-mode">(write mode)</span>';

    animateBlurIn(sanitizedContentP);

    removeTooltips();
    removeMarkTags();
    setCursorToEnd(originalInput);
    originalInput.focus();
  }
}

// This function toggles the sanitization of the text.
async function toggleSanitization() {
  let toggleSwitch = document.getElementById("toggle-mode");
  let originalInput = document.getElementById("original-input");
  let sanitizeButton = document.getElementById("sanitize-button");
  let sanitizedHeader = document.getElementById("sanitized-header");

  if (currentlySanitizing) {
    currentlySanitizing = !currentlySanitizing;
    sanitizeButton.textContent = "Sanitize";
    sanitizeButton.classList.remove("loading");
    toggleSwitch.disabled = false;
    toggleSwitch.parentNode.style.opacity = 1.0;
    toggleSwitch.parentNode.style.cursor = "pointer";
    document.getElementById("switch-slider").style.cursor = "pointer";
    clearInterval(sanitizationDurationInterval);
    counter = 0;
    sanitizedHeader.innerHTML = `Sanitized <span class="current-mode"></span>`;
    if (sanitizedContentP.innerHTML === "") {
      sanitizedStatusOverlay.innerHTML =
        "<p>Press the <i>Sanitize</i> button to start.</p>";
      sanitizedStatusOverlay.style.opacity = 1.0;
    }

    socket.off();
    sanitizationRequestController.abort();
    stopGeneratingOnServer();
  } else {
    currentlySanitizing = !currentlySanitizing;
    sanitizeButton.textContent = "Stop";
    sanitizeButton.classList.add("loading");
    toggleSwitch.disabled = true;
    toggleSwitch.parentNode.style.opacity = 0.5;
    toggleSwitch.parentNode.style.cursor = "not-allowed";
    document.getElementById("switch-slider").style.cursor = "not-allowed";

    sanitizedStatusOverlay.style.opacity = 0.0;
    sanitizedStatusOverlay.style.display = "none";

    let counter = 0;
    sanitizationDurationInterval = setInterval(() => {
      counter += 10; // Increment by 10 milliseconds
      const formattedCounter = (counter / 1000).toFixed(2);
      sanitizedHeader.innerHTML = `Sanitizing... <span class="current-mode">(${formattedCounter} seconds)</span>`;

      if (counter >= 120000) { // Two minute timeout
        clearInterval(sanitizationDurationInterval);
        counter = 0;
        console.log("Timeout");
        toggleSanitization();
      }
    }, 10); // every 10 milliseconds

    if (toggleSwitch.checked === false) {
      await toggleMode();
      toggleSwitch.checked = true;
    }

    removeTooltips();
    let plainText = originalInput.textContent;
    let markAnnotation = originalInput.innerHTML.toString();
    addTooltips();
    updateWordListeners();

    sanitizedContentP.innerText = "";
    sanitizeText(plainText, markAnnotation);
  }
}

// This function sends the text and its annotation to the server to be sanitized.
// The sanitized text is then displayed in the user interface.
function sanitizeText(plainText, markAnnotation) {
  // Event listener for 'generated_text' event
  socket.off();
  socket.on("generated_text", function (data) {
    if (currentlySanitizing) {
      sanitizedContentP.innerText = data.text;
    }
    if (sanitizedStatusOverlay.style.opacity !== 0.0) {
      sanitizedStatusOverlay.style.opacity = 0.0;
      setTimeout(function () {
        sanitizedStatusOverlay.innerHTML =
          "<p>Press the <i>Sanitize</i> button to start.</p>";
      }, 700);
    }
  });
  socket.on("status", function (data) {
    if (currentlySanitizing) {
      console.log(data.text);
      sanitizedStatusOverlay.style.display = "flex";
      if (data.text.includes("Wikidata")) {
        sanitizedStatusOverlay.style.opacity = 1.0;
        sanitizedStatusOverlay.innerHTML = `<p>Removing words and dependent phrases, finding inflections and hypernyms on WordNet, searching Wikidata for generalizations...</p>`;
      }
    }
  });

  return new Promise((resolve, reject) => {
    if (sanitizationRequestController) {
      sanitizationRequestController.abort();
    }
    sanitizationRequestController = new AbortController();
    const signal = sanitizationRequestController.signal;
    fetch(backendBaseUrl + "/sanitize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: plainText,
        annotation: markAnnotation,
        temperature: parseFloat(temperatureValue.innerText).toFixed(1),
        noRepeatBigram: noRepeatBigram.checked,
        extremeLengthPenalty: extremeLengthPenalty.checked,
      }),
      signal: signal,
    })
      .then((response) => response.text())
      .then((data) => {
        setTimeout(function () {
          toggleSanitization();
          resolve(data);
          console.log(data);
        }, 500);
      }) // {"status": "complete"}
      .catch((error) => {
        if (error.name === "AbortError") {
          console.log("Fetch aborted");
        } else {
          console.error("Error:", error);
        }
        reject("");
      });
  });
}

// This function sends a request to the server to stop any ongoing text generation.
// It is called when the user stops the sanitization process before it is finished.
function stopGeneratingOnServer() {
  return new Promise((resolve, reject) => {
    fetch(backendBaseUrl + "/stop_generating", {
      method: "POST",
    })
      .then((response) => response.text())
      .then((data) => resolve(data)) // {"status": "stopped"}
      .catch((error) => {
        if (error.name === "AbortError") {
          console.log("Fetch aborted");
        } else {
          console.error("Error:", error);
        }
        reject("");
      });
  });
}
