// Define elements
let appContainer = document.getElementById("app-container");
let appContainerShadow = document.getElementById("app-container-shadow");
let toggleSwitch = document.getElementById("toggle-mode");
let burgerMenuIcon = document.getElementById("hamburger");
let overlay = document.getElementById("overlay");
let offlineOverlay = document.getElementById("overlay-offline");
let modelLoadingOverlay = document.getElementById("overlay-model-loading");
let originalInput = document.getElementById("original-input");
let clientsOverlay = document.getElementById("overlay-clients");
let sanitizeButton = document.getElementById("sanitize-button");
let sanitizedContentP = document.getElementById("sanitized-text");
let sanitizedHeader = document.getElementById("sanitized-header");
let sanitizedStatusOverlay = document.getElementById(
  "sanitized-content-status"
);
let t5BaseSwitch = document.getElementById("FLAN_T5_Base");
let t5XlSwitch = document.getElementById("FLAN_T5_XL");
let temperatureSwitch = document.getElementById("temperature");
let temperatureValue = document.getElementById("temperature-value");
let noRepeatBigram = document.getElementById("no-repeat-bigram");
let extremeLengthPenalty = document.getElementById("extreme-length-penalty");
let memoryUsageP = document.getElementById("memory-usage");

// Prevent double clicking words and highlighting them
document.addEventListener(
  "mousedown",
  function (event) {
    var clickedElement = event.target;
    if (
      event.detail > 1 &&
      writeMode == false &&
      clickedElement.tagName != "P"
    ) {
      event.preventDefault();
    }
  },
  false
);

// Add ALT/Option+A shortcut for toggling modes
document.addEventListener("keydown", function (event) {
  if (event.altKey && (event.code === 65 || event.keyCode === 65)) {
    event.preventDefault();
    toggleMode();
    toggleSwitch.checked = !toggleSwitch.checked;
  }
});

// Add ALT/Option+C shortcut for clearing annotation
document.addEventListener("keydown", function (event) {
  if (event.altKey && (event.code === 67 || event.keyCode === 67)) {
    event.preventDefault();
    clearAnnotation();
  }
});

// Initial state for the toggle switch and event listener
toggleSwitch.checked = false;
toggleSwitch.addEventListener("change", toggleMode);

// Event listeners for burger menu icon and overlay
burgerMenuIcon.addEventListener("click", toggleMenu);
overlay.addEventListener("click", toggleMenu);

// Placeholder text setup
// https://www.noslang.com/reverse/
// During every evening when I do my cleaning, after everyone else has left the building 2, a tall guy enters the building, uses one of the terminals. Most of the times he wears a plaid shirt. On April 5th, I approached him and realized it was John...
originalInput.classList.add("placeholder");
let placeholderText =
  "During evry evening wn I do my cleaning, after ev1 else has left da building 2, a tall guy enters da building, uses one of da terminals. Most of da times he wears a plaid shirt. On April 5th, I approached him and realized it was John...";
originalInput.textContent = placeholderText;

function handlePlaceholder(event) {
  if (
    originalInput.textContent.trim() === "" &&
    !originalInput.classList.contains("placeholder")
  ) {
    originalInput.classList.add("placeholder");
    originalInput.textContent = placeholderText;
    setCursorToEnd(originalInput);
  } else {
    originalInput.classList.remove("placeholder");
  }
}

function handlePlaceholder2(event) {
  if (originalInput.classList.contains("placeholder")) {
    event.preventDefault();
    setCursorToEnd(originalInput);
  }
}

// Keydown event
function handleKeydown(event) {
  if (event.key === "Enter") {
    event.preventDefault();
    alert(
      "Currently new lines are not supported, because they cause issues with the tokenization..."
    );
  }

  const invalidKeys = new Set([
    "ArrowLeft",
    "ArrowRight",
    "ArrowUp",
    "ArrowDown",
    "Home",
    "End",
    "PageUp",
    "PageDown",
    "Tab",
    "Enter",
    "Escape",
    "Backspace",
    "Delete",
  ]);

  if (
    originalInput.classList.contains("placeholder") &&
    !event.altKey &&
    !event.ctrlKey &&
    !event.metaKey &&
    !invalidKeys.has(event.key)
  ) {
    originalInput.classList.remove("placeholder");
    originalInput.textContent = "";
  }
}

// Event listener for paste event
function handlePaste(event) {
  event.preventDefault(); // Prevent default paste behavior

  if (originalInput.classList.contains("placeholder")) {
    originalInput.classList.remove("placeholder");
    originalInput.textContent = "";
  }

  // Get the plain text from the clipboard
  let clipboardData = event.clipboardData || window.clipboardData;
  let plainText = clipboardData.getData("text/plain");
  plainText = plainText.replace(/\s{2,}/g, " "); // Remove all extra whitespaces
  // Insert the modified text into the input
  document.execCommand("insertText", false, plainText);
}

// Pseudo Shadow, because we want rounded corners and overflow
var copySize = function () {
  appContainerShadow.style.width = window
    .getComputedStyle(appContainer)
    .getPropertyValue("width");
  appContainerShadow.style.height = window
    .getComputedStyle(appContainer)
    .getPropertyValue("height");
};
copySize();
var resizeObserver = new ResizeObserver(copySize);
resizeObserver.observe(appContainer);

// Function to update modelOptions
function updateState(key, value) {
  modelOptions[key] = value;
}

function changeLanguageModel() {
  if (writeMode == false) {
    // Go back to write mode
    toggleMode();
    toggleSwitch.checked = !toggleSwitch.checked;
  }
  modelLoadingOverlay.innerHTML =
    "<p>Loading the language model into memory...</p>";
  modelLoadingOverlay.style.opacity = "1";
  modelLoadingOverlay.style.visibility = "visible";

  let counter = 0;
  const counterInterval = setInterval(() => {
    counter += 10; // Increment by 10 milliseconds
    const formattedCounter = (counter / 1000).toFixed(2);
    modelLoadingOverlay.innerHTML = `<p>Loading the language model into memory... <span class="current-mode">(${formattedCounter} seconds)</span></p>`;
  }, 10); // every 10 milliseconds

  set_model(modelOptions.languageModel)
    .then((data) => {
      clearInterval(counterInterval); // Clear the interval when finished loading
      modelLoadingOverlay.style.opacity = "0";
      modelLoadingOverlay.style.visibility = "hidden";

      get_model()
        .then((data) => {
          memoryUsageP.innerHTML = `Memory usage: ${data["memory_usage"]} GB / ${data["system_memory"]} GB`;
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    })
    .catch((error) => {
      clearInterval(counterInterval); // Clear the interval in case of an error
      modelLoadingOverlay.innerHTML =
        "<p>Error loading language model into memory.</p>";
      modelLoadingOverlay.style.opacity = "1";
      modelLoadingOverlay.style.visibility = "visible";
      console.error("An error occurred:", error);
    });
}

// Mobile optimization
function mobileOptimization() {
  const tabsContainer = document.getElementsByClassName('tabs')[0];
  const firstTabContainer = document.getElementsByClassName('tabcontainer')[0];
  const secondTabContainer = document.getElementsByClassName('tabcontainer')[1];
  const originalContent = document.getElementById('original-content');
  const sanitizedContent = document.getElementById('sanitized-content');
  if (window.innerWidth < 700) {
      if (firstTabContainer && originalContent) {
          originalContent.insertBefore(firstTabContainer, originalContent.firstChild);
      }
      if (secondTabContainer && sanitizedContent) {
          sanitizedContent.insertBefore(secondTabContainer, sanitizedContent.firstChild);
      }
  } else {
      if (firstTabContainer && secondTabContainer) {
          tabsContainer.appendChild(firstTabContainer);
          tabsContainer.appendChild(secondTabContainer);
      }
  }
}
window.addEventListener('resize', mobileOptimization);
mobileOptimization();

// Attach event listeners to the language model switches
t5BaseSwitch.addEventListener("change", function () {
  t5XlSwitch.checked = !this.checked;
  updateState(
    "languageModel",
    this.checked
      ? "Backend/chatgpt_paraphrases_out_1000000_base"
      : "Backend/chatgpt_paraphrases_out_100000_xl"
  );
  changeLanguageModel();
});

t5XlSwitch.addEventListener("change", function () {
  t5BaseSwitch.checked = !this.checked;
  updateState(
    "languageModel",
    this.checked
      ? "Backend/chatgpt_paraphrases_out_100000_xl"
      : "Backend/chatgpt_paraphrases_out_1000000_base"
  );
  changeLanguageModel();
});

noRepeatBigram.checked = true;
extremeLengthPenalty.checked = false;
temperatureSwitch.value = 0.5;

// Attach event listener to the temperature slider
temperatureSwitch.addEventListener("input", function () {
  temperatureValue.innerText = parseFloat(this.value).toFixed(1);
  updateState("temperature", parseFloat(this.value).toFixed(1));
});

// Sanitize button event listener
sanitizeButton.addEventListener("click", toggleSanitization);

// Add event listeners to original input
originalInput.addEventListener("input", handlePlaceholder);
originalInput.addEventListener("mousedown", handlePlaceholder2);
originalInput.addEventListener("keydown", handleKeydown);
originalInput.addEventListener("paste", handlePaste);

// Set cursor to end and focus on original input
setCursorToEnd(originalInput);
originalInput.focus();
