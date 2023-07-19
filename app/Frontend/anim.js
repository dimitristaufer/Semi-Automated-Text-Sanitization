// Extract Blur Value
function extractBlurValue(element) {
  let filterValue = getComputedStyle(element).filter;
  let blurMatch = filterValue.match(/blur\((\d+(\.\d+)?)px\)/);
  return blurMatch ? parseFloat(blurMatch[1]) : 0;
}

// Animation Function
function animateBlur(element, endBlur, step) {
  let currentBlur = extractBlurValue(element);

  const interval = setInterval(() => {
    currentBlur -= step;
    if (currentBlur <= endBlur) {
      currentBlur = endBlur;
      clearInterval(interval);
    }
    element.style.filter = `blur(${currentBlur}px)`;
  }, 10);
}

// Animate Blur In
function animateBlurIn(element) {
  animateBlur(element, 8, 0.2);
}

// Animate Blur Out
function animateBlurOut(element) {
  animateBlur(element, 0, 0.2);
}
