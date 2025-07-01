const videoPreview = document.getElementById("videoPreview");
const captureCanvas = document.getElementById("captureCanvas");
const ctx = captureCanvas.getContext("2d");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const wordOutput = document.getElementById("wordOutput");
const statusIndicator = document.getElementById("processingStatus");
const clearWordBtn = document.getElementById("clearWordBtn");

let stream = null;
let isPredicting = false;
let lastLetter = "";
let lastLetterTime = 0;
let builtWord = "";

const predictionInterval = 800; // 800ms between predictions
const addLetterDelay = 1000; // at least 1s before adding a letter again

// Navigation buttons
document.getElementById("startBtn").onclick = () => {
  homePage.classList.remove("active");
  cameraPage.classList.add("active");
  initCamera();
};

document.getElementById("cameraLink").onclick = () => {
  homePage.classList.remove("active");
  cameraPage.classList.add("active");
  initCamera();
};

document.getElementById("homeLink").onclick = () => {
  homePage.classList.add("active");
  cameraPage.classList.remove("active");
  stopCamera();
};

document.getElementById("backBtn").onclick = () => {
  homePage.classList.add("active");
  cameraPage.classList.remove("active");
  stopCamera();
};

clearWordBtn.onclick = () => {
  builtWord = "";
  wordOutput.textContent = "";
};

// Add rectangle frame overlay
const predictionFrame = document.createElement("div");
predictionFrame.id = "predictionFrame";
document.querySelector(".camera-container").appendChild(predictionFrame);

// Camera setup
function initCamera() {
  captureCanvas.width = 224;
  captureCanvas.height = 224;

  navigator.mediaDevices
    .getUserMedia({ video: { facingMode: "user" } })
    .then((s) => {
      stream = s;
      videoPreview.srcObject = stream;
      videoPreview.play();
      requestAnimationFrame(predictLoop);
    })
    .catch((err) => {
      console.error("Camera error:", err);
    });
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  isPredicting = false;
}

// Prediction loop
function predictLoop() {
  if (!stream) return;

  const now = Date.now();
  if (now - lastLetterTime > predictionInterval) {
    captureAndPredict();
    lastLetterTime = now;
  }
  requestAnimationFrame(predictLoop);
}

function captureAndPredict() {
  if (isPredicting) return;

  // Draw video frame to canvas
  ctx.drawImage(
    videoPreview,
    videoPreview.videoWidth / 2 - 112, // crop center region
    videoPreview.videoHeight / 2 - 112,
    224,
    224,
    0,
    0,
    224,
    224
  );

  const imageData = captureCanvas.toDataURL("image/jpeg", 0.8);
  const imgData = ctx.getImageData(0, 0, 224, 224).data;

  // Check average brightness
  let sum = 0;
  for (let i = 0; i < imgData.length; i += 4) {
    sum += imgData[i]; // R channel is fine
  }
  const avgBrightness = sum / (imgData.length / 4);

  if (avgBrightness < 30) {
    // frame too dark or no hand
    predictionText.textContent = "No hand detected";
    confidenceText.textContent = "";
    isPredicting = false;
    return;
  }

  // Proceed with prediction
  isPredicting = true;
  statusIndicator.textContent = "Processing...";

  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image: imageData }),
  })
    .then((res) => res.json())
    .then((data) => {
      predictionText.textContent = data.prediction;
      confidenceText.textContent = `Confidence: ${data.confidence}`;
      statusIndicator.textContent = "Active";

      // If new letter different from last or enough time passed
      const currentTime = Date.now();
      if (
        data.prediction !== lastLetter &&
        currentTime - lastLetterTime > addLetterDelay
      ) {
        builtWord += data.prediction.charAt(0); // Only letter part
        wordOutput.textContent = builtWord;
        lastLetter = data.prediction;
        lastLetterTime = currentTime;
      }
    })
    .catch((err) => {
      console.error("Prediction error:", err);
      predictionText.textContent = "Prediction failed";
    })
    .finally(() => {
      isPredicting = false;
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // UI Elements
    const homeLink = document.getElementById('homeLink');
    const cameraLink = document.getElementById('cameraLink');
    const startBtn = document.getElementById('startBtn');
    const backBtn = document.getElementById('backBtn');
    const clearWordBtn = document.getElementById('clearWordBtn');
    const homePage = document.getElementById('homePage');
    const cameraPage = document.getElementById('cameraPage');
    const videoFeed = document.getElementById('videoFeed');
    const formedWord = document.getElementById('formedWord');
    const predictionText = document.getElementById('predictionText');
    const confidenceText = document.getElementById('confidenceText');
    const cameraPlaceholder = document.getElementById('cameraPlaceholder');

    let wordUpdateInterval = null;
    let isCapturing = false;

    // Navigation functions
    function showHomePage() {
        homePage.classList.add('active');
        cameraPage.classList.remove('active');
        homeLink.classList.add('active');
        cameraLink.classList.remove('active');
        stopCapture();
    }

    function showCameraPage() {
        homePage.classList.remove('active');
        cameraPage.classList.add('active');
        homeLink.classList.remove('active');
        cameraLink.classList.add('active');
        startCapture();
    }

    // Video capture functions
    function startCapture() {
        if (!isCapturing) {
            isCapturing = true;
            if (cameraPlaceholder) {
                cameraPlaceholder.style.display = 'none';
            }
            if (videoFeed) {
                videoFeed.style.display = 'block';
            }
            startWordUpdates();
        }
    }

    function stopCapture() {
        isCapturing = false;
        if (cameraPlaceholder) {
            cameraPlaceholder.style.display = 'flex';
        }
        if (videoFeed) {
            videoFeed.style.display = 'none';
        }
        stopWordUpdates();
    }

    // Word formation functions
    function updateWord() {
        if (!isCapturing) return;

        fetch('/get_word')
            .then(response => response.json())
            .then(data => {
                if (data.word !== undefined) {
                    formedWord.textContent = data.word;
                }
                if (data.prediction !== undefined) {
                    predictionText.textContent = data.prediction;
                }
                if (data.confidence !== undefined) {
                    confidenceText.textContent = `Confidence: ${data.confidence}%`;
                }
            })
            .catch(error => console.error('Error getting word:', error));
    }

    function startWordUpdates() {
        // Update immediately
        updateWord();
        // Then update every 100ms for smoother updates
        if (!wordUpdateInterval) {
            wordUpdateInterval = setInterval(updateWord, 100);
        }
    }

    function stopWordUpdates() {
        if (wordUpdateInterval) {
            clearInterval(wordUpdateInterval);
            wordUpdateInterval = null;
        }
        // Clear displays
        formedWord.textContent = '';
        predictionText.textContent = 'Waiting for gesture...';
        confidenceText.textContent = 'Confidence: --';
    }

    // Event listeners
    homeLink.addEventListener('click', showHomePage);
    cameraLink.addEventListener('click', showCameraPage);
    startBtn.addEventListener('click', showCameraPage);
    backBtn.addEventListener('click', showHomePage);

    clearWordBtn.addEventListener('click', function() {
        fetch('/clear_word')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    formedWord.textContent = '';
                }
            })
            .catch(error => console.error('Error clearing word:', error));
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        // Press Spacebar to clear word
        if (event.code === 'Space' && isCapturing) {
            event.preventDefault();
            clearWordBtn.click();
        }
        // Press Escape to go back to home
        if (event.code === 'Escape' && isCapturing) {
            backBtn.click();
        }
    });

    // Error handling for video feed
    if (videoFeed) {
        videoFeed.addEventListener('error', function() {
            console.error('Error loading video feed');
            cameraPlaceholder.style.display = 'flex';
            cameraPlaceholder.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i>
                <p>Error loading camera feed. Please check camera permissions and try again.</p>
            `;
        });
    }
});
