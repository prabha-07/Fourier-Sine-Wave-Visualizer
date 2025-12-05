// DOM Elements
let elements = {};
let elementsInitialized = false;

const initializeElements = () => {
  if (elementsInitialized) return;
  elements = {
    fileInput: document.getElementById("file-input"),
    play: document.getElementById("btn-play"),
    pause: document.getElementById("btn-pause"),
    stop: document.getElementById("btn-stop"),
    reset: document.getElementById("btn-reset"),
    status: document.getElementById("status"),
    progressBar: document.getElementById("progress-bar"),
    statFile: document.getElementById("stat-file"),
    statDuration: document.getElementById("stat-duration"),
    statSampleRate: document.getElementById("stat-sample-rate"),
    statFFT: document.getElementById("stat-fft"),
    freqRows: document.getElementById("frequency-rows"),
    canvas: document.getElementById("visualizer"),
  };
  
  if (!elements.canvas) {
    console.error("Canvas element not found!");
    return;
  }
  
  elements.ctx = elements.canvas.getContext("2d");
  elementsInitialized = true;
  console.log("All elements initialized successfully");
};

// Constants
const MAX_COMPONENTS = 48;

// Application State
const STATE = {
  audioCtx: null,
  source: null,
  audioBuffer: null,
  analyser: null,
  animationId: null,
  playing: false,
  startTime: 0,
  startOffset: 0,
  components: [],
  topFrequencies: [],
  fftSize: null,
  fileName: null,
};

// Utility Functions
const formatSeconds = (seconds) => {
  if (!Number.isFinite(seconds)) return "–";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
};

const setStatus = (message) => {
  if (elements.status) {
    elements.status.textContent = message;
  }
};

const toggleButtons = (enabled) => {
  [elements.play, elements.pause, elements.stop, elements.reset].forEach((btn) => {
    if (btn) btn.disabled = !enabled;
  });
};

const updateStats = () => {
  if (elements.statFile) elements.statFile.textContent = STATE.fileName ?? "–";
  if (elements.statDuration) {
    elements.statDuration.textContent = STATE.audioBuffer ? formatSeconds(STATE.audioBuffer.duration) : "–";
  }
  if (elements.statSampleRate) {
    elements.statSampleRate.textContent = STATE.audioBuffer ? `${STATE.audioBuffer.sampleRate.toLocaleString()} Hz` : "–";
  }
  if (elements.statFFT) {
    elements.statFFT.textContent = STATE.fftSize ? STATE.fftSize.toLocaleString() : "–";
  }
};

const clearFrequencyTable = () => {
  if (elements.freqRows) {
    elements.freqRows.innerHTML = '<p class="empty">Upload an audio file to populate this table.</p>';
  }
};

const renderFrequencyTable = () => {
  if (!elements.freqRows) return;
  
  if (!STATE.topFrequencies.length) {
    clearFrequencyTable();
    return;
  }

  const rows = STATE.topFrequencies
    .map(
      (entry, index) => `
        <div>
          <span>${index + 1}</span>
          <span>${entry.freq.toFixed(2)}</span>
          <span>${entry.amplitude.toFixed(3)}</span>
        </div>`
    )
    .join("");

  elements.freqRows.innerHTML = rows;
};

// Audio Playback Functions
const stopSource = () => {
  if (STATE.source) {
    try {
      STATE.source.onended = null;
      STATE.source.stop();
    } catch (err) {
      console.warn("Source stop error", err);
    }
    STATE.source.disconnect();
    STATE.source = null;
  }
};

const cancelAnimation = () => {
  if (STATE.animationId) {
    cancelAnimationFrame(STATE.animationId);
    STATE.animationId = null;
  }
};

const resetPlaybackState = () => {
  STATE.playing = false;
  STATE.startOffset = 0;
  STATE.startTime = 0;
  if (elements.progressBar) {
    elements.progressBar.style.width = "0%";
  }
};

const ensureAudioContext = () => {
  if (!STATE.audioCtx) {
    // Cross-browser compatibility
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    STATE.audioCtx = new AudioContextClass();
    console.log("AudioContext created, state:", STATE.audioCtx.state);
    
    // Auto-resume on user interaction (production fix)
    const resumeOnInteraction = async () => {
      if (STATE.audioCtx && STATE.audioCtx.state === "suspended") {
        try {
          await STATE.audioCtx.resume();
          console.log("AudioContext auto-resumed on interaction");
        } catch (error) {
          console.warn("AudioContext resume failed:", error);
        }
      }
      // Remove listeners after first successful resume
      document.removeEventListener("click", resumeOnInteraction);
      document.removeEventListener("touchstart", resumeOnInteraction);
      document.removeEventListener("keydown", resumeOnInteraction);
    };
    
    // Listen for any user interaction to enable audio
    document.addEventListener("click", resumeOnInteraction, { once: true });
    document.addEventListener("touchstart", resumeOnInteraction, { once: true });
    document.addEventListener("keydown", resumeOnInteraction, { once: true });
  }
  return STATE.audioCtx;
};

const updateProgress = () => {
  if (!STATE.audioBuffer) return;
  const duration = STATE.audioBuffer.duration;
  const elapsed = STATE.playing
    ? STATE.audioCtx.currentTime - STATE.startTime
    : STATE.startOffset;
  const progress = Math.min((elapsed / duration) * 100, 100);
  if (elements.progressBar) {
    elements.progressBar.style.width = `${progress}%`;
  }
};

// Canvas Drawing Functions
const drawPlaceholder = () => {
  const { ctx, canvas } = elements;
  if (!ctx || !canvas) return;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(2,6,23,0.85)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(148, 163, 184, 0.45)";
  ctx.font = "600 20px 'Space Grotesk', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Upload audio to see the sine wave decomposition", canvas.width / 2, canvas.height / 2);
};

const drawWaves = () => {
  const { ctx, canvas } = elements;
  if (!ctx || !canvas) return;
  
  if (!STATE.components.length || !STATE.audioBuffer) {
    drawPlaceholder();
    return;
  }

  const baselineGradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  baselineGradient.addColorStop(0, "rgba(8,47,73,0.9)");
  baselineGradient.addColorStop(1, "rgba(15,23,42,0.9)");
  ctx.fillStyle = baselineGradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const now = STATE.playing ? STATE.audioCtx.currentTime - STATE.startTime : STATE.startOffset;
  const rows = STATE.components.length;
  const rowHeight = canvas.height / rows;

  STATE.components.forEach((component, index) => {
    const rowTop = index * rowHeight;
    const centerY = rowTop + rowHeight / 2;
    const amplitudePx = rowHeight * 0.45 * component.amplitude;
    const hue = 180 + (component.freq / 20000) * 120;
    const alpha = 0.35 + component.amplitude * 0.6;
    ctx.strokeStyle = `hsla(${hue}, 75%, 60%, ${alpha})`;
    ctx.lineWidth = Math.max(1, component.amplitude * 3);
    ctx.beginPath();

    const freqScale = component.freq / 100;

    for (let x = 0; x <= canvas.width; x += 4) {
      const progress = x / canvas.width;
      const phase = 2 * Math.PI * (freqScale * progress + now * component.freq);
      const y = centerY + Math.sin(phase) * amplitudePx;
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    ctx.fillStyle = `rgba(226,232,240,${Math.min(1, component.amplitude + 0.2)})`;
    ctx.font = "600 11px 'Space Grotesk', sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`${component.freq.toFixed(1)} Hz`, canvas.width - 12, centerY - rowHeight * 0.4);
  });

  if (STATE.playing) {
    STATE.animationId = requestAnimationFrame(() => {
      updateProgress();
      drawWaves();
    });
  }
};

const handleResize = () => {
  const { canvas } = elements;
  if (!canvas) return;
  
  const containerWidth = canvas.parentElement.clientWidth;
  canvas.width = Math.min(1200, containerWidth - 10);
  canvas.height = 600;
  drawWaves();
};

// API Call to Flask Backend
const computeFFT = async (file) => {
  try {
    setStatus("Uploading and processing audio on server...");
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Create AbortController for timeout handling
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 110000); // 110 seconds (just under 120s gunicorn timeout)
    
    let response;
    try {
      response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
      clearTimeout(timeoutId);
    } catch (fetchError) {
      clearTimeout(timeoutId);
      if (fetchError.name === 'AbortError') {
        throw new Error('Request timed out. The audio file may be too large or complex. Try a shorter or smaller file.');
      }
      throw fetchError;
    }
    
    if (!response.ok) {
      let errorMessage = 'Server error';
      try {
        const error = await response.json();
        errorMessage = error.error || errorMessage;
      } catch (e) {
        // If response is not JSON, try to get text
        try {
          errorMessage = await response.text();
        } catch (e2) {
          errorMessage = `Server returned status ${response.status}`;
        }
      }
      throw new Error(errorMessage);
    }
    
    const data = await response.json();
    
    console.log("Received FFT data from Python:", data);
    
    // Use the data from Python backend
    STATE.components = data.components || [];
    STATE.topFrequencies = data.topFrequencies || [];
    STATE.fftSize = data.fftSize;
    STATE.fileName = data.fileName;
    
    // Update audio buffer stats if available
    if (data.sampleRate && STATE.audioBuffer) {
      // Stats already set from audioBuffer
    }
    
    renderFrequencyTable();
    updateStats();
    drawWaves();
    
    setStatus("Analysis complete! Press play to hear the track and animate the sine waves.");
    
  } catch (error) {
    console.error("Error calling API:", error);
    setStatus(`Error: ${error.message}`);
    throw error;
  }
};

// Audio Playback Handlers
const prepareSource = () => {
  stopSource();
  cancelAnimation();
  const ctx = ensureAudioContext();
  const source = ctx.createBufferSource();
  source.buffer = STATE.audioBuffer;
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);
  analyser.connect(ctx.destination);
  STATE.source = source;
  STATE.analyser = analyser;
};

const handlePlay = async () => {
  if (!STATE.audioBuffer) {
    console.error("No audio buffer available");
    setStatus("Error: No audio file loaded.");
    return;
  }
  
  const ctx = ensureAudioContext();
  
  // Production fix: Always ensure AudioContext is running
  if (ctx.state !== "running") {
    try {
      console.log("Resuming AudioContext, current state:", ctx.state);
      await ctx.resume();
      
      // Wait a moment and verify it's running
      await new Promise(resolve => setTimeout(resolve, 50));
      
      if (ctx.state !== "running") {
        // Try one more time
        await ctx.resume();
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      
      if (ctx.state !== "running") {
        throw new Error(`AudioContext state is ${ctx.state}, expected "running"`);
      }
      
      console.log("AudioContext is now running");
    } catch (error) {
      console.error("Failed to resume AudioContext:", error);
      setStatus("Audio not ready. Please click anywhere on the page, then try play again.");
      return;
    }
  }
  
  // Stop any existing playback
  stopSource();
  cancelAnimation();
  
  // Create new audio source
  try {
    prepareSource();
    
    if (!STATE.source) {
      throw new Error("Failed to create audio source");
    }
    
    // Calculate start time
    STATE.startTime = ctx.currentTime - STATE.startOffset;
    
    // Start playback with error handling
    try {
      if (STATE.startOffset > 0) {
        STATE.source.start(0, STATE.startOffset);
      } else {
        STATE.source.start(0);
      }
      
      console.log("Audio playback started successfully");
      
      // Handle playback end
      STATE.source.onended = () => {
        console.log("Playback ended");
        STATE.playing = false;
        STATE.startOffset = 0;
        cancelAnimation();
        updateProgress();
        drawWaves();
        setStatus("Playback finished.");
      };
      
      STATE.playing = true;
      setStatus("Playing in sync with the sine wave visualizer.");
      updateProgress();
      drawWaves();
      
    } catch (playError) {
      console.error("Error starting audio playback:", playError);
      setStatus(`Playback error: ${playError.message}. Please try again.`);
      STATE.playing = false;
      stopSource();
    }
    
  } catch (error) {
    console.error("Error preparing audio source:", error);
    setStatus(`Error: ${error.message}`);
    STATE.playing = false;
  }
};

const handlePause = () => {
  if (!STATE.playing) return;
  STATE.startOffset = STATE.audioCtx.currentTime - STATE.startTime;
  stopSource();
  STATE.playing = false;
  cancelAnimation();
  setStatus("Paused. Resume to keep the visualizer in sync.");
  drawWaves();
};

const handleStop = () => {
  stopSource();
  cancelAnimation();
  resetPlaybackState();
  STATE.playing = false;
  setStatus("Playback stopped.");
  drawWaves();
};

const handleReset = () => {
  handleStop();
  STATE.components = [];
  STATE.topFrequencies = [];
  STATE.audioBuffer = null;
  STATE.fileName = null;
  STATE.fftSize = null;
  clearFrequencyTable();
  updateStats();
  toggleButtons(false);
  // Clear the file input so new files can be selected
  if (elements.fileInput) {
    elements.fileInput.value = '';
  }
  setStatus("Visualizer reset. Upload a new file to begin.");
  drawPlaceholder();
};

// File Handling
const handleFile = async (file) => {
  try {
    if (!file) {
      setStatus("No file selected.");
      return;
    }
    
    if (!file.type.startsWith("audio/")) {
      const audioExtensions = ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac', '.webm'];
      const fileName = file.name.toLowerCase();
      const isAudioExtension = audioExtensions.some(ext => fileName.endsWith(ext));
      
      if (!isAudioExtension) {
        setStatus("Please select an audio file (MP3, WAV, OGG, etc.)");
        return;
      }
    }
    
    STATE.fileName = file.name;
    setStatus("Decoding audio for playback...");
    
    // Decode audio in browser for playback
    const ctx = ensureAudioContext();
    const arrayBuffer = await file.arrayBuffer();
    
    setStatus("Loading audio...");
    try {
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
      STATE.audioBuffer = audioBuffer;
    } catch (decodeError) {
      // Handle audio decoding errors
      if (decodeError.message && decodeError.message.includes('pattern')) {
        throw new Error('Invalid audio file format. Please try a different audio file (MP3, WAV, OGG, etc.)');
      }
      throw new Error(`Failed to decode audio: ${decodeError.message || 'Unknown error'}`);
    }
    
    resetPlaybackState();
    toggleButtons(true);
    STATE.playing = false;
    STATE.startOffset = 0;
    
    // Now compute FFT on server (Python)
    await computeFFT(file);
    
  } catch (error) {
    console.error("Error processing file:", error);
    setStatus(`Error: ${error.message || "Unable to process this audio file."}`);
  }
};

const onFileInputChange = (event) => {
  console.log("=== onFileInputChange CALLED ===");
  const [file] = event.target.files || [];
  if (file) {
    console.log("✓ File found! Name:", file.name);
    handleFile(file);
  } else {
    console.warn("✗ No file in files array");
    setStatus("No file selected. Please try again.");
  }
};

// Wire up controls
const wireControls = () => {
  if (!elements.fileInput) {
    console.error("File input element not found!");
    return;
  }
  
  console.log("Wiring file input change handler");
  elements.fileInput.addEventListener("change", onFileInputChange);
  
  const dropzone = document.querySelector(".dropzone");
  if (dropzone) {
    console.log("Dropzone found, setting up handlers");
    
    dropzone.addEventListener("click", (event) => {
      console.log("Dropzone clicked");
      if (event.target !== elements.fileInput && 
          !event.target.closest('input') &&
          elements.fileInput) {
        console.log("Triggering file input click");
        setTimeout(() => {
          try {
            elements.fileInput.click();
          } catch (err) {
            console.error("Error clicking file input:", err);
          }
        }, 0);
      }
    });

    dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropzone.style.borderColor = "rgba(56, 189, 248, 0.6)";
    });

    dropzone.addEventListener("dragleave", (event) => {
      event.preventDefault();
      dropzone.style.borderColor = "rgba(248, 250, 252, 0.3)";
    });

    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropzone.style.borderColor = "rgba(248, 250, 252, 0.3)";
      const file = event.dataTransfer?.files?.[0];
      if (file) {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        elements.fileInput.files = dataTransfer.files;
        handleFile(file);
      }
    });
  }

  if (elements.play) elements.play.addEventListener("click", handlePlay);
  if (elements.pause) elements.pause.addEventListener("click", handlePause);
  if (elements.stop) elements.stop.addEventListener("click", handleStop);
  if (elements.reset) elements.reset.addEventListener("click", handleReset);
};

// Initialize
const init = () => {
  console.log("=== FOURIER VISUALIZER INIT STARTING ===");
  
  const doInit = () => {
    console.log("=== STARTING INITIALIZATION ===");
    try {
      initializeElements();
      console.log("Elements initialized:", elements);
      
      if (!elements.fileInput) {
        console.error("CRITICAL: File input element not found!");
        return;
      }
      
      wireControls();
      console.log("Controls wired");
      toggleButtons(false);
      drawPlaceholder();
      handleResize();
      window.addEventListener("resize", handleResize);
      setStatus("Ready! Upload an audio file to begin.");
      console.log("=== APP INITIALIZED SUCCESSFULLY ===");
    } catch (error) {
      console.error("CRITICAL ERROR during initialization:", error);
    }
  };
  
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", doInit);
  } else {
    setTimeout(doInit, 100);
  }
};

// Start initialization
console.log("Script loaded, calling init()...");
init();