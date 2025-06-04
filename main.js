// Variables globales
const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const demosSection = document.getElementById("demos");
const optionsDiv = document.querySelector('.video-options');
const cameraOptions = document.querySelector('.video-options>select');
const enableWebcamButton = document.getElementById("webcamButton");
const hideCameraBtn = document.getElementById("hide-camera");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext('2d');
const statusDisplay = document.getElementById("status");
const tempCanvas = document.createElement('canvas');
tempCanvas.id = 'id_canvas';
document.body.appendChild(tempCanvas);

let localStream = null;
let modelWorker = null;
let stopModel = false;
let modelLoaded = false;
let frameId = 0;

// Configuración del modelo actualizada
const MODEL_CONFIG = {
  inputSize: [320, 320],
  scoreThreshold: 0.65,
  maxResults: 10,
  iouThreshold: 0.5
};

// Mapeo de clases
const CLASS_MAPPING = {
  0: 'INE_Frente',
  1: 'INE_Reverso'
};

// Inicialización
document.addEventListener('DOMContentLoaded', async () => {
  await setupCameraSelection();
  await initModelWorker();
});

// 1. Configurar selección de cámara
async function setupCameraSelection() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    cameraOptions.innerHTML = videoDevices.map(device => 
      `<option value="${device.deviceId}">${device.label || `Cámara ${videoDevices.indexOf(device) + 1}`}</option>`
    ).join('');
  } catch (error) {
    console.error("Error al enumerar dispositivos:", error);
  }
}

// 2. Inicializar el Web Worker
async function initModelWorker() {
  modelWorker = new Worker('./worker-qf16.js');
  
  modelWorker.onmessage = (e) => {
    const { event, data, id, debugInfo } = e.data;
    
    if (event === 'model-ready') {
      modelLoaded = true;
      demosSection.classList.remove("invisible");
      enableWebcamButton.disabled = false;
      updateStatus("Modelo QF16 cargado. Haz clic en 'Activar Cámara'.");
    } 
    else if (event === 'prediction' && id === frameId) {
      renderDetections(data);
      logDetectionDetails(data);
      
      if (debugInfo) {
        console.log("📊 Debug info:", debugInfo);
      }
    }
    else if (event === 'error') {
      console.error("Error en el worker:", data);
      updateStatus("Error en el modelo. Ver consola para detalles.", true);
    }
  };
  
  // Inicializar el modelo en el worker
  modelWorker.postMessage({
    event: 'init',
    modelPath: './assets/tflite/ssd_mobilenetv2_lite_320x320_docs_idmex_v3_qf16_metadata.tflite',
    config: MODEL_CONFIG,
    classMapping: CLASS_MAPPING
  });
}

// 3. Activar la cámara
async function enableCam() {
  if (!modelLoaded) {
    alert("El modelo aún no ha terminado de cargar");
    return;
  }

  try {
    const constraints = {
      audio: false,
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        deviceId: cameraOptions.value ? { exact: cameraOptions.value } : undefined,
        facingMode: "environment"
      }
    };

    localStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = localStream;

    enableWebcamButton.classList.add("removed");
    hideCameraBtn.classList.remove("removed");
    optionsDiv.classList.add("removed");

    await new Promise(resolve => {
      video.onloadeddata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        resolve();
      };
    });

    stopModel = false;
    updateStatus("Detección activa. Mostrando resultados en tiempo real...");
    predictWebcam();

  } catch (error) {
    console.error("Error al acceder a la cámara:", error);
    handleCameraError(error);
  }
}

// 4. Preprocesar frame para el worker
function preprocessFrameForWorker(videoElement) {
  const [targetWidth, targetHeight] = MODEL_CONFIG.inputSize;
  const tempCtx = tempCanvas.getContext('2d');
  
  tempCanvas.width = targetWidth;
  tempCanvas.height = targetHeight;
  
  // Limpiar y rellenar con negro
  tempCtx.fillStyle = 'black';
  tempCtx.fillRect(0, 0, targetWidth, targetHeight);

  // Calcular escala manteniendo relación de aspecto
  const videoRatio = videoElement.videoWidth / videoElement.videoHeight;
  const targetRatio = targetWidth / targetHeight;
  
  let drawWidth, drawHeight, offsetX, offsetY;
  
  if (videoRatio > targetRatio) {
    drawHeight = targetHeight;
    drawWidth = videoElement.videoWidth * (targetHeight / videoElement.videoHeight);
    offsetX = (targetWidth - drawWidth) / 2;
    offsetY = 0;
  } else {
    drawWidth = targetWidth;
    drawHeight = videoElement.videoHeight * (targetWidth / videoElement.videoWidth);
    offsetX = 0;
    offsetY = (targetHeight - drawHeight) / 2;
  }

  // Dibujar el video escalado y centrado
  tempCtx.drawImage(videoElement, offsetX, offsetY, drawWidth, drawHeight);

  // Obtener ImageData y transferir el buffer para mejor rendimiento
  const imageData = tempCtx.getImageData(0, 0, targetWidth, targetHeight);
  return {
    width: targetWidth,
    height: targetHeight,
    data: imageData.data
  };
}

// 5. Bucle de predicción
async function predictWebcam() {
  if (stopModel) return;
  
  frameId++;
  const currentFrameId = frameId;
  
  try {
    const frameData = preprocessFrameForWorker(video);
    
    // Enviar frame al worker (con transferencia de buffer)
    modelWorker.postMessage({
      event: 'predict',
      frame: frameData,
      id: currentFrameId
    }, [frameData.data.buffer]);
    
    // Control de FPS (~10fps)
    setTimeout(predictWebcam, 100);
    
  } catch (error) {
    console.error("Error en predictWebcam:", error);
    updateStatus("Error durante la predicción. Ver consola para detalles.", true);
    setTimeout(predictWebcam, 200); // Reintentar
  }
}

// 6. Mostrar detecciones en consola
function logDetectionDetails(predictions) {
  console.log(`📊 Detecciones: ${predictions.length}`);
  predictions.forEach((det, i) => {
    const className = CLASS_MAPPING[det.class] || `Clase ${det.class}`;
    console.groupCollapsed(`Detección ${i+1}: ${className} (${(det.score * 100).toFixed(2)}%)`);
    console.log("Clase ID:", det.class);
    console.log("Precisión:", det.score);
    console.log("Coordenadas (normalizadas):", det.bbox);
    
    // Calcular coordenadas en píxeles
    const [ymin, xmin, ymax, xmax] = det.bbox;
    const pxCoords = {
      x: Math.round(xmin * canvas.width),
      y: Math.round(ymin * canvas.height),
      width: Math.round((xmax - xmin) * canvas.width),
      height: Math.round((ymax - ymin) * canvas.height)
    };
    console.log("Coordenadas (píxeles):", pxCoords);
    console.groupEnd();
  });
}

// 7. Dibujar área guía
function drawGuideArea() {
  const guideWidth = canvas.width * 0.6;
  const guideHeight = canvas.height * 0.4;
  const guideX = (canvas.width - guideWidth) / 2;
  const guideY = (canvas.height - guideHeight) / 2;

  ctx.strokeStyle = 'yellow';
  ctx.lineWidth = 3;
  ctx.setLineDash([5, 5]);
  ctx.strokeRect(guideX, guideY, guideWidth, guideHeight);
  ctx.setLineDash([]);
}

// 8. Renderizar detecciones en el canvas
function renderDetections(predictions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGuideArea();

  if (predictions.length === 0) {
    ctx.fillStyle = "#FF0000";
    ctx.font = '20px Arial';
    ctx.fillText("No se detectó ningún documento", 20, 40);
    return;
  }

  let detectedINEFrente = false;
  let bestDetection = null;
  let bestScore = 0;

  // Encontrar la mejor detección
  predictions.forEach(pred => {
    if (pred.score > bestScore) {
      bestScore = pred.score;
      bestDetection = pred;
    }
  });

  predictions.forEach(pred => {
    const [ymin, xmin, ymax, xmax] = pred.bbox;
    const x = xmin * canvas.width;
    const y = ymin * canvas.height;
    const width = (xmax - xmin) * canvas.width;
    const height = (ymax - ymin) * canvas.height;

    if ([x, y, width, height].some(isNaN)) return;

    const labelName = CLASS_MAPPING[pred.class] || `Clase ${pred.class}`;
    const label = `${labelName} ${(pred.score * 100).toFixed(1)}%`;
    const isBest = pred === bestDetection;
    const color = isBest ? '#00FF00' : (labelName === 'INE_Frente' ? '#FF0000' : '#0000FF');

    // Dibujar cuadro de detección
    ctx.strokeStyle = color;
    ctx.lineWidth = isBest ? 4 : 2;
    ctx.strokeRect(x, y, width, height);

    // Dibujar etiqueta
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y > 10 ? y - 20 : y, textWidth + 10, 20);
    ctx.fillStyle = "#FFFFFF";
    ctx.font = '14px Arial';
    ctx.fillText(label, x + 5, y > 10 ? y - 5 : y + 15);

    // Verificar INE_Frente con score > 0.7 dentro del área guía
    if (labelName === 'INE_Frente' && pred.score > 0.7 && isBoxInsideGuide(x, y, width, height)) {
      detectedINEFrente = true;
    }
  });

  if (detectedINEFrente) {
    updateStatus("✅ INE_Frente detectada correctamente dentro del área. Cámara detenida.");
    hideCam();
  }
}

// Verificar si está dentro del área guía
function isBoxInsideGuide(x, y, width, height) {
  const guideWidth = canvas.width * 0.6;
  const guideHeight = canvas.height * 0.4;
  const guideX = (canvas.width - guideWidth) / 2;
  const guideY = (canvas.height - guideHeight) / 2;

  return (
    x >= guideX &&
    y >= guideY &&
    x + width <= guideX + guideWidth &&
    y + height <= guideY + guideHeight
  );
}

// 9. Ocultar cámara
function hideCam() {
  if (!localStream) return;
  
  localStream.getTracks().forEach(track => track.stop());
  localStream = null;
  video.srcObject = null;

  enableWebcamButton.classList.remove("removed");
  hideCameraBtn.classList.add("removed");
  optionsDiv.classList.remove("removed");
  stopModel = true;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Helper: Actualizar estado UI
function updateStatus(message, isError = false) {
  if (statusDisplay) {
    statusDisplay.textContent = message;
    statusDisplay.style.color = isError ? 'red' : 'white';
  }
  console.log(message);
}

// Helper: Manejo de errores
function handleCameraError(error) {
  let errorMessage = "Error de cámara: ";
  if (error.name === 'NotAllowedError') {
    errorMessage += "Permiso denegado. Por favor habilita el acceso a la cámara.";
  } else if (error.name === 'NotFoundError') {
    errorMessage += "No se encontró una cámara conectada.";
  } else {
    errorMessage += error.message || error.toString();
  }
  updateStatus(errorMessage, true);
}

// Event listeners
enableWebcamButton.addEventListener('click', enableCam);
hideCameraBtn.addEventListener('click', hideCam);
cameraOptions.addEventListener('change', () => {
  if (localStream) {
    hideCam();
    enableCam();
  }
});