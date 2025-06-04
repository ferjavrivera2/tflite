// Variables globales
const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const demosSection = document.getElementById("demos");
const optionsDiv = document.querySelector('.video-options');
const cameraOptions = document.querySelector('.video-options>select');
const enableWebcamButton = document.getElementById("webcamButton");
const hideCameraBtn = document.getElementById("hide-camera");
const canvas = document.getElementById("canvas");
const imageContainer = document.getElementById("image-container");
const ctx = canvas.getContext('2d');
const statusDisplay = document.getElementById("status");
const tempCanvas = document.createElement('canvas');
tempCanvas.id = 'id_canvas';
document.body.appendChild(tempCanvas);

let localStream = null;
let tfliteModel = null;
let stopModel = false;
let modelLoaded = false;

// Configuración del modelo actualizada
const MODEL_CONFIG = {
  inputSize: [320, 320],      // Tamaño de entrada esperado por el modelo
  scoreThreshold: 0.9,       // Umbral más alto para mayor precisión
  maxResults: 10,             // Máximo número de detecciones
  outputFormat: 'auto',       // Formato de salida automático
  iouThreshold: 0.5           // Umbral de supresión no máxima
};

// Mapeo de clases actualizado (verificar con el nuevo modelo)
const CLASS_MAPPING = {
  0: 'INE_Frente',
  1: 'INE_Reverso',
  // Agregar más clases si el modelo detecta otros tipos
};

// Inicialización
document.addEventListener('DOMContentLoaded', async () => {
  await setupCameraSelection();
  await loadModel();
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

// 2. Cargar modelo TFLite actualizado
async function loadModel() {
  try {
    const modelPath = './assets/tflite/ssd_mobilenetv2_lite_320x320_docs_idmex_v3_qf16_metadata.tflite';
    
    // Configuración actualizada para modelo cuantizado
    tfliteModel = await tflite.loadTFLiteModel(modelPath, {
      experimentalNormalize: false,
      inputType: 'uint8',      // Modelo cuantizado usa uint8
      outputType: 'float32',   // Las salidas suelen ser float32
      numThreads: 4            // Usar más hilos para mejor rendimiento
    });
    
    console.log("Nuevo modelo cargado exitosamente:", tfliteModel);
    modelLoaded = true;
    demosSection.classList.remove("invisible");
    enableWebcamButton.disabled = false;
    updateStatus("Modelo v3 cargado. Haz clic en 'Activar Cámara'.");
  } catch (error) {
    console.error("Error al cargar el nuevo modelo:", error);
    updateStatus("Error al cargar el modelo v3. Verifica la consola.", true);
  }
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
    demosSection.classList.remove("invisible");

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

// 4. Procesar cuadro para modelo (optimizado para el nuevo modelo)
function preprocessFrame(videoElement) {
  const [targetWidth, targetHeight] = MODEL_CONFIG.inputSize;
  const tempCtx = tempCanvas.getContext('2d');
  
  // Configurar canvas temporal
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

  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(tempCanvas); // tf.Tensor3D uint8
    return tensor.expandDims(0); // [1, height, width, 3]
  });
}


// Función para aplicar supresión no máxima (NMS)
function applyNMS(predictions, iouThreshold) {
  const selected = [];
  const active = new Array(predictions.length).fill(true);
  
  for (let i = 0; i < predictions.length; i++) {
    if (active[i]) {
      selected.push(predictions[i]);
      
      for (let j = i + 1; j < predictions.length; j++) {
        if (active[j]) {
          const iou = calculateIOU(predictions[i].bbox, predictions[j].bbox);
          if (iou > iouThreshold) {
            active[j] = false;
          }
        }
      }
    }
  }
  
  return selected;
}

// Función para calcular Intersección sobre Unión (IoU)
function calculateIOU(box1, box2) {
  const [ymin1, xmin1, ymax1, xmax1] = box1;
  const [ymin2, xmin2, ymax2, xmax2] = box2;
  
  const xLeft = Math.max(xmin1, xmin2);
  const yTop = Math.max(ymin1, ymin2);
  const xRight = Math.min(xmax1, xmax2);
  const yBottom = Math.min(ymax1, ymax2);
  
  if (xRight < xLeft || yBottom < yTop) {
    return 0.0;
  }
  
  const intersectionArea = (xRight - xLeft) * (yBottom - yTop);
  const box1Area = (xmax1 - xmin1) * (ymax1 - ymin1);
  const box2Area = (xmax2 - xmin2) * (ymax2 - ymin2);
  
  return intersectionArea / (box1Area + box2Area - intersectionArea);
}

async function processOutput(outputTensors) {
  try {
    console.log("Estructura completa de outputTensors:");
    Object.keys(outputTensors).forEach(key => {
      console.log(`Tensor ${key}:`, outputTensors[key]);
    });

    // Asignación basada en los logs
    const numDetections = 10; // Fijo según tus logs
    const classesData = await outputTensors['StatefulPartitionedCall:1'].array();
    const scoresData = await outputTensors['StatefulPartitionedCall:2'].array();
    const boxesData = await outputTensors['StatefulPartitionedCall:3'].array();

    console.log("Datos crudos - Clases:", classesData[0]);
    console.log("Datos crudos - Scores:", scoresData[0]);
    console.log("Datos crudos - Boxes:", boxesData[0]);

    const predictions = [];
    
    for (let i = 0; i < numDetections; i++) {
      const score = scoresData[0][i];
      const classId = classesData[0][i];
      const bbox = boxesData[0][i];

      // Solo para debugging - mostrar todas las detecciones sin filtro
      console.log(`Detección ${i}: Clase=${classId}, Score=${score}, BBox=${bbox}`);

      if (score >= MODEL_CONFIG.scoreThreshold) {
        predictions.push({
          bbox: [bbox[0], bbox[1], bbox[2], bbox[3]], // [ymin, xmin, ymax, xmax]
          score: score,
          class: classId
        });
      }
    }

    console.log('Detecciones antes de NMS:', predictions);
    
    // Ordenar y aplicar NMS
    predictions.sort((a, b) => b.score - a.score);
    const nmsPredictions = applyNMS(predictions, MODEL_CONFIG.iouThreshold);

    console.log('Detecciones después de NMS:', nmsPredictions);
    return nmsPredictions;
  } catch (error) {
    console.error("Error en processOutput:", error);
    return [];
  }
}

// 6. Renderizar detecciones (versión mejorada para debugging)
// 6. Renderizar detecciones (versión mejorada similar al segundo ejemplo)
function renderDetections(predictions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGuideArea();

  if (predictions.length === 0) {
    ctx.fillStyle = "#FF0000";
    ctx.font = '20px Arial';
    ctx.fillText("No se detectó ningún documento", 20, 40);
    return;
  }

  const videoWidth = canvas.width;
  const videoHeight = canvas.height;
  let bestDetection = null;
  let bestScore = 0;

  // Encontrar la mejor detección
  predictions.forEach(pred => {
    if (pred.score > bestScore) {
      bestScore = pred.score;
      bestDetection = pred;
    }
  });

  // Dibujar todas las detecciones
  predictions.forEach(pred => {
    const [ymin, xmin, ymax, xmax] = pred.bbox;
    const x = xmin * videoWidth;
    const y = ymin * videoHeight;
    const width = (xmax - xmin) * videoWidth;
    const height = (ymax - ymin) * videoHeight;

    if ([x, y, width, height].some(isNaN)) return;

    const labelName = CLASS_MAPPING[pred.class] || `Clase ${pred.class}`;
    const label = `${labelName} ${(pred.score * 100).toFixed(1)}%`;
    const isBest = pred === bestDetection;
    const color = isBest ? '#00FF00' : (labelName === 'INE_Frente' ? '#FF0000' : '#0000FF');

    // Dibujar cuadro de detección
    ctx.strokeStyle = color;
    ctx.lineWidth = isBest ? 4 : 2;
    ctx.strokeRect(x, y, width, height);

    // Dibujar fondo para la etiqueta
    ctx.fillStyle = color;
    const textWidth = ctx.measureText(label).width;
    ctx.fillRect(x, y > 10 ? y - 20 : y, textWidth + 10, 20);
    
    // Dibujar texto
    ctx.fillStyle = "#FFFFFF";
    ctx.font = '14px Arial';
    ctx.fillText(label, x + 5, y > 10 ? y - 5 : y + 15);

    // Verificar si está en el área guía
    if (isBest && isBoxInsideGuide(x, y, width, height)) {
      if (labelName === 'INE_Frente' && pred.score > 0.7) {
        updateStatus("INE_Frente detectada correctamente dentro del área. Cámara detenida.");
        hideCam();
      }
    }
  });

  updateStatus(`Detecciones: ${predictions.length} | Mejor: ${bestDetection ? CLASS_MAPPING[bestDetection.class] : 'N/A'} (${bestScore.toFixed(2)})`);
}
// Dibujar área guía
function drawGuideArea() {
  const guideWidthRatio = 0.6;
  const guideHeightRatio = 0.4;

  const guideWidth = canvas.width * guideWidthRatio;
  const guideHeight = canvas.height * guideHeightRatio;
  const guideX = (canvas.width - guideWidth) / 2;
  const guideY = (canvas.height - guideHeight) / 2;

  ctx.strokeStyle = 'yellow';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);
  ctx.strokeRect(guideX, guideY, guideWidth, guideHeight);
  ctx.setLineDash([]);
}

// Verificar si está dentro del área guía
function isBoxInsideGuide(x, y, width, height) {
  const guideWidthRatio = 0.6;
  const guideHeightRatio = 0.4;

  const guideWidth = canvas.width * guideWidthRatio;
  const guideHeight = canvas.height * guideHeightRatio;
  const guideX = (canvas.width - guideWidth) / 2;
  const guideY = (canvas.height - guideHeight) / 2;

  return (
    x >= guideX &&
    y >= guideY &&
    x + width <= guideX + guideWidth &&
    y + height <= guideY + guideHeight
  );
}

// Resto de funciones
function updateStatus(message, isError = false) {
  if (statusDisplay) {
    statusDisplay.textContent = message;
    statusDisplay.style.color = isError ? 'red' : 'white';
  }
  console.log(message);
}

function hideCam() {
  if (!localStream) return;
  localStream.getTracks().forEach(track => track.stop());
  localStream = null;
  video.srcObject = null;

  enableWebcamButton.classList.remove("removed");
  hideCameraBtn.classList.add("removed");
  optionsDiv.classList.remove("removed");
  stopModel = true;
  clearDetections();
}

function clearDetections() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

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

// Bucle de predicción optimizado
async function predictWebcam() {
  if (!tfliteModel || stopModel) return;

  try {
    const startTime = performance.now();
    const inputTensor = preprocessFrame(video);
    
    // Realizar predicción
    const outputTensors = await tfliteModel.predict(inputTensor);
    inputTensor.dispose();
    
    // Procesar resultados
    const predictions = await processOutput(outputTensors);
    
    // Renderizar
    renderDetections(predictions);
    
    // Liberar tensores
    Object.values(outputTensors).forEach(t => t.dispose());
    
    // Calcular FPS
    const endTime = performance.now();
    const fps = 1000 / (endTime - startTime);
    console.log(`FPS: ${fps.toFixed(1)}`);
    
    // Continuar el bucle
    requestAnimationFrame(predictWebcam);

  } catch (error) {
    console.error("Error en predictWebcam:", error);
    updateStatus("Error durante la predicción. Ver consola para detalles.", true);
  }
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