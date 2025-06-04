// Variables globales --logs aparece cuando se detecta una id y te dice que tipo y que porcentaje tiene
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

let localStream = null;
let tfliteModel = null;
let stopModel = false;
let modelLoaded = false;

// Configuración del modelo
const MODEL_CONFIG = {
  inputSize: [320, 320],
  scoreThreshold: 0.5,
  maxResults: 10,
  outputFormat: 'auto'
};

// Mapeo de clases del modelo
const CLASS_MAPPING = {
  0: 'INE_Frente',
  1: 'INE_Reverso',
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

// 2. Cargar modelo TFLite
async function loadModel() {
  try {
    const modelPath = './assets/tflite/ssd_mobilenetv2_lite_320x320_docs_idmex_v3_qi8_metadata.tflite';
    tfliteModel = await tflite.loadTFLiteModel(modelPath, {
      experimentalNormalize: false,
      inputType: 'uint8',
      outputType: 'float32'
    });
    console.log("Modelo cargado exitosamente:", tfliteModel);
    modelLoaded = true;
    demosSection.classList.remove("invisible");
    enableWebcamButton.disabled = false;
    updateStatus("Modelo cargado. Haz clic en 'Activar Cámara'.");
  } catch (error) {
    console.error("Error al cargar el modelo:", error);
    updateStatus("Error al cargar el modelo. Verifica la consola.", true);
  }
}

// 3. Activar la cámara y empezar predicción
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

// 4. Procesar cuadro para modelo (preprocesamiento)
function preprocessFrame(videoElement) {
  const canvasTemp = document.createElement('canvas');
  const ctxTemp = canvasTemp.getContext('2d');
  canvasTemp.width = MODEL_CONFIG.inputSize[0];
  canvasTemp.height = MODEL_CONFIG.inputSize[1];
  ctxTemp.drawImage(videoElement, 0, 0, canvasTemp.width, canvasTemp.height);

  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(canvasTemp); // Esto ya regresa tf.Tensor3D uint8
    const inputTensor = tensor.expandDims(0); // [1, height, width, 3]
    return inputTensor;
  });
}

// 5. Procesar salida del modelo y buscar INE_Frente con score alto
async function processOutput(outputTensors) {
  try {
    Object.entries(outputTensors).forEach(([name, tensor], i) => {
      console.log(`Tensor ${i}:`, name, tensor.shape, tensor.dtype);
    });

    // Asignamos los nombres tal cual como corresponden:
    const nameScores = Object.keys(outputTensors).find(k => k.endsWith('02'));
    const nameBoxes = Object.keys(outputTensors).find(k => k.endsWith('0'));
    const nameNumDetections = Object.keys(outputTensors).find(k => k.endsWith('03'));
    const nameClasses = Object.keys(outputTensors).find(k => k.endsWith('01'));

    const scoresTensor = outputTensors[nameScores];
    const boxesTensor = outputTensors[nameBoxes];
    const numDetectionsTensor = outputTensors[nameNumDetections];
    const classesTensor = outputTensors[nameClasses];

    const [scoresData, boxesData, numDetectionsData, classesData] = await Promise.all([
      scoresTensor.array(),
      boxesTensor.array(),
      numDetectionsTensor.array(),
      classesTensor.array()
    ]);

    const predictions = [];
    const num = Math.min(numDetectionsData[0], MODEL_CONFIG.maxResults || 10);

    for (let i = 0; i < num; i++) {
      const score = scoresData[0][i];
      if (score >= MODEL_CONFIG.scoreThreshold) {
        const bbox = boxesData[0][i];
        const classId = classesData[0][i];

        predictions.push({
          bbox: [bbox[1], bbox[0], bbox[3], bbox[2]], // [xMin, yMin, xMax, yMax]
          score: score,
          class: classId
        });
      }
    }

    console.log('Boxes:', boxesData);
    console.log('Scores:', scoresData);
    console.log('Classes:', classesData);

    return predictions;
  } catch (error) {
    console.error("Error en processOutput:", error);
    return [];
  }
}

// NUEVA: Dibujar área guía amarilla en el canvas
function drawGuideArea() {
  const guideWidthRatio = 0.6;
  const guideHeightRatio = 0.4;

  const guideWidth = canvas.width * guideWidthRatio;
  const guideHeight = canvas.height * guideHeightRatio;
  const guideX = (canvas.width - guideWidth) / 2;
  const guideY = (canvas.height - guideHeight) / 2;

  ctx.strokeStyle = 'yellow';
  ctx.lineWidth = 3;
  ctx.strokeRect(guideX, guideY, guideWidth, guideHeight);
}

// NUEVA: Verificar si un cuadro está dentro del área guía
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

// MODIFICADA: renderDetections con área guía y verificación espacial
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
  let detectedINEFrente = false;

  predictions.forEach(pred => {
    const [ymin, xmin, ymax, xmax] = pred.bbox;
    const x = xmin * videoWidth;
    const y = ymin * videoHeight;
    const width = (xmax - xmin) * videoWidth;
    const height = (ymax - ymin) * videoHeight;

    if ([x, y, width, height].some(isNaN)) return;

    const labelName = CLASS_MAPPING[pred.class] || 'Desconocido';
    const label = `${labelName} ${(pred.score * 100).toFixed(1)}%`;
    const color = labelName === 'INE_Frente' ? '#FF0000' : '#0000FF';

    // Dibujar cuadro de detección
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, width, height);

    ctx.fillStyle = color;
    ctx.fillRect(x, y > 10 ? y - 20 : y, ctx.measureText(label).width + 10, 20);
    ctx.fillStyle = "#FFFFFF";
    ctx.font = '14px Arial';
    ctx.fillText(label, x + 5, y > 10 ? y - 5 : y + 15);

    // Verificar INE_Frente con score > 0.7 y dentro del área guía
    if (labelName === 'INE_Frente' && pred.score > 0.7) {
      if (isBoxInsideGuide(x, y, width, height)) {
        detectedINEFrente = true;
      } else {
        updateStatus("INE_Frente detectada pero fuera del área guía.");
      }
    }
  });

  if (detectedINEFrente) {
    updateStatus("INE_Frente detectada correctamente dentro del área. Cámara detenida.");
    hideCam();
  } else {
    updateStatus(`Detecciones: ${predictions.length}`);
    

  }
}

// 7. Actualizar estado UI
function updateStatus(message, isError = false) {
  if (statusDisplay) {
    statusDisplay.textContent = message;
    statusDisplay.style.color = isError ? 'red' : 'white';
  }
  console.log(message);
}

// 8. Ocultar cámara y limpiar recursos
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

// 9. Limpiar canvas de detecciones
function clearDetections() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// 10. Manejo de errores de cámara
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

// 11. Bucle de predicción con la cámara activa
async function predictWebcam() {
  if (!tfliteModel || stopModel) return;

  try {
    const inputTensor = preprocessFrame(video);

    // ⚠️ Hacer la predicción
    const outputTensors = await tfliteModel.predict(inputTensor);

    // 🧹 Liberar inputTensor lo antes posible
    inputTensor.dispose();

    // 🐞 Mostrar salida de diagnóstico
    console.log("📤 Output infos:", tfliteModel.outputs);
    console.log("📦 Output tensors:", outputTensors);

    // Procesar resultados
    const predictions = await processOutput(outputTensors);

    // Mostrar info detallada de las detecciones en consola
    console.log("Detecciones:", predictions.length);
    predictions.forEach((det, i) => {
      console.log(`Detección ${i+1}:`);
      console.log(` Clase ID: ${det.class}`);
      console.log(` Score: ${(det.score * 100).toFixed(2)}%`);
      console.log(` Caja (xMin, yMin, xMax, yMax):`, det.bbox);
    });

    const validPredictions = predictions.filter(pred => pred.score >= MODEL_CONFIG.scoreThreshold);

    // Renderizar resultados en canvas
    renderDetections(validPredictions);

    // Continuar el bucle de predicción
    requestAnimationFrame(predictWebcam);

    // ⚠️ Importante: liberar todos los tensores de salida
    Object.values(outputTensors).forEach(t => t.dispose());

  } catch (error) {
    console.error("Error en predictWebcam:", error);
    updateStatus("Error durante la predicción. Ver consola para detalles.", true);
  }
}



// Eventos de UI
enableWebcamButton.addEventListener('click', enableCam);
hideCameraBtn.addEventListener('click', hideCam);
cameraOptions.addEventListener('change', () => {
  if (localStream) {
    hideCam();
    enableCam();
  }
});
