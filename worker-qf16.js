// Importar TF.js y TFLite
try {
  importScripts(
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js",
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/tf-tflite.min.js"
  );
} catch(error) {
  postMessage({ event: "error", data: "Error loading TF libraries" });
  console.error(error);
}

let model = null;
let config = null;
let classMapping = null;

// Escuchar mensajes del hilo principal
self.addEventListener('message', async (e) => {
  const { event, modelPath, frame, id, config: sentConfig, classMapping: sentClassMapping } = e.data;
  
  if (event === 'init') {
    config = sentConfig;
    classMapping = sentClassMapping;
    await loadModel(modelPath);
  } 
  else if (event === 'predict' && model) {
    await predictFrame(frame, id);
  }
});

// Cargar modelo TFLite
async function loadModel(modelPath) {
  try {
    tflite.setWasmPath("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/");
    
    model = await tflite.loadTFLiteModel(modelPath, {
      experimentalNormalize: false,
      inputType: 'uint8',
      outputType: 'float32',
      numThreads: 4
    });
    
    postMessage({ 
      event: 'model-ready',
      data: 'Model loaded successfully'
    });
  } catch (error) {
    postMessage({ 
      event: 'error', 
      data: `Model loading error: ${error.message}` 
    });
    console.error("Error loading model:", error);
  }
}

// Procesar frame y hacer predicción
async function predictFrame(frameData, id) {
  try {
    // 1. Convertir ImageData a tensor
    const inputTensor = tf.tidy(() => {
      const tensor = tf.tensor3d(
        new Uint8Array(frameData.data), 
        [frameData.height, frameData.width, 4] // RGBA
      );
      return tensor.slice([0, 0, 0], [-1, -1, 3]).expandDims(0); // RGB
    });

    // 2. Hacer predicción
    const outputTensors = await model.predict(inputTensor);
    inputTensor.dispose();

    // 3. Procesar resultados
    const predictions = await processOutput(outputTensors);
    
    // 4. Enviar resultados al main thread
    postMessage({
      event: 'prediction',
      data: predictions,
      id: id,
      debugInfo: {
        outputShapes: Object.fromEntries(
          Object.entries(outputTensors).map(([k, v]) => [k, v.shape])
      }
    });
    
    // 5. Liberar tensores
    Object.values(outputTensors).forEach(t => t.dispose());
    
  } catch (error) {
    console.error("Error during prediction:", error);
    postMessage({
      event: 'error',
      data: `Prediction error: ${error.message}`,
      id: id
    });
  }
}

// Procesar salida del modelo
async function processOutput(outputTensors) {
  try {
    // Identificar tensores relevantes
    const nameScores = Object.keys(outputTensors).find(k => k.includes('scores') || k.endsWith('02'));
    const nameBoxes = Object.keys(outputTensors).find(k => k.includes('boxes') || k.endsWith('0'));
    const nameClasses = Object.keys(outputTensors).find(k => k.includes('classes') || k.endsWith('01'));
    
    if (!nameScores || !nameBoxes || !nameClasses) {
      throw new Error("No se encontraron los tensores esperados en la salida del modelo");
    }

    const [scoresData, boxesData, classesData] = await Promise.all([
      outputTensors[nameScores].array(),
      outputTensors[nameBoxes].array(),
      outputTensors[nameClasses].array()
    ]);

    const predictions = [];
    const numDetections = Math.min(scoresData[0].length, config.maxResults);

    for (let i = 0; i < numDetections; i++) {
      const score = scoresData[0][i];
      if (score >= config.scoreThreshold) {
        const bbox = boxesData[0][i];
        const classId = classesData[0][i];

        predictions.push({
          bbox: [bbox[1], bbox[0], bbox[3], bbox[2]], // [ymin, xmin, ymax, xmax]
          score: score,
          class: classId
        });
      }
    }

    // Ordenar detecciones por score (mayor a menor)
    predictions.sort((a, b) => b.score - a.score);

    // Aplicar supresión no máxima (NMS)
    const nmsPredictions = applyNMS(predictions, config.iouThreshold);

    return nmsPredictions;
  } catch (error) {
    console.error("Error in processOutput:", error);
    return [];
  }
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