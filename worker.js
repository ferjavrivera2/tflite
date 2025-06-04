// Importar TF.js y TFLite
try {
  importScripts(
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js",
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/tf-tflite.min.js"
  );
} catch(error) {
  postMessage({ event: "error", data: "Error loading TF libraries" });
  console.warn(error);
}

let model = null;

// Mapeo de clases (debe coincidir con el main thread)
const CLASS_MAPPING = {
  0: 'INE_Frente',
  1: 'INE_Reverso'
};

// Escuchar mensajes del hilo principal
self.addEventListener('message', async (e) => {
  const { event, modelPath, frame, id } = e.data;
  
  if (event === 'init') {
    await loadModel(modelPath);
  } 
  else if (event === 'predict' && model) {
    await predictFrame(frame, id);
  }
});

// Cargar modelo TFLite QF16
async function loadModel(modelPath) {
  try {
    tflite.setWasmPath("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/");
    
    // Configuración específica para modelo QF16
    model = await tflite.loadTFLiteModel(modelPath, {
      experimentalNormalize: false,
      inputType: 'uint8',      // Mantener uint8 para entrada
      outputType: 'float32'    // Mantener float32 para salida
    });
    
    postMessage({ 
      event: 'model-ready',
      data: 'QF16 Model loaded successfully'
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
    // 1. Convertir ImageData a tensor - mantener uint8 para QF16
    const inputTensor = tf.tidy(() => {
      const tensor = tf.tensor3d(
        new Uint8Array(frameData.data), 
        [frameData.height, frameData.width, 4] // RGBA
      );
      
      // Para este modelo QF16: mantener uint8, solo extraer RGB
      const rgbTensor = tensor.slice([0, 0, 0], [-1, -1, 3]); // Solo RGB
      return rgbTensor.expandDims(0); // Batch dimension, mantener uint8
    });

    console.log("📤 Input tensor (QF16):", inputTensor.shape, inputTensor.dtype);

    // 2. Hacer predicción
    const outputTensors = await model.predict(inputTensor);
    
    // 3. Debug: Mostrar info de tensores
    console.log("📤 Output infos:", model.outputs);
    console.log("📦 Output tensors:", outputTensors);
    
    Object.entries(outputTensors).forEach(([name, tensor], i) => {
      console.log(`Tensor ${i}:`, name, tensor.shape, tensor.dtype);
    });

    inputTensor.dispose();

    // 4. Procesar resultados
    const predictions = await processOutput(outputTensors);
    
    // 5. Debug: Mostrar detecciones crudas
    console.log("Detecciones QF16:", predictions.length);
    predictions.forEach((det, i) => {
      console.log(`Detección ${i+1}:`);
      console.log(` Clase ID: ${det.class}`);
      console.log(` Score: ${(det.score * 100).toFixed(2)}%`);
      console.log(` Caja:`, det.bbox);
    });

    // Liberar tensores
    Object.values(outputTensors).forEach(t => t.dispose());
    
    // 6. Enviar resultados al main thread
    postMessage({
      event: 'prediction',
      data: predictions,
      id: id,
      debugInfo: {
        outputShapes: Object.fromEntries(
          Object.entries(outputTensors).map(([k, v]) => [k, v.shape])
        )
      }
    });
    
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
    // Identificar tensores de salida (adaptado a tu modelo)
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

    // Debug: Mostrar datos crudos
    console.log("Boxes raw (QF16):", boxesData);
    console.log("Scores raw (QF16):", scoresData);
    console.log("Classes raw (QF16):", classesData);
    console.log("NumDetections (QF16):", numDetectionsData);

    const predictions = [];
    const num = Math.min(numDetectionsData[0], 10); 

    for (let i = 0; i < num; i++) {
      const score = scoresData[0][i];
      if (score >= 0.5) { 
        const bbox = boxesData[0][i];
        const classId = classesData[0][i];

        predictions.push({
          bbox: [bbox[1], bbox[0], bbox[3], bbox[2]], 
          score: score,
          class: classId
        });
      }
    }

    return predictions;
  } catch (error) {
    console.error("Error in processOutput:", error);
    return [];
  }
}