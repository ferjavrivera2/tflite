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
      experimentalNormalize: false,  // No normalizar automáticamente
      inputType: 'float32',          // Cambiar a float32 para entrada normalizada
      outputType: 'float32'          // Mantener float32 para salida
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
    // 1. Convertir ImageData a tensor - NORMALIZAR para el modelo
    const inputTensor = tf.tidy(() => {
      const tensor = tf.tensor3d(
        new Uint8Array(frameData.data), 
        [frameData.height, frameData.width, 4] // RGBA
      );
      
      // Extraer solo RGB y convertir a float32
      const rgbTensor = tensor.slice([0, 0, 0], [-1, -1, 3]); // Solo RGB
      
      // CRÍTICO: Normalizar de [0,255] a [0,1] para el modelo
      const normalizedTensor = rgbTensor.cast('float32').div(255.0);
      
      return normalizedTensor.expandDims(0); // Batch dimension
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

// Procesar salida del modelo - CORREGIDO
async function processOutput(outputTensors) {
  try {
    // Mapeo directo basado en los nombres reales de salida de tu modelo
    // Según tu info del modelo:
    // StatefulPartitionedCall:0 -> número de detecciones
    // StatefulPartitionedCall:1 -> scores 
    // StatefulPartitionedCall:2 -> clases
    // StatefulPartitionedCall:3 -> cajas delimitadoras
    
    const numDetectionsTensor = outputTensors['StatefulPartitionedCall:0'];
    const scoresTensor = outputTensors['StatefulPartitionedCall:1'];
    const classesTensor = outputTensors['StatefulPartitionedCall:2'];
    const boxesTensor = outputTensors['StatefulPartitionedCall:3'];

    // Verificar que todos los tensores existen
    if (!numDetectionsTensor || !scoresTensor || !classesTensor || !boxesTensor) {
      console.error("Tensores faltantes:", {
        numDetections: !!numDetectionsTensor,
        scores: !!scoresTensor,
        classes: !!classesTensor,
        boxes: !!boxesTensor
      });
      return [];
    }

    const [numDetectionsData, scoresData, classesData, boxesData] = await Promise.all([
      numDetectionsTensor.array(),
      scoresTensor.array(),
      classesTensor.array(),
      boxesTensor.array()
    ]);

    // Debug: Mostrar datos crudos COMPLETOS
    console.log("NumDetections (QF16):", numDetectionsData);
    console.log("Scores raw (QF16):", scoresData[0]); // Solo el primer batch
    console.log("Classes raw (QF16):", classesData[0]); // Solo el primer batch
    console.log("Boxes raw (QF16) - primeras 3:", boxesData[0].slice(0, 3)); // Solo las primeras 3 cajas

    const predictions = [];
    
    // El número de detecciones puede ser un escalar o array
    const numDetections = Array.isArray(numDetectionsData) ? numDetectionsData[0] : numDetectionsData;
    const maxDetections = Math.min(Math.floor(numDetections), 10);

    console.log("Número de detecciones a procesar:", maxDetections);

    for (let i = 0; i < maxDetections; i++) {
      const score = scoresData[0][i];
      const classId = Math.floor(classesData[0][i]);
      const bbox = boxesData[0][i];
      
      // Debug: Mostrar TODOS los scores y clases para diagnosticar
      console.log(`Detección ${i}: Score=${score.toFixed(4)}, Class=${classId}, BBox=${bbox ? bbox.map(x => x.toFixed(4)) : 'undefined'}`);
      
      // Restaurar threshold normal después de normalización
      if (score >= 0.5) { // Volver al threshold original
        // Verificar que bbox tiene 4 elementos
        if (bbox && bbox.length >= 4) {
          predictions.push({
            bbox: [bbox[0], bbox[1], bbox[2], bbox[3]], // ymin, xmin, ymax, xmax
            score: score,
            class: classId
          });
          console.log(`✅ Detección válida añadida: ${CLASS_MAPPING[classId] || 'Desconocido'} con score ${score.toFixed(4)}`);
        }
      }
    }

    console.log("Predicciones válidas encontradas:", predictions.length);
    return predictions;
    
  } catch (error) {
    console.error("Error in processOutput:", error);
    console.error("OutputTensors keys:", Object.keys(outputTensors));
    return [];
  }
}