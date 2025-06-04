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
    
    // Para modelos QF16, usar configuración más simple
    model = await tflite.loadTFLiteModel(modelPath);
    
    console.log("📊 Modelo cargado - Info de entrada:", model.inputs);
    console.log("📊 Modelo cargado - Info de salida:", model.outputs);
    
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
    // 1. Probar con normalización para ver si mejoran los scores
    const inputTensor = tf.tidy(() => {
      const tensor = tf.tensor3d(
        new Uint8Array(frameData.data), 
        [frameData.height, frameData.width, 4] // RGBA
      );
      
      const rgbTensor = tensor.slice([0, 0, 0], [-1, -1, 3]);
      
      // PROBAR: Normalizar la entrada [0,1]
      const normalizedTensor = rgbTensor.div(255.0);
      return normalizedTensor.expandDims(0);
      
      // Si no funciona, volver a:
      // return rgbTensor.expandDims(0);
    });

    console.log("📤 Input tensor (QF16 - Normalizado):", inputTensor.shape, inputTensor.dtype);

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
    // Identificar tensores basándose en las shapes que vimos:
    // StatefulPartitionedCall:0 [1] - num_detections
    // StatefulPartitionedCall:1 [1, 10] - scores 
    // StatefulPartitionedCall:2 [1, 10] - classes
    // StatefulPartitionedCall:3 [1, 10, 4] - boxes
    
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

    // Debug: Investigar si los scores necesitan post-procesamiento
    console.log("📊 Primeros 5 scores (raw):", scoresData[0].slice(0, 5));
    console.log("📊 Score máximo encontrado:", Math.max(...scoresData[0]));
    console.log("📊 Score mínimo encontrado:", Math.min(...scoresData[0]));
    
    // Probar si los scores necesitan sigmoid o softmax
    const rawScores = scoresData[0].slice(0, 5);
    const sigmoidScores = rawScores.map(s => 1 / (1 + Math.exp(-s)));
    console.log("📊 Scores con sigmoid:", sigmoidScores);

    // Debug: Mostrar los primeros scores y boxes para diagnóstico
    console.log("📊 Primeras 5 clases:", classesData[0].slice(0, 5));
    console.log("📊 Primeras 2 cajas:", boxesData[0].slice(0, 2));

    const predictions = [];
    const num = Math.min(numDetectionsData[0], 10); 

    console.log("📊 Procesando", num, "detecciones");

    for (let i = 0; i < num; i++) {
      const rawScore = scoresData[0][i];
      
      // Investigar si necesitamos escalar los logits
      // Probar diferentes escalas para ver cuál funciona mejor
      const scaledScore = rawScore * 10; // Multiplicar por 10 para amplificar
      const score = 1 / (1 + Math.exp(-scaledScore));
      
      const classId = classesData[0][i];
      const bbox = boxesData[0][i];
      
      // Mapear clases a las correctas
      let mappedClassId = classId;
      if (classId === 2) mappedClassId = 0; // INE_Frente
      if (classId === 5) mappedClassId = 1; // INE_Reverso
      
      console.log(`Detección ${i}: rawScore=${rawScore}, scaled=${scaledScore.toFixed(2)}, sigmoid=${(score*100).toFixed(1)}%, class=${classId} -> ${mappedClassId}`);
      
      // Threshold más bajo para incluir también INE_Reverso
      if (score >= 0.53 && (classId === 0 || classId === 1 || classId === 2 || classId === 5)) {
        
        // 🚨 ENVIAR DETECCIÓN INMEDIATA AL MAIN THREAD
        const className = mappedClassId === 0 ? 'INE_Frente' : 'INE_Reverso';
        const confidence = (score * 100).toFixed(1);
        
        // Enviar mensaje inmediato para mostrar en consola principal
        postMessage({
          event: 'immediate-detection',
          data: {
            class: mappedClassId,
            className: className,
            confidence: confidence,
            timestamp: new Date().toLocaleTimeString()
          }
        });
        
        predictions.push({
          bbox: [bbox[1], bbox[0], bbox[3], bbox[2]], 
          score: score,
          class: mappedClassId
        });
        console.log(`✅ Detección ${i} agregada con score ${(score*100).toFixed(1)}%, clase ${mappedClassId}`);
      } else {
        console.log(`❌ Detección ${i} rechazada: score=${(score*100).toFixed(1)}%, class=${classId}`);
      }
    }

    return predictions;
  } catch (error) {
    console.error("Error in processOutput:", error);
    return [];
  }
}