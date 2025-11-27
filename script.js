// Constants
const IMAGE_SIZE = 128; // Standard input size for many models (ResNet, MobileNet, etc.)
// Adjust this if your models expect a different size

let currentModel = null;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultArea = document.getElementById('resultArea');
const probBar = document.getElementById('probBar');
const resultText = document.getElementById('resultText');
const confidenceText = document.getElementById('confidenceText');
const modelSelect = document.getElementById('modelSelect');

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

modelSelect.addEventListener('change', async (e) => {
    const modelPath = e.target.value;
    if (modelPath) {
        await loadModel(modelPath);
        // If an image is already loaded, re-run inference
        if (imagePreview.src && imagePreview.src !== window.location.href) {
            predict();
        }
    }
});

// Functions
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.style.display = 'block';
        resultArea.style.display = 'none';

        // Wait for image to load before predicting
        imagePreview.onload = () => {
            if (currentModel) {
                predict();
            } else {
                alert('Please select a model first.');
            }
        };
    };
    reader.readAsDataURL(file);
}

async function loadModel(path) {
    loadingOverlay.style.display = 'flex';
    try {
        // Dispose previous model to free memory
        if (currentModel) {
            currentModel.dispose();
        }

        console.log(`Loading model from ${path}...`);

        // Try direct loading first (for properly converted models)
        try {
            currentModel = await tf.loadLayersModel(path);
            console.log('Model loaded successfully (direct load)');
            loadingOverlay.style.display = 'none';
            return;
        } catch (directError) {
            console.warn('Direct load failed, trying manual patching...', directError);
        }

        // Manual patching approach for Keras 3 models
        try {
            const handler = tf.io.browserHTTPRequest(path);
            const modelArtifacts = await handler.load();

            // Deep patch for Keras 3 compatibility
            if (modelArtifacts.modelTopology?.model_config?.config?.layers) {
                const layers = modelArtifacts.modelTopology.model_config.config.layers;
                
                layers.forEach((layer, idx) => {
                    // Fix batch_shape -> batchInputShape
                    if (layer.config?.batch_shape) {
                        layer.config.batchInputShape = layer.config.batch_shape;
                        delete layer.config.batch_shape;
                    }

                    // Fix dtype object -> string
                    if (layer.config?.dtype && typeof layer.config.dtype === 'object') {
                        layer.config.dtype = layer.config.dtype.config?.name || 'float32';
                    }

                    // Fix inbound_nodes (Keras 3 -> TFJS format)
                    if (layer.inbound_nodes && Array.isArray(layer.inbound_nodes)) {
                        const newInboundNodes = [];
                        
                        layer.inbound_nodes.forEach(node => {
                            if (node && node.args && Array.isArray(node.args)) {
                                // Keras 3 format: extract keras_history from each arg
                                const nodeConnections = [];
                                node.args.forEach(arg => {
                                    if (arg?.config?.keras_history) {
                                        nodeConnections.push(arg.config.keras_history);
                                    }
                                });
                                if (nodeConnections.length > 0) {
                                    newInboundNodes.push(nodeConnections);
                                }
                            } else if (Array.isArray(node)) {
                                // Already in TFJS format
                                newInboundNodes.push(node);
                            }
                        });
                        
                        layer.inbound_nodes = newInboundNodes;
                    }

                    // Fix initializers, regularizers, constraints
                    ['kernel_initializer', 'bias_initializer', 'kernel_regularizer', 
                     'bias_regularizer', 'activity_regularizer', 'kernel_constraint', 
                     'bias_constraint'].forEach(key => {
                        if (layer.config?.[key] && typeof layer.config[key] === 'object') {
                            const val = layer.config[key];
                            if (val.module && val.class_name) {
                                layer.config[key] = {
                                    className: val.class_name,
                                    config: val.config || {}
                                };
                            }
                        }
                    });
                });
            }

            // Load patched model
            currentModel = await tf.loadLayersModel(tf.io.fromMemory(modelArtifacts));
            console.log('Model loaded successfully (with patching)');

        } catch (patchError) {
            console.error("Patching failed:", patchError);
            throw new Error('Unable to load model. Please reconvert using the latest tensorflowjs converter.');
        }
    } catch (error) {
        console.error("Final load error:", error);
        alert('Failed to load model. The model format may be incompatible. Please reconvert your Keras model using:\n\ntensorflowjs_converter --input_format=keras your_model.h5 output_dir/');
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

async function predict() {
    if (!currentModel || !imagePreview) return;

    loadingOverlay.style.display = 'flex';

    // Small delay to allow UI to update
    await new Promise(resolve => setTimeout(resolve, 100));

    try {
        tf.tidy(() => {
            // Preprocess image
            let tensor = tf.browser.fromPixels(imagePreview)
                .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE]) // Resize
                .toFloat();

            // Normalize (assuming 0-1 or -1 to 1 depending on model training)
            // Common practice: tensor.div(255.0)
            tensor = tensor.div(255.0);

            // Expand dimensions to match batch size [1, 224, 224, 3]
            const batched = tensor.expandDims(0);

            // Inference
            const prediction = currentModel.predict(batched);
            const score = prediction.dataSync()[0]; // Assuming binary classification output [0-1]

            displayResult(score);
        });
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error during prediction. Check console for details.');
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

function displayResult(score) {
    resultArea.style.display = 'block';

    // Assuming score > 0.5 is AI, < 0.5 is Real
    // Adjust this logic based on your specific model's training
    const isAI = score > 0.7;
    const percentage = (score * 100).toFixed(1);

    probBar.style.width = `${percentage}%`;

    if (isAI) {
        resultText.textContent = 'Likely AI Generated';
        resultText.style.color = '#ef4444'; // Red
        probBar.style.background = 'linear-gradient(90deg, #f87171, #ef4444)';
    } else {
        resultText.textContent = 'Likely Real Image';
        resultText.style.color = '#22c55e'; // Green
        probBar.style.background = 'linear-gradient(90deg, #4ade80, #22c55e)';
    }

    confidenceText.textContent = `Confidence Score: ${percentage}%`;
}
