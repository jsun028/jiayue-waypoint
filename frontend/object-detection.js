// =====================================================
// object-detection-manager.js - Object detection functionality
// =====================================================

const ObjectDetectionManager = {
    // Detection state
    model: null,
    isModelLoading: false,
    isDetecting: false,

    /**
     * Load TensorFlow.js object detection model
     */
    async loadModel() {
        try {
            this.isModelLoading = true;
            this.updateUI();
            
            // Load COCO-SSD model (pre-trained on COCO dataset)
            this.model = await cocoSsd.load();
            
            this.isModelLoading = false;
            this.updateUI();
            console.log('Object detection model loaded successfully');
        } catch (error) {
            console.error('Failed to load object detection model:', error);
            this.isModelLoading = false;
            this.updateUI();
            alert('Failed to load object detection model. Make sure TensorFlow.js libraries are loaded.');
        }
    },

    /**
     * Update detection button UI state
     */
    updateUI() {
        const detectBtn = document.getElementById('detectObjectsBtn');
        if (!detectBtn) return;
        
        detectBtn.disabled = false;
        if (this.isModelLoading) {
            detectBtn.textContent = 'Loading Model...';
            detectBtn.disabled = true;
        } else if (this.isDetecting) {
            detectBtn.textContent = 'Detecting...';
            detectBtn.disabled = true;
        } else if (this.model) {
            detectBtn.textContent = '🔍 Detect Objects';
            detectBtn.disabled = currentKeyframeIndex < 0;
        } else {
            detectBtn.textContent = 'Model Failed';
            detectBtn.disabled = true;
        }
    },

    /**
     * Run object detection on current keyframe
     */
    async detectOnCurrentKeyframe() {
        if (!this.model || currentKeyframeIndex < 0 || this.isDetecting) {
            return;
        }
        
        try {
            this.isDetecting = true;
            this.updateUI();
            
            // Create a canvas to capture the current video frame
            const detectionCanvas = document.createElement('canvas');
            const ctx = detectionCanvas.getContext('2d');

            // Set canvas size to match video's natural dimensions  
            const videoDimensions = VideoController.getVideoDimensions();
            detectionCanvas.width = videoDimensions.width;
            detectionCanvas.height = videoDimensions.height;
                    
            // Draw current video frame to canvas
            ctx.drawImage(video, 0, 0, videoDimensions.width, videoDimensions.height);
            
            // Run object detection
            const predictions = await this.model.detect(detectionCanvas);
            
            // Convert predictions to annotations and add to current keyframe
            CanvasManager.addDetectedObjects(predictions);
            
            this.isDetecting = false;
            this.updateUI();
            
            console.log(`Detected ${predictions.length} objects:`, predictions);
            
        } catch (error) {
            console.error('Object detection failed:', error);
            this.isDetecting = false;
            this.updateUI();
            alert('Object detection failed. Please try again.');
        }
    },

    /**
     * Clear all detected objects from current keyframe
     */
    clearDetectedObjects() {
        if (currentKeyframeIndex < 0) return;
        
        CanvasManager.clearDetectedObjects();
        this.updateUI();
    },

    /**
     * Run detection on multiple keyframes
     * @param {Array} keyframeIndices - Array of keyframe indices to process
     */
    async detectBatch(keyframeIndices = null) {
        if (!this.model) {
            alert('Object detection model not loaded.');
            return;
        }

        // Use all keyframes if none specified
        if (!keyframeIndices) {
            keyframeIndices = Array.from({length: annotationData.keyframes.length}, (_, i) => i);
        }

        if (keyframeIndices.length === 0) {
            alert('No keyframes to process.');
            return;
        }

        const originalKeyframeIndex = currentKeyframeIndex;
        let processedCount = 0;
        let totalDetected = 0;

        try {
            this.isDetecting = true;
            this.updateUI();

            for (const index of keyframeIndices) {
                // Seek to keyframe
                VideoController.seekToKeyframe(index);
                
                // Wait a bit for video to seek
                await new Promise(resolve => setTimeout(resolve, 100));

                // Detect objects
                const detectionCanvas = document.createElement('canvas');
                const ctx = detectionCanvas.getContext('2d');
                const videoDimensions = VideoController.getVideoDimensions();
                detectionCanvas.width = videoDimensions.width;
                detectionCanvas.height = videoDimensions.height;
                ctx.drawImage(video, 0, 0, videoDimensions.width, videoDimensions.height);

                const predictions = await this.model.detect(detectionCanvas);
                const filteredPredictions = this.filterPredictions(predictions);

                if (filteredPredictions.length > 0) {
                    CanvasManager.addDetectedObjects(filteredPredictions);
                    totalDetected += filteredPredictions.length;
                }

                processedCount++;
                console.log(`Processed keyframe ${index + 1}/${annotationData.keyframes.length}`);
            }

            // Return to original keyframe
            if (originalKeyframeIndex >= 0) {
                VideoController.seekToKeyframe(originalKeyframeIndex);
            }

            this.isDetecting = false;
            this.updateUI();

            alert(`Batch detection complete!\nProcessed: ${processedCount} keyframes\nTotal objects detected: ${totalDetected}`);

        } catch (error) {
            console.error('Batch detection failed:', error);
            this.isDetecting = false;
            this.updateUI();
            alert('Batch detection failed. Please try again.');
        }
    },

    /**
     * Filter predictions based on confidence and other criteria
     * @param {Array} predictions - Raw predictions from model
     * @param {number} minConfidence - Minimum confidence threshold
     * @returns {Array} Filtered predictions
     */
    filterPredictions(predictions, minConfidence = 0.3) {
        return predictions
            .filter(pred => pred.score >= minConfidence)
            .sort((a, b) => b.score - a.score); // Sort by confidence descending
    },

    /**
     * Get model information
     * @returns {Object} Model information
     */
    getModelInfo() {
        return {
            loaded: !!this.model,
            loading: this.isModelLoading,
            detecting: this.isDetecting,
            type: 'COCO-SSD'
        };
    },

    /**
     * Set minimum confidence threshold for detections
     * @param {number} threshold - Confidence threshold (0-1)
     */
    setConfidenceThreshold(threshold) {
        if (threshold >= 0 && threshold <= 1) {
            this.confidenceThreshold = threshold;
            console.log(`Detection confidence threshold set to: ${threshold}`);
        } else {
            console.warn('Confidence threshold must be between 0 and 1');
        }
    },

    /**
     * Reset detection state
     */
    reset() {
        this.isDetecting = false;
        this.updateUI();
    },

    /**
     * Check if detection is available
     * @returns {boolean} Whether detection can be performed
     */
    isAvailable() {
        return !!(this.model && !this.isModelLoading && !this.isDetecting && currentKeyframeIndex >= 0);
    },

    /**
     * Get current detection status
     * @returns {string} Current status
     */
    getStatus() {
        if (this.isModelLoading) return 'loading_model';
        if (this.isDetecting) return 'detecting';
        if (!this.model) return 'model_failed';
        if (currentKeyframeIndex < 0) return 'no_keyframe_selected';
        return 'ready';
    }
};

// Make ObjectDetectionManager available globally
window.ObjectDetectionManager = ObjectDetectionManager;