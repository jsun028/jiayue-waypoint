// =====================================================
// canvas-manager.js - Canvas operations and Fabric.js handling
// =====================================================

const CanvasManager = {
    /**
     * Setup Fabric.js canvas
     */
    setup() {
        fabricCanvas = new fabric.Canvas('annotationCanvas', {
            selection: true,
            backgroundColor: 'transparent',
            skipTargetFind: false
        });

        this.resize();
        this.setupEventListeners();
    },

    /**
     * Setup canvas event listeners
     */
    setupEventListeners() {
        fabricCanvas.on('mouse:down', this.onMouseDown.bind(this));
        fabricCanvas.on('mouse:move', this.onMouseMove.bind(this));
        fabricCanvas.on('mouse:up', this.onMouseUp.bind(this));
        fabricCanvas.on('object:added', updateJSON);
        fabricCanvas.on('object:removed', updateJSON);
        fabricCanvas.on('object:modified', updateJSON);
    },

    /**
     * Handle mouse down events
     * @param {Object} e - Fabric.js event object
     */
    onMouseDown(e) {
        if (currentMode === 'select') return;
        
        isDrawing = true;
        const pointer = fabricCanvas.getPointer(e.e);
        
        if (currentMode === 'point') {
            this.addPointAnnotation(pointer.x, pointer.y);
        } else {
            this.startDrawingShape(pointer.x, pointer.y);
        }
    },

    /**
     * Handle mouse move events
     * @param {Object} e - Fabric.js event object
     */
    onMouseMove(e) {
        if (!isDrawing || currentMode === 'point' || currentMode === 'select') return;
        
        const pointer = fabricCanvas.getPointer(e.e);
        this.updateDrawingShape(pointer.x, pointer.y);
    },

    /**
     * Handle mouse up events
     */
    onMouseUp() {
        if (isDrawing) {
            isDrawing = false;
            this.finishDrawingShape();
        }
    },

    /**
     * Start drawing a new shape
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     */
    startDrawingShape(x, y) {
        startPoint = { x, y };

        if (currentMode === 'rect') {
            activeShape = new fabric.Rect({
                left: x,
                top: y,
                width: 0,
                height: 0,
                fill: 'rgba(102, 126, 234, 0.3)',
                stroke: '#667eea',
                strokeWidth: 2
            });
        }

        if (activeShape) {
            fabricCanvas.add(activeShape);
        }
    },

    /**
     * Update shape while drawing
     * @param {number} x - Current X coordinate
     * @param {number} y - Current Y coordinate
     */
    updateDrawingShape(x, y) {
        if (!activeShape || !startPoint) return;

        if (currentMode === 'rect') {
            const width = Math.abs(x - startPoint.x);
            const height = Math.abs(y - startPoint.y);
            const left = Math.min(x, startPoint.x);
            const top = Math.min(y, startPoint.y);
            
            activeShape.set({
                left: left,
                top: top,
                width: width,
                height: height
            });
        }
        
        fabricCanvas.renderAll();
    },

    /**
     * Finish drawing and create annotation object
     */
    finishDrawingShape() {
        if (activeShape) {
            const label = prompt('Enter object label:') || 'object';
            const objectCount = this.getObjectCountForClass(label) + 1;
            const displayId = `${label}${objectCount}`;

            // Add metadata to the shape
            activeShape.set({
                id: 'manual_' + displayId + '_' + Date.now(),
                timestamp: video.currentTime,
                label: `${displayId} (manual)`,
                displayId: displayId
            });

            fabricCanvas.renderAll();

            if (currentKeyframeIndex >= 0) {
                const keyframe = annotationData.keyframes[currentKeyframeIndex];
                keyframe.objects.push({
                    id: activeShape.id,
                    label: activeShape.label,
                    timestamp: activeShape.timestamp,
                    type: 'bounding_box',
                    coordinates: {
                        x: Math.round(activeShape.left),
                        y: Math.round(activeShape.top),
                        width: Math.round(activeShape.width * (activeShape.scaleX || 1)),
                        height: Math.round(activeShape.height * (activeShape.scaleY || 1))
                    },
                    constraints: []
                });
            }

            activeShape = null;
            startPoint = null;
            setAnnotationMode('select');
            displayConstraints();
            updateJSON();
        }
    },

    /**
     * Add point annotation
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     */
    addPointAnnotation(x, y) {
        if (currentKeyframeIndex < 0) {
            alert("Select a keyframe first.");
            setAnnotationMode('select');
            return;
        }

        const point = new fabric.Circle({
            left: x - 5,
            top: y - 5,
            radius: 5,
            fill: '#f39c12',
            stroke: '#e67e22',
            strokeWidth: 2,
            id: 'point_' + Date.now(),
            timestamp: video.currentTime,
            label: prompt('Enter point label:') || 'point'
        });
        
        fabricCanvas.add(point);

        // Add to current keyframe.objects
        const keyframe = annotationData.keyframes[currentKeyframeIndex];
        keyframe.objects.push({
            id: point.id,
            label: point.label,
            timestamp: point.timestamp,
            type: 'point',
            coordinates: {
                x: Math.round(point.left + point.radius),
                y: Math.round(point.top + point.radius)
            },
            constraints: []
        });

        isDrawing = false;
        setAnnotationMode('select');
        displayConstraints();
        updateJSON();
    },

    /**
     * Load keyframe annotations onto canvas
     * @param {Object} keyframe - Keyframe data
     */
    loadKeyframeAnnotations(keyframe) {
        isLoadingKeyframe = true;   // disable JSON update temporarily
        fabricCanvas.clear();

        keyframe.objects.forEach(objData => {
            let obj;
            
            if (objData.type === 'bounding_box') {
                obj = new fabric.Rect({
                    left: objData.coordinates.x,
                    top: objData.coordinates.y,
                    width: objData.coordinates.width,
                    height: objData.coordinates.height,
                    fill: 'rgba(102, 126, 234, 0.3)',
                    stroke: '#667eea',
                    strokeWidth: 2
                });
            } else if (objData.type === 'point') {
                obj = new fabric.Circle({
                    left: objData.coordinates.x - 5,
                    top: objData.coordinates.y - 5,
                    radius: 5,
                    fill: '#f39c12',
                    stroke: '#e67e22',
                    strokeWidth: 2
                });
            }
            
            if (obj) {
                obj.set({
                    id: objData.id,
                    label: objData.label,
                    timestamp: objData.timestamp,
                });
                fabricCanvas.add(obj);
            }
        });

        isLoadingKeyframe = false;  // allow JSON update again
        fabricCanvas.renderAll();
    },

    /**
     * Resize canvas to match video dimensions
     */
    resize() {
        const videoElement = document.getElementById('mainVideo');
        const canvas = document.getElementById('annotationCanvas');

        if (!videoElement || !canvas || !fabricCanvas) return;

        const rect = videoElement.getBoundingClientRect();

        canvas.width = rect.width;
        canvas.height = rect.height;

        fabricCanvas.setDimensions({
            width: rect.width,
            height: rect.height
        });
    },

    /**
     * Clear all objects from canvas
     */
    clear() {
        if (fabricCanvas) {
            fabricCanvas.clear();
        }
    },

    /**
     * Add object to canvas
     * @param {fabric.Object} obj - Fabric.js object to add
     */
    addObject(obj) {
        if (fabricCanvas) {
            fabricCanvas.add(obj);
            fabricCanvas.renderAll();
        }
    },

    /**
     * Remove object from canvas
     * @param {fabric.Object} obj - Fabric.js object to remove
     */
    removeObject(obj) {
        if (fabricCanvas) {
            fabricCanvas.remove(obj);
            fabricCanvas.renderAll();
        }
    },

    /**
     * Get all objects on canvas
     * @returns {Array} Array of fabric objects
     */
    getObjects() {
        return fabricCanvas ? fabricCanvas.getObjects() : [];
    },

    /**
     * Get currently selected object
     * @returns {fabric.Object|null} Selected object or null
     */
    getActiveObject() {
        return fabricCanvas ? fabricCanvas.getActiveObject() : null;
    },

    getObjectCountForClass(className) {
        if (currentKeyframeIndex < 0) return 0;
        
        const keyframe = annotationData.keyframes[currentKeyframeIndex];
        console.log("getObjectCountForClass", keyframe.objects);
        return keyframe.objects.filter(obj => 
            obj.label.startsWith(className)
        ).length;
    },

    /**
     * Set canvas interaction mode
     * @param {string} mode - Mode ('select', 'rect', 'point')
     */
    setMode(mode) {
        if (!fabricCanvas) return;

        const upperCanvasEl = document.querySelector('.upper-canvas');
        if (!upperCanvasEl) return;

        if (mode === 'select') {
            fabricCanvas.isDrawingMode = false;
            fabricCanvas.selection = true;
            upperCanvasEl.style.setProperty("pointer-events", "none", "important");
            fabricCanvas.defaultCursor = 'default';
        } else {
            if (currentKeyframeIndex >= 0) {
                fabricCanvas.isDrawingMode = false;
                fabricCanvas.selection = false;
                upperCanvasEl.style.setProperty("pointer-events", "auto", "important");
                fabricCanvas.defaultCursor = 'crosshair';
            } else {
                alert("Select a keyframe first.");
                setAnnotationMode('select');
            }
        }
    },

};


// Make CanvasManager available globally
window.CanvasManager = CanvasManager;