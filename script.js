// Global state
let annotationData = {
    video_id: null,
    keyframes: [],
    inter_frame_constraints: []
};

let currentMode = 'select';
let canvas, video, fabricCanvas;
let currentKeyframeIndex = -1;
let isDrawing = false;

// Initialize the application
window.onload = function() {
    video = document.getElementById('mainVideo');
    canvas = document.getElementById('annotationCanvas');
    
    setupVideo();
    setupCanvas();
    setupEventListeners();
    // Initialize default mode
    setAnnotationMode('select');
    
    // Load demo video
    annotationData.video_id = 'demo_big_buck_bunny';
};

function setupVideo() {
    video.addEventListener('loadedmetadata', function() {
        resizeCanvas();
        updateTimeDisplay();
        updateTimelineMarkers(); 
        setupTimelineClick(); 
    });

    video.addEventListener('timeupdate', function() {
        updateTimeDisplay();
        updateProgressBar();
        highlightCurrentKeyframe();
    });

    video.addEventListener('click', function(e) {
        if (currentMode !== 'select') {
            e.preventDefault();
        }
    });
}

function setupCanvas() {
    
    fabricCanvas = new fabric.Canvas('annotationCanvas', {
        selection: true,
        backgroundColor: 'transparent', // This is key!
        skipTargetFind: false
    });

    resizeCanvas();

    fabricCanvas.on('mouse:down', function(e) {
        if (currentMode === 'select') return;
        
        isDrawing = true;
        const pointer = fabricCanvas.getPointer(e.e);
        
        if (currentMode === 'point') {
            addPointAnnotation(pointer.x, pointer.y);
        } else {
            startDrawingShape(pointer.x, pointer.y);
        }
    });

    fabricCanvas.on('mouse:move', function(e) {
        if (!isDrawing || currentMode === 'point' || currentMode === 'select') return;
        
        const pointer = fabricCanvas.getPointer(e.e);
        updateDrawingShape(pointer.x, pointer.y);
    });

    fabricCanvas.on('mouse:up', function() {
        if (isDrawing) {
            isDrawing = false;
            finishDrawingShape();
        }
    });

    fabricCanvas.on('object:added', updateJSON);
    fabricCanvas.on('object:removed', updateJSON);
    fabricCanvas.on('object:modified', updateJSON);
}

// Ensure canvas sizing matches video exactly
function resizeCanvas() {
    const video = document.getElementById('mainVideo');
    const canvas = document.getElementById('annotationCanvas');
    
    canvas.width = video.offsetWidth;
    canvas.height = video.offsetHeight;
    
    fabricCanvas.setDimensions({
        width: video.offsetWidth,
        height: video.offsetHeight
    });
}


function setupEventListeners() {
    document.getElementById('videoFile').addEventListener('change', handleFileUpload);
    window.addEventListener('resize', resizeCanvas);
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        video.src = url;
        annotationData.video_id = file.name.replace(/\.[^/.]+$/, "");
        clearAllAnnotations();
    }
}

function setAnnotationMode(mode) {
    currentMode = mode;
    
    // Update button states
    document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(mode + 'Btn').classList.add('active');
    
    // Update canvas interaction mode
    fabricCanvas.selection = mode === 'select';
    fabricCanvas.skipTargetFind = mode !== 'select';
    
    if (mode === 'select') {
        fabricCanvas.defaultCursor = 'default';
    } else {
        fabricCanvas.defaultCursor = 'crosshair';
    }
}

let startPoint = null;
let activeShape = null;

function startDrawingShape(x, y) {
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
}

function updateDrawingShape(x, y) {
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
}

function finishDrawingShape() {
    if (activeShape) {
        // Add metadata to the shape
        activeShape.set({
            id: 'obj_' + Date.now(),
            timestamp: video.currentTime,
            label: prompt('Enter object label:') || 'unlabeled'
        });
        
        fabricCanvas.renderAll();
        activeShape = null;
        startPoint = null;
        setAnnotationMode('select');
    }
}

function addPointAnnotation(x, y) {
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
    isDrawing = false;
    setAnnotationMode('select');
}

function addKeyframe() {
    const timestamp = video.currentTime;
    
    // Check if keyframe already exists at this time (within 0.5 seconds)
    const existingKeyframe = annotationData.keyframes.find(kf => 
        Math.abs(kf.timestamp - timestamp) < 0.5
    );
    
    if (existingKeyframe) {
        alert('Keyframe already exists at this time!');
        return;
    }
    
    const keyframe = {
        id: 'keyframe_' + Date.now(),
        timestamp: timestamp,
        duration: 0,
        objects: [],
        constraints: {
            frame_level: [],
            inter_frame: []
        }
    };
    
    annotationData.keyframes.push(keyframe);
    annotationData.keyframes.sort((a, b) => a.timestamp - b.timestamp);
    
    updateKeyframeList();
    updateTimelineMarkers(); 
    updateKeyframeDropdowns();
    updateJSON();
}

function updateKeyframeDropdowns() {
    const fromSelect = document.getElementById('fromKeyframe');
    const toSelect = document.getElementById('toKeyframe');
    
    // Store current selections
    const currentFrom = fromSelect.value;
    const currentTo = toSelect.value;
    
    // Clear and repopulate
    fromSelect.innerHTML = '<option value="">Select keyframe...</option>';
    toSelect.innerHTML = '<option value="">Select keyframe...</option>';
    
    annotationData.keyframes.forEach((keyframe, index) => {
        const option = `<option value="${index}">Keyframe ${index + 1} (${formatTime(keyframe.timestamp)})</option>`;
        fromSelect.innerHTML += option;
        toSelect.innerHTML += option;
    });
    
    // Restore selections if still valid
    if (currentFrom !== '' && parseInt(currentFrom) < annotationData.keyframes.length) {
        fromSelect.value = currentFrom;
    }
    if (currentTo !== '' && parseInt(currentTo) < annotationData.keyframes.length) {
        toSelect.value = currentTo;
    }
    
    updateInterFrameUI();
}

function updateInterFrameUI() {
    const fromIndex = document.getElementById('fromKeyframe').value;
    const toIndex = document.getElementById('toKeyframe').value;
    const selectedDisplay = document.getElementById('selectedKeyframes');
    
    if (fromIndex !== '' && toIndex !== '') {
        const fromTime = formatTime(annotationData.keyframes[parseInt(fromIndex)].timestamp);
        const toTime = formatTime(annotationData.keyframes[parseInt(toIndex)].timestamp);
        selectedDisplay.textContent = `Keyframe ${parseInt(fromIndex) + 1} (${fromTime}) → Keyframe ${parseInt(toIndex) + 1} (${toTime})`;
        selectedDisplay.style.color = '#27ae60';
    } else if (fromIndex !== '' || toIndex !== '') {
        selectedDisplay.textContent = 'Select both keyframes';
        selectedDisplay.style.color = '#f39c12';
    } else {
        selectedDisplay.textContent = 'None';
        selectedDisplay.style.color = '#666';
    }
}

function updateTimelineMarkers() {
    const markersContainer = document.getElementById('keyframeMarkers');
    const timelineBar = document.getElementById('timelineBar');
    
    if (!markersContainer || !video.duration) return;
    
    // Clear existing markers
    markersContainer.innerHTML = '';
    
    // Add markers for each keyframe
    annotationData.keyframes.forEach((keyframe, index) => {
        const marker = document.createElement('div');
        marker.className = 'keyframe-marker';
        marker.title = `Keyframe ${index + 1} - ${formatTime(keyframe.timestamp)}`;
        
        // Position marker based on timestamp
        const position = (keyframe.timestamp / video.duration) * 100;
        marker.style.left = position + '%';
        
        // Add click handler to seek to keyframe
        marker.onclick = (e) => {
            e.stopPropagation();
            seekToKeyframe(index);
        };
        
        markersContainer.appendChild(marker);
    });
}

function seekToKeyframe(index) {
    const keyframe = annotationData.keyframes[index];
    video.currentTime = keyframe.timestamp;
    currentKeyframeIndex = index;
    highlightCurrentKeyframe();
    highlightTimelineMarker(index);
    loadKeyframeAnnotations(keyframe);
    displayConstraints();
}

function highlightTimelineMarker(activeIndex) {
    document.querySelectorAll('.keyframe-marker').forEach((marker, index) => {
        marker.classList.toggle('active', index === activeIndex);
    });
}

// Add timeline click functionality
function setupTimelineClick() {
    const timelineBar = document.getElementById('timelineBar');
    
    timelineBar.addEventListener('click', (e) => {
        const rect = timelineBar.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const percentage = clickX / rect.width;
        const newTime = percentage * video.duration;
        
        if (newTime >= 0 && newTime <= video.duration) {
            video.currentTime = newTime;
        }
    });
}

function updateKeyframeList() {
    const list = document.getElementById('keyframeList');
    list.innerHTML = '';
    
    annotationData.keyframes.forEach((keyframe, index) => {
        const div = document.createElement('div');
        div.className = 'keyframe-item';
        div.innerHTML = `
            <strong>Keyframe ${index + 1}</strong><br>
            Time: ${formatTime(keyframe.timestamp)}<br>
            Objects: ${keyframe.objects.length}
        `;
        div.onclick = () => seekToKeyframe(index);
        list.appendChild(div);
    });
}

function highlightCurrentKeyframe() {
    document.querySelectorAll('.keyframe-item').forEach((item, index) => {
        item.classList.toggle('active', index === currentKeyframeIndex);
    });
}

function loadKeyframeAnnotations(keyframe) {
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
                timestamp: objData.timestamp
            });
            fabricCanvas.add(obj);
        }
    });
}

function addFrameConstraint() {
    const input = document.getElementById('frameConstraint');
    const constraint = input.value.trim();
    
    if (!constraint) return;
    
    if (currentKeyframeIndex < 0) {
        alert('Please select a keyframe first by clicking on one in the timeline or sidebar.');
        return;
    }

    
    const keyframe = annotationData.keyframes[currentKeyframeIndex];
    if (!keyframe.constraints.frame_level.includes(constraint)) {
        keyframe.constraints.frame_level.push(constraint);
        input.value = '';
        displayConstraints();
        updateJSON();
    }
}

function addInterFrameConstraint() {
    const input = document.getElementById('interFrameConstraint');
    const constraint = input.value.trim();
    const fromIndex = parseInt(document.getElementById('fromKeyframe').value);
    const toIndex = parseInt(document.getElementById('toKeyframe').value);
    
    if (!constraint) {
        alert('Please enter a constraint description.');
        return;
    }
    
    if (isNaN(fromIndex) || isNaN(toIndex)) {
        alert('Please select both from and to keyframes.');
        return;
    }
    
    if (fromIndex === toIndex) {
        alert('From and to keyframes must be different.');
        return;
    }
    
    const interFrameConstraint = {
        id: 'constraint_' + Date.now(),
        from_keyframe_id: annotationData.keyframes[fromIndex].id,
        to_keyframe_id: annotationData.keyframes[toIndex].id,
        constraint_type: constraint,
        from_timestamp: annotationData.keyframes[fromIndex].timestamp,
        to_timestamp: annotationData.keyframes[toIndex].timestamp
    };
    
    // Check for duplicate constraints
    const exists = annotationData.inter_frame_constraints.some(existing => 
        existing.from_keyframe_id === interFrameConstraint.from_keyframe_id &&
        existing.to_keyframe_id === interFrameConstraint.to_keyframe_id &&
        existing.constraint_type === interFrameConstraint.constraint_type
    );
    
    if (!exists) {
        annotationData.inter_frame_constraints.push(interFrameConstraint);
        input.value = '';
        
        // Reset selections
        document.getElementById('fromKeyframe').value = '';
        document.getElementById('toKeyframe').value = '';
        updateInterFrameUI();
        
        displayConstraints();
        updateJSON();
    } else {
        alert('This constraint already exists between these keyframes.');
    }
}


function displayConstraints() {
    const frameDiv = document.getElementById('frameConstraints');
    const interDiv = document.getElementById('interFrameConstraints');
    const seletedKF = document.getElementById('intra-selected');
    
    // Display frame-level constraints for current keyframe
    if (currentKeyframeIndex >= 0) {
        console.log("Here");
        const keyframe = annotationData.keyframes[currentKeyframeIndex];
        frameDiv.innerHTML = keyframe.constraints.frame_level.map(c => 
            `<span style="display: inline-block; background: #667eea; color: white; padding: 4px 8px; margin: 2px; border-radius: 4px; font-size: 0.8em;">${c}</span>`
        ).join('');
        seletedKF.textContent = `Keyframe ${currentKeyframeIndex+1}`
    } else {
        frameDiv.innerHTML = '<em style="color: #999;">Select a keyframe to view constraints</em>';
    }
    
    // Display inter-frame constraints
    interDiv.innerHTML = annotationData.inter_frame_constraints.map(c => {
        const fromIndex = annotationData.keyframes.findIndex(kf => kf.id === c.from_keyframe_id) + 1;
        const toIndex = annotationData.keyframes.findIndex(kf => kf.id === c.to_keyframe_id) + 1;
        return `<div style="background: #f8f9fa; padding: 8px; margin: 4px 0; border-radius: 6px; border-left: 3px solid #e74c3c;">
            <strong>${c.constraint_type}</strong><br>
            <small>KF${fromIndex} → KF${toIndex}</small>
        </div>`;
    }).join('');
}

function updateJSON() {
    // Update current keyframe with canvas objects
    if (currentKeyframeIndex >= 0) {
        const keyframe = annotationData.keyframes[currentKeyframeIndex];
        keyframe.objects = fabricCanvas.getObjects().map(obj => {
            const objData = {
                id: obj.id || 'obj_' + Date.now(),
                label: obj.label || 'unlabeled',
                timestamp: obj.timestamp || video.currentTime
            };
            
            if (obj.type === 'rect') {
                objData.type = 'bounding_box';
                objData.coordinates = {
                    x: Math.round(obj.left),
                    y: Math.round(obj.top),
                    width: Math.round(obj.width * obj.scaleX),
                    height: Math.round(obj.height * obj.scaleY)
                };
            } else if (obj.type === 'circle') {
                objData.type = 'point';
                objData.coordinates = {
                    x: Math.round(obj.left + obj.radius),
                    y: Math.round(obj.top + obj.radius)
                };
            }
            
            return objData;
        });
    }
    
    document.getElementById('jsonOutput').textContent = JSON.stringify(annotationData, null, 2);
}

function clearAnnotations() {
    fabricCanvas.clear();
    annotationData.keyframes = [];
    annotationData.inter_frame_constraints = [];
    currentKeyframeIndex = -1;
    updateKeyframeList();
    updateTimelineMarkers();
    displayConstraints();
    updateJSON();
}

function clearAllAnnotations() {
    if (confirm('Are you sure you want to clear all annotations?')) {
        clearAnnotations();
    }
}

function updateTimeDisplay() {
    document.getElementById('currentTime').textContent = formatTime(video.currentTime);
    document.getElementById('duration').textContent = formatTime(video.duration || 0);
    
    // Update timeline markers when video metadata loads
    if (video.duration && annotationData.keyframes.length > 0) {
        updateTimelineMarkers();
    }
}

function updateProgressBar() {
    const progress = (video.currentTime / video.duration) * 100;
    document.getElementById('progressFill').style.width = progress + '%';
}

function formatTime(seconds) {
    if (isNaN(seconds)) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function copyToClipboard() {
    const jsonText = document.getElementById('jsonOutput').textContent;
    navigator.clipboard.writeText(jsonText).then(() => {
        alert('JSON copied to clipboard!');
    }).catch(() => {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = jsonText;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        alert('JSON copied to clipboard!');
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT') return;
    
    switch(e.key) {
        case ' ':
            e.preventDefault();
            video.paused ? video.play() : video.pause();
            break;
        case 'k':
            addKeyframe();
            break;
        case 'r':
            setAnnotationMode('rect');
            break;
        case 'p':
            setAnnotationMode('point');
            break;
        case 'v':
            setAnnotationMode('select');
            break;
        case 'Delete':
        case 'Backspace':
            if (currentMode === 'select') {
                fabricCanvas.remove(fabricCanvas.getActiveObject());
            }
            break;
    }
});


