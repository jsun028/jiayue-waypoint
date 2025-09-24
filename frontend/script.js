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
    
    VideoController.setupVideo();
    CanvasManager.setup();
    setupEventListeners();
    // Load ML model
    ObjectDetectionManager.loadModel();
    // Initialize default mode
    setAnnotationMode('select');
    
    // Load demo video
    annotationData.video_id = 'demo_big_buck_bunny';
};


function setupEventListeners() {
    document.getElementById('videoFile').addEventListener(
        'change', VideoController.handleFileUpload);

    // Window resize
    window.addEventListener('resize', CanvasManager.resize);
}



function setAnnotationMode(mode) {
    currentMode = mode;

    // update button states    
    document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(mode + 'Btn').classList.add('active');

    // Use CanvasManager for canvas mode changes
    CanvasManager.setMode(mode);
}

function addKeyframe() {
    const timestamp = VideoController.getCurrentTime();
    
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

    // Find the index of the newly added keyframe after sorting
    currentKeyframeIndex = annotationData.keyframes.findIndex(kf => kf.id === keyframe.id);
    
    updateKeyframeList();
    VideoController.updateTimelineMarkers(); 
    updateKeyframeDropdowns();
    VideoController.highlightCurrentKeyframe();
    updateJSON();
    ObjectDetectionManager.updateUI();
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
        const option = `<option value="${index}">Keyframe ${index + 1} (
            ${VideoController.formatTime(keyframe.timestamp)})</option>`;
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
        const fromTime = VideoController.formatTime(
            annotationData.keyframes[parseInt(fromIndex)].timestamp);
        const toTime = VideoController.formatTime(
            annotationData.keyframes[parseInt(toIndex)].timestamp);
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


// Add timeline click functionality
function setupTimelineClick() {
    const timelineBar = document.getElementById('timelineBar');

    // Check if the timeline bar element exists
    if (!timelineBar) {
        console.warn('Timeline bar element not found');
        return;
    }
    
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
            <div class="kf-content">
                <strong>Keyframe ${index + 1}</strong><br>
                Time: ${VideoController.formatTime(keyframe.timestamp)}<br>
                Objects: ${keyframe.objects.length}
            </div>
            <button class="delete-kf-btn" data-index="${index}">❌</button>
        `;
        div.querySelector('.delete-kf-btn').onclick = (e) => {
            e.stopPropagation(); // 클릭 시 keyframe seek 방지
            deleteKeyframe(index);
        };
        div.onclick = () => VideoController.seekToKeyframe(index);
        list.appendChild(div);
    });
}

function deleteKeyframe(index) {
    if (index < 0 || index >= annotationData.keyframes.length) return;

    const removed = annotationData.keyframes.splice(index, 1)[0];

    if (currentKeyframeIndex === index) {
        fabricCanvas.clear();
        currentKeyframeIndex = -1;
    }

    annotationData.inter_frame_constraints = annotationData.inter_frame_constraints.filter(c =>
        c.from_keyframe_id !== removed.id && c.to_keyframe_id !== removed.id
    );

    updateKeyframeList();
    VideoController.updateTimelineMarkers();
    updateKeyframeDropdowns();
    displayConstraints();
    updateJSON();
    ObjectDetectionManager.updateUI();
}



let isLoadingKeyframe = false;


function addObjectConstraint(objectId) {
    const keyframe = annotationData.keyframes[currentKeyframeIndex];
    const obj = keyframe.objects.find(o => o.id === objectId);
    if (!obj) return;

    if (!obj.constraints) obj.constraints = [];
    const input = document.getElementById(`objConstraint_${objectId}`);
    const constraint = input.value.trim();
    if (constraint && !obj.constraints.includes(constraint)) {
        obj.constraints.push(constraint);
        input.value = '';
        displayConstraints();
        updateJSON();
    }
}

function deleteObjectConstraint(objectId, idx) {
    const keyframe = annotationData.keyframes[currentKeyframeIndex];
    const obj = keyframe.objects.find(o => o.id === objectId);
    if (!obj) return;

    obj.constraints.splice(idx, 1);
    displayConstraints();
    updateJSON();
}

function addFrameConstraint() {
    if (currentKeyframeIndex < 0) {
        alert('Please select a keyframe first.');
        return;
    }

    const input = document.getElementById('frameConstraintInput');
    if (!input) return;

    const constraint = input.value.trim();
    if (!constraint) return;

    const keyframe = annotationData.keyframes[currentKeyframeIndex];
    if (!keyframe.constraints) keyframe.constraints = { frame_level: [], inter_frame: [] };

    if (!keyframe.constraints.frame_level.includes(constraint)) {
        keyframe.constraints.frame_level.push(constraint);
        input.value = ''; 
        displayConstraints(); 
        updateJSON();
    }
}


function deleteFrameConstraint(idx) {
    const keyframe = annotationData.keyframes[currentKeyframeIndex];
    keyframe.constraints.frame_level.splice(idx, 1);
    displayConstraints();
    updateJSON();
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

function deleteInterFrameConstraint(idx) {
    if (idx < 0 || idx >= annotationData.inter_frame_constraints.length) return;

    annotationData.inter_frame_constraints.splice(idx, 1);

    updateInterFrameUI();
    updateJSON();
}


function displayConstraints() {
    const frameDiv = document.getElementById('frameConstraints');
    const selectedKF = document.getElementById('intra-selected');

    if (currentKeyframeIndex >= 0) {
        const keyframe = annotationData.keyframes[currentKeyframeIndex];
        selectedKF.textContent = `Keyframe ${currentKeyframeIndex + 1}`;

        let html = "";

        // 🔹 Keyframe-level constraints
        html += `<h4>Keyframe Constraints</h4>`;
        html += keyframe.constraints.frame_level.map((c, idx) =>
            `<span class="constraint-badge">
                ${c}
                <button class="delete-constraint-btn" onclick="deleteFrameConstraint(${idx})">x</button>
            </span>`
        ).join('');
        html += `
            <div style="margin-top:6px;">
                <input type="text" id="frameConstraintInput"
                    placeholder="e.g., good_lighting" class="constraint-input">
                <button class="btn btn-primary" type="button"
                        onclick="addFrameConstraint()" style="margin-top:6px;">Add</button>
            </div>
            `;


        // Object-level constraints
        html += `<h4 style="margin-top:15px;">Object Constraints</h4>`;
        if (keyframe.objects.length === 0) {
            html += `<em>No objects in this keyframe</em>`;
        } else {
            keyframe.objects.forEach(obj => {
                if (!obj.constraints) obj.constraints = []; 
                html += `
                    <div style="margin:6px 0; padding:6px; border:1px solid #ccc; border-radius:6px;">
                        <strong>${obj.label} (${obj.type})</strong><br>
                        ${obj.constraints.length > 0
                            ? obj.constraints.map((c, idx) => `
                                <span class="constraint-badge">
                                  ${c}
                                  <button class="delete-constraint-btn" onclick="deleteObjectConstraint('${obj.id}', ${idx})">x</button>
                                </span>
                              `).join('')
                            : '<em>No constraints</em>'}
                        <div style="margin-top:4px;">
                            <input type="text" id="objConstraint_${obj.id}" placeholder="e.g., color=red, velocity>5" class="constraint-input">
                            <button class="btn btn-primary" onclick="addObjectConstraint('${obj.id}')" style="margin-top:4px;">Add</button>
                        </div>
                    </div>
                `;
            });
        }

        frameDiv.innerHTML = html;
    } else {
        selectedKF.textContent = 'None';
        frameDiv.innerHTML = '<em style="color:#999;">Select a keyframe to view constraints</em>';
    }
}

function updateJSON() {
    if (isLoadingKeyframe) return;

    if (currentKeyframeIndex >= 0) {
        const keyframe = annotationData.keyframes[currentKeyframeIndex];

        // update current canvas objects with existing object constraints
        keyframe.objects = fabricCanvas.getObjects().map(obj => {
            const existingObj = keyframe.objects.find(o => o.id === obj.id);

            const objData = {
                id: obj.id || 'obj_' + Date.now(),
                label: obj.label || 'unlabeled',
                timestamp: keyframe.timestamp,
                constraints: existingObj?.constraints || []
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

    document.getElementById('jsonOutput').textContent =
        JSON.stringify(annotationData, null, 2);
}



function clearAnnotations() {
    CanvasManager.clear();
    annotationData.keyframes = [];
    annotationData.inter_frame_constraints = [];
    currentKeyframeIndex = -1;
    updateKeyframeList();
    VideoController.updateTimelineMarkers();
    displayConstraints();
    updateJSON();
}

function clearAllAnnotations() {
    if (confirm('Are you sure you want to clear all annotations?')) {
        clearAnnotations();
    }
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


