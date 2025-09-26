// =====================================================
// video-controller.js - Video player and timeline operations
// =====================================================

const VideoController = {
    /**
     * Setup video event listeners
     */
    setupVideo() {
        video.addEventListener('loadedmetadata', this.onLoadedMetadata.bind(this));
        video.addEventListener('timeupdate', this.onTimeUpdate.bind(this));
        video.addEventListener('click', this.onVideoClick.bind(this));
        video.addEventListener('play', this.onVideoPlay.bind(this));
    },

    /**
     * Handle video metadata loaded event
     */
    onLoadedMetadata() {
        CanvasManager.resize();
        this.updateTimeDisplay();
        this.updateTimelineMarkers(); 
        this.setupTimelineClick(); 
    },

    /**
     * Handle video time update event
     */
    onTimeUpdate() {
        this.updateTimeDisplay();
        this.updateProgressBar();
        this.highlightCurrentKeyframe();
    },

    /**
     * Handle video click events
     * @param {Event} e - Click event
     */
    onVideoClick(e) {
        if (currentMode !== 'select') {
            e.preventDefault();
        }
    },

    /**
     * Handle video play event
     */
    onVideoPlay() {
        currentKeyframeIndex = -1;   // release keyframe
        fabricCanvas.clear();        // clear canvas
        this.highlightCurrentKeyframe();  // update sidebar UI
        document.getElementById('intra-selected').textContent = 'None';
    },

    /**
     * Setup timeline click functionality
     */
    setupTimelineClick() {
        const timelineBar = document.getElementById('timelineBar');

        // Check if the timeline bar element exists
        if (!timelineBar) {
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
    },

    /**
     * Update time display
     */
    updateTimeDisplay() {
        const currentTimeEl = document.getElementById('currentTime');
        const durationEl = document.getElementById('duration');
        
        if (currentTimeEl) {
            currentTimeEl.textContent = this.formatTime(video.currentTime);
        }
        if (durationEl) {
            durationEl.textContent = this.formatTime(video.duration || 0);
        }
        
        // Update timeline markers when video metadata loads
        if (video.duration && annotationData.keyframes.length > 0) {
            this.updateTimelineMarkers();
        }
    },

    /**
     * Update progress bar
     */
    updateProgressBar() {
        const progressFill = document.getElementById('progressFill');
        if (!progressFill || !video.duration) return;
        
        const progress = (video.currentTime / video.duration) * 100;
        progressFill.style.width = Math.max(0, Math.min(100, progress)) + '%';
    },

    /**
     * Update timeline markers for keyframes
     */
    updateTimelineMarkers() {
        const markersContainer = document.getElementById('keyframeMarkers');
        
        if (!markersContainer || !video.duration) return;
        
        // Clear existing markers
        markersContainer.innerHTML = '';
        
        // Add markers for each keyframe
        annotationData.keyframes.forEach((keyframe, index) => {
            const marker = document.createElement('div');
            marker.className = 'keyframe-marker';
            marker.title = `Keyframe ${index + 1} - ${this.formatTime(keyframe.timestamp)}`;
            
            // Position marker based on timestamp
            const position = (keyframe.timestamp / video.duration) * 100;
            marker.style.left = Math.max(0, Math.min(100, position)) + '%';
            
            // Add visual indicator for detected objects
            const detectedCount = keyframe.objects.filter(obj => obj.detection_data?.auto_detected).length;
            if (detectedCount > 0) {
                marker.classList.add('has-detection');
                marker.title += ` (${detectedCount} AI objects)`;
            }
            
            // Add click handler to seek to keyframe
            marker.onclick = (e) => {
                e.stopPropagation();
                this.seekToKeyframe(index);
            };
            
            markersContainer.appendChild(marker);
        });
    },

    /**
     * Seek to a specific keyframe
     * @param {number} index - Index of keyframe to seek to
     */
    seekToKeyframe(index) {
        currentKeyframeIndex = index;
        const keyframe = annotationData.keyframes[index];
        video.currentTime = keyframe.timestamp;
        
        CanvasManager.loadKeyframeAnnotations(keyframe);

        this.highlightCurrentKeyframe();
        this.highlightTimelineMarker(index);
        displayConstraints();
        
        console.log(`Seeked to keyframe ${index + 1} at ${keyframe.timestamp}s`);
    },

    /**
     * Highlight current keyframe in the sidebar
     */
    highlightCurrentKeyframe() {
        document.querySelectorAll('.keyframe-item').forEach((item, index) => {
            item.classList.toggle('active', index === currentKeyframeIndex);
        });
    },

    /**
     * Highlight active timeline marker
     * @param {number} activeIndex - Index of active keyframe
     */
    highlightTimelineMarker(activeIndex) {
        document.querySelectorAll('.keyframe-marker').forEach((marker, index) => {
            marker.classList.toggle('active', index === activeIndex);
        });
    },

    /**
     * Handle file upload for video
     * @param {Event} event - File input change event
     */
    handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            // Create a URL for the video file
            const url = URL.createObjectURL(file);
            
            // Set the video's source to the new URL
            video.src = url;
            
            // Load the new video
            video.load();

            // Update video_id and clear old annotations
            annotationData.video_id = file.name.replace(/\.[^/.]+$/, "");
            clearAnnotations(); // Use clearAnnotations to avoid confirmation dialog
            
            console.log(`Loaded video: ${annotationData.video_id}`);
        }
    },

    /**
     * Format time in MM:SS format
     * @param {number} seconds - Time in seconds
     * @returns {string} Formatted time string
     */
    formatTime(seconds) {
        if (isNaN(seconds)) return '00:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    },

    /**
     * Get current video time
     * @returns {number} Current time in seconds
     */
    getCurrentTime() {
        return video.currentTime || 0;
    },

    /**
     * Get video duration
     * @returns {number} Duration in seconds
     */
    getDuration() {
        return video.duration || 0;
    },

    /**
     * Set video time
     * @param {number} time - Time in seconds to seek to
     */
    setCurrentTime(time) {
        if (video && time >= 0 && time <= video.duration) {
            video.currentTime = time;
        }
    },

    /**
     * Get video dimensions
     * @returns {Object} Video width and height
     */
    getVideoDimensions() {
        return {
            width: video ? video.videoWidth : 0,
            height: video ? video.videoHeight : 0
        };
    },

};

// Make VideoController available globally
window.VideoController = VideoController;