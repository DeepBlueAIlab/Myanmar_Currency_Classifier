let webcamStarted = false;
let webcamStream = null;
let predictionInterval = null;

function startWebcam() {
    if (!webcamStarted) {
        webcamStarted = true;
        const video = document.createElement('video');
        document.getElementById('webcam-container').appendChild(video);

        navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
            webcamStream = stream;
            video.srcObject = stream;
            video.play();
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            predictionInterval = setInterval(() => {
                sendFrameToServer(video);
            }, 1000);
        }).catch(error => {
            console.error("Error accessing webcam: ", error);
            webcamStarted = false;
        });
    }
}

function stopWebcam() {
    if (webcamStarted) {
        webcamStarted = false;
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
        }

        clearInterval(predictionInterval);
        predictionInterval = null;

        const videoElement = document.querySelector('video');
        if (videoElement) {
            videoElement.pause();
            videoElement.srcObject = null;
            videoElement.remove();
        }

        const displayDiv = document.getElementById('webcam-prediction');
        displayDiv.innerHTML = '';
    }

    // Re-enabling the start button
    document.getElementById('startBtn').disabled = false;
}

function sendFrameToServer(video) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg').split(',')[1];
    
    fetch('/predict-webcam', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
    })
    .then(response => response.json())
    .then(data => {
        const displayDiv = document.getElementById('webcam-prediction');
        displayDiv.innerHTML = `
        <h3 class="text-light mt-3">
            Predicted Banknote: ${data.label} with confidence of ${data.confidence.toFixed(2)}
        </h3>
        `;
    }).catch(error => {
        console.error("Error sending frame to server: ", error);
    });
}

function clearUploadedPrediction() {
    const predictionDiv = document.getElementById('image-prediction');
    if (predictionDiv) {
        predictionDiv.innerHTML = '';
    }
}
