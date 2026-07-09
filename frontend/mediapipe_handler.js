/**
 * mediapipe_handler.js
 *
 * Initialises MediaPipe Face Mesh, attaches it to the webcam video element,
 * and emits normalised 468-landmark JSON payloads via the onFrame callback.
 *
 * Depends on globals injected by the MediaPipe CDN script tags in index.html:
 *   window.FaceMesh, window.Camera
 */

const MEDIAPIPE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/';

export class MediaPipeHandler {
  /**
   * @param {HTMLVideoElement} videoEl  - The <video id="webcam"> element.
   * @param {function}         onFrame  - Called with (landmarks, frameIndex, timestamp).
   *                                      landmarks: Array of 468 {x, y, z} objects.
   */
  constructor(videoEl, onFrame) {
    this._videoEl    = videoEl;
    this._onFrame    = onFrame;
    this._frameIndex = 0;
    this._active     = false;
    this._faceMesh   = null;
    this._camera     = null;

    // Offscreen canvas for capturing 224×224 RGB frames to send to the backend.
    // The model (EfficientNet-B0 + LSTM) was trained on raw 224×224 RGB video frames,
    // not on landmark renderings — so we send the actual pixel data for correct inference.
    this._canvas = document.createElement('canvas');
    this._canvas.width  = 224;
    this._canvas.height = 224;
    this._ctx = this._canvas.getContext('2d');
  }

  /** Initialise Face Mesh model and start the camera. */
  async start() {
    this._faceMesh = new window.FaceMesh({
      locateFile: (file) => `${MEDIAPIPE_CDN}${file}`,
    });

    this._faceMesh.setOptions({
      maxNumFaces:        1,
      // refineLandmarks adds 10 iris points (468 -> 478); backend expects exactly 468.
      refineLandmarks:    false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence:  0.5,
    });

    this._faceMesh.onResults((results) => this._onResults(results));

    this._camera = new window.Camera(this._videoEl, {
      onFrame: async () => {
        if (this._active) {
          await this._faceMesh.send({ image: this._videoEl });
        }
      },
      width: 640,
      height: 480,
    });

    this._active = true;
    await this._camera.start();
  }

  /** Stop the camera and free MediaPipe resources. */
  stop() {
    this._active = false;
    if (this._camera) {
      this._camera.stop();
      this._camera = null;
    }
    if (this._faceMesh) {
      this._faceMesh.close();
      this._faceMesh = null;
    }
    this._frameIndex = 0;
  }

  /** @private Process a single Face Mesh result frame. */
  _onResults(results) {
    if (!this._active) return;
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;

    // Take only the first detected face.
    const rawLandmarks = results.multiFaceLandmarks[0];

    // Normalise: MediaPipe already returns x,y in [0,1] relative to frame
    // dimensions, and z as a relative depth value. We strip any extra fields.
    const landmarks = rawLandmarks.map(({ x, y, z }) => ({
      x: parseFloat(x.toFixed(6)),
      y: parseFloat(y.toFixed(6)),
      z: parseFloat(z.toFixed(6)),
    }));

    // Capture the current video frame as a 224×224 JPEG for the video model.
    this._ctx.drawImage(this._videoEl, 0, 0, 224, 224);
    const frameJpeg = this._canvas.toDataURL('image/jpeg', 0.6).split(',')[1];

    this._onFrame(landmarks, this._frameIndex++, Date.now() / 1000, frameJpeg);
  }
}
