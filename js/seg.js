/* SETUP MEDIAPIPE HOLISTIC INSTANCE */
let video = document.querySelector("video.input_video")
video.width = window.outerWidth
video.height = video.width * 1.3333333 //window.outerHeight

const canvasElement = document.getElementById("output_canvas");
canvasElement.width = video.width
canvasElement.height = video.height
const canvasCtx = canvasElement.getContext("2d");
const maskElement = document.getElementById("mask_canvas");
maskElement.width = video.width
maskElement.height = video.height
const maskCtx = maskElement.getContext("2d");

const gestureOutput = document.getElementById("gesture_output");

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
// Create task for image file processing:
import {
  ImageSegmenter,
  GestureRecognizer,
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.1.0-alpha-16";
let faceLandmarker;
let gestureRecognizer;
let imageSegmenter;

const legendColors = [
  [255, 197, 0, 255], // Vivid Yellow
  [128, 62, 117, 255], // Strong Purple
  [255, 104, 0, 255], // Vivid Orange
  [166, 189, 215, 255], // Very Light Blue
  [193, 0, 32, 255], // Vivid Red
  [206, 162, 98, 255], // Grayish Yellow
  [129, 112, 102, 255], // Medium Gray
  [0, 125, 52, 255], // Vivid Green
  [246, 118, 142, 255], // Strong Purplish Pink
  [0, 83, 138, 255], // Strong Blue
  [255, 112, 92, 255], // Strong Yellowish Pink
  [83, 55, 112, 255], // Strong Violet
  [255, 142, 0, 255], // Vivid Orange Yellow
  [179, 40, 81, 255], // Strong Purplish Red
  [244, 200, 0, 255], // Vivid Greenish Yellow
  [127, 24, 13, 255], // Strong Reddish Brown
  [147, 170, 0, 255], // Vivid Yellowish Green
  [89, 51, 21, 255], // Deep Yellowish Brown
  [241, 58, 19, 255], // Vivid Reddish Orange
  [35, 44, 22, 255], // Dark Olive Green
  [0, 161, 194, 255] // Vivid Blue
];

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}
  // getUsermedia parameters.
const constraints = {
   video: true,
   video: { facingMode: "user" }
    };


const creatImageSegmenter = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
  );
  imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        //"https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite",
        //"https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
       "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    outputCategoryMask: true,
    outputConfidenceMasks: false
  });
};
creatImageSegmenter();

const createFaceLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
    },
    outputFaceBlendshapes: true,
    runningMode: "VIDEO",
    numFaces: 9
  });
};
createFaceLandmarker();

const creatGestureLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
  );
  gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 18
  });
};
creatGestureLandmarker();

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/

function callbackForVideo(result) {
  var canvas = document.createElement('canvas');
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  var ctx = canvas.getContext('2d');
  //let imageData = canvasCtx.getImageData(
  let imageData = ctx.getImageData(
    0,
    0,
    //canvasElement.width,
    canvas.width,
    //canvasElement.height
    canvas.height
  ).data;
  const mask = result.categoryMask.getAsFloat32Array();
  let j = 0;
  for (let i = 0; i < mask.length; ++i) {
    const maskVal = Math.round(mask[i] * 255.0);
    const legendColor = legendColors[maskVal % legendColors.length];
    imageData[j] = (legendColor[0] + imageData[j]) / 2;
    imageData[j + 1] = (legendColor[1] + imageData[j + 1]) / 2;
    imageData[j + 2] = (legendColor[2] + imageData[j + 2]) / 2;
    imageData[j + 3] = (legendColor[3] + imageData[j + 3]) / 2;
    j += 4;
  }
  const uint8Array = new Uint8ClampedArray(imageData.buffer);
  const dataNew = new ImageData(
    uint8Array,
    canvas.width,
    canvas.height
  );
  ctx.putImageData(dataNew, 0, 0)

  var resizedcanvas = document.createElement('canvas');
  resizedcanvas.width = window.outerWidth
  resizedcanvas.height = window.outerHeight
  var resizedctx = resizedcanvas.getContext('2d');
  //resizedctx.drawImage(canvas, 0, 0, canvasElement.width, canvasElement.height)
  resizedctx.drawImage(canvas, 0, 0, maskElement.width, maskElement.height)
  //maskCtx.putImageData(resizedctx.getImageData(0,0, canvasElement.width, canvasElement.height), 0, 0)
  maskCtx.putImageData(resizedctx.getImageData(0,0, maskElement.width, maskElement.height), 0, 0)
}


let lastVideoTime = -1;
let results = undefined;
let gestureResults = undefined;

async function predictWebcam() {
  //const webcamElement = document.getElementById("webcam");
  // Now let's start detecting the stream.
  let nowInMs = Date.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    imageSegmenter.segmentForVideo(video, nowInMs, callbackForVideo);
    results = faceLandmarker.detectForVideo(video, nowInMs );
    gestureResults = gestureRecognizer.recognizeForVideo(video, nowInMs);
  }

  //canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  const drawingUtils = new DrawingUtils(canvasCtx);

  if (results.faceLandmarks) {
    for (const landmarks of results.faceLandmarks) {
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        { color: "#C0C0C070", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
        { color: "#30FF30" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
        { color: "#30FF30" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LIPS,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
        { color: "#30FF30" }
      );
    }
  }
  if (gestureResults.landmarks) {
    for (const landmarks of gestureResults.landmarks) {
      drawingUtils.drawConnectors(
        landmarks,
        GestureRecognizer.HAND_CONNECTIONS,
        {
          color: "#00FF00",
          lineWidth: 5
        }
      );
      drawingUtils.drawLandmarks(landmarks, {
        color: "#FF0000",
        lineWidth: 2
      });
    }
  }

  if (gestureResults.gestures.length > 0) {
    gestureOutput.style.display = "block";
    gestureOutput.style.width = video.videoWidth;
    var i = 0;

   for (const gestures of gestureResults.gestures){
    //  const categoryName = results.gestures[0][0].categoryName;
      const categoryName = gestureResults.gestures[i][0].categoryName;
      const categoryScore = parseFloat(
     // results.gestures[0][0].score * 100
        gestureResults.gestures[i][0].score * 100
      ).toFixed(2);
     //  const handedness = results.handednesses[0][0].displayName;
      const handedness = gestureResults.handednesses[i][0].displayName;
   
   /*
   if (i == 0){
    gestureOutput.innerText = `[${i}]Gesture: ${categoryName},  ${categoryScore} %,  ${handedness}\n`;
   } else{
    gestureOutput.innerText += `[${i}]Gesture: ${categoryName},  ${categoryScore} %,  ${handedness}\n`;
   }*/
      if (i == 0){
        gestureOutput.innerText = `[${i}] ${categoryName}:  ${handedness}`;
      } else{
        gestureOutput.innerText += `,[${i}] ${categoryName}:  ${handedness}`;
      }
      i++;
    }
  } else {
    gestureOutput.style.display = "none";
  }
  // Call this function again to keep predicting when the browser is ready.
  window.requestAnimationFrame(predictWebcam);
}

/////
///// DEMO
/////
// add 'Tap + Hold to Add to Photos' prompt when user takes a photo
window.addEventListener('mediarecorder-photocomplete', () => {
  document.getElementById('overlay').style.display = 'block'
})

// hide 'Tap + Hold to Add to Photos' prompt when user dismisses preview modal
window.addEventListener('mediarecorder-previewclosed', () => {
  document.getElementById('overlay').style.display = 'none'
})

 // Activate the webcam stream.
 navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
   video.srcObject = stream;
   video.addEventListener("loadeddata", predictWebcam);
 });

const picture = document.querySelector("#picture")
/**
   * シャッターボタン
   */
   document.querySelector("#save").addEventListener("click", () => {
    const ctx = picture.getContext("2d")
      picture.width = video.height
      picture.height = video.width

    // 演出的な目的で一度映像を止めてSEを再生する
    video.pause()  // 映像を停止
    //se.play()      // シャッター音
    setTimeout( () => {
      video.play()    // 0.5秒後にカメラ再開
    }, 500);

    // canvasに画像を貼り付ける
    ctx.drawImage(video, 0, 0, picture.width, picture.height);
    ctx.drawImage(canvasElement, 0, 0, picture.width, picture.height);
    ctx.drawImage(maskElement, 0, 0, picture.width, picture.height);

    var base64Image = document.getElementById('picture').toDataURL()
  resizeImage(base64Image, function(base64) {
 
    var object = {
        //"url": dataUrl
        "url": base64Image
    }
    var result = prompt(JSON.stringify(object))
  })
  return false
  })


/**
 * 画像のリサイズ
 * @param  {string}   base64   [base64]
 * @param  {Function} callback [Function]
 * @return {string}            [base64]
 */
const resizeImage = function(base64, callback) {
    const MIN_SIZE = 800;
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    var image = new Image();
    image.crossOrigin = "Anonymous";
    image.onload = function(event){
        var dstWidth, dstHeight;
        if (this.width > this.height) {
            dstWidth = MIN_SIZE;
            dstHeight = this.height * MIN_SIZE / this.width;
        } else {
            dstHeight = MIN_SIZE;
            dstWidth = this.width * MIN_SIZE / this.height;
        }
        canvas.width = dstWidth;
        canvas.height = dstHeight;
        ctx.drawImage(this, 0, 0, this.width, this.height, 0, 0, dstWidth, dstHeight);
        callback(canvas.toDataURL());
    };
    image.src = base64;
};
 
 /**
 * base64からBlobにコンパイル
 * @param  {string} base64 [base64]
 * @return {string}        [blob]
 */
function base64toBlob(base64) {
  var bin = atob(base64.replace(/^.*,/, ''));
  var buffer = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) {
    buffer[i] = bin.charCodeAt(i);
  }
  try{
    var blob = new Blob([buffer.buffer], {
      type: 'image/jpeg'
    });
  }catch (e){
    return false;
  }
  return blob;
}
/*
document.getElementById('save').addEventListener('click', function() {
  var base64Image = document.getElementById('picture').toDataURL()
  resizeImage(base64Image, function(base64) {
    var blob = base64toBlob(base64Image)
    var url = (window.URL || window.webkitURL);
    var dataUrl = url.createObjectURL(blob)
 
    var object = {
        //"url": dataUrl
        "url": base64Image
    }
    var result = prompt(JSON.stringify(object))
  })
  return false
})
*/
