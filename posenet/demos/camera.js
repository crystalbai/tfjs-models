/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posenet from '@tensorflow-models/posenet';
import dat from 'dat.gui';
import Stats from 'stats.js';

import {drawBoundingBox, drawKeypoints, createPoseVector, cosineDistanceMatching, drawSkeleton, isMobile, toggleLoadingUI, tryResNetButtonName, tryResNetButtonText, updateTryResNetButtonDatGuiCss} from './demo_util';

const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();

// const gt_keypoints = JSON.parse("[{\"score\":0.9993378520011902,\"part\":\"nose\",\"position\":{\"x\":260.54107666015625,\"y\":93.54180908203125}},{\"score\":0.999618649482727,\"part\":\"leftEye\",\"position\":{\"x\":245.94584147135419,\"y\":79.91317749023438}},{\"score\":0.999249279499054,\"part\":\"rightEye\",\"position\":{\"x\":276.66829427083337,\"y\":80.54864501953125}},{\"score\":0.984531044960022,\"part\":\"leftEar\",\"position\":{\"x\":222.8551025390625,\"y\":93.85414632161456}},{\"score\":0.985697329044342,\"part\":\"rightEar\",\"position\":{\"x\":296.9931640625,\"y\":96.161376953125}},{\"score\":0.999459445476532,\"part\":\"leftShoulder\",\"position\":{\"x\":191.51460774739587,\"y\":201.64638264973956}},{\"score\":0.9996799826622009,\"part\":\"rightShoulder\",\"position\":{\"x\":337.8243611653646,\"y\":197.4298095703125}},{\"score\":0.9969266057014465,\"part\":\"leftElbow\",\"position\":{\"x\":162.71073404947919,\"y\":321.08451334635413}},{\"score\":0.9985221028327942,\"part\":\"rightElbow\",\"position\":{\"x\":379.58994547526044,\"y\":311.21561686197913}},{\"score\":0.994284451007843,\"part\":\"leftWrist\",\"position\":{\"x\":145.43326822916669,\"y\":441.3509114583333}},{\"score\":0.9983605742454529,\"part\":\"rightWrist\",\"position\":{\"x\":415.46531168619794,\"y\":416.8245849609375}},{\"score\":0.9795545935630798,\"part\":\"leftHip\",\"position\":{\"x\":219.13444010416669,\"y\":410.942138671875}},{\"score\":0.9871241450309753,\"part\":\"rightHip\",\"position\":{\"x\":329.55029296875,\"y\":405.956787109375}},{\"score\":0.5951845645904541,\"part\":\"leftKnee\",\"position\":{\"x\":212.55814615885419,\"y\":560.9640706380208}},{\"score\":0.3492562472820282,\"part\":\"rightKnee\",\"position\":{\"x\":330.12620035807294,\"y\":569.2143961588541}},{\"score\":0.017244666814804077,\"part\":\"leftAnkle\",\"position\":{\"x\":189.892822265625,\"y\":561.7104085286458}},{\"score\":0.0058021098375320435,\"part\":\"rightAnkle\",\"position\":{\"x\":207.15718587239587,\"y\":547.4464111328125}}]")

let gt_keypoints = JSON.parse("[{\"score\": 0.9981367588043213, \"part\": \"nose\", \"position\": {\"x\": 269.9845810039419, \"y\": 128.59750973231533}}, {\"score\": 0.9969700574874878, \"part\": \"leftEye\", \"position\": {\"x\": 288.33018564932377, \"y\": 113.48109451129807}}, {\"score\": 0.9938220977783203, \"part\": \"rightEye\", \"position\": {\"x\": 254.50813884724747, \"y\": 111.61868993571213}}, {\"score\": 0.7406206727027893, \"part\": \"leftEar\", \"position\": {\"x\": 310.28090664360525, \"y\": 120.45185836239511}}, {\"score\": 0.8841767311096191, \"part\": \"rightEar\", \"position\": {\"x\": 225.64682827335105, \"y\": 119.25885369034785}}, {\"score\": 0.9750819802284241, \"part\": \"leftShoulder\", \"position\": {\"x\": 337.8243611653646, \"y\": 200.8179327304283}}, {\"score\": 0.7116842269897461, \"part\": \"rightShoulder\", \"position\": {\"x\": 191.51460774739587, \"y\": 197.42980957031247}}, {\"score\": 0.9506950378417969, \"part\": \"leftElbow\", \"position\": {\"x\": 374.6327404568937, \"y\": 316.4945623498969}}, {\"score\": 0.9578353762626648, \"part\": \"rightElbow\", \"position\": {\"x\": 136.05771865929444, \"y\": 300.9504424262101}}, {\"score\": 0.5046786069869995, \"part\": \"leftWrist\", \"position\": {\"x\": 397.52424973031725, \"y\": 421.65612196352254}}, {\"score\": 0.9162625074386597, \"part\": \"rightWrist\", \"position\": {\"x\": 118.93416330525818, \"y\": 404.72299780643954}}, {\"score\": 0.9555754661560059, \"part\": \"leftHip\", \"position\": {\"x\": 314.01663233310194, \"y\": 409.41661667008907}}, {\"score\": 0.9543619751930237, \"part\": \"rightHip\", \"position\": {\"x\": 208.31408978981582, \"y\": 410.942138671875}}, {\"score\": 0.9460088610649109, \"part\": \"leftKnee\", \"position\": {\"x\": 313.3556259961397, \"y\": 601.0336949254065}}, {\"score\": 0.9692136645317078, \"part\": \"rightKnee\", \"position\": {\"x\": 226.26852614932793, \"y\": 603.5518799666479}}, {\"score\": 0.8375222682952881, \"part\": \"leftAnkle\", \"position\": {\"x\": 275.8595179846701, \"y\": 781.6670204323323}}, {\"score\": 0.8201625943183899, \"part\": \"rightAnkle\", \"position\": {\"x\": 233.9587667864883, \"y\": 782.2519435684072}}]")


const gtKeypointsList = ["[{\"score\": 0.9884622693061829, \"part\": \"nose\", \"position\": {\"x\": 239.42941335021817, \"y\": 135.63652339286844}}, {\"score\": 0.9870627522468567, \"part\": \"leftEye\", \"position\": {\"x\": 259.9866861154069, \"y\": 119.2678566042892}}, {\"score\": 0.9510775804519653, \"part\": \"rightEye\", \"position\": {\"x\": 218.73459315881033, \"y\": 121.56412493391474}}, {\"score\": 0.8054677248001099, \"part\": \"leftEar\", \"position\": {\"x\": 293.0000644153366, \"y\": 130.25120367220597}}, {\"score\": 0.2604908347129822, \"part\": \"rightEar\", \"position\": {\"x\": 206.65172036820417, \"y\": 132.78787768268884}}, {\"score\": 0.9949918389320374, \"part\": \"leftShoulder\", \"position\": {\"x\": 337.8243611653646, \"y\": 205.31788485796}}, {\"score\": 0.772001326084137, \"part\": \"rightShoulder\", \"position\": {\"x\": 197.1726051871393, \"y\": 197.4298095703125}}, {\"score\": 0.4606063961982727, \"part\": \"leftElbow\", \"position\": {\"x\": 404.5640449744809, \"y\": 297.83348700179835}}, {\"score\": 0.6210929155349731, \"part\": \"rightElbow\", \"position\": {\"x\": 99.42898277930144, \"y\": 146.00490463022186}}, {\"score\": 0.9807297587394714, \"part\": \"leftWrist\", \"position\": {\"x\": 368.1344655646007, \"y\": 395.86100343397675}}, {\"score\": 0.7574535012245178, \"part\": \"rightWrist\", \"position\": {\"x\": -54.23180690221915, \"y\": 75.28747396400567}}, {\"score\": 0.8437752723693848, \"part\": \"leftHip\", \"position\": {\"x\": 306.7802471742272, \"y\": 407.82130085879544}}, {\"score\": 0.9275577664375305, \"part\": \"rightHip\", \"position\": {\"x\": 191.5146077473959, \"y\": 410.942138671875}}, {\"score\": 0.9751166701316833, \"part\": \"leftKnee\", \"position\": {\"x\": 448.3296074209169, \"y\": 566.0246513889133}}, {\"score\": 0.9585945010185242, \"part\": \"rightKnee\", \"position\": {\"x\": 173.5935137483881, \"y\": 584.8021470411572}}, {\"score\": 0.9299298524856567, \"part\": \"leftAnkle\", \"position\": {\"x\": 415.5970828050014, \"y\": 747.8287360726383}}, {\"score\": 0.5320151448249817, \"part\": \"rightAnkle\", \"position\": {\"x\": 149.9005431371687, \"y\": 744.2586389403449}}]",
"[{\"score\": 0.9965842962265015, \"part\": \"nose\", \"position\": {\"x\": 253.5464475740273, \"y\": 132.47873851107738}}, {\"score\": 0.9884806871414185, \"part\": \"leftEye\", \"position\": {\"x\": 270.007364979413, \"y\": 117.3624753241416}}, {\"score\": 0.9886897206306458, \"part\": \"rightEye\", \"position\": {\"x\": 233.7987353669667, \"y\": 117.54409769018744}}, {\"score\": 0.7728598117828369, \"part\": \"leftEar\", \"position\": {\"x\": 290.04887010770636, \"y\": 125.5898711780969}}, {\"score\": 0.6010012030601501, \"part\": \"rightEar\", \"position\": {\"x\": 214.17392613807039, \"y\": 128.19407245179366}}, {\"score\": 0.9726883172988892, \"part\": \"leftShoulder\", \"position\": {\"x\": 337.8243611653646, \"y\": 197.4298095703125}}, {\"score\": 0.9860156774520874, \"part\": \"rightShoulder\", \"position\": {\"x\": 191.51460774739584, \"y\": 212.9393539087667}}, {\"score\": 0.9697409272193909, \"part\": \"leftElbow\", \"position\": {\"x\": 400.4982408223558, \"y\": 303.7772950575426}}, {\"score\": 0.9865039587020874, \"part\": \"rightElbow\", \"position\": {\"x\": 147.54796618470397, \"y\": 315.3377698861636}}, {\"score\": 0.7318808436393738, \"part\": \"leftWrist\", \"position\": {\"x\": 376.77192356500507, \"y\": 388.0409635735275}}, {\"score\": 0.45608246326446533, \"part\": \"rightWrist\", \"position\": {\"x\": 92.15984543234683, \"y\": 406.5499301955294}}, {\"score\": 0.9853435754776001, \"part\": \"leftHip\", \"position\": {\"x\": 306.31710407392313, \"y\": 409.3102012003752}}, {\"score\": 0.9727784395217896, \"part\": \"rightHip\", \"position\": {\"x\": 199.74893003632445, \"y\": 410.942138671875}}, {\"score\": 0.8992089629173279, \"part\": \"leftKnee\", \"position\": {\"x\": 429.5154843189406, \"y\": 555.7750090111325}}, {\"score\": 0.984283447265625, \"part\": \"rightKnee\", \"position\": {\"x\": 180.33084440503262, \"y\": 596.257929745072}}, {\"score\": 0.8944122195243835, \"part\": \"leftAnkle\", \"position\": {\"x\": 394.542433382863, \"y\": 737.1624535757342}}, {\"score\": 0.706839382648468, \"part\": \"rightAnkle\", \"position\": {\"x\": 172.52576127165617, \"y\": 754.2225840535341}}]",
"[{\"score\": 0.9850581884384155, \"part\": \"nose\", \"position\": {\"x\": 266.73675711429775, \"y\": 104.1127484987035}}, {\"score\": 0.9744524359703064, \"part\": \"leftEye\", \"position\": {\"x\": 280.9633354586377, \"y\": 87.3802924779132}}, {\"score\": 0.9730096459388733, \"part\": \"rightEye\", \"position\": {\"x\": 253.43991753218398, \"y\": 84.73738664947692}}, {\"score\": 0.522997260093689, \"part\": \"leftEar\", \"position\": {\"x\": 307.47185957341003, \"y\": 100.13713467353352}}, {\"score\": 0.5789024829864502, \"part\": \"rightEar\", \"position\": {\"x\": 231.6633520349741, \"y\": 100.75994061591476}}, {\"score\": 0.9920985698699951, \"part\": \"leftShoulder\", \"position\": {\"x\": 337.8243611653646, \"y\": 201.463427860798}}, {\"score\": 0.9944802522659302, \"part\": \"rightShoulder\", \"position\": {\"x\": 200.62556795807683, \"y\": 197.4298095703125}}, {\"score\": 0.9871802926063538, \"part\": \"leftElbow\", \"position\": {\"x\": 346.9462417167575, \"y\": 319.1695798460662}}, {\"score\": 0.9700820446014404, \"part\": \"rightElbow\", \"position\": {\"x\": 174.4818910546703, \"y\": 316.65936362631686}}, {\"score\": 0.9287230372428894, \"part\": \"leftWrist\", \"position\": {\"x\": 352.43054178265186, \"y\": 429.07700319344207}}, {\"score\": 0.5781751275062561, \"part\": \"rightWrist\", \"position\": {\"x\": 143.60727163902325, \"y\": 424.2418029318508}}, {\"score\": 0.4209105372428894, \"part\": \"leftHip\", \"position\": {\"x\": 295.28514217285283, \"y\": 410.942138671875}}, {\"score\": 0.9768547415733337, \"part\": \"rightHip\", \"position\": {\"x\": 191.51460774739587, \"y\": 403.6523331119057}}, {\"score\": 0.9639926552772522, \"part\": \"leftKnee\", \"position\": {\"x\": 331.26554966086223, \"y\": 613.458117270475}}, {\"score\": 0.9594760537147522, \"part\": \"rightKnee\", \"position\": {\"x\": 183.2448806541509, \"y\": 617.5793200109204}}, {\"score\": 0.7726997137069702, \"part\": \"leftAnkle\", \"position\": {\"x\": 409.9483750981821, \"y\": 787.3688965162924}}, {\"score\": 0.8538125157356262, \"part\": \"rightAnkle\", \"position\": {\"x\": 176.62655919479278, \"y\": 789.230606765148}}]",
"[{\"score\": 0.9910212159156799, \"part\": \"nose\", \"position\": {\"x\": 241.33104220076018, \"y\": 119.50664685727185}}, {\"score\": 0.9879887104034424, \"part\": \"leftEye\", \"position\": {\"x\": 260.40455438821857, \"y\": 100.31650543356847}}, {\"score\": 0.9836134910583496, \"part\": \"rightEye\", \"position\": {\"x\": 219.4923996344529, \"y\": 103.42536838838171}}, {\"score\": 0.8451476097106934, \"part\": \"leftEar\", \"position\": {\"x\": 297.9281526350917, \"y\": 113.48073390177504}}, {\"score\": 0.2963601052761078, \"part\": \"rightEar\", \"position\": {\"x\": 203.6032908098465, \"y\": 120.64235372011709}}, {\"score\": 0.9924599528312683, \"part\": \"leftShoulder\", \"position\": {\"x\": 337.8243611653646, \"y\": 197.4298095703125}}, {\"score\": 0.9844498038291931, \"part\": \"rightShoulder\", \"position\": {\"x\": 194.8928458035657, \"y\": 202.12116865427095}}, {\"score\": 0.6733474135398865, \"part\": \"leftElbow\", \"position\": {\"x\": 362.6063758259943, \"y\": 331.44723428082085}}, {\"score\": 0.9068714380264282, \"part\": \"rightElbow\", \"position\": {\"x\": 157.4949675459623, \"y\": 336.0174696272823}}, {\"score\": 0.4237756133079529, \"part\": \"leftWrist\", \"position\": {\"x\": 264.0809070011592, \"y\": 329.3253799660698}}, {\"score\": 0.6783806085586548, \"part\": \"rightWrist\", \"position\": {\"x\": 62.0891050596604, \"y\": 354.692284786044}}, {\"score\": 0.9669419527053833, \"part\": \"leftHip\", \"position\": {\"x\": 330.54439014340903, \"y\": 405.91052812177065}}, {\"score\": 0.924975574016571, \"part\": \"rightHip\", \"position\": {\"x\": 191.51460774739587, \"y\": 410.942138671875}}, {\"score\": 0.8180175423622131, \"part\": \"leftKnee\", \"position\": {\"x\": 399.6874354731543, \"y\": 624.1328374099442}}, {\"score\": 0.8094072937965393, \"part\": \"rightKnee\", \"position\": {\"x\": 153.2387408919891, \"y\": 627.9436112006811}}, {\"score\": 0.8337874412536621, \"part\": \"leftAnkle\", \"position\": {\"x\": 500.47966932890967, \"y\": 784.3729175464725}}, {\"score\": 0.7481302618980408, \"part\": \"rightAnkle\", \"position\": {\"x\": 20.316270102332197, \"y\": 777.8977711754957}}]"]

let gtKeypointsIdx = 0;

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const defaultQuantBytes = 2;

const defaultMobileNetMultiplier = isMobile() ? 0.50 : 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 500;

const defaultResNetMultiplier = 1.0;
const defaultResNetStride = 32;
const defaultResNetInputResolution = 250;

const guiState = {
  algorithm: 'multi-pose',
  input: {
    architecture: 'MobileNetV1',
    outputStride: defaultMobileNetStride,
    inputResolution: defaultMobileNetInputResolution,
    multiplier: defaultMobileNetMultiplier,
    quantBytes: defaultQuantBytes
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
  },
  net: null,
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({width: 300});

  let architectureController = null;
  guiState[tryResNetButtonName] = function() {
    architectureController.setValue('ResNet50')
  };
  gui.add(guiState, tryResNetButtonName).name(tryResNetButtonText);
  updateTryResNetButtonDatGuiCss();

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController =
      gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  architectureController =
      input.add(guiState.input, 'architecture', ['MobileNetV1', 'ResNet50']);
  guiState.architecture = guiState.input.architecture;
  // Input resolution:  Internally, this parameter affects the height and width
  // of the layers in the neural network. The higher the value of the input
  // resolution the better the accuracy but slower the speed.
  let inputResolutionController = null;
  function updateGuiInputResolution(
      inputResolution,
      inputResolutionArray,
  ) {
    if (inputResolutionController) {
      inputResolutionController.remove();
    }
    guiState.inputResolution = inputResolution;
    guiState.input.inputResolution = inputResolution;
    inputResolutionController =
        input.add(guiState.input, 'inputResolution', inputResolutionArray);
    inputResolutionController.onChange(function(inputResolution) {
      guiState.changeToInputResolution = inputResolution;
    });
  }

  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  let outputStrideController = null;
  function updateGuiOutputStride(outputStride, outputStrideArray) {
    if (outputStrideController) {
      outputStrideController.remove();
    }
    guiState.outputStride = outputStride;
    guiState.input.outputStride = outputStride;
    outputStrideController =
        input.add(guiState.input, 'outputStride', outputStrideArray);
    outputStrideController.onChange(function(outputStride) {
      guiState.changeToOutputStride = outputStride;
    });
  }

  // Multiplier: this parameter affects the number of feature map channels in
  // the MobileNet. The higher the value, the higher the accuracy but slower the
  // speed, the lower the value the faster the speed but lower the accuracy.
  let multiplierController = null;
  function updateGuiMultiplier(multiplier, multiplierArray) {
    if (multiplierController) {
      multiplierController.remove();
    }
    guiState.multiplier = multiplier;
    guiState.input.multiplier = multiplier;
    multiplierController =
        input.add(guiState.input, 'multiplier', multiplierArray);
    multiplierController.onChange(function(multiplier) {
      guiState.changeToMultiplier = multiplier;
    });
  }

  // QuantBytes: this parameter affects weight quantization in the ResNet50
  // model. The available options are 1 byte, 2 bytes, and 4 bytes. The higher
  // the value, the larger the model size and thus the longer the loading time,
  // the lower the value, the shorter the loading time but lower the accuracy.
  let quantBytesController = null;
  function updateGuiQuantBytes(quantBytes, quantBytesArray) {
    if (quantBytesController) {
      quantBytesController.remove();
    }
    guiState.quantBytes = +quantBytes;
    guiState.input.quantBytes = +quantBytes;
    quantBytesController =
        input.add(guiState.input, 'quantBytes', quantBytesArray);
    quantBytesController.onChange(function(quantBytes) {
      guiState.changeToQuantBytes = +quantBytes;
    });
  }

  function updateGui() {
    if (guiState.input.architecture === 'MobileNetV1') {
      updateGuiInputResolution(
          defaultMobileNetInputResolution,
          [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]);
      updateGuiOutputStride(defaultMobileNetStride, [8, 16]);
      updateGuiMultiplier(defaultMobileNetMultiplier, [0.50, 0.75, 1.0]);
    } else {  // guiState.input.architecture === "ResNet50"
      updateGuiInputResolution(
          defaultResNetInputResolution,
          [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]);
      updateGuiOutputStride(defaultResNetStride, [32, 16]);
      updateGuiMultiplier(defaultResNetMultiplier, [1.0]);
    }
    updateGuiQuantBytes(defaultQuantBytes, [1, 2, 4]);
  }

  updateGui();
  input.open();
  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);

  let multi = gui.addFolder('Multi Pose Detection');
  multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
      .min(1)
      .max(20)
      .step(1);
  multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0);
  multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0);
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0);
  multi.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showBoundingBox');
  output.open();


  architectureController.onChange(function(architecture) {
    // if architecture is ResNet50, then show ResNet50 options
    updateGui();
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function(value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        multi.close();
        single.open();
        break;
      case 'multi-pose':
        single.close();
        multi.open();
        break;
    }
  });
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');

  // since images are being fed from a webcam, we want to feed in the
  // original image and then just flip the keypoints' x coordinates. If instead
  // we flip the image, then correcting left-right keypoint pairs requires a
  // permutation on all the keypoints.
  const flipPoseHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();
      toggleLoadingUI(true);
      guiState.net = await posenet.load({
        architecture: guiState.changeToArchitecture,
        outputStride: guiState.outputStride,
        inputResolution: guiState.inputResolution,
        multiplier: guiState.multiplier,
      });
      toggleLoadingUI(false);
      guiState.architecture = guiState.changeToArchitecture;
      guiState.changeToArchitecture = null;
    }

    if (guiState.changeToMultiplier) {
      guiState.net.dispose();
      toggleLoadingUI(true);
      guiState.net = await posenet.load({
        architecture: guiState.architecture,
        outputStride: guiState.outputStride,
        inputResolution: guiState.inputResolution,
        multiplier: +guiState.changeToMultiplier,
        quantBytes: guiState.quantBytes
      });
      toggleLoadingUI(false);
      guiState.multiplier = +guiState.changeToMultiplier;
      guiState.changeToMultiplier = null;
    }

    if (guiState.changeToOutputStride) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();
      toggleLoadingUI(true);
      guiState.net = await posenet.load({
        architecture: guiState.architecture,
        outputStride: +guiState.changeToOutputStride,
        inputResolution: guiState.inputResolution,
        multiplier: guiState.multiplier,
        quantBytes: guiState.quantBytes
      });
      toggleLoadingUI(false);
      guiState.outputStride = +guiState.changeToOutputStride;
      guiState.changeToOutputStride = null;
    }

    if (guiState.changeToInputResolution) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();
      toggleLoadingUI(true);
      guiState.net = await posenet.load({
        architecture: guiState.architecture,
        outputStride: guiState.outputStride,
        inputResolution: +guiState.changeToInputResolution,
        multiplier: guiState.multiplier,
        quantBytes: guiState.quantBytes
      });
      toggleLoadingUI(false);
      guiState.inputResolution = +guiState.changeToInputResolution;
      guiState.changeToInputResolution = null;
    }

    if (guiState.changeToQuantBytes) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();
      toggleLoadingUI(true);
      guiState.net = await posenet.load({
        architecture: guiState.architecture,
        outputStride: guiState.outputStride,
        inputResolution: guiState.inputResolution,
        multiplier: guiState.multiplier,
        quantBytes: guiState.changeToQuantBytes
      });
      toggleLoadingUI(false);
      guiState.quantBytes = guiState.changeToQuantBytes;
      guiState.changeToQuantBytes = null;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;
    switch (guiState.algorithm) {
      case 'single-pose':
        const pose = await guiState.net.estimatePoses(video, {
          flipHorizontal: flipPoseHorizontal,
          decodingMethod: 'single-person'
        });
        poses = poses.concat(pose);
        minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
        break;
      case 'multi-pose':
        let all_poses = await guiState.net.estimatePoses(video, {
          flipHorizontal: flipPoseHorizontal,
          decodingMethod: 'multi-person',
          maxDetections: guiState.multiPoseDetection.maxPoseDetections,
          scoreThreshold: guiState.multiPoseDetection.minPartConfidence,
          nmsRadius: guiState.multiPoseDetection.nmsRadius
        });

        poses = poses.concat(all_poses);
        minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
        break;
    }

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
    poses.forEach(({score, keypoints}) => {
      if (score >= minPoseConfidence) {
        var poseVector = createPoseVector(keypoints);
        var gtVector = createPoseVector(gt_keypoints)
        var similarityScore = cosineDistanceMatching(poseVector, gtVector);
        if (similarityScore < 0.35) {
          console.log("sim score", poseVector, similarityScore);
          // gt_keypoints = JSON.parse(gtKeypointsList[gtKeypointsIdx]);
          // gtKeypointsIdx = (gtKeypointsIdx + 1) % gtKeypointsList.length;
          // console.log(gtKeypointsIdx);
        }
        drawKeypoints(gt_keypoints, minPartConfidence, ctx);
        drawSkeleton(gt_keypoints, minPartConfidence, ctx);
        drawBoundingBox(gt_keypoints, ctx);
        // debugger;
        if (guiState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showBoundingBox) {
          drawBoundingBox(keypoints, ctx);
        }
      }
    });

    // End monitoring code for frames per second
    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  toggleLoadingUI(true);
  const net = await posenet.load({
    architecture: guiState.input.architecture,
    outputStride: guiState.input.outputStride,
    inputResolution: guiState.input.inputResolution,
    multiplier: guiState.input.multiplier,
    quantBytes: guiState.input.quantBytes
  });
  toggleLoadingUI(false);

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  setupGui([], net);
  setupFPS();
  detectPoseInRealTime(video, net);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
