const express = require("express");
const multer = require("multer");
const AWS = require("aws-sdk");
const path = require("path");
const canvas = require("canvas");
const faceapi = require("face-api.js");
const cors = require("cors");
const { Readable } = require("stream");
require('dotenv').config();

// Monkey-patch face-api.js to use canvas in Node.js
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();

// Middleware
app.use(cors({ 
    origin: "*", 
    methods: "GET,POST,PUT,DELETE", 
    allowedHeaders: "Content-Type, Authorization" 
}));

app.use(bodyParser.json({ limit: "50mb" }));  // Parse JSON bodies (increased size)
app.use(bodyParser.urlencoded({ extended: true, limit: "50mb" }));  // Parse URL-encoded bodies

// AWS S3 Configuration
// const s3 = new AWS.S3({
//   accessKeyId: process.env.ACCESS_KEY_ID,
//   secretAccessKey: process.env.SECRET_ACCESS_KEY,
//   region: process.env.REGION,
// });
// const bucketName = "sazs-attendance";

const s3 = new AWS.S3({
  accessKeyId: "AKIAXYKJTMARMYSGNCMD",
  secretAccessKey: "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",
  region: "ap-south-1",
});
const bucketName = "sazs-attendance";

// Multer setup for in-memory uploads
const upload = multer({ storage: multer.memoryStorage() });

// Initialize face-api.js models
const initializeModels = async () => {
  try {
    const modelPath = path.join(__dirname, "models");
    console.log("Loading models from:", modelPath);
    await faceapi.nets.tinyFaceDetector.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
    console.log("Models loaded successfully");
  } catch (error) {
    console.error("Error loading models:", error.message);
    throw error;
  }
};

// Load labeled face descriptors from S3
const loadLabeledDescriptors = async () => {
  console.log("calls");
  
  try {
    // List all files under the "labels/" directory
    const files = await s3
      .listObjectsV2({ Bucket: bucketName, Prefix: "labels" })
      .promise();
      
      console.log("Found label files:", files);
    if (!files.Contents || files.Contents.length === 0) {
      console.log("No label files found in S3.");
      return []; // Return an empty array if no labels are found
    }


    return Promise.all(
      files.Contents.map(async (file) => {
        const label = file.Key.split("/")[1]; // Extract label name from file key
        const descriptors = [];

        try {
          const imageStream = s3.getObject({ Bucket: bucketName, Key: file.Key }).createReadStream();
          console.log("imageStream",imageStream);
          
          const buffer = await streamToBuffer(imageStream);
          const img = await canvas.loadImage(buffer);
          console.log("img",img);
          
          const detection = await faceapi
            .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceDescriptor();
          if (detection) descriptors.push(detection.descriptor);
        } catch (err) {
          console.error(`Error processing file ${file.Key}:`, err.message);
        }

        return new faceapi.LabeledFaceDescriptors(label, descriptors);
      })
    );
  } catch (error) {
    console.error("Error loading labeled descriptors from S3:", error.message);
    return [];
  }
};


// Utility to convert stream to buffer
const streamToBuffer = (stream) =>
  new Promise((resolve, reject) => {
    const chunks = [];
    stream.on("data", (chunk) => chunks.push(chunk));
    stream.on("end", () => resolve(Buffer.concat(chunks)));
    stream.on("error", (err) => reject(err));
  });

let faceMatcher;

// Initialize face-api.js and load descriptors
initializeModels()
  .then(loadLabeledDescriptors)
  .then((labeledDescriptors) => {
    faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);
    console.log("Face matcher initialized");
  })
  .catch((error) => {
    console.error("Error initializing face matcher:", error.message);
  });

// Upload labeled images to S3 and reload descriptors
app.post("/loadimages", upload.single("image"), async (req, res) => {
  try {
    const { name } = req.body;

    if (!name || !req.file) {
      return res.status(400).json({ success: false, error: "Name and image are required" });
    }

    // Upload the image to S3
    const fileName = `${Date.now()}-${req.file.originalname}`;
    const s3Key = `labels/${name}/${fileName}`;
    await s3
      .upload({
        Bucket: bucketName,
        Key: s3Key,
        Body: req.file.buffer,
        ContentType: req.file.mimetype,
      })
      .promise();

    console.log(`Uploaded file to S3: ${s3Key}`);

    // Reload labeled descriptors
    const labeledDescriptors = await loadLabeledDescriptors();
    faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);
    console.log("Labeled descriptors reloaded successfully");

    res.json({
      success: true,
      message: "Image uploaded and labeled descriptors reloaded successfully",
      fileKey: s3Key,
    });
  } catch (error) {
    console.error("Error uploading labeled image to S3:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Verify face using images from S3
app.post("/api/verify-face", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: "No file uploaded" });
    }

    // Load and process the uploaded image
    const img = await canvas.loadImage(req.file.buffer).catch((err) => {
      throw new Error("Error loading image: " + err.message);
    });

    const detections = await faceapi
      .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();

    // Compare the descriptors with the labeled descriptors
    const results = detections.map((detection) => {
      const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
      return bestMatch.toString(); // e.g., "John (distance: 0.45)" or "unknown"
    });

    res.json({ success: true, matches: results });
  } catch (error) {
    console.error("Error verifying face:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Start the server
const PORT = 3004;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

app.get("/", (req, res) => {
  res.send("Hello, Vercel!");
});
