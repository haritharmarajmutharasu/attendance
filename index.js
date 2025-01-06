const express = require("express");
const fs = require("fs");
const multer = require("multer");
const path = require("path");
const canvas = require("canvas");
const faceapi = require("face-api.js");
const cors = require('cors');

// Monkey-patch face-api.js to use canvas in Node.js
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });


// Initialize face-api.js models

const app = express();
app.use(express.json());
app.use(cors());

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

// Dynamically load labeled face descriptors from folder structure
const loadLabeledDescriptors = async () => {
  const labelsDir = path.join(__dirname, "labels");
  const labelFolders = fs.readdirSync(labelsDir).filter((folder) => {
    return fs.statSync(path.join(labelsDir, folder)).isDirectory();
  });

  console.log("Found labels:", labelFolders);

  return Promise.all(
    labelFolders.map(async (label) => {
      const descriptors = [];
      const dir = path.join(labelsDir, label); // Path to label's folder

      const files = fs.readdirSync(dir).filter((file) => file.endsWith(".jpg") || file.endsWith(".png") || file.endsWith(".jpeg"));
      for (const file of files) {
        try {
          const img = await canvas.loadImage(path.join(dir, file));
          const detection = await faceapi
            .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceDescriptor();
          if (detection) descriptors.push(detection.descriptor);
        } catch (err) {
          console.error(`Error processing file ${file}:`, err.message);
        }
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptors);
    })
  );
};

// Multer setup for file uploads
const upload = multer({ dest: "uploads/" });
const uploadLabeledImage = multer({ dest: "temp/" }); // Temporary directory for labeled images

// Express app setup



let faceMatcher; // To store the face matcher after loading descriptors

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

// Endpoint to verify face
app.post("/api/verify-face", upload.single("image"), async (req, res) => {
  try {
    const imagePath = req.file.path;

    // Validate image file exists
    if (!fs.existsSync(imagePath)) {
      return res.status(400).json({ success: false, error: "File not found" });
    }

    // Load and process the uploaded image
    const img = await canvas.loadImage(imagePath).catch((err) => {
      throw new Error("Error loading image: " + err.message);
    });

    const detections = await faceapi
      .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();

    // Clean up uploaded file
    fs.unlinkSync(imagePath);

    // Compare the descriptors with the labeled descriptors
    const results = detections.map((detection) => {
      const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
      return bestMatch.toString(); // e.g., "HTR (distance: 0.45)" or "unknown"
    });

    res.json({ success: true, matches: results });
  } catch (error) {
    console.error("Error processing the face verification:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Endpoint to upload labeled images and reload descriptors
app.post("/api/load-labeled-descriptors", uploadLabeledImage.single("image"), async (req, res) => {
  try {
    const { name } = req.body;

    console.log("req.body",req.body);
    
    // Validate request body
    if (!name || !req.file) {
      return res.status(400).json({ success: false, error: "Name and image are required" });
    }

    // Create a folder for the given name if it doesn't exist
    const labelDir = path.join(__dirname, "labels", name);
    if (!fs.existsSync(labelDir)) {
      fs.mkdirSync(labelDir, { recursive: true });
      console.log(`Created directory: ${labelDir}`);
    }

    // Move the uploaded file to the created folder
    const ext = path.extname(req.file.originalname); // Preserve file extension
    const newFilePath = path.join(labelDir, `${Date.now()}${ext}`); // Use a timestamp to ensure unique filenames
    fs.renameSync(req.file.path, newFilePath);

    console.log(`File saved to: ${newFilePath}`);

    // Reload the labeled descriptors to include the newly added data
    const labeledDescriptors = await loadLabeledDescriptors();
    faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);
    console.log("Labeled descriptors reloaded successfully");

    res.json({
      success: true,
      message: "Image uploaded and labeled descriptors reloaded successfully",
      folder: labelDir,
      file: newFilePath,
    });
  } catch (error) {
    console.error("Error processing labeled image upload:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

app.get("/", (req, res) => {
  res.send("Hello, Vercel!");
});