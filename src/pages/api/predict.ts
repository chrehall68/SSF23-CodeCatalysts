import * as tf from '@tensorflow/tfjs-node';
import type { NextApiRequest, NextApiResponse } from 'next';
import formidable, { IncomingForm, Fields, Files, File } from 'formidable';
import fs from 'fs';

export const config = {
  api: {
    bodyParser: false,
  },
};

let model: tf.LayersModel | null = null;

const loadModel = async () => {
  if (!model) {
    model = await tf.loadLayersModel('file://path-to-your-model/model.h5');
  }
};

const readFile = (file: File): Promise<Buffer> =>
  new Promise((resolve, reject) => {
    fs.readFile(file.filepath, (err, data) => {
      if (err) {
        reject(err);
      } else {
        resolve(data);
      }
    });
  });

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  await loadModel(); // Ensure the model is loaded

  if (req.method !== 'POST') {
    return res.status(405).end(); // Method Not Allowed
  }

  const form = new IncomingForm();

  form.parse(req, async (err: any, fields: Fields, files: Files) => {
    if (err) {
      return res.status(500).json({ error: 'Failed to parse form data' });
    }

    const { text } = fields as { text?: string };
    const sentiment = await processInput(text, files);

    res.status(200).json({ sentiment });
  });
}

async function processInput(text?: string, files?: Files): Promise<string> {
  let sentiment = 'Neutral'; // Default sentiment

  if (text) {
    // Example: Tokenize and process text input
    const tokenizedText = tokenizeText(text); // Implement this based on your model
    const inputTensor = tf.tensor([tokenizedText]);
    const prediction = model!.predict(inputTensor) as tf.Tensor;
    sentiment = prediction.dataSync()[0] > 0.5 ? 'Positive' : 'Negative';
  }

  if (files && files.image) {
    const imageFile = files.image as unknown as File;
    const imageBuffer = await readFile(imageFile);

    // Load and preprocess image (example: resizing and normalizing)
    const imageTensor = tf.node.decodeImage(imageBuffer, 3)
      .resizeNearestNeighbor([224, 224]) // Resize to the input size expected by the model
      .toFloat()
      .div(tf.scalar(255.0)) // Normalize to [0, 1]
      .expandDims();

    const imagePrediction = model!.predict(imageTensor) as tf.Tensor;
    const imageSentiment = imagePrediction.dataSync()[0];
    sentiment = imageSentiment > 0.5 ? 'Positive' : 'Negative';
  }

  return sentiment;
}

function tokenizeText(text: string) {
  // Implement tokenization logic based on your model’s requirements
  return [/* tokenized text data */];
}
