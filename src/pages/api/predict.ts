import type { NextApiRequest, NextApiResponse } from 'next';
import formidable, { IncomingForm, Fields, Files, File } from 'formidable';
import axios from 'axios';
import fs from 'fs';

export const config = {
  api: {
    bodyParser: false, // We disable bodyParser because we're handling file uploads
  },
};

const HUGGINGFACE_API_URL = 'https://api-inference.huggingface.co/models/chreh/bert-discrimination-classifier';
const HUGGINGFACE_API_TOKEN = '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'; // You need to get this from Hugging Face

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

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).end(); // Method Not Allowed
  }

  const form = new IncomingForm();

  form.parse(req, async (err: any, fields: Fields, files: Files) => {
    if (err) {
      return res.status(500).json({ error: 'Failed to parse form data' });
    }

    const { text } = fields as { text?: string };
    let sentiment;

    if (text) {
      sentiment = await analyzeText(text);
    } else if (files && files.image) {
      const imageFile = files.image as unknown as File;
      sentiment = await analyzeImage(imageFile);
    }

    if (!sentiment) {
      return res.status(400).json({ error: 'No valid input provided' });
    }

    res.status(200).json({ sentiment });
  });
}

// Function to analyze text using Hugging Face model
async function analyzeText(text: string): Promise<string> {
  try {
    const response = await axios.post(
      HUGGINGFACE_API_URL,
      {
        inputs: text,
      },
      {
        headers: {
          Authorization: `Bearer ${HUGGINGFACE_API_TOKEN}`,
          'Content-Type': 'application/json',
        },
      }
    );

    const sentiment = response.data[0].label; // Get the sentiment label from the model response
    return sentiment;
  } catch (error) {
    console.error('Error analyzing text:', error);
    return 'Error';
  }
}

// Function to analyze an image using Hugging Face model
async function analyzeImage(imageFile: File): Promise<string> {
  try {
    const imageBuffer = await readFile(imageFile);

    const response = await axios.post(
      HUGGINGFACE_API_URL,
      imageBuffer,
      {
        headers: {
          Authorization: `Bearer ${HUGGINGFACE_API_TOKEN}`,
          'Content-Type': 'application/octet-stream',
        },
      }
    );

    const sentiment = response.data[0].label; // Get the sentiment label from the model response
    return sentiment;
  } catch (error) {
    console.error('Error analyzing image:', error);
    return 'Error';
  }
}
