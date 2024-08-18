"use client";


import { useState } from 'react';

export default function Home() {
  const [inputText, setInputText] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [result, setResult] = useState<{ sentiment: string } | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const formData = new FormData();
    if (inputText) {
      formData.append('text', inputText);
    }
    if (selectedImage) {
      formData.append('image', selectedImage);
    }

    // Send the data to the API
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();
    setResult(data);
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedImage(e.target.files[0]);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Sentiment Analysis</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          className="border p-2 w-full"
          rows={3}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type something to analyze sentiment..."
        />
        <div className="my-4">
          <input type="file" accept="image/*" onChange={handleImageChange} />
        </div>
        <button className="bg-blue-500 text-white py-2 px-4 mt-4" type="submit">
          Analyze
        </button>
      </form>
      {result && (
        <div className="mt-4">
          <h2 className="text-xl font-semibold">Result:</h2>
          <p className="text-lg">{result.sentiment}</p>
        </div>
      )}
    </div>
  );
}
