"use client";

import { SetStateAction, useState } from "react";
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"


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
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl font-bold">Sentiment Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit}>
            <Textarea
              className="mb-4"
              rows={3}
              value={inputText}
              onChange={(e: { target: { value: SetStateAction<string>; }; }) => setInputText(e.target.value)}
              placeholder="Type something to analyze sentiment..."
            />
            <div className="mb-4">
              <Input type="file" accept="image/*" onChange={handleImageChange} />
            </div>
            <Button type="submit">Analyze</Button>
          </form>

          {result && (
            <div className="mt-4">
              <h2 className="text-xl font-semibold">Result:</h2>
              <p className="text-lg">{result.sentiment}</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
