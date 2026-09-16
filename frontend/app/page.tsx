'use client';

import { useState } from 'react';

type Verse = {
  id: string;
  book: string;
  chapter: number;
  verse: number;
  text: string;
};

type QueryResponse = {
  answer: string;
  verses: Verse[];
};

export default function HomePage() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<string | null>(null);
  const [verses, setVerses] = useState<Verse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setAnswer(null);
    setVerses([]);
    const q = question.trim();
    if (!q) {
      setError('Please enter a question.');
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: q }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Request failed: ${res.status} ${text}`);
      }

      const data: QueryResponse = await res.json();
      setAnswer(data.answer);
      setVerses(data.verses);
    } catch (err: any) {
      console.error(err);
      setError(err.message ?? 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen flex flex-col items-center justify-start p-6 gap-6 bg-slate-50">
      <div className="w-full max-w-2xl bg-white shadow-md rounded-xl p-6 border border-slate-200">
        <h1 className="text-2xl font-semibold mb-2">
          Bible Reference AI (MVP)
        </h1>
        <p className="text-sm text-slate-600 mb-4">
          Ask a question and I&apos;ll answer based on the embedded Bible verses.
        </p>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3 mb-4">
          <textarea
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={3}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. What does the Bible say about God creating the world?"
          />
          <button
            type="submit"
            disabled={loading}
            className="self-end px-4 py-2 rounded-md bg-blue-600 text-white text-sm font-medium disabled:bg-slate-400"
          >
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </form>

        {error && (
          <div className="mb-3 text-sm text-red-600">
            {error}
          </div>
        )}

        {answer && (
          <div className="mb-4">
            <h2 className="text-lg font-semibold mb-1">Answer</h2>
            <p className="text-sm leading-relaxed whitespace-pre-line">
              {answer}
            </p>
          </div>
        )}

        {verses.length > 0 && (
          <div>
            <h2 className="text-lg font-semibold mb-1">Referenced Verses</h2>
            <ul className="space-y-2">
              {verses.map((v, index) => (
                <li key={v.id || `${v.book}-${v.chapter}-${v.verse}-${index}`} className="text-sm">
                  <span className="font-medium">
                    {v.book} {v.chapter}:{v.verse}
                  </span>
                  {': '}
                  <span>{v.text}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </main>
  );
}