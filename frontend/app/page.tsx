'use client';

import { useState } from 'react';

type Verse = {
  id: string;
  usfm?: string | null;
  book: string;
  book_zh?: string | null;
  chapter: number;
  verse: number;
  text: string; // in the language of the question
  text_en?: string | null;
  text_zh?: string | null;
  translation?: string | null;
  score?: number | null; // cosine similarity between the question and this verse
};

type QueryMeta = {
  cache: 'exact' | 'semantic' | 'miss';
  language: 'en' | 'zh';
  translation?: string | null;
  embedding_cached: boolean;
  semantic_similarity?: number | null;
  timings_ms: Record<string, number>;
};

type QueryResponse = {
  answer: string;
  verses: Verse[];
  meta?: QueryMeta | null;
};

const TRANSLATION_NAMES: Record<string, string> = {
  bsb: 'Berean Standard Bible',
  cuv: '和合本（神版）',
};

// Licensed editions are never stored here; a link to YouVersion lets the
// reader open the same verse in them. 111 = NIV, 46 = 新標點和合本 神版.
function youVersionUrl(v: Verse, lang: 'en' | 'zh'): string | null {
  if (!v.usfm) return null;
  const version = lang === 'zh' ? 46 : 111;
  return `https://www.bible.com/bible/${version}/${v.usfm}.${v.chapter}.${v.verse}`;
}

function describeMeta(meta: QueryMeta): string {
  const ms = Math.round(meta.timings_ms.total ?? 0).toLocaleString();
  if (meta.cache === 'exact') return `⚡ cached · ${ms} ms`;
  if (meta.cache === 'semantic') {
    const sim = meta.semantic_similarity != null ? ` (similarity ${meta.semantic_similarity.toFixed(2)})` : '';
    return `⚡ similar question cached${sim} · ${ms} ms`;
  }
  return `generated · ${ms} ms`;
}

export default function HomePage() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<string | null>(null);
  const [verses, setVerses] = useState<Verse[]>([]);
  const [meta, setMeta] = useState<QueryMeta | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setAnswer(null);
    setVerses([]);
    setMeta(null);
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
      setMeta(data.meta ?? null);
    } catch (err: any) {
      console.error(err);
      setError(err.message ?? 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  const lang: 'en' | 'zh' = meta?.language ?? 'en';
  const translationName = meta?.translation ? TRANSLATION_NAMES[meta.translation] ?? meta.translation : null;

  return (
    <main className="min-h-screen flex flex-col items-center justify-start p-6 gap-6 bg-slate-50">
      <div className="w-full max-w-2xl bg-white shadow-md rounded-xl p-6 border border-slate-200">
        <h1 className="text-2xl font-semibold mb-2">
          Bible Reference AI
        </h1>
        <p className="text-sm text-slate-600 mb-4">
          Ask in English or 中文. Answers are grounded in the Berean Standard Bible and the 和合本（神版）,
          with every verse shown in both.
        </p>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3 mb-4">
          <textarea
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={3}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. What does the Bible say about forgiveness?  /  圣经关于饶恕是怎么说的？"
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
            <div className="flex items-baseline justify-between gap-3 mb-1">
              <h2 className="text-lg font-semibold">Answer</h2>
              {meta && (
                <span
                  className={`text-xs px-2 py-0.5 rounded-full ${
                    meta.cache === 'miss' ? 'bg-slate-100 text-slate-600' : 'bg-emerald-100 text-emerald-700'
                  }`}
                  title={Object.entries(meta.timings_ms)
                    .map(([k, v]) => `${k}: ${Math.round(v)} ms`)
                    .join('\n')}
                >
                  {describeMeta(meta)}
                </span>
              )}
            </div>
            <p className="text-sm leading-relaxed whitespace-pre-line">
              {answer}
            </p>
          </div>
        )}

        {verses.length > 0 && (
          <div>
            <div className="flex items-baseline justify-between gap-3 mb-1">
              <h2 className="text-lg font-semibold">Referenced Verses</h2>
              {translationName && (
                <span className="text-xs text-slate-500">searched in {translationName}</span>
              )}
            </div>
            <ul className="space-y-3">
              {verses.map((v, index) => {
                const primaryBook = lang === 'zh' && v.book_zh ? v.book_zh : v.book;
                const secondaryBook = lang === 'zh' ? v.book : v.book_zh ?? v.book;
                const secondaryText = lang === 'zh' ? v.text_en : v.text_zh;
                const link = youVersionUrl(v, lang);
                return (
                  <li key={v.id || `${v.book}-${v.chapter}-${v.verse}-${index}`} className="text-sm">
                    <div>
                      <span className="font-medium">
                        {primaryBook} {v.chapter}:{v.verse}
                      </span>
                      {': '}
                      <span>{v.text}</span>
                      {v.score != null && (
                        <span className="ml-2 text-xs text-slate-400" title="cosine similarity to your question">
                          {v.score.toFixed(2)}
                        </span>
                      )}
                    </div>
                    {secondaryText && (
                      <div className="text-slate-500 mt-0.5">
                        <span className="font-medium">
                          {secondaryBook} {v.chapter}:{v.verse}
                        </span>
                        {': '}
                        <span>{secondaryText}</span>
                      </div>
                    )}
                    {link && (
                      <a
                        href={link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-blue-600 hover:underline"
                      >
                        {lang === 'zh' ? '在 YouVersion 读新标点和合本 ↗' : 'Read in the NIV on YouVersion ↗'}
                      </a>
                    )}
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
    </main>
  );
}
