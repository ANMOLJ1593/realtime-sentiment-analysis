import { useState, useCallback, useMemo, useRef } from "react";
import "./App.css";

interface SentimentData {
  label: string;
  score: number;
  time_taken?: number;
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<SentimentData | null>(null);
  const [loading, setLoading] = useState(false);

  const apiUrl = import.meta.env.VITE_API_URL;

  // 🔹 Debounce timer reference
  const debounceRef = useRef<number | null>(null);

  const analyzeSentiment = useCallback(
    async (inputText: string) => {
      const trimmed = inputText.trim();
      if (!trimmed) {
        setResult(null);
        return;
      }

      setLoading(true);
      try {
        const res = await fetch(`${apiUrl}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: trimmed }),
        });

        if (!res.ok) {
          throw new Error(`HTTP error ${res.status}`);
        }

        const data = await res.json();

        // HF returns array → normalize
        if (Array.isArray(data)) {
          setResult({
            label: data[0].label,
            score: data[0].score,
          });
        } else {
          setResult(data);
        }
      } catch (err) {
        console.error("Prediction failed:", err);
        setResult(null);
      } finally {
        setLoading(false);
      }
    },
    [apiUrl]
  );

  // ✅ Debounced input handler
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setText(val);

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = window.setTimeout(() => {
      analyzeSentiment(val);
    }, 800); // wait 800ms after typing stops
  };

  const sentimentColor = useMemo(() => {
    if (!result) return "";
    const l = result.label.toLowerCase();
    if (l === "positive") return "from-green-400 to-emerald-500";
    if (l === "negative") return "from-red-400 to-rose-500";
    return "from-yellow-400 to-amber-500";
  }, [result]);

  return (
    <main className="min-h-screen flex items-center justify-center bg-slate-950 font-sans">
      <div className="w-full max-w-2xl mx-auto p-4 sm:p-6 space-y-8">

        {/* Header */}
        <header className="text-center">
          <h1 className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-500 to-pink-500">
            Sentiment Lens
          </h1>
          <p className="text-slate-400 mt-2 text-sm sm:text-base">
            See the emotion behind every word, instantly.
          </p>
        </header>

        {/* Input */}
        <div className="relative">
          <textarea
            value={text}
            onChange={handleChange}
            placeholder="Start typing to analyze sentiment…"
            rows={5}
            className="w-full rounded-2xl bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 p-4 pr-12 text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-indigo-500 resize-none transition"
          />
          <span className="absolute bottom-4 right-4 text-xs text-slate-500">
            {text.length}
          </span>
        </div>

        {/* Loader */}
        {loading && (
          <div className="animate-pulse space-y-3">
            <div className="h-10 rounded-xl bg-slate-800/50" />
            <div className="h-10 rounded-xl bg-slate-800/50" />
            <div className="h-10 rounded-xl bg-slate-800/50" />
          </div>
        )}

        {/* Result */}
        {result && (
          <section className="space-y-5">
            <h2 className="text-center text-xl font-medium text-slate-200">
              Analysis
            </h2>

            <div className="flex items-center justify-between rounded-2xl bg-slate-800/40 p-4 border border-slate-700/50">
              <span className="text-slate-300 font-medium">Sentiment</span>
              <span className={`px-3 py-1 rounded-full text-sm font-semibold bg-gradient-to-r text-white ${sentimentColor}`}>
                {result.label}
              </span>
            </div>

            <div className="rounded-2xl bg-slate-800/40 p-4 border border-slate-700/50">
              <div className="flex items-center justify-between mb-2">
                <span className="text-slate-300 font-medium">Confidence</span>
                <span className="font-mono text-indigo-300">
                  {(result.score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-700/50 rounded-full h-1.5">
                <div
                  className="bg-gradient-to-r from-indigo-500 to-purple-500 h-1.5 rounded-full"
                  style={{ width: `${result.score * 100}%` }}
                />
              </div>
            </div>
          </section>
        )}

        {!loading && !result && text.trim() === "" && (
          <div className="text-center text-slate-500 pt-8">
            <p className="text-sm">Write something above to begin.</p>
          </div>
        )}

      </div>
    </main>
  );
}



// import { useState, useCallback, useMemo } from "react";
// import "./App.css";

// interface SentimentData {
//   label: string;
//   score: number;
//   time_taken: number;
// }

// // export default function App() {
// //   const [text, setText] = useState("");
// //   const [result, setResult] = useState<SentimentData | null>(null);
// //   const [loading, setLoading] = useState(false);
// //   const apiUrl = "http://localhost:8000";
// export default function App() {
//   const [text, setText] = useState("");
//   const [result, setResult] = useState<SentimentData | null>(null);
//   const [loading, setLoading] = useState(false);

//   const apiUrl = import.meta.env.VITE_API_URL;


//   const analyzeSentiment = useCallback(
//     async (inputText: string) => {
//       const trimmed = inputText.trim();
//       if (!trimmed) {
//         setResult(null);
//         return;
//       }
//       setLoading(true);
//       try {
//         const res = await fetch(`${apiUrl}/predict`, {
//           method: "POST",
//           headers: { "Content-Type": "application/json" },
//           body: JSON.stringify({ text: trimmed }),
//         });
//         if (!res.ok) throw new Error("Network error");
//         const data: SentimentData = await res.json();
//         setResult(data);
//       } catch {
//         setResult(null);
//       } finally {
//         setLoading(false);
//       }
//     },
//     [apiUrl]
//   );

//   const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
//     const val = e.target.value;
//     setText(val);
//     analyzeSentiment(val);
//   };

//   const sentimentColor = useMemo(() => {
//     if (!result) return "";
//     const l = result.label.toLowerCase();
//     if (l === "positive") return "from-green-400 to-emerald-500";
//     if (l === "negative") return "from-red-400 to-rose-500";
//     return "from-yellow-400 to-amber-500";
//   }, [result]);

//   return (
//     <main className="min-h-screen flex items-center justify-center bg-slate-950 font-sans">
//       <div className="w-full max-w-2xl mx-auto p-4 sm:p-6 space-y-8">
//         {/* Header */}
//         <header className="text-center">
//           <h1 className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-500 to-pink-500">
//             Sentiment Lens
//           </h1>
//           <p className="text-slate-400 mt-2 text-sm sm:text-base">
//             See the emotion behind every word, instantly.
//           </p>
//         </header>

//         {/* Input */}
//         <div className="relative">
//           <textarea
//             value={text}
//             onChange={handleChange}
//             placeholder="Start typing to analyze sentiment…"
//             rows={5}
//             className="w-full rounded-2xl bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 p-4 pr-12 text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none transition"
//           />
//           <span className="absolute bottom-4 right-4 text-xs text-slate-500">
//             {text.length}
//           </span>
//         </div>

//         {/* Skeleton / Loader */}
//         {loading && (
//           <div className="animate-pulse space-y-3">
//             <div className="h-10 rounded-xl bg-slate-800/50" />
//             <div className="h-10 rounded-xl bg-slate-800/50" />
//             <div className="h-10 rounded-xl bg-slate-800/50" />
//           </div>
//         )}

//         {/* Results */}
//         {result && (
//           <section className="space-y-5">
//             <h2 className="text-center text-xl font-medium text-slate-200">
//               Analysis
//             </h2>

//             {/* Card: Sentiment */}
//             <div className="flex items-center justify-between rounded-2xl bg-slate-800/40 backdrop-blur-sm p-4 border border-slate-700/50">
//               <span className="text-slate-300 font-medium">Sentiment</span>
//               <span
//                 className={`px-3 py-1 rounded-full text-sm font-semibold bg-gradient-to-r text-white ${sentimentColor}`}
//               >
//                 {result.label}
//               </span>
//             </div>

//             {/* Card: Confidence */}
//             <div className="rounded-2xl bg-slate-800/40 backdrop-blur-sm p-4 border border-slate-700/50">
//               <div className="flex items-center justify-between mb-2">
//                 <span className="text-slate-300 font-medium">Confidence</span>
//                 <span className="font-mono text-indigo-300">
//                   {(result.score * 100).toFixed(1)}%
//                 </span>
//               </div>
//               <div className="w-full bg-slate-700/50 rounded-full h-1.5">
//                 <div
//                   className="bg-gradient-to-r from-indigo-500 to-purple-500 h-1.5 rounded-full transition-all duration-500"
//                   style={{ width: `${result.score * 100}%` }}
//                 />
//               </div>
//             </div>

//             {/* Card: Latency */}
//             <div className="flex items-center justify-between rounded-2xl bg-slate-800/40 backdrop-blur-sm p-4 border border-slate-700/50">
//               <span className="text-slate-300 font-medium">Processing time</span>
//               <span className="font-mono text-purple-300">
//                 {result.time_taken} ms
//               </span>
//             </div>
//           </section>
//         )}

//         {/* Empty State */}
//         {!loading && !result && text.trim() === "" && (
//           <div className="flex flex-col items-center text-center text-slate-500 pt-8">
//             <EmptyIcon className="w-24 h-24 mb-4 text-slate-700" />
//             <p className="text-sm">Write something above to begin.</p>
//           </div>
//         )}
//       </div>
//     </main>
//   );
// }

// /* Simple SVG illustration */
// const EmptyIcon = (props: React.SVGProps<SVGSVGElement>) => (
//   <svg
//     viewBox="0 0 24 24"
//     fill="none"
//     stroke="currentColor"
//     strokeWidth="1"
//     strokeLinecap="round"
//     strokeLinejoin="round"
//     {...props}
//   >
//     <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" />
//     <path d="M8 12h8M12 8v8" />
//   </svg>
// );