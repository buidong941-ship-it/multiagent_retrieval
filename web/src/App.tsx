import { useState } from 'react';
import axios from 'axios';
import { Search, Loader2, AlertCircle, Clock, Film, LayoutGrid, FileText, Package, Cpu, Zap, Type } from 'lucide-react';
import { ResultsGrid } from './components/ResultsGrid';
import { VideoModal } from './components/VideoModal';
import type { RetrievalResponse } from './types';
import './App.css';

const MODES = [
  { id: 'agent',      icon: '🧠', label: 'Agent (ReAct)',    badge: 'AI',   isAgent: true },
  { id: 'fusion',     icon: '🔮', label: 'Auto Fusion',      badge: null,   isAgent: false },
  { id: 'siglip_jina_sequence',icon: '🔥', label: 'SigLIP + Jina Sequence (PRF)', badge: null,  isAgent: false },
  { id: 'siglip_jina_parallel',icon: '⚡', label: 'SigLIP + Jina Parallel (RRF)', badge: 'New',  isAgent: false },
  { id: 'clip',       icon: '🖼️', label: 'CLIP Semantic',    badge: null,   isAgent: false },
  { id: 'action',     icon: '🏃', label: 'Action Search',    badge: null,   isAgent: false },
  { id: 'ocr_bm25',   icon: '🤖', label: 'OCR + BM25',       badge: null,   isAgent: false },
  { id: 'direct_ocr', icon: '🔤', label: 'Direct OCR',       badge: null,   isAgent: false },
  { id: 'object',     icon: '📦', label: 'YOLO Objects',     badge: null,   isAgent: false },
];

function App() {
  const [query, setQuery] = useState('');
  const [query2, setQuery2] = useState('');
  const [topK, setTopK] = useState(100);
  const [mode, setMode] = useState('agent');
  const [resultsData, setResultsData] = useState<RetrievalResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeVideo, setActiveVideo] = useState<{ videoId: string; timestamp: number } | null>(null);

  const activeMode = MODES.find(m => m.id === mode)!;

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    const finalQuery = (mode === 'siglip_jina_sequence' || mode === 'siglip_jina_parallel') && query2.trim() ? `${query.trim()} || ${query2.trim()}` : query.trim();
    if (!finalQuery || isLoading) return;
    setIsLoading(true);
    setError(null);
    setResultsData(null);
    try {
      const response = await axios.post<RetrievalResponse>('http://localhost:8000/api/v1/retrieve', {
        query: finalQuery,
        top_k: topK,
        mode,
      });
      setResultsData(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Đã xảy ra lỗi khi tìm kiếm.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-layout">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1 className="sidebar-logo-title">AIC 2026</h1>
          <p className="sidebar-logo-sub">Video Retrieval System</p>
        </div>

        <div className="sidebar-section-label">Chế độ tìm kiếm</div>

        <nav className="sidebar-modes">
          {MODES.map(m => (
            <button
              key={m.id}
              className={`mode-item ${mode === m.id ? 'active' : ''} ${m.isAgent && mode === m.id ? 'agent-mode' : ''}`}
              onClick={() => setMode(m.id)}
            >
              <span className="mode-icon">{m.icon}</span>
              <span className="mode-label">{m.label}</span>
              {m.badge && <span className="mode-badge">{m.badge}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-status">
            <div className="status-dot" />
            <span>Backend connected</span>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="main-wrapper">
        {/* Topbar */}
        <header className="topbar">
          <p className="topbar-title">Tìm kiếm khung hình video</p>
          <span className="topbar-mode-badge">
            {activeMode.icon} {activeMode.label}
          </span>
        </header>

        <main className="page-content">
          {/* Search Box */}
          <div className="search-area animate-fade-in">
            <form className="search-box" onSubmit={handleSearch}>
              <div className="search-inputs-container">
                <div className="search-input-row">
                  <div className="search-icon-wrap">
                    <Search size={18} strokeWidth={2.5} />
                  </div>
                  <input
                    type="text"
                    className="search-input"
                    placeholder={(mode === 'siglip_jina_sequence' || mode === 'siglip_jina_parallel') ? `Hành động 1 (VD: người đàn ông đi vào)…` : `Tìm kiếm video (VD: người đàn ông mặc áo đỏ chạy trên đường)…`}
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    autoFocus
                  />
                </div>
                
                {(mode === 'siglip_jina_sequence' || mode === 'siglip_jina_parallel') && (
                  <>
                    <div className="search-divider">
                      <span className="search-divider-text">SAU ĐÓ</span>
                    </div>
                    <div className="search-input-row">
                      <div className="search-icon-wrap">
                        <Search size={18} strokeWidth={2.5} />
                      </div>
                      <input
                        type="text"
                        className="search-input"
                        placeholder="Hành động 2 (VD: người đàn ông ngồi xuống)..."
                        value={query2}
                        onChange={e => setQuery2(e.target.value)}
                      />
                    </div>
                  </>
                )}
              </div>
              <div className="search-controls">
                <div className="topk-group">
                  <span className="topk-label">Top K</span>
                  <select
                    className="topk-select"
                    value={topK}
                    onChange={e => setTopK(Number(e.target.value))}
                  >
                    {[50, 100, 150, 200, 250, 300].map(v => (
                      <option key={v} value={v}>{v}</option>
                    ))}
                  </select>
                </div>
                <button
                  type="submit"
                  className="search-btn"
                  disabled={isLoading || !query.trim()}
                >
                  {isLoading
                    ? <><Loader2 size={15} className="spinner-icon" /> Đang tìm…</>
                    : <><Search size={15} /> Tìm kiếm</>
                  }
                </button>
              </div>
            </form>
          </div>

          {/* Loading shimmer */}
          {isLoading && <div className="loading-bar" />}

          {/* Error */}
          {error && (
            <div className="error-banner animate-fade-in">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {/* Results */}
          {resultsData && !isLoading && (
            <div className="animate-fade-in">
              <div className="results-header">
                <h2 className="results-title">Kết quả tìm kiếm</h2>
                <div className="results-meta">
                  <span className="results-count-pill">{resultsData.total_results} khung hình</span>
                  <span className="latency-pill">⚡ {resultsData.latency_ms}ms</span>
                </div>
              </div>

              {/* Query chips */}
              <div className="chips-row">
                {resultsData.parsed_query.objects?.map((o, i) => (
                  <span key={`o-${i}`} className="chip chip-obj">📦 {o}</span>
                ))}
                {resultsData.parsed_query.actions?.map((a, i) => (
                  <span key={`a-${i}`} className="chip chip-act">🏃 {a}</span>
                ))}
                {resultsData.parsed_query.ocr?.map((t, i) => (
                  <span key={`r-${i}`} className="chip chip-ocr">🔤 {t}</span>
                ))}
              </div>

              <ResultsGrid
                results={resultsData.results}
                onPlayVideo={(videoId, timestamp) => setActiveVideo({ videoId, timestamp })}
              />
            </div>
          )}

          {/* Empty state */}
          {!resultsData && !isLoading && !error && (
            <div className="empty-state animate-fade-in">
              <div className="empty-icon-ring">
                <Film size={40} color="var(--accent)" />
              </div>
              <h3>Nhập câu truy vấn để bắt đầu</h3>
              <p>
                Hệ thống kết hợp <strong>AI Agent</strong>, tìm kiếm ngữ nghĩa (CLIP),
                nhận diện vật thể (YOLO) và trích xuất chữ viết (OCR)
                để tìm ra khung hình chính xác nhất trong hàng nghìn video.
              </p>
            </div>
          )}
        </main>
      </div>

      {/* Video Modal */}
      {activeVideo && (
        <VideoModal
          videoId={activeVideo.videoId}
          timestamp={activeVideo.timestamp}
          onClose={() => setActiveVideo(null)}
        />
      )}
    </div>
  );
}

export default App;
