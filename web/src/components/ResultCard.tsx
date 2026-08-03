import { PlayCircle, Clock, ArrowRight } from 'lucide-react';
import type { FrameResult } from '../types';

interface Props {
  result: FrameResult;
  rank: number;
  onPlayVideo?: (videoId: string, timestamp: number) => void;
}

export const ResultCard: React.FC<Props> = ({ result, rank, onPlayVideo }) => {
  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  const pct = (result.score * 100).toFixed(1);
  const isPair = result.source === 'siglip_jina_pair';
  const imgUrl = `http://localhost:8000/api/v1/frame/${result.frame_id}`;

  const renderImage = (url: string, alt: string) => (
    <img
      src={url}
      alt={alt}
      className="card-img"
      loading="lazy"
      onError={e => {
        (e.target as HTMLImageElement).src =
          'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNkMWQ1ZGIiIHN0cm9rZS13aWR0aD0iMS41Ij48cmVjdCB4PSIzIiB5PSIzIiB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHJ4PSIyIi8+PGNpcmNsZSBjeD0iOC41IiBjeT0iOC41IiByPSIxLjUiLz48cGF0aCBkPSJtMjEgMTUtNS01LTExIDExIi8+PC9zdmc+';
      }}
    />
  );

  return (
    <div
      className="result-card animate-fade-in"
      style={{ animationDelay: `${Math.min(rank * 0.04, 0.6)}s` }}
    >
      <div className="card-img-wrap">
        {isPair && result.metadata?.pair_left && result.metadata?.pair_right ? (
          <div className="pair-img-container">
            <div className="pair-img-half">
              {renderImage(`http://localhost:8000/api/v1/frame/${result.metadata.pair_left}`, `Frame ${result.metadata.pair_left}`)}
            </div>
            <div className="pair-connector">
              <div className="connector-line"></div>
              <ArrowRight size={18} className="connector-icon" />
            </div>
            <div className="pair-img-half">
              {renderImage(`http://localhost:8000/api/v1/frame/${result.metadata.pair_right}`, `Frame ${result.metadata.pair_right}`)}
            </div>
          </div>
        ) : (
          renderImage(imgUrl, `Frame ${result.frame_idx} — ${result.video_id}`)
        )}
        
        <div className="card-overlay">
          <div className="card-overlay-top">
            <span className="rank-badge">#{rank + 1}</span>
            <span className="score-badge">{pct}%</span>
          </div>
          <button
            className="play-btn"
            title="Xem video tại vị trí này"
            onClick={e => { e.stopPropagation(); onPlayVideo?.(result.video_id, result.timestamp); }}
          >
            <PlayCircle size={22} />
          </button>
        </div>
      </div>

      <div className="card-body">
        <h3 className="card-video-name" title={result.video_id}>{result.video_id}</h3>
        <div className="card-meta-row">
          <div className="card-meta-item">
            <Clock size={12} />
            <span>{fmt(result.timestamp)}</span>
          </div>
          <span className="source-tag">{isPair ? 'Temporal Pair' : (result.source || 'clip')}</span>
        </div>
      </div>
    </div>
  );
};
