import { ResultCard } from './ResultCard';
import type { FrameResult } from '../types';

interface Props {
  results: FrameResult[];
  onPlayVideo?: (videoId: string, timestamp: number) => void;
}

export const ResultsGrid: React.FC<Props> = ({ results, onPlayVideo }) => (
  <div className="results-grid">
    {results.map((r, i) => (
      <ResultCard key={r.frame_id} result={r} rank={i} onPlayVideo={onPlayVideo} />
    ))}
  </div>
);
