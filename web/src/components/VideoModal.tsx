import { X } from 'lucide-react';

interface Props {
  videoId: string;
  timestamp: number;
  onClose: () => void;
}

export const VideoModal: React.FC<Props> = ({ videoId, timestamp, onClose }) => {
  const videoUrl = `http://localhost:8000/api/v1/video/${videoId}#t=${Math.floor(timestamp)}`;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-box" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">{videoId}</h2>
          <button className="modal-close-btn" onClick={onClose} aria-label="Đóng">
            <X size={16} />
          </button>
        </div>
        <div className="modal-video-wrap">
          <video
            src={videoUrl}
            controls
            autoPlay
            style={{ width: '100%', height: '100%', display: 'block' }}
          />
        </div>
      </div>
    </div>
  );
};
