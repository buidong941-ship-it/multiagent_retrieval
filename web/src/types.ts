export interface FrameResult {
  frame_id: string;
  video_id: string;
  frame_idx: number;
  timestamp: number;
  frame_path: string;
  score: number;
  source: string;
  metadata?: Record<string, any>;
}

export interface ParsedQuery {
  objects: string[];
  ocr: string[];
  actions: string[];
  attributes: string[];
  translated_query: string;
}

export interface RetrievalResponse {
  query: string;
  parsed_query: ParsedQuery;
  total_results: number;
  results: FrameResult[];
  latency_ms: number;
}

export interface SearchRequest {
  query: string;
  top_k?: number;
  use_temporal?: boolean;
  mode?: string;
}
