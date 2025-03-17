export interface Detection {
  id?: string;
  timestamp: string | number;
  imageUrl?: string;
  threatDetected: boolean;
  threatType?: string | null;
  confidence: number;
  source?: string;
  location?: string;
}

export interface VideoFeed {
  id: string;
  name: string;
  status: 'active' | 'inactive';
  url: string;
  poster?: string;
}
