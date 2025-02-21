export interface Detection {
  id: string;
  timestamp: Date;
  confidence: number;
  type: 'weapon';
  location: string;
  imageUrl?: string;
}

export interface VideoFeed {
  id: string;
  name: string;
  status: 'active' | 'inactive';
  url: string;
  poster?: string;
}
