import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { Detection, VideoFeed } from '../interfaces/detection.interface';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private detections = new BehaviorSubject<Detection[]>([
    {
      id: '1',
      timestamp: new Date(),
      confidence: 0.98,
      type: 'weapon',
      location: 'Terminal 1',
      imageUrl: './../../assets/detection1.jpg'
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 300000),
      confidence: 0.89,
      type: 'weapon',
      location: 'Terminal 2',
      imageUrl: './../../assets/detection2.jpg'
    }
  ]);

  private videoFeeds = new BehaviorSubject<VideoFeed[]>([
    {
      id: '1',
      name: 'Terminal 1 Security',
      status: 'active',
      url: './../../assets/cam1.jpg',
      poster: './../../assets/cam1-poster.jpg'
    },
    {
      id: '2',
      name: 'Terminal 2 Security',
      status: 'active',
      url: './../../assets/cam2.jpg',
      poster: './../../assets/cam2-poster.jpg'
    },
    {
      id: '3',
      name: 'Main Entrance',
      status: 'active',
      url: './../../assets/cam3.jpg',
      poster: './../../assets/cam3-poster.jpg'
    }
  ]);

  getDetections() {
    return this.detections.asObservable();
  }

  getVideoFeeds() {
    return this.videoFeeds.asObservable();
  }

  addDetection(detection: Detection) {
    const current = this.detections.value;
    this.detections.next([detection, ...current]);
  }
}
