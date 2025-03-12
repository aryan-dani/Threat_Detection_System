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
      imageUrl: 'assets\\detection1.jpg'
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 300000),
      confidence: 0.89,
      type: 'weapon',
      location: 'Terminal 2',
      imageUrl: 'assets\\detection2.jpg'
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 300000),
      confidence: 0.89,
      type: 'weapon',
      location: 'Terminal 3',
      imageUrl: 'assets\\detection2.jpg'
    },
  ]);

  getDetections() {
    return this.detections.asObservable();
  }

  addDetection(detection: Detection) {
    const current = this.detections.value;
    this.detections.next([detection, ...current]);
  }
}
