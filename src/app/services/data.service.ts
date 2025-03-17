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
      timestamp: Date.now(),
      threatDetected: true,
      threatType: 'gun',
      confidence: 0.88,
      location: 'Terminal 1',
      imageUrl: 'assets\\detection1.jpg'
    },
    {
      id: '2',
      timestamp: Date.now() - 300000,
      threatDetected: true,
      threatType: 'knife',
      confidence: 0.89,
      location: 'Terminal 2',
      imageUrl: 'assets\\detection4.jpg'
    },
    {
      id: '3',
      timestamp: Date.now() - 600000,
      threatDetected: true,
      threatType: 'gun',
      confidence: 0.97,
      location: 'Terminal 3',
      imageUrl: 'assets\\detection3.jpg'
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
