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
    {
      id: '4',
      timestamp: Date.now() - 900000,
      threatDetected: true,
      threatType: 'pistol',
      confidence: 0.76,
      location: 'Security Gate A',
      imageUrl: 'assets\\detection2.jpg'
    },
    {
      id: '5',
      timestamp: Date.now() - 1200000,
      threatDetected: true,
      threatType: 'gun',
      confidence: 0.92,
      location: 'Parking Area B',
      imageUrl: 'assets\\detection1.jpg'
    },
    {
      id: '6',
      timestamp: Date.now() - 1800000,
      threatDetected: true,
      threatType: 'knife',
      confidence: 0.82,
      location: 'Food Court',
      imageUrl: 'assets\\detection4.jpg'
    }
  ]);

  getDetections() {
    return this.detections.asObservable();
  }

  addDetection(detection: Detection) {
    const current = this.detections.value;
    this.detections.next([detection, ...current]);
  }
}
