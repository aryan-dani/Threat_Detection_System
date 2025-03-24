import { Injectable } from '@angular/core';
import { Observable, BehaviorSubject } from 'rxjs';
import { io, Socket } from 'socket.io-client';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class WebSocketService {
  private socket: Socket;
  private connectionStatus = new BehaviorSubject<boolean>(false);

  constructor() {
    this.socket = io(environment.websocketUrl);
    
    this.socket.on('connect', () => {
      this.connectionStatus.next(true);
      console.log('WebSocket connected');
    });
    
    this.socket.on('disconnect', () => {
      this.connectionStatus.next(false);
      console.log('WebSocket disconnected');
    });
    
    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.connectionStatus.next(false);
    });
  }

  // Get connection status as observable
  public getConnectionStatus(): Observable<boolean> {
    return this.connectionStatus.asObservable();
  }

  // Listen to specific events from the server
  public on<T>(event: string): Observable<T> {
    return new Observable<T>(observer => {
      this.socket.on(event, (data: T) => {
        observer.next(data);
      });
      
      return () => {
        this.socket.off(event);
      };
    });
  }

  // Emit events to the server
  public emit(event: string, data: any): void {
    this.socket.emit(event, data);
  }

  // Subscribe to real-time detections
  public getDetections(): Observable<any> {
    return this.on<any>('detection');
  }

  // Subscribe to alerts
  public getAlerts(): Observable<any> {
    return this.on<any>('alert');
  }

  // Disconnect socket on service destroy
  public disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
    }
  }
}
