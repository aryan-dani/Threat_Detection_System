import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Observable, timer, Subject, EMPTY, of, OperatorFunction } from 'rxjs';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { HttpClient } from '@angular/common/http';
import { catchError, tap, retryWhen, delayWhen, takeUntil, retry, take, timeout } from 'rxjs/operators';

export interface CPUInfo {
  model: string;
  speed: number;
  usage: string;
  times: {
    user: number;
    nice: number;
    sys: number;
    idle: number;
    irq: number;
  };
}

export interface SystemMetrics {
  online: boolean;
  status: string;
  cpuLoad: string;
  storage: string;
  memory: string;
  networkLatency: number;
  networkInterfaces: any;
  uptime: number;
  platform: string;
  arch: string;
  totalMemory: string;
  freeMemory: string;
  cpuInfo: any[];
  hostname: string;
  activeProcesses: number;
  lastChecked: Date;
}

interface MetricsError {
  message: string;
  code: string;
  timestamp: Date;
}

interface HealthStatus {
  status: 'healthy' | 'warning' | 'critical';
  message: string;
  lastUpdated: Date;
}

@Injectable({
  providedIn: 'root'
})
export class SystemMonitorService implements OnDestroy {
  private readonly API_ENDPOINT = 'http://localhost:3000';
  private readonly WS_ENDPOINT = 'ws://localhost:3000';
  private socket: WebSocket | null = null;
  private metricsSubject = new BehaviorSubject<SystemMetrics | null>(null);
  private destroy$ = new Subject<void>();
  private reconnectAttempts = 0;
  private readonly MAX_RECONNECT_ATTEMPTS = 5;
  private healthStatus = new BehaviorSubject<HealthStatus>({
    status: 'healthy',
    message: 'System is operating normally',
    lastUpdated: new Date()
  });

  constructor(private http: HttpClient) {
    this.initializeConnection();
  }

  private initializeConnection() {
    console.log('Initializing connection...');
    this.fetchMetricsHttp().pipe(
      take(1),
      catchError(error => {
        console.error('HTTP connection failed:', error);
        return EMPTY;
      })
    ).subscribe({
      next: (metrics) => {
        if (metrics) {
          this.metricsSubject.next(metrics);
          this.updateHealthStatus(metrics);
          this.connect(); // Only try WebSocket after HTTP success
        }
      },
      error: () => this.handleError(new Error('Failed to connect to server'))
    });
  }

  private fetchMetricsHttp() {
    return this.http.get<SystemMetrics>(`${this.API_ENDPOINT}/api/system-metrics`).pipe(
      retry({ count: 3, delay: 1000 }),
      catchError(error => {
        if (error.status === 0) {
          console.error('Backend service is not running');
          this.healthStatus.next({
            status: 'critical',
            message: 'Backend monitoring service is not running',
            lastUpdated: new Date()
          });
        }
        throw error;
      })
    );
  }

  private connect() {
    try {
      if (this.socket) {
        this.socket.close();
      }

      console.log('Attempting WebSocket connection...');
      this.socket = new WebSocket(this.WS_ENDPOINT);

      this.socket.onopen = () => {
        console.log('WebSocket connected successfully');
        this.reconnectAttempts = 0;
      };

      this.socket.onmessage = (event) => {
        try {
          const metrics = JSON.parse(event.data) as SystemMetrics;
          this.metricsSubject.next(metrics);
          this.updateHealthStatus(metrics);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.socket.onclose = () => {
        console.log('WebSocket connection closed');
        this.handleDisconnect();
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.handleError(error);
      };

    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
      this.handleError(error);
    }
  }

  getMetrics(): Observable<SystemMetrics | null> {
    // If no data is available, trigger a fetch
    if (!this.metricsSubject.value) {
      this.refreshMetrics();
    }
    return this.metricsSubject.asObservable();
  }

  getHealthStatus(): Observable<HealthStatus> {
    return this.healthStatus.asObservable();
  }

  refreshMetrics(): Observable<SystemMetrics | null> {
    // For simplicity, always fetch metrics using HTTP call, relying on the backend updates via WebSocket separately
    return this.fetchMetricsHttp().pipe(
      timeout(5000),
      retry({
        count: 2,
        delay: 1000,
        resetOnSuccess: true
      }) as OperatorFunction<unknown, SystemMetrics | null>,
      catchError(error => {
        console.error('Error fetching metrics:', error);
        return of(null);
      })
    );
  }

  private handleDisconnect() {
    if (this.reconnectAttempts < this.MAX_RECONNECT_ATTEMPTS) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      
      setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.MAX_RECONNECT_ATTEMPTS})...`);
        this.connect();
      }, delay);
    } else {
      this.healthStatus.next({
        status: 'critical',
        message: 'Connection lost to monitoring service',
        lastUpdated: new Date()
      });
    }
  }

  private updateHealthStatus(metrics: SystemMetrics) {
    const cpuLoad = parseFloat(metrics.cpuLoad);
    const memoryUsage = parseFloat(metrics.memory);
    const storageUsage = parseFloat(metrics.storage);

    let status: 'healthy' | 'warning' | 'critical' = 'healthy';
    let message = 'System is operating normally';

    if (cpuLoad > 85 || memoryUsage > 90 || storageUsage > 90) {
      status = 'critical';
      message = 'Critical system resources detected';
    } else if (cpuLoad > 70 || memoryUsage > 75 || storageUsage > 80) {
      status = 'warning';
      message = 'System resources are under heavy load';
    }

    this.healthStatus.next({
      status,
      message,
      lastUpdated: new Date()
    });
  }

  private handleError(error: any) {
    const metricError: MetricsError = {
      message: error.message || 'Unknown error occurred',
      code: error.status || '500',
      timestamp: new Date()
    };
    console.error('System monitoring error:', metricError);
  }

  reconnect() {
    this.disconnect();
    this.reconnectAttempts = 0;
    this.connect();
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
    this.disconnect();
  }
}