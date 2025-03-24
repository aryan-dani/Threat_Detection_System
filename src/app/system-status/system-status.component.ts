import { Component, OnInit, OnDestroy, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { HttpClientModule } from '@angular/common/http';
import { SystemMonitorService } from '../services/system-monitor.service';
import { Subscription, catchError, of, Subject } from 'rxjs';
import { FormsModule } from '@angular/forms';
import { trigger, transition, style, animate, state, query, stagger } from '@angular/animations';
import { takeUntil } from 'rxjs/operators';

interface ActivityLogEntry {
  type: 'error' | 'warning' | 'info';
  severity: 'critical' | 'moderate' | 'low';
  message: string;
  timestamp: Date;
  source?: string;
}

interface SystemMetrics {
  cpuLoad: string;
  memory: string;
  storage: string;
  uptime: number;
  networkLatency: number;
  status: string;
  online: boolean;
  lastChecked: Date;
}

interface MetricItem {
  label: string;
  value: string;
  icon: string;
  status: string;
  trend?: 'up' | 'down';
  chartData?: any;
}

@Component({
  selector: 'app-system-status',
  standalone: true,
  imports: [CommonModule, MatIconModule, HttpClientModule, FormsModule],
  providers: [SystemMonitorService],
  templateUrl: './system-status.component.html',
  styleUrl: '../styling/component/system-status.component.scss',
  animations: [
    trigger('fadeInOut', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(-10px)' }),
        animate('300ms ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
      ]),
      transition(':leave', [
        animate('200ms ease-in', style({ opacity: 0, transform: 'translateY(10px)' }))
      ])
    ]),
    trigger('listAnimation', [
      transition('* => *', [
        query(':enter', [
          style({ opacity: 0, transform: 'translateY(-15px)' }),
          stagger(60, [
            animate('300ms ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
          ])
        ], { optional: true })
      ])
    ]),
    trigger('cardState', [
      state('normal', style({
        transform: 'perspective(1000px) rotateX(0) rotateY(0) translateZ(0)',
        transition: 'all 0.3s ease'
      })),
      state('hover', style({
        transform: 'perspective(1000px) rotateX({{rotateX}}deg) rotateY({{rotateY}}deg) translateZ(10px)',
        transition: 'all 0.3s ease'
      }), { params: { rotateX: 0, rotateY: 0 } })
    ])
  ]
})
export class SystemStatusComponent implements OnInit, OnDestroy {
  systemHealth: SystemMetrics | null = null;
  private metricsSubscription?: Subscription;
  private lastLogTimes: Map<string, number> = new Map();
  private readonly destroy$ = new Subject<void>();

  systemMetrics: MetricItem[] = [];
  activityLog: ActivityLogEntry[] = [];
  selectedSeverity: 'all' | 'critical' | 'moderate' | 'low' = 'all';
  severityOptions: Array<'all' | 'critical' | 'moderate' | 'low'> = ['all', 'critical', 'moderate', 'low'];

  private errorRetryCount = 0;
  private readonly MAX_RETRY_ATTEMPTS = 3;
  isError = false;
  errorMessage = '';

  // New property to throttle system health checks
  private lastSystemHealthCheck: number = 0;

  constructor(
    private systemMonitorService: SystemMonitorService,
    private ngZone: NgZone
  ) {}

  ngOnInit() {
    // Subscribe to metrics updates from the monitoring service
    this.metricsSubscription = this.systemMonitorService.getMetrics().subscribe((metrics) => {
      if (metrics) {
        this.updateSystemMetrics(metrics);
        // Throttle system health check to run every 1 minute
        if (Date.now() - this.lastSystemHealthCheck >= 60000) {
          this.checkSystemHealth(metrics);
          this.lastSystemHealthCheck = Date.now();
        }
      }
    });
    this.setupMouseEffects();
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
    
    if (this.metricsSubscription) {
      this.metricsSubscription.unsubscribe();
    }
    this.systemMonitorService.disconnect();
    
    // Clean up event listeners with proper type casting
    document.querySelectorAll('.metric-card').forEach(card => {
      const element = card as HTMLElement;
      element.removeEventListener('mousemove', (e: Event) => {
        this.handleMouseMove(e as MouseEvent);
      });
      element.removeEventListener('mouseleave', (e: Event) => {
        this.handleMouseLeave(e as MouseEvent);
      });
    });
  }

  refreshStatus() {
    this.isError = false;
    this.systemMonitorService.refreshMetrics().pipe(
      catchError(error => {
        this.handleError(error);
        return of(null);
      })
    ).subscribe(metrics => {
      if (metrics) {
        this.updateSystemMetrics(metrics);
        this.checkSystemHealth(metrics);
      }
    });
  }

  refreshLogs() {
    // Clear cooldown times to allow immediate logging
    this.lastLogTimes.clear();
    this.checkSystemHealth(this.systemHealth!);
  }

  getOverallHealthStatus(): string {
    if (!this.systemHealth) return 'unknown';
    
    const cpuLoad = parseFloat(this.systemHealth.cpuLoad);
    const memoryUsage = parseFloat(this.systemHealth.memory);
    const storageUsage = parseFloat(this.systemHealth.storage);
    
    if (cpuLoad > 85 || memoryUsage > 90 || storageUsage > 90) {
      return 'critical';
    } else if (cpuLoad > 70 || memoryUsage > 75 || storageUsage > 80) {
      return 'warning';
    }
    return 'healthy';
  }

  private setupMouseEffects() {
    requestAnimationFrame(() => {
      document.querySelectorAll('.metric-card').forEach(card => {
        (card as HTMLElement).addEventListener('mousemove', (e: Event) => {
          this.handleMouseMove(e as MouseEvent);
        });
        (card as HTMLElement).addEventListener('mouseleave', (e: Event) => {
          this.handleMouseLeave(e as MouseEvent);
        });
      });
    });
  }

  private handleMouseMove = (e: MouseEvent) => {
    const card = e.currentTarget as HTMLElement;
    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    const rotateX = ((y - centerY) / centerY) * 10;
    const rotateY = ((centerX - x) / centerX) * 10;
    
    card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`;
  };

  private handleMouseLeave = (e: MouseEvent) => {
    const card = e.currentTarget as HTMLElement;
    card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateZ(0)';
  };

  private updateSystemMetrics(metrics: SystemMetrics) {
    this.systemHealth = metrics;
    const memoryUsage = parseFloat(metrics.memory).toFixed(1);
    const cpuLoad = parseFloat(metrics.cpuLoad).toFixed(1);
    // Remove storage and use network latency
    this.systemMetrics = [
      {
        label: 'CPU Load',
        value: `${cpuLoad}%`,
        icon: 'memory',
        status: this.getStatusClass(+cpuLoad),
        trend: this.determineTrend(+cpuLoad, +this.systemMetrics[0]?.value?.replace('%', '') || 0)
      },
      {
        label: 'Memory Usage',
        value: `${memoryUsage}%`,
        icon: 'storage',
        status: this.getStatusClass(+memoryUsage),
        trend: this.determineTrend(+memoryUsage, +this.systemMetrics[1]?.value?.replace('%', '') || 0)
      },
      {
        label: 'Network Latency',
        value: `${metrics.networkLatency}ms`,
        icon: 'network_check',
        status: this.getNetworkStatus(metrics.networkLatency)
      },
      {
        label: 'Uptime',
        value: this.formatUptime(metrics.uptime),
        icon: 'timer',
        status: 'info'
      }
    ];
  }

  private determineTrend(currentValue: number, previousValue: number): 'up' | 'down' | undefined {
    if (!previousValue) return undefined;
    const difference = currentValue - previousValue;
    if (Math.abs(difference) < 0.1) return undefined;
    return difference > 0 ? 'up' : 'down';
  }

  private checkSystemHealth(metrics: SystemMetrics) {
    if (!metrics) return;

    const cpuLoad = parseFloat(metrics.cpuLoad);
    const memUsage = parseFloat(metrics.memory);
    const storageUsage = parseFloat(metrics.storage);
    const latency = metrics.networkLatency;
    
    // CPU monitoring with threshold checks
    if (cpuLoad > 85) {
      this.addActivityLog('error', `Critical CPU usage: ${cpuLoad.toFixed(1)}%`, 'critical', 'CPU Monitor');
    } else if (cpuLoad > 70) {
      this.addActivityLog('warning', `High CPU usage: ${cpuLoad.toFixed(1)}%`, 'moderate', 'CPU Monitor');
    }
    
    // Memory monitoring with predictive warning
    const memoryTrend = this.calculateTrend('memory', memUsage);
    if (memUsage > 90) {
      this.addActivityLog('error', `Critical memory usage: ${memUsage.toFixed(1)}%`, 'critical', 'Memory Monitor');
    } else if (memUsage > 75 || (memUsage > 65 && memoryTrend === 'up')) {
      this.addActivityLog('warning', `High memory usage: ${memUsage.toFixed(1)}%`, 'moderate', 'Memory Monitor');
    }
    
    // Storage monitoring with capacity prediction
    if (storageUsage > 90) {
      this.addActivityLog('error', `Storage space critical: ${storageUsage.toFixed(1)}%`, 'critical', 'Storage Monitor');
    } else if (storageUsage > 80) {
      const daysUntilFull = this.predictDaysUntilFull(storageUsage);
      this.addActivityLog('warning', 
        `Storage space high: ${storageUsage.toFixed(1)}% (Est. ${daysUntilFull} days until full)`, 
        'moderate', 
        'Storage Monitor'
      );
    }

    // Network latency monitoring with adaptive thresholds
    if (latency > 200) {
      this.addActivityLog('error', `High network latency: ${latency}ms`, 'critical', 'Network Monitor');
    } else if (latency > 100) {
      this.addActivityLog('warning', `Network latency elevated: ${latency}ms`, 'moderate', 'Network Monitor');
    }

    // System status changes
    if (!metrics.online && this.systemHealth?.online) {
      this.addActivityLog('error', 'System went offline', 'critical', 'System Monitor');
    } else if (metrics.online && !this.systemHealth?.online) {
      this.addActivityLog('info', 'System back online', 'low', 'System Monitor');
    }
  }

  private storageUsageHistory: number[] = [];
  private readonly STORAGE_HISTORY_SIZE = 24; // Keep 24 hours of data

  private predictDaysUntilFull(currentUsage: number): number {
    this.storageUsageHistory.push(currentUsage);
    if (this.storageUsageHistory.length > this.STORAGE_HISTORY_SIZE) {
      this.storageUsageHistory.shift();
    }

    if (this.storageUsageHistory.length < 2) return 0;

    const usageRate = (this.storageUsageHistory[this.storageUsageHistory.length - 1] - 
                      this.storageUsageHistory[0]) / this.storageUsageHistory.length;
    
    if (usageRate <= 0) return 999; // No growth or negative growth
    
    const remainingPercentage = 100 - currentUsage;
    return Math.round(remainingPercentage / (usageRate * 24)); // Convert to days
  }

  private metricsHistory: Map<string, number[]> = new Map();
  
  private calculateTrend(metric: string, currentValue: number): 'up' | 'down' | 'stable' {
    const history = this.metricsHistory.get(metric) || [];
    history.push(currentValue);
    
    if (history.length > 10) history.shift(); // Keep last 10 readings
    this.metricsHistory.set(metric, history);
    
    if (history.length < 3) return 'stable';
    
    const recentAvg = history.slice(-3).reduce((a, b) => a + b, 0) / 3;
    const oldAvg = history.slice(0, -3).reduce((a, b) => a + b, 0) / (history.length - 3);
    
    const difference = recentAvg - oldAvg;
    if (Math.abs(difference) < 0.5) return 'stable';
    return difference > 0 ? 'up' : 'down';
  }

  private shouldLogWarning(type: string, severity: string): boolean {
    const key = `${type}-${severity}`;
    const lastTime = this.lastLogTimes.get(key) || 0;
    const currentTime = Date.now();
    const timeDiff = currentTime - lastTime;
    
    // Different cooldown periods based on severity
    const cooldown = severity === 'critical' ? 60000 : // 1 minute for critical
                    severity === 'moderate' ? 180000 : // 3 minutes for moderate
                    300000; // 5 minutes for low

    if (timeDiff >= cooldown) {
      this.lastLogTimes.set(key, currentTime);
      return true;
    }
    return false;
  }

  private addActivityLog(type: 'error' | 'warning' | 'info', message: string, severity: 'critical' | 'moderate' | 'low', source?: string) {
    if (this.shouldLogWarning(message, severity)) {
      this.ngZone.run(() => {
        const newEntry: ActivityLogEntry = {
          type,
          message,
          severity,
          timestamp: new Date(),
          source
        };
        
        this.activityLog.unshift(newEntry);
        
        if (this.activityLog.length > 100) {
          this.activityLog.pop();
        }

        // Removed auto highlight animation after log addition
      });
    }
  }

  get filteredActivityLog(): ActivityLogEntry[] {
    if (this.selectedSeverity === 'all') {
      return this.activityLog;
    }
    return this.activityLog.filter(entry => entry.severity === this.selectedSeverity);
  }

  getLogCountBySeverity(severity: string): number {
    return this.activityLog.filter(entry => entry.severity === severity).length;
  }

  public formatUptime(uptime: number): string {
    const days = Math.floor(uptime / 86400);
    const hours = Math.floor((uptime % 86400) / 3600);
    const minutes = Math.floor((uptime % 3600) / 60);
    
    const parts = [];
    if (days > 0) parts.push(`${days}d`);
    if (hours > 0 || days > 0) parts.push(`${hours}h`);
    parts.push(`${minutes}m`);
    
    return parts.join(' ');
  }

  getStatusClass(value: number): string {
    if (value < 50) return 'healthy';
    if (value < 75) return 'warning';
    return 'critical';
  }

  getActivityIcon(type: string): string {
    switch (type) {
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      default:
        return 'info';
    }
  }

  getSeverityIcon(severity: string): string {
    switch (severity) {
      case 'critical':
        return 'error';
      case 'moderate':
        return 'warning';
      case 'low':
        return 'info';
      default:
        return 'all_inbox';
    }
  }

  getSafeCpuLoad(): number {
    return this.systemHealth?.cpuLoad ? parseFloat(this.systemHealth.cpuLoad) : 0;
  }

  // Error handling and recovery
  private handleError(error: any) {
    console.error('System monitoring error:', error);
    
    if (++this.errorRetryCount <= this.MAX_RETRY_ATTEMPTS) {
      this.addActivityLog('warning', 
        `Connection issue, retrying (${this.errorRetryCount}/${this.MAX_RETRY_ATTEMPTS})...`, 
        'moderate', 
        'System Monitor'
      );
      
      // Exponential backoff retry
      setTimeout(() => this.refreshStatus(), Math.pow(2, this.errorRetryCount) * 1000);
    } else {
      this.addActivityLog('error', 
        'Failed to connect to monitoring service after multiple attempts', 
        'critical', 
        'System Monitor'
      );
      this.errorRetryCount = 0;
    }
  }

  // New helper method for network status
  getNetworkStatus(latency: number): string {
    if (latency < 100) return 'healthy';
    if (latency < 200) return 'moderate';
    return 'critical';
  }
}
