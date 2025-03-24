import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LivefeedComponent } from '../../livefeed/livefeed.component';
import { DetectionLogComponent } from '../../detection-log/detection-log.component';
import { UploadFileComponent } from '../../upload-file/upload-file.component';
import { DataService } from '../../services/data.service';

interface VideoFeed {
  id: string;
  name: string;
  status: 'active' | 'inactive';
  url: string;
}

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule, 
    LivefeedComponent, 
    DetectionLogComponent, 
    UploadFileComponent
  ],
  templateUrl: './dashboard.component.html',
  styleUrls: ['../../styling/component/dashboard.component.scss']
})
export class DashboardComponent implements OnInit {
  totalDetections: number = 0;
  activeAlerts: number = 0;
  systemHealth: number = 98;
  detectionTrend: number = 15;
  alertTrend: number = -5;

  // Performance metrics
  cpuUsage: number = 45;
  memoryUsage: number = 60;
  storageUsage: number = 32;

  recentActivities = [
    {
      title: 'High-risk threat detected',
      time: new Date(),
      icon: 'warning',
      color: '#ef4444'
    },
    {
      title: 'System scan completed',
      time: new Date(Date.now() - 1800000),
      icon: 'task_alt',
      color: '#22c55e'
    },
    {
      title: 'Database backup',
      time: new Date(Date.now() - 3600000),
      icon: 'backup',
      color: '#3498db'
    },
    {
      title: 'Configuration updated',
      time: new Date(Date.now() - 7200000),
      icon: 'settings',
      color: '#f59e0b'
    }
  ];

  constructor(private dataService: DataService) {}

  ngOnInit() {
    this.loadDashboardData();
    this.startPerformanceMonitoring();
  }

  private loadDashboardData() {
    this.dataService.getDetections().subscribe(detections => {
      this.totalDetections = detections.length;
      this.activeAlerts = detections.filter(d => d.confidence > 0.8).length;
    });
  }

  private startPerformanceMonitoring() {
    // Simulate real-time updates
    setInterval(() => {
      this.cpuUsage = Math.min(100, this.cpuUsage + (Math.random() * 10 - 5));
      this.memoryUsage = Math.min(100, this.memoryUsage + (Math.random() * 8 - 4));
      this.storageUsage = Math.min(100, this.storageUsage + (Math.random() * 2 - 1));
    }, 5000);
  }
}
