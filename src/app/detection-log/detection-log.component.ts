import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Detection } from '../interfaces/detection.interface';
import { DataService } from '../services/data.service';

@Component({
  selector: 'app-detection-log',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './detection-log.component.html',
  styleUrls: ['../styling/component/detection-log.component.scss']
})
export class DetectionLogComponent implements OnInit {
  detections: Detection[] = [];
  isFilterOpen = false;
  confidenceThreshold = 0;
  sortBy: 'time' | 'confidence' = 'time';
  expanded: string[] = [];

  constructor(private dataService: DataService) {}

  ngOnInit() {
    this.dataService.getDetections().subscribe(
      detections => this.detections = detections
    );
  }

  toggleFilter() {
    this.isFilterOpen = !this.isFilterOpen;
  }

  toggleExpanded(id: string | undefined) {
    if (!id) return; // Skip if id is undefined
    
    const index = this.expanded.indexOf(id);
    if (index === -1) {
      this.expanded.push(id);
    } else {
      this.expanded.splice(index, 1);
    }
  }

  isExpanded(id: string | undefined): boolean {
    return id ? this.expanded.includes(id) : false;
  }

  get filteredDetections() {
    return this.detections
      .filter(d => d.confidence >= this.confidenceThreshold)
      .sort((a, b) => {
        if (this.sortBy === 'time') {
          // Handle timestamp as a number directly
          const aTime = typeof a.timestamp === 'number' ? a.timestamp : Date.parse(String(a.timestamp));
          const bTime = typeof b.timestamp === 'number' ? b.timestamp : Date.parse(String(b.timestamp));
          return bTime - aTime;
        }
        return b.confidence - a.confidence;
      });
  }
}
