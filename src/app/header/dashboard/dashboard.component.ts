import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LivefeedComponent } from '../../livefeed/livefeed.component';
import { DetectionLogComponent } from '../../detection-log/detection-log.component';
import { UploadFileComponent } from '../../upload-file/upload-file.component';

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
export class DashboardComponent {
}
