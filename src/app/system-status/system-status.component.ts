import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-system-status',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="system-status">
      <div class="status-item">
        <span class="status-dot" [class.active]="systemHealth.online"></span>
        System Status: {{systemHealth.status}}
      </div>
      <div class="status-item">
        <span class="material-icons">memory</span>
        CPU Load: {{systemHealth.cpuLoad}}%
      </div>
      <div class="status-item">
        <span class="material-icons">storage</span>
        Storage: {{systemHealth.storage}}%
      </div>
      <div class="status-badge {{systemHealth.performance}}">
        Performance: {{systemHealth.performance}}
      </div>
    </div>
  `,
  styleUrls: ['./system-status.component.scss']
})
export class SystemStatusComponent {
  systemHealth = {
    online: true,
    status: 'Operational',
    cpuLoad: 45,
    storage: 68,
    performance: 'optimal'
  };
}
