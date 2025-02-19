import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-system-status',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="status-bar">
      <div class="status-item">
        <span class="status-dot" [class.active]="systemActive"></span>
        System Status: {{systemActive ? 'Active' : 'Inactive'}}
      </div>
      <div class="status-item">
        <span class="material-icons">wifi</span>
        Network: {{networkStrength}}%
      </div>
      <div class="status-item">
        <span class="material-icons">memory</span>
        Processing Load: {{processingLoad}}%
      </div>
    </div>
  `,
  styles: [`
    .status-bar {
      display: flex;
      gap: 2rem;
      padding: 0.75rem;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 0.5rem;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      margin-bottom: 1rem;
    }
    // ... additional styles ...
  `]
})
export class SystemStatusComponent {
  systemActive = true;
  networkStrength = 95;
  processingLoad = 45;
}
