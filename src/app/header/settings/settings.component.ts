import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

interface SettingSection {
  icon: string;
  title: string;
  description: string;
  isExpanded: boolean;
}

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [CommonModule, FormsModule, MatButtonModule, MatIconModule],
  templateUrl: './settings.component.html',
  styleUrls: ['../../styling/component/settings.component.scss']
})
export class SettingsComponent {
  sensitivity = 75;
  confidence = 90;
  notificationTypes = ['Email Alerts', 'Push Notifications', 'SMS Alerts', 'Desktop Notifications'];
  nightMode = false;
  realTimeProcessing = true;

  settings: SettingSection[] = [
    {
      icon: 'videocam',
      title: 'Camera Configuration',
      description: 'Manage camera settings, resolution, and frame rate',
      isExpanded: false
    },
    {
      icon: 'shield',
      title: 'Detection Settings',
      description: 'Configure detection sensitivity and algorithms',
      isExpanded: false
    },
    {
      icon: 'notifications',
      title: 'Notifications',
      description: 'Configure alert settings and notification preferences',
      isExpanded: false
    },
    {
      icon: 'group',
      title: 'User Management',
      description: 'Manage user access and permissions',
      isExpanded: false
    }
  ];

  constructor(private snackBar: MatSnackBar) {}

  private showSnackbar(message: string) {
    const snackBarRef = this.snackBar.open(message, 'X', {
      duration: 5000,
      horizontalPosition: 'left',
      verticalPosition: 'top',
      panelClass: ['custom-snackbar']
    });

    setTimeout(() => {
      const snackBarEl = document.querySelector('.custom-snackbar');
      if (snackBarEl) {
        snackBarEl.classList.add('hide');
      }
    }, 4700);
  }

  toggleSection(index: number) {
    this.settings[index].isExpanded = !this.settings[index].isExpanded;
  }

  saveSettings() {
    this.showSnackbar('Settings saved successfully');
  }

  addNewUser() {
    this.showSnackbar('New user added successfully');
  }

  managePermissions() {
    this.showSnackbar('Managing user permissions...');
  }
}
