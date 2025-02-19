import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar } from '@angular/material/snack-bar';

interface Report {
  id: string;
  title: string;
  date: Date;
}

interface ScheduledReport {
  id: string;
  title: string;
  frequency: string;
  nextRun: Date;
}

@Component({
  selector: 'app-reports',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule],
  templateUrl: './reports.component.html',
  styleUrls: ['../../styling/component/reports.component.scss']
})
export class ReportsComponent {
  reports: Report[] = [
    {
      id: '1',
      title: 'Monthly Security Summary - March 2024',
      date: new Date('2024-03-31'),
    },
    {
      id: '2',
      title: 'Weekly Detection Report',
      date: new Date('2024-03-24'),
    },
    {
      id: '3',
      title: 'Incident Analysis Q1 2024',
      date: new Date('2024-03-15'),
    },
  ];

  scheduledReports: ScheduledReport[] = [
    {
      id: '1',
      title: 'Monthly Security Overview',
      frequency: 'Monthly',
      nextRun: new Date('2024-04-01'),
    },
    {
      id: '2',
      title: 'Weekly Detection Summary',
      frequency: 'Weekly',
      nextRun: new Date('2024-03-31'),
    },
  ];

  constructor(private snackBar: MatSnackBar) {}

  private showSnackbar(message: string) {
    const snackBarRef = this.snackBar.open(message, 'X', {
      duration: 5000,
      horizontalPosition: 'left',
      verticalPosition: 'top',
      panelClass: ['custom-snackbar']
    });

    // Add hide animation class before closing
    setTimeout(() => {
      const snackBarEl = document.querySelector('.custom-snackbar');
      if (snackBarEl) {
        snackBarEl.classList.add('hide');
      }
    }, 4700); // Slightly before duration ends
  }

  generateNewReport() {
    this.showSnackbar('Generating new report...');
  }

  downloadReport(report: Report) {
    this.showSnackbar(`Downloading ${report.title}...`);
  }

  editScheduledReport(report: ScheduledReport) {
    this.showSnackbar(`Editing schedule for ${report.title}`);
  }

  disableScheduledReport(report: ScheduledReport) {
    this.showSnackbar(`Disabling scheduled report: ${report.title}`);
  }
}
