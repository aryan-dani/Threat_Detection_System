import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ThreatDetectionService, ThreatDetectionResult } from '../services/threat-detection.service';
import { HttpClientModule } from '@angular/common/http';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MatSnackBarModule } from '@angular/material/snack-bar';
import { finalize } from 'rxjs/operators';

@Component({
  selector: 'app-upload-file',
  standalone: true,
  imports: [CommonModule, HttpClientModule, MatSnackBarModule],
  templateUrl: './upload-file.component.html',
  styleUrls: ['../styling/component/upload-file.component.scss']
})
export class UploadFileComponent {
  isDragging = false;
  isProcessing = false;
  detectionResult: ThreatDetectionResult | null = null;
  imagePreview: string | null = null;
  error: string | null = null;
  
  // Max file size in bytes (10MB)
  private readonly MAX_FILE_SIZE = 10 * 1024 * 1024;

  constructor(
    private threatDetectionService: ThreatDetectionService,
    private snackBar: MatSnackBar
  ) {}

  handleDragOver(event: DragEvent) {
    event.preventDefault();
    this.isDragging = true;
  }

  handleDragLeave() {
    this.isDragging = false;
  }

  handleDrop(event: DragEvent) {
    event.preventDefault();
    this.isDragging = false;
    const files = Array.from(event.dataTransfer?.files || []);
    if (files.length > 0) {
      this.processFile(files[0]);
    }
  }

  handleFileSelect(event: Event) {
    const files = Array.from((event.target as HTMLInputElement).files || []);
    if (files.length > 0) {
      this.processFile(files[0]);
    }
  }

  processFile(file: File) {
    // Reset previous results
    this.detectionResult = null;
    this.error = null;
    
    // Check if file is an image
    if (!file.type.startsWith('image/')) {
      this.snackBar.open('Please upload an image file', 'Close', { duration: 3000 });
      return;
    }
    
    // Check file size
    if (file.size > this.MAX_FILE_SIZE) {
      this.snackBar.open(
        `File too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum size is ${this.MAX_FILE_SIZE / (1024 * 1024)} MB.`, 
        'Close', 
        { duration: 5000 }
      );
      return;
    }

    this.isProcessing = true;
    
    // Create image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      this.imagePreview = e.target?.result as string;
    };
    reader.onerror = () => {
      this.snackBar.open('Error reading file', 'Close', { duration: 3000 });
      this.isProcessing = false;
    };
    reader.readAsDataURL(file);

    // Process with threat detection service
    this.threatDetectionService.detectThreatFromFile(file)
      .pipe(
        finalize(() => {
          this.isProcessing = false;
        })
      )
      .subscribe({
        next: (result) => {
          this.detectionResult = result;
          
          // Show notification based on result
          if (result.threat_detected) {
            this.snackBar.open(
              `Threat detected: ${result.threat_type} (${(result.confidence * 100).toFixed(1)}% confidence)`, 
              'Close', 
              { duration: 5000, panelClass: 'threat-alert' }
            );
          } else {
            this.snackBar.open('No threats detected', 'Close', { duration: 3000 });
          }
        },
        error: (error) => {
          console.error('Error processing image:', error);
          this.error = error.message || 'An error occurred while processing the image';
          // Fix: Ensure this.error is not null when passed to snackBar.open()
          const errorMessage = this.error || 'An error occurred while processing the image';
          this.snackBar.open(errorMessage, 'Close', { duration: 6000, panelClass: 'error-alert' });
        }
      });
  }
}
