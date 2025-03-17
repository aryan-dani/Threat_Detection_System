import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, timeout, retry } from 'rxjs/operators';
import { TimeoutError } from 'rxjs';

export interface ThreatDetectionResult {
  threat_detected: boolean;
  threat_type: string | null;
  confidence: number;
  timestamp: number;
  processing_time_ms?: number;
}

@Injectable({
  providedIn: 'root'
})
export class ThreatDetectionService {
  private apiUrl = 'http://localhost:3000/api';
  
  constructor(private http: HttpClient) { }

  /**
   * Detect threats in an image using base64 encoding
   */
  detectThreatFromBase64(base64Image: string): Observable<ThreatDetectionResult> {
    return this.http.post<ThreatDetectionResult>(`${this.apiUrl}/detect-threat`, {
      image: base64Image
    }).pipe(
      timeout(90000), // 90 second timeout for large images
      retry(1),       // Retry once if it fails
      catchError(this.handleError)
    );
  }

  /**
   * Detect threats in an uploaded file
   */
  detectThreatFromFile(file: File): Observable<ThreatDetectionResult> {
    // Validate file size before uploading
    if (file.size > 20 * 1024 * 1024) { // 20MB limit
      return throwError(() => new Error('File size exceeds 20MB limit. Please upload a smaller image.'));
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    return this.http.post<ThreatDetectionResult>(`${this.apiUrl}/detect-threat/upload`, formData).pipe(
      timeout(90000), // 90 second timeout for large images
      retry(1),       // Retry once if it fails
      catchError(this.handleError)
    );
  }
  
  /**
   * Error handler for HTTP requests
   */
  private handleError(error: HttpErrorResponse) {
    let errorMessage = 'Unknown error occurred';
    
    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Error: ${error.error.message}`;
    } else if (error.status === 0) {
      // Network error or server not reachable
      errorMessage = 'Unable to connect to the server. Please check your connection or try again later.';
    } else if (error.status === 504 || error instanceof TimeoutError) {
      // Gateway timeout or request timeout
      errorMessage = 'The request took too long to process. Please try with a smaller image.';
    } else {
      // Server-side error
      errorMessage = `Server returned code ${error.status}, message: ${error.message}`;
      
      // Additional details if available
      if (error.error && error.error.details) {
        errorMessage += ` - ${error.error.details}`;
      }
    }
    
    console.error('Threat detection error:', errorMessage);
    return throwError(() => new Error(errorMessage));
  }
}