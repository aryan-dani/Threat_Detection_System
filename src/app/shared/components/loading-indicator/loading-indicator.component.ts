import { Component, OnInit } from '@angular/core';
import { LoadingService } from '../../../core/services/loading.service';
import { Observable, of } from 'rxjs';
import { trigger, state, style, transition, animate } from '@angular/animations';

@Component({
  selector: 'app-loading-indicator',
  template: `
    <div class="loading-overlay" *ngIf="isLoading$ | async" [@fadeInOut]>
      <mat-progress-spinner 
        color="accent" 
        mode="indeterminate" 
        diameter="50">
      </mat-progress-spinner>
      <p class="loading-text">Processing...</p>
    </div>
  `,
  styles: [`
    .loading-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.5);
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      z-index: 9999;
    }
    .loading-text {
      color: white;
      margin-top: 15px;
      font-weight: 500;
    }
  `],
  animations: [
    trigger('fadeInOut', [
      state('void', style({
        opacity: 0
      })),
      transition('void <=> *', animate(300))
    ])
  ]
})
export class LoadingIndicatorComponent implements OnInit {
  isLoading$: Observable<boolean> = of(false);

  constructor(private loadingService: LoadingService) { }

  ngOnInit(): void {
    this.isLoading$ = this.loadingService.getLoading();
  }
}
