import { Directive, ElementRef, Input, OnChanges, SimpleChanges, Renderer2 } from '@angular/core';

@Directive({
  selector: '[appHighlightDetection]'
})
export class HighlightDetectionDirective implements OnChanges {
  @Input() appHighlightDetection: any; // Confidence level or detection data
  @Input() threshold = 0.6; // Default confidence threshold
  
  private defaultBorderColor = 'transparent';
  private warningBorderColor = '#FFA500';
  private alertBorderColor = '#FF0000';
  
  constructor(private el: ElementRef, private renderer: Renderer2) {
    this.setBorder(this.defaultBorderColor, 0);
  }
  
  ngOnChanges(changes: SimpleChanges): void {
    if (changes['appHighlightDetection']) {
      this.updateHighlight();
    }
  }
  
  private updateHighlight(): void {
    if (!this.appHighlightDetection) {
      this.setBorder(this.defaultBorderColor, 0);
      return;
    }
    
    const confidence = typeof this.appHighlightDetection === 'number' 
      ? this.appHighlightDetection 
      : this.appHighlightDetection.confidence || 0;
      
    if (confidence > this.threshold) {
      // High confidence - red border with pulse animation
      this.setBorder(this.alertBorderColor, 3);
      this.addPulseAnimation();
    } else if (confidence > this.threshold * 0.7) {
      // Medium confidence - orange border
      this.setBorder(this.warningBorderColor, 2);
      this.removePulseAnimation();
    } else {
      // Low confidence - no highlight
      this.setBorder(this.defaultBorderColor, 0);
      this.removePulseAnimation();
    }
  }
  
  private setBorder(color: string, width: number): void {
    this.renderer.setStyle(this.el.nativeElement, 'border', `${width}px solid ${color}`);
    this.renderer.setStyle(this.el.nativeElement, 'transition', 'border 0.3s ease-in-out');
  }
  
  private addPulseAnimation(): void {
    this.renderer.addClass(this.el.nativeElement, 'pulse-animation');
  }
  
  private removePulseAnimation(): void {
    this.renderer.removeClass(this.el.nativeElement, 'pulse-animation');
  }
}
