import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-upload-file',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './upload-file.component.html',
  styleUrls: ['../styling/component/upload-file.component.scss']
})
export class UploadFileComponent {
  isDragging = false;

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
    console.log('Files dropped:', files);
  }

  handleFileSelect(event: Event) {
    const files = Array.from((event.target as HTMLInputElement).files || []);
    console.log('Files selected:', files);
  }
}
