import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar } from '@angular/material/snack-bar';

interface Feed {
  id: number;
  name: string;
  status: string;
  imageUrl: string;
}

@Component({
  selector: 'app-livefeed',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule],
  templateUrl: './livefeed.component.html',
  styleUrl: '../styling/component/livefeed.component.scss'
})
export class LivefeedComponent implements OnInit {
  allCamerasActive = true;
  emergencyClickCount = 0;
  feeds: Feed[] = [
    {
      id: 1,
      name: 'Terminal 1 Security',
      status: 'Inactive',
      imageUrl: 'assets\\camera1.jpg'
    },
    {
      id: 2,
      name: 'Terminal 2 Security',
      status: 'Active',
      imageUrl: 'assets\\camera2.jpeg'
    },
    {
      id: 3,
      name: 'Main Entrance',
      status: 'Active',
      imageUrl: 'assets\\premium_photo-1661963237186-6ad19c923d24.avif'
    },
    {
      id: 4,
      name: 'Parking Area',
      status: 'Active',
      imageUrl: 'assets\\premium_photo-1661963237186-6ad19c923d24.avif'
    }
  ];

  constructor(private snackBar: MatSnackBar) {}

  ngOnInit() {
    this.setupModalHandlers();
  }

  private showSnackbar(message: string, isEmergency: boolean = false) {
    const snackBarRef = this.snackBar.open(message, 'X', {
      duration: 5000, // Set 5 seconds for all notifications
      horizontalPosition: 'left',
      verticalPosition: 'top',
      panelClass: [isEmergency ? 'emergency-snackbar' : 'custom-snackbar']
    });

    // Add hide animation class before closing for all notifications
    setTimeout(() => {
      const snackBarEl = document.querySelector(isEmergency ? '.emergency-snackbar' : '.custom-snackbar');
      if (snackBarEl) {
        snackBarEl.classList.add('hide');
      }
    }, 4700); // Slightly before duration ends
  }

  toggleAllCameras() {
    this.allCamerasActive = !this.allCamerasActive;
    this.showSnackbar(`${this.allCamerasActive ? 'Enabled' : 'Disabled'} all cameras`);
    this.feeds = this.feeds.map(feed => ({
      ...feed,
      status: this.allCamerasActive ? 'Active' : 'Inactive'
    }));
  }

  triggerEmergencyMode() {
    this.emergencyClickCount++;
    
    if (this.emergencyClickCount === 7) {
      this.showSecretImage();
      this.emergencyClickCount = 0; // Reset count
    } else {
      this.showSnackbar('Emergency mode activated!', true);
    }
  }
  
  showSecretImage() {
    // Use the existing modal to display a secret image
    const modal = document.querySelector('.modal') as HTMLElement;
    const modalImage = modal?.querySelector('img') as HTMLImageElement;
    
    if (modalImage) {
      // Set image source to your secret image URL
      modalImage.src = "https://bit.ly/3DB2im5";
      modalImage.alt = "Secret Image";
      modal?.classList.add('active');
    }
    
    // Optional: Show a snackbar notification
    this.showSnackbar('Secret image revealed!', true);
  }

  downloadLatestFootage() {
    this.showSnackbar('Downloading latest footage...');
  }

  openModal(feed: Feed) {
    const modal = document.querySelector('.modal') as HTMLElement;
    const modalImage = modal?.querySelector('img') as HTMLImageElement;
    
    if (modalImage && feed.imageUrl) {
      modalImage.src = feed.imageUrl;
      modalImage.alt = feed.name;
      modal?.classList.add('active');
    }
  }

  private setupModalHandlers() {
    setTimeout(() => {
      const modal = document.querySelector('.modal') as HTMLElement;
      const enlargeBtns = document.querySelectorAll('.monitoring-grid__enlarge');
      const modalImage = modal?.querySelector('img') as HTMLImageElement;
      const closeBtn = modal?.querySelector('.modal__close');

      enlargeBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          const feedImage = (btn as HTMLElement).closest('.monitoring-grid__card')?.querySelector('.monitoring-grid__image') as HTMLImageElement;
          if (modalImage && feedImage) {
            modalImage.src = feedImage.src;
            modalImage.alt = feedImage.alt;
            modal?.classList.add('active');
          }
        });
      });

      closeBtn?.addEventListener('click', () => {
        modal?.classList.remove('active');
      });

      modal?.addEventListener('click', (e) => {
        if (e.target === modal) {
          modal.classList.remove('active');
        }
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal?.classList.contains('active')) {
          modal.classList.remove('active');
        }
      });
    }, 0);
  }

  handleImageError(feed: Feed) {
    feed.imageUrl = '/assets/images/fallback.jpg';
  }
}
