import { Component } from '@angular/core';
import { RouterOutlet, RouterModule } from '@angular/router';
import { HeaderComponent } from './header/header.component';
import { CommonModule } from '@angular/common';
import { DetectionLogComponent } from './detection-log/detection-log.component';
import { FooterComponent } from './footer/footer.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    RouterOutlet,
    HeaderComponent,
    DetectionLogComponent,
    FooterComponent
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./styling/component/app.component.scss']
})
export class AppComponent {
  title = 'Weapon Detection System';
}
