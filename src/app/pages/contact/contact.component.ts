import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { trigger, state, style, animate, transition, keyframes } from '@angular/animations';

@Component({
  selector: 'app-contact',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './contact.component.html',
  styleUrls: ['./contact.component.scss'],
  animations: [
    trigger('toastAnimation', [
      state('show', style({
        opacity: 1,
        transform: 'translateY(0)'
      })),
      state('hide', style({
        opacity: 0,
        transform: 'translateY(-100%)'
      })),
      transition('hide => show', [
        animate('800ms ease-out', keyframes([
          style({ opacity: 0, transform: 'translateY(-100%) scale(0.8)', offset: 0 }),
          style({ opacity: 0.5, transform: 'translateY(10%) scale(1.1)', offset: 0.3 }),
          style({ opacity: 0.8, transform: 'translateY(-5%) scale(0.95)', offset: 0.8 }),
          style({ opacity: 1, transform: 'translateY(0) scale(1)', offset: 1.0 })
        ]))
      ]),
      transition('show => hide', [
        animate('600ms ease-in', keyframes([
          style({ opacity: 1, transform: 'translateY(0) rotate(0deg)', offset: 0 }),
          style({ opacity: 0.8, transform: 'translateY(10%) rotate(2deg)', offset: 0.2 }),
          style({ opacity: 0, transform: 'translateY(-100%) rotate(-5deg)', offset: 1.0 })
        ]))
      ])
    ]),
    trigger('successAnimation', [
      transition(':enter', [
        animate('1s', keyframes([
          style({ opacity: 0, transform: 'scale(0.3) rotate(0deg)', offset: 0 }),
          style({ opacity: 0.5, transform: 'scale(1.2) rotate(5deg)', offset: 0.3 }),
          style({ opacity: 0.8, transform: 'scale(0.9) rotate(-3deg)', offset: 0.6 }),
          style({ opacity: 1, transform: 'scale(1) rotate(0deg)', offset: 1.0 })
        ]))
      ])
    ]),
    trigger('formAnimation', [
      transition(':enter', [
        animate('800ms ease-out', keyframes([
          style({ opacity: 0, transform: 'translateY(50px) scale(0.8)', offset: 0 }),
          style({ opacity: 0.5, transform: 'translateY(20px) scale(0.9)', offset: 0.5 }),
          style({ opacity: 1, transform: 'translateY(0) scale(1)', offset: 1.0 })
        ]))
      ])
    ])
  ]
})
export class ContactComponent {
  contactForm: FormGroup;
  submitted = false;
  formSuccess = false;
  
  // Toast properties
  showToast = false;
  toastMessage = '';
  toastType: 'success' | 'error' | 'info' = 'info';
  toastIcon = '';
  
  constructor(private fb: FormBuilder) {
    this.contactForm = this.fb.group({
      name: ['', [Validators.required]],
      email: ['', [Validators.required, Validators.email]],
      subject: ['', [Validators.required]],
      message: ['', [Validators.required, Validators.minLength(20)]]
    });
  }
  
  onSubmit() {
    this.submitted = true;
    
    if (this.contactForm.valid) {
      // In a real app, we would send this data to a backend service
      console.log('Form submitted:', this.contactForm.value);
      
      // Show loading toast
      this.showToastNotification('Sending your message...', 'info');
      
      // Simulate successful submission with a longer timeout to show the animation
      setTimeout(() => {
        this.formSuccess = true;
        this.contactForm.reset();
        this.submitted = false;
        
        // Show success toast with exciting animation
        this.showToastNotification('Message sent successfully! 🎉 We\'ll get back to you soon.', 'success');
        
        // Reset form success state after a while
        setTimeout(() => {
          this.formSuccess = false;
        }, 5000);
      }, 2000);
    } else {
      // Show error toast if form is invalid
      this.showToastNotification('Please fix the errors in the form before submitting! 🛑', 'error');
      
      // Add a shake animation to invalid fields
      const invalidControls = document.querySelectorAll('.ng-invalid.ng-touched');
      invalidControls.forEach(element => {
        element.classList.add('shake-animation');
        setTimeout(() => {
          element.classList.remove('shake-animation');
        }, 1000);
      });
    }
  }
  
  showToastNotification(message: string, type: 'success' | 'error' | 'info') {
    this.toastMessage = message;
    this.toastType = type;
    
    // Set appropriate icon
    switch(type) {
      case 'success':
        this.toastIcon = '✓';
        break;
      case 'error':
        this.toastIcon = '✗';
        break;
      case 'info':
        this.toastIcon = 'ℹ';
        break;
    }
    
    this.showToast = true;
    
    // Auto-hide the toast after time based on message length
    const displayTime = Math.max(3000, message.length * 100);
    setTimeout(() => {
      this.showToast = false;
    }, displayTime);
  }
  
  hideToast() {
    this.showToast = false;
  }
  
  get f() {
    return this.contactForm.controls;
  }
}