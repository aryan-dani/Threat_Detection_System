import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';

@Component({
  selector: 'app-contact',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './contact.component.html',
  styleUrls: ['./contact.component.scss']
})
export class ContactComponent {
  contactForm: FormGroup;
  submitted = false;
  formSuccess = false;
  
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
      
      // Simulate successful submission
      setTimeout(() => {
        this.formSuccess = true;
        this.contactForm.reset();
        this.submitted = false;
      }, 1000);
    }
  }
  
  get f() {
    return this.contactForm.controls;
  }
}