import { trigger, transition, style, animate, query, group } from '@angular/animations';

export const fadeAnimation = trigger('fadeAnimation', [
  transition('* <=> *', [
    query(':enter', [style({ opacity: 0 })], { optional: true }),
    query(':leave', [style({ opacity: 1 }), animate('0.3s', style({ opacity: 0 }))], { optional: true }),
    query(':enter', [style({ opacity: 0 }), animate('0.3s', style({ opacity: 1 }))], { optional: true })
  ])
]);

export const slideInAnimation = trigger('slideInAnimation', [
  transition('* <=> *', [
    query(':enter, :leave', style({ position: 'fixed', width: '100%' }), { optional: true }),
    group([
      query(':enter', [
        style({ transform: 'translateX(100%)' }),
        animate('0.5s ease-in-out', style({ transform: 'translateX(0%)' }))
      ], { optional: true }),
      query(':leave', [
        style({ transform: 'translateX(0%)' }),
        animate('0.5s ease-in-out', style({ transform: 'translateX(-100%)' }))
      ], { optional: true })
    ])
  ])
]);

export const zoomAnimation = trigger('zoomAnimation', [
  transition(':enter', [
    style({ transform: 'scale(0.5)', opacity: 0 }),
    animate('0.3s ease-out', style({ transform: 'scale(1)', opacity: 1 }))
  ]),
  transition(':leave', [
    animate('0.3s ease-in', style({ transform: 'scale(0.5)', opacity: 0 }))
  ])
]);

export const bounceAnimation = trigger('bounceAnimation', [
  transition(':enter', [
    style({ transform: 'translateY(50px)', opacity: 0 }),
    animate('0.5s cubic-bezier(0.17, 0.89, 0.32, 1.25)', style({ transform: 'translateY(0)', opacity: 1 }))
  ])
]);

export const alertAnimation = trigger('alertAnimation', [
  transition(':enter', [
    style({ transform: 'translateY(-20px)', opacity: 0 }),
    animate('0.3s ease-out', style({ transform: 'translateY(0)', opacity: 1 }))
  ]),
  transition(':leave', [
    animate('0.3s ease-in', style({ opacity: 0 }))
  ])
]);
