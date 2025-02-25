import { Routes } from '@angular/router';

export const routes: Routes = [
  { 
    path: '', 
    redirectTo: '/dashboard', 
    pathMatch: 'full' 
  },
  {
    path: 'dashboard',
    loadComponent: () => import('./header/dashboard/dashboard.component').then(m => m.DashboardComponent)
  },
  {
    path: 'analytics',
    loadComponent: () => import('./header/analytics/analytics.component').then(m => m.AnalyticsComponent)
  },
  {
    path: 'reports',
    loadComponent: () => import('./header/reports/reports.component').then(m => m.ReportsComponent)
  },
  {
    path: 'settings',
    loadComponent: () => import('./header/settings/settings.component').then(m => m.SettingsComponent)
  },
  {
    path: 'system-status',
    loadComponent: () => import('./system-status/system-status.component').then(m => m.SystemStatusComponent)
  },
  { 
    path: '**', 
    redirectTo: '/dashboard' 
  }
];
