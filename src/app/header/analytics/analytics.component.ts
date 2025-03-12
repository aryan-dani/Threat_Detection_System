import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgChartsModule } from 'ng2-charts';
import { ChartConfiguration, ChartOptions } from 'chart.js';

interface StatCard {
  title: string;
  value: string;
  change: string;
  icon: string;
  isPositive?: boolean;
  highlight?: boolean;  // New property for red highlight
}

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [CommonModule, NgChartsModule],
  templateUrl: './analytics.component.html',
  styleUrls: ['../../styling/component/analytics.component.scss']
})
export class AnalyticsComponent implements OnInit {
  // Line Chart Data
  public lineChartData: ChartConfiguration<'line'>['data'] = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    datasets: [
      {
        data: [35, 48, 42, 55, 72, 65, 77, 85, 92, 88, 95, 98],
        label: 'Detections',
        fill: true,
        tension: 0.4,
        borderColor: '#2563eb', // Blue
        backgroundColor: 'rgba(37, 99, 235, 0.1)',
        borderWidth: 2,
        pointBackgroundColor: '#2563eb',
        pointBorderColor: '#ffffff',
        pointHoverRadius: 6
      }
    ]
  };

  // Pie Chart Data
  public pieChartData: ChartConfiguration<'pie'>['data'] = {
    labels: ['Main Terminal', 'Security Gate', 'Baggage Area', 'Parking Zone', 'Public Transit'],
    datasets: [{
      data: [35, 25, 20, 15, 5],
      backgroundColor: [
        'rgba(37, 99, 235, 0.8)', // Blue
        'rgba(168, 85, 247, 0.8)', // Purple
        'rgba(236, 72, 153, 0.8)', // Pink
        'rgba(234, 179, 8, 0.8)',  // Yellow
        'rgba(34, 197, 94, 0.8)'   // Green
      ],
      borderWidth: 2,
      borderColor: '#ffffff'
    }]
  };

  // Bar Chart Data
  public barChartData: ChartConfiguration<'bar'>['data'] = {
    labels: ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
    datasets: [
      {
        data: [15, 8, 12, 45, 68, 72, 56, 32],
        label: 'Detections by Hour',
        backgroundColor: 'rgba(37, 99, 235, 0.8)',
        borderColor: '#2563eb',
        borderWidth: 1,
        borderRadius: 4,
        hoverBackgroundColor: 'rgba(37, 99, 235, 1)'
      }
    ]
  };

  // Updated Pie Chart Options
  public pieChartOptions: ChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'right',
        align: 'center',
        labels: {
          color: '#404040',
          font: {
            family: "'Poppins', sans-serif",
            size: 11,
            weight: 500
          },
          padding: 10,
          boxWidth: 10
        }
      }
    }
  };

  // Common Chart Options
  public commonOptions: ChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'bottom',
        labels: {
          color: '#404040',
          font: {
            family: "'Poppins', sans-serif",
            size: 11,
            weight: 500
          },
          padding: 10,
          boxWidth: 10
        }
      },
      tooltip: {
        backgroundColor: 'rgba(32, 32, 32, 0.9)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        padding: 8,
        cornerRadius: 6,
        boxWidth: 8
      }
    },
    scales: {
      x: {
        display: true, // Only for line and bar charts
        grid: {
          display: false
        },
        ticks: {
          padding: 8,
          font: {
            size: 10
          },
          maxRotation: 45, // Add rotation to x-axis labels to save horizontal space
          minRotation: 45,  // Add rotation to x-axis labels
          autoSkip: true,   // Automatically skip labels if they overlap
          maxTicksLimit: 12 // Limit the number of ticks to avoid overcrowding
        }
      },
      y: {
        display: true, // Only for line and bar charts
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        ticks: {
          padding: 8,
          font: {
            size: 10
          }
        }
      }
    },
    layout: {
      padding: {
        left: 5,
        right: 5,
        top: 5,
        bottom: 5
      }
    },
    resizeDelay: 100 // Add small delay for better resize handling
  };

  stats: StatCard[] = [
    {
      title: 'Total Detections',
      value: '1,247',
      change: '+18%',
      icon: 'trending_up',
      highlight: true  // New property for red highlight
    },
    {
      title: 'Average Confidence',
      value: '96.8%',
      change: '+3.2%',
      icon: 'pie_chart',
      highlight: true  // New property for red highlight
    },
    {
      title: 'False Positives',
      value: '0.8%',
      change: '-0.7%',
      isPositive: true,
      icon: 'analytics'
    }
  ];

  ngOnInit() {
    // Any initialization logic
  }
}
