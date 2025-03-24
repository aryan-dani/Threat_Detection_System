import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class LoadingService {
  private loadingSubject = new BehaviorSubject<boolean>(false);
  private loadingMap = new Map<string, boolean>();

  constructor() {}

  setLoading(loading: boolean, url: string): void {
    if (loading) {
      this.loadingMap.set(url, loading);
      this.loadingSubject.next(true);
    } else if (!loading && this.loadingMap.has(url)) {
      this.loadingMap.delete(url);
      this.loadingSubject.next(this.loadingMap.size > 0);
    }
  }

  getLoading(): Observable<boolean> {
    return this.loadingSubject.asObservable();
  }
}
