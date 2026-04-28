"use client";

import { useCallback, useState } from "react";

export interface Loadable<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useLoadable<T>(initial: T | null = null): [Loadable<T>, (loader: () => Promise<T>) => Promise<T | null>] {
  const [state, setState] = useState<Loadable<T>>({ data: initial, loading: false, error: null });

  const run = useCallback(async (loader: () => Promise<T>): Promise<T | null> => {
    setState((current) => ({ ...current, loading: true, error: null }));
    try {
      const data = await loader();
      setState({ data, loading: false, error: null });
      return data;
    } catch (error) {
      const message = error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다.";
      setState((current) => ({ ...current, loading: false, error: message }));
      return null;
    }
  }, []);

  return [state, run];
}
