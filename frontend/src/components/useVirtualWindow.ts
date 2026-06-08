import { type UIEvent, useMemo, useState } from "react";

interface VirtualWindowOptions {
  overscan?: number;
  rowHeight: number;
  viewportHeight: number;
}

export function useVirtualWindow<T>(
  items: T[],
  { overscan = 4, rowHeight, viewportHeight }: VirtualWindowOptions,
) {
  const [scrollTop, setScrollTop] = useState(0);
  const windowState = useMemo(() => {
    const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
    const count = Math.ceil(viewportHeight / rowHeight) + overscan * 2;
    const end = Math.min(items.length, start + count);
    return {
      bottomPad: Math.max(0, (items.length - end) * rowHeight),
      end,
      items: items.slice(start, end),
      start,
      topPad: start * rowHeight,
    };
  }, [items, overscan, rowHeight, scrollTop, viewportHeight]);

  const onScroll = (event: UIEvent<HTMLElement>) => {
    setScrollTop(event.currentTarget.scrollTop);
  };

  return { ...windowState, onScroll };
}
