import "@testing-library/jest-dom/vitest";

class NoopObserver { observe() {} unobserve() {} disconnect() {} takeRecords() { return []; } }
// jsdom has neither; framer-motion's useInView and the canvases use them
(globalThis as any).IntersectionObserver ??= NoopObserver;
(globalThis as any).ResizeObserver ??= NoopObserver;
