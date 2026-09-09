import { describe, expect, it } from "vitest";
import { impactForSwing, stageSensitivity } from "./sensitivity";

const stage = (scores: Record<string, number>) => ({ stage: "test", scores, winner: "a" });

describe("stageSensitivity", () => {
  it("ignores negative failed-probe sentinels", () => {
    expect(stageSensitivity(stage({ failed: -1, a: 100, b: 80 }))?.candidateCount).toBe(2);
  });

  it("requires two valid candidates", () => {
    expect(stageSensitivity(stage({ failed: -1, a: 100 }))).toBeNull();
  });

  it("handles ties deterministically", () => {
    expect(stageSensitivity(stage({ a: 100, b: 100 }))).toMatchObject({ bestLabel: "a", worstLabel: "a", swingPct: 0 });
  });

  it("marks a positive result over a zero baseline as non-comparable, not unbounded", () => {
    const bar = stageSensitivity(stage({ a: 100, b: 0 }));
    expect(bar?.swingPct).toBeNull();
    expect(bar?.impact).toBe("non_comparable");
  });
});

describe("impactForSwing", () => {
  it("classifies exact threshold boundaries centrally", () => {
    expect(impactForSwing(4.999)).toBe("low");
    expect(impactForSwing(5)).toBe("medium");
    expect(impactForSwing(19.999)).toBe("medium");
    expect(impactForSwing(20)).toBe("high");
    expect(impactForSwing(null)).toBe("non_comparable");
  });
});