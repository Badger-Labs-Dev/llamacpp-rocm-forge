import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App", () => {
  it("announces its initial loading state", () => {
    const html = renderToStaticMarkup(<App />);
    expect(html).toContain('role="status"');
    expect(html).toContain('aria-live="polite"');
    expect(html).toContain("Loading results…");
  });
});
