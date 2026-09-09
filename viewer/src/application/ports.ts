import type { ResultsFile } from "../domain/results";

export interface ResultsCatalog {
  load(): Promise<ResultsFile>;
}
