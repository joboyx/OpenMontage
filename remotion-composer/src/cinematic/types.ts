// "none" renders the clip ungraded: no default filter, tone gradient, vignette or texture.
export type CinematicTone = "cold" | "steel" | "void" | "neutral" | "none";

export interface CinematicBaseScene {
  id: string;
  startSeconds: number;
  durationSeconds: number;
}

export interface CinematicVideoScene extends CinematicBaseScene {
  kind: "video";
  src: string;
  tone?: CinematicTone;
  trimBeforeSeconds?: number;
  trimAfterSeconds?: number;
  playbackRate?: number;
  filter?: string;
  fadeInFrames?: number;
  fadeOutFrames?: number;
}

export interface CinematicTitleScene extends CinematicBaseScene {
  kind: "title";
  text: string;
  accent?: string;
  intensity?: number;
  backgroundSrc?: string;
  backgroundTrimBeforeSeconds?: number;
  backgroundTrimAfterSeconds?: number;
  variant?: "plate" | "overlay";
}

export interface CinematicEndCardLine {
  text: string;
  // Poppins weight (default 700).
  weight?: 400 | 700;
  // CSS colour (default "#5B3A23").
  color?: string;
  // Font size in px at the render resolution (default 72).
  fontSize?: number;
  // Extra space in px above this line (default 0), to set uneven line gaps.
  marginTop?: number;
}

// Flat end card: Poppins lines centred on a solid background, no grain or vignette.
export interface CinematicEndCardScene extends CinematicBaseScene {
  kind: "endcard";
  lines: CinematicEndCardLine[];
  // CSS colour (default "#F7F0E3").
  backgroundColor?: string;
}

export type CinematicScene =
  | CinematicVideoScene
  | CinematicTitleScene
  | CinematicEndCardScene;

export interface CinematicSoundtrack {
  src: string;
  volume?: number;
  trimBeforeSeconds?: number;
  trimAfterSeconds?: number;
  fadeInSeconds?: number;
  fadeOutSeconds?: number;
}

export interface CinematicWordCaption {
  word: string;
  startMs: number;
  endMs: number;
  pageBreakAfter?: boolean;
  // Per-word (speaker) colour; a page never mixes two colours.
  color?: string;
}

export interface CinematicCaptionConfig {
  words: CinematicWordCaption[];
  wordsPerPage?: number;
  fontSize?: number;
  color?: string;
  highlightColor?: string;
  backgroundColor?: string;
  // Distance in px from the frame bottom to the caption box (default 80).
  bottomOffset?: number;
  // Corner radius in px of the caption box (default 12).
  borderRadius?: number;
}

export interface CinematicRendererProps {
  [key: string]: unknown;
  scenes: CinematicScene[];
  titleFontSize?: number;
  titleWidth?: number;
  signalLineCount?: number;
  soundtrack?: CinematicSoundtrack;
  music?: CinematicSoundtrack;
  captions?: CinematicCaptionConfig;
}
