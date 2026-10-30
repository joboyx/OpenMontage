import React, { Fragment } from "react";
import {
  AbsoluteFill,
  Sequence,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

// Word-level caption for TikTok-style highlight display
export interface WordCaption {
  word: string;
  startMs: number;
  endMs: number;
  // Force a page break after this word (e.g. sentence or scene boundaries).
  // Useful for CJK captions where pages should align with clause boundaries.
  pageBreakAfter?: boolean;
  /**
   * Optional per-word colour, for example to colour-code speakers. A word
   * with a colour ignores the global `color` / `highlightColor`: upcoming
   * words show it at 60% opacity, past words at full strength, and the
   * active word at full strength with a stronger glow and a small scale pop.
   * A page never holds words of two different colours.
   */
  color?: string;
}

type CaptionOverlayProps = {
  words: WordCaption[];
  // How many words to show at once in a "page"
  wordsPerPage?: number;
  fontSize?: number;
  color?: string;
  highlightColor?: string;
  backgroundColor?: string;
  fontFamily?: string;
  // Separator rendered between words. Space-delimited languages want the
  // default " "; CJK languages (no inter-word spacing) should pass "".
  wordSeparator?: string;
  // Distance in px from the frame bottom to the caption box (default 80).
  bottomOffset?: number;
  // Corner radius in px of the caption box (default 12; use a large value for a pill).
  borderRadius?: number;
  // Base text direction. "auto" (default) picks RTL when most words of the
  // captions are Arabic/Hebrew/Persian/Urdu. Each word is an
  // inline-block, which the bidi algorithm treats as a neutral object, so
  // without an explicit direction RTL words are laid out left-to-right and
  // the sentence reads backwards.
  direction?: "ltr" | "rtl" | "auto";
};

const RTL_CHAR = /[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]/;
const LTR_CHAR = /[A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF]/;

function detectDirection(words: WordCaption[]): "ltr" | "rtl" {
  // Count words, not first letter: Arabic captions often open with a Latin
  // brand name ("OpenMontage يحوّل ...") and must still read right-to-left.
  let rtl = 0;
  let ltr = 0;
  for (const w of words) {
    if (RTL_CHAR.test(w.word)) rtl++;
    else if (LTR_CHAR.test(w.word)) ltr++;
  }
  return rtl > ltr ? "rtl" : "ltr";
}

interface CaptionPage {
  words: WordCaption[];
  startMs: number;
  endMs: number;
}

function buildPages(words: WordCaption[], wordsPerPage: number): CaptionPage[] {
  const pages: CaptionPage[] = [];
  let pageWords: WordCaption[] = [];
  const flush = () => {
    if (pageWords.length === 0) return;
    pages.push({
      words: pageWords,
      startMs: pageWords[0].startMs,
      endMs: pageWords[pageWords.length - 1].endMs,
    });
    pageWords = [];
  };
  for (const w of words) {
    // A page shows one speaker only: start a new page when the colour changes.
    // Words without a colour compare equal (undefined), so they page as before.
    if (pageWords.length > 0 && pageWords[pageWords.length - 1].color !== w.color) flush();
    pageWords.push(w);
    if (pageWords.length >= wordsPerPage || w.pageBreakAfter) flush();
  }
  flush();
  return pages;
}

// A word that carries its own (speaker) colour. Rendered inside a page Sequence,
// so frame 0 is the page start.
const SpeakerWord: React.FC<{
  word: WordCaption;
  page: CaptionPage;
  isActive: boolean;
  isPast: boolean;
  color: string;
}> = ({ word, page, isActive, isPast, color }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Scale pop that starts when the word becomes active (small overshoot, then 1.06).
  const wordStartFrame = Math.round(((word.startMs - page.startMs) / 1000) * fps);
  const pop = isActive
    ? spring({
        frame: frame - wordStartFrame,
        fps,
        config: { damping: 10, stiffness: 220 },
      })
    : 0;
  const scale = 1 + 0.06 * pop;

  return (
    <span
      style={{
        display: "inline-block",
        whiteSpace: "nowrap",
        color,
        opacity: isActive || isPast ? 1 : 0.6,
        transform: `scale(${scale})`,
        textShadow: isActive
          ? `0 0 26px color-mix(in srgb, ${color} 80%, transparent), 0 0 8px color-mix(in srgb, ${color} 50%, transparent), 0 2px 4px rgba(0,0,0,0.5)`
          : "0 2px 4px rgba(0,0,0,0.5)",
      }}
    >
      {word.word}
    </span>
  );
};

const PageRenderer: React.FC<{
  page: CaptionPage;
  fontSize: number;
  color: string;
  highlightColor: string;
  backgroundColor: string;
  fontFamily: string;
  wordSeparator: string;
  bottomOffset: number;
  borderRadius: number;
  direction: "ltr" | "rtl";
}> = ({
  page,
  fontSize,
  color,
  highlightColor,
  backgroundColor,
  fontFamily,
  wordSeparator,
  bottomOffset,
  borderRadius,
  direction,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const currentMs = page.startMs + (frame / fps) * 1000;

  // Spring entrance
  const entrance = spring({
    frame,
    fps,
    config: { damping: 18, stiffness: 120 },
  });

  return (
    <AbsoluteFill
      style={{
        justifyContent: "flex-end",
        alignItems: "center",
        paddingBottom: bottomOffset,
      }}
    >
      <div
        dir={direction}
        style={{
          direction,
          opacity: entrance,
          transform: `translateY(${interpolate(entrance, [0, 1], [20, 0])}px)`,
          backgroundColor,
          borderRadius,
          padding: "14px 28px",
          maxWidth: "80%",
          textAlign: "center",
        }}
      >
        <span
          style={{
            fontSize,
            fontWeight: 700,
            fontFamily,
            lineHeight: 1.4,
            whiteSpace: "pre-wrap",
          }}
        >
          {page.words.map((w, i) => {
            const isActive = w.startMs <= currentMs && w.endMs > currentMs;
            const isPast = w.endMs <= currentMs;
            if (w.color) {
              return (
                <Fragment key={`${w.startMs}-${i}`}>
                  <SpeakerWord
                    word={w}
                    page={page}
                    isActive={isActive}
                    isPast={isPast}
                    color={w.color}
                  />
                  {i < page.words.length - 1 ? wordSeparator : ""}
                </Fragment>
              );
            }
            return (
              <Fragment key={`${w.startMs}-${i}`}>
                <span
                  style={{
                    // Keep each word unbroken so lines wrap only at word
                    // boundaries. For space-delimited text this matches the
                    // previous behavior; for CJK it prevents mid-word breaks.
                    display: "inline-block",
                    whiteSpace: "nowrap",
                    color: isActive ? highlightColor : isPast ? color : `${color}99`,
                    transition: "none", // CSS transitions forbidden in Remotion
                    textShadow: isActive
                      ? `0 0 20px ${highlightColor}66, 0 2px 4px rgba(0,0,0,0.5)`
                      : "0 2px 4px rgba(0,0,0,0.5)",
                  }}
                >
                  {w.word}
                </span>
                {i < page.words.length - 1 ? wordSeparator : ""}
              </Fragment>
            );
          })}
        </span>
      </div>
    </AbsoluteFill>
  );
};

export const CaptionOverlay: React.FC<CaptionOverlayProps> = ({
  words,
  wordsPerPage = 6,
  fontSize = 42,
  color = "#F8FAFC",
  highlightColor = "#22D3EE",
  backgroundColor = "rgba(15, 23, 42, 0.75)",
  fontFamily = "Space Grotesk, Inter, system-ui, sans-serif",
  wordSeparator = " ",
  bottomOffset = 80,
  borderRadius = 12,
  direction = "auto",
}) => {
  const { fps } = useVideoConfig();
  const pages = buildPages(words, wordsPerPage);
  const resolvedDirection =
    direction === "auto" ? detectDirection(words) : direction;

  return (
    <AbsoluteFill>
      {pages.map((page, i) => {
        const fromFrame = Math.round((page.startMs / 1000) * fps);
        const nextStart = pages[i + 1]?.startMs ?? page.endMs + 500;
        const duration = Math.max(
          1,
          Math.round(((nextStart - page.startMs) / 1000) * fps)
        );

        return (
          <Sequence key={i} from={fromFrame} durationInFrames={duration}>
            <PageRenderer
              page={page}
              fontSize={fontSize}
              color={color}
              highlightColor={highlightColor}
              backgroundColor={backgroundColor}
              fontFamily={fontFamily}
              wordSeparator={wordSeparator}
              bottomOffset={bottomOffset}
              borderRadius={borderRadius}
              direction={resolvedDirection}
            />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
