# ECOM — In-Store Reaction — Typography V1

Measurements are from the approved V2 title artwork and its editable source, then carried into the native Fusion Text+ template. The approved Resolve timeline used a transparent ProRes 4444 caption movie rather than a live Text+ node, so the original values were recovered from the title artwork/source and confirmed against the V2 frame.

## Text treatment

- Canvas: 1080 × 1920, vertical 9:16.
- Typeface: Arial Narrow, Bold (the installed macOS `Arial Narrow Bold` face).
- Main line: 88 px source artwork size; centered at x=540 px; top near y=250 px. The approved wording occupied about 600 px of width after a horizontal fit adjustment.
- Second line: 72 px source artwork size; left aligned within the centered second-row group, with the word beginning near x=391 px and top near y=350 px.
- Character spacing: 0 additional tracking.
- Fill: white `#FFFFFF`.
- Outline: near-black `#050505`, 6 px, fully opaque.
- Shadow: black, 72% opacity, approximately 2 px offset, approximately 1.2 px Gaussian softness.
- Line alignment: main line centered; second-line word is aligned as part of the centered text-and-emoji composition.
- Line spacing: static two-row layout; main glyphs end near y=303 px and second-row word begins near y=350 px.
- Animation: none. V2 title remains static for its duration.

## Emoji treatment and placement

- The approved V2 emoji artwork is preserved as a transparent PNG overlay (219 × 74 px), placed at x=470 px, y=333 px on the 1080 × 1920 canvas.
- Its visual sequence is the approved pink hearts, smiling face, and telephone receiver artwork. The emoji sprite uses Apple Color Emoji source artwork at 80 px, cropped and resized to 74 px high.
- Emoji position is fixed in the supplied full-frame overlay. Reposition or replace the overlay when changing the second-line wording or emoji sequence.

## Fusion template implementation

- The editable Fusion title exposes `Main line`, `Second line`, and `Emoji overlay image` controls.
- Native Text+ controls use Arial Narrow Bold, white fill, black outline, and black shadow. The title has no entrance or exit animation.
- Text+ size values are normalized Fusion controls (`0.074` main line and `0.061` second line), rather than point sizes; they render on the 1080 × 1920 timeline.
- The template references the PNG in this reusable-assets folder by absolute local path. Keep the folder in place on this Mac, or update the Loader path after moving the assets to another machine.
- The exact original font identity was not embedded in the raster title, so Arial Narrow Bold is the recovered installed face used by the approved source artwork.
