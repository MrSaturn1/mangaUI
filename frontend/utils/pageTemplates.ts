// utils/pageTemplates.ts

export interface PageTemplate {
  id: string;
  name: string;
  description: string;
  columns: number;
  rows: number;
  panelCount: number;
  layoutFunction: (pageWidth: number, pageHeight: number, gap: number) => {
    id: string;
    x: number;
    y: number;
    width: number;
    height: number;
  }[];
}

// Create some predefined templates
export const pageTemplates: PageTemplate[] = [
  {
    id: "standard-2x3",
    name: "Standard 2×3",
    description: "6 equal panels in a 2×3 grid",
    columns: 2,
    rows: 3,
    panelCount: 6,
    layoutFunction: (pageWidth, pageHeight, gap) => {
      const panelWidth = (pageWidth - (gap * (2 + 1))) / 2;
      const panelHeight = (pageHeight - (gap * (3 + 1))) / 3;
      const panels = [];
      
      for (let row = 0; row < 3; row++) {
        for (let col = 0; col < 2; col++) {
          panels.push({
            id: `panel-${row * 2 + col}`,
            x: gap + col * (panelWidth + gap),
            y: gap + row * (panelHeight + gap),
            width: panelWidth,
            height: panelHeight,
          });
        }
      }
      
      return panels;
    }
  },
  {
    id: "manga-dynamic",
    name: "Manga Dynamic",
    description: "5 panels with varied sizes for dynamic storytelling",
    columns: 2,
    rows: 3,
    panelCount: 5,
    layoutFunction: (pageWidth, pageHeight, gap) => {
      const halfWidth = (pageWidth - (gap * 3)) / 2;
      const thirdHeight = (pageHeight - (gap * 4)) / 3;
      
      return [
        // Large panel in top-left
        {
          id: `panel-0`,
          x: gap,
          y: gap,
          width: halfWidth,
          height: thirdHeight * 2 + gap,
        },
        // Small panel in top-right
        {
          id: `panel-1`,
          x: gap * 2 + halfWidth,
          y: gap,
          width: halfWidth,
          height: thirdHeight,
        },
        // Medium panel in middle-right
        {
          id: `panel-2`,
          x: gap * 2 + halfWidth,
          y: gap * 2 + thirdHeight,
          width: halfWidth,
          height: thirdHeight,
        },
        // Wide panel at bottom
        {
          id: `panel-3`,
          x: gap,
          y: gap * 3 + thirdHeight * 2,
          width: pageWidth - (gap * 2),
          height: thirdHeight,
        },
      ];
    }
  },
  {
    id: "cinematic",
    name: "Cinematic",
    description: "3 wide panels for a cinematic feel",
    columns: 1,
    rows: 3,
    panelCount: 3,
    layoutFunction: (pageWidth, pageHeight, gap) => {
      const panelHeight = (pageHeight - (gap * 4)) / 3;
      
      return [
        {
          id: `panel-0`,
          x: gap,
          y: gap,
          width: pageWidth - (gap * 2),
          height: panelHeight,
        },
        {
          id: `panel-1`,
          x: gap,
          y: gap * 2 + panelHeight,
          width: pageWidth - (gap * 2),
          height: panelHeight,
        },
        {
          id: `panel-2`,
          x: gap,
          y: gap * 3 + panelHeight * 2,
          width: pageWidth - (gap * 2),
          height: panelHeight,
        },
      ];
    }
  },
  {
    id: "splash-focus",
    name: "Splash Focus",
    description: "One large splash panel with 3 smaller support panels",
    columns: 2,
    rows: 2,
    panelCount: 4,
    layoutFunction: (pageWidth, pageHeight, gap) => {
      const quarterWidth = (pageWidth - (gap * 3)) / 4;
      const halfHeight = (pageHeight - (gap * 3)) / 2;
      
      return [
        // Large splash panel taking up 3/4 of top
        {
          id: `panel-0`,
          x: gap,
          y: gap,
          width: quarterWidth * 3 + gap * 2,
          height: halfHeight,
        },
        // Small panel top-right
        {
          id: `panel-1`,
          x: gap * 3 + quarterWidth * 3,
          y: gap,
          width: quarterWidth,
          height: halfHeight / 2 - gap / 2,
        },
        // Small panel middle-right
        {
          id: `panel-2`,
          x: gap * 3 + quarterWidth * 3,
          y: gap + halfHeight / 2 + gap / 2,
          width: quarterWidth,
          height: halfHeight / 2 - gap / 2,
        },
        // Wide bottom panel
        {
          id: `panel-3`,
          x: gap,
          y: gap * 2 + halfHeight,
          width: pageWidth - (gap * 2),
          height: halfHeight,
        },
      ];
    }
  },
  {
    id: "vertical-flow",
    name: "Vertical Flow",
    description: "4 panels in ascending height for vertical storytelling",
    columns: 1,
    rows: 4,
    panelCount: 4,
    layoutFunction: (pageWidth, pageHeight, gap) => {
      const totalGaps = gap * 5;
      const availableHeight = pageHeight - totalGaps;
      
      // Heights increase: 15%, 20%, 25%, 40%
      const heights = [
        availableHeight * 0.15,
        availableHeight * 0.20,
        availableHeight * 0.25,
        availableHeight * 0.40
      ];
      
      let currentY = gap;
      const panels = [];
      
      for (let i = 0; i < 4; i++) {
        panels.push({
          id: `panel-${i}`,
          x: gap,
          y: currentY,
          width: pageWidth - (gap * 2),
          height: heights[i],
        });
        currentY += heights[i] + gap;
      }
      
      return panels;
    }
  },
  {
    id: "l-shape",
    name: "L-Shape Layout",
    description: "5 panels in an L-shaped composition",
    columns: 3,
    rows: 2,
    panelCount: 5,
    layoutFunction: (pageWidth, pageHeight, gap) => {
      const thirdWidth = (pageWidth - (gap * 4)) / 3;
      const halfHeight = (pageHeight - (gap * 3)) / 2;
      
      return [
        // Top row - 3 equal panels
        {
          id: `panel-0`,
          x: gap,
          y: gap,
          width: thirdWidth,
          height: halfHeight,
        },
        {
          id: `panel-1`,
          x: gap * 2 + thirdWidth,
          y: gap,
          width: thirdWidth,
          height: halfHeight,
        },
        {
          id: `panel-2`,
          x: gap * 3 + thirdWidth * 2,
          y: gap,
          width: thirdWidth,
          height: halfHeight,
        },
        // Bottom left - large panel
        {
          id: `panel-3`,
          x: gap,
          y: gap * 2 + halfHeight,
          width: thirdWidth * 2 + gap,
          height: halfHeight,
        },
        // Bottom right - medium panel
        {
          id: `panel-4`,
          x: gap * 3 + thirdWidth * 2,
          y: gap * 2 + halfHeight,
          width: thirdWidth,
          height: halfHeight,
        },
      ];
    }
  },
  {
    id: "action-sequence",
    name: "Action Sequence",
    description: "6 panels optimized for action sequences",
    columns: 3,
    rows: 3,
    panelCount: 6,
    layoutFunction: (pageWidth, pageHeight, gap) => {
      const thirdWidth = (pageWidth - (gap * 4)) / 3;
      const thirdHeight = (pageHeight - (gap * 4)) / 3;
      
      return [
        // Top row - wide establishing shot
        {
          id: `panel-0`,
          x: gap,
          y: gap,
          width: pageWidth - (gap * 2),
          height: thirdHeight,
        },
        // Middle row - 3 quick action panels
        {
          id: `panel-1`,
          x: gap,
          y: gap * 2 + thirdHeight,
          width: thirdWidth,
          height: thirdHeight,
        },
        {
          id: `panel-2`,
          x: gap * 2 + thirdWidth,
          y: gap * 2 + thirdHeight,
          width: thirdWidth,
          height: thirdHeight,
        },
        {
          id: `panel-3`,
          x: gap * 3 + thirdWidth * 2,
          y: gap * 2 + thirdHeight,
          width: thirdWidth,
          height: thirdHeight,
        },
        // Bottom row - 2 aftermath panels
        {
          id: `panel-4`,
          x: gap,
          y: gap * 3 + thirdHeight * 2,
          width: (pageWidth - (gap * 3)) / 2,
          height: thirdHeight,
        },
        {
          id: `panel-5`,
          x: gap * 2 + (pageWidth - (gap * 3)) / 2,
          y: gap * 3 + thirdHeight * 2,
          width: (pageWidth - (gap * 3)) / 2,
          height: thirdHeight,
        },
      ];
    }
  },
  {
    id: "dialogue-heavy",
    name: "Dialogue Heavy",
    description: "4 tall panels perfect for character conversations",
    columns: 2,
    rows: 2,
    panelCount: 4,
    layoutFunction: (pageWidth, pageHeight, gap) => {
      const halfWidth = (pageWidth - (gap * 3)) / 2;
      const halfHeight = (pageHeight - (gap * 3)) / 2;
      
      return [
        {
          id: `panel-0`,
          x: gap,
          y: gap,
          width: halfWidth,
          height: halfHeight,
        },
        {
          id: `panel-1`,
          x: gap * 2 + halfWidth,
          y: gap,
          width: halfWidth,
          height: halfHeight,
        },
        {
          id: `panel-2`,
          x: gap,
          y: gap * 2 + halfHeight,
          width: halfWidth,
          height: halfHeight,
        },
        {
          id: `panel-3`,
          x: gap * 2 + halfWidth,
          y: gap * 2 + halfHeight,
          width: halfWidth,
          height: halfHeight,
        },
      ];
    }
  },
  {
    id: "experimental-grid",
    name: "Experimental Grid",
    description: "7 panels in an asymmetric experimental layout",
    columns: 4,
    rows: 3,
    panelCount: 7,
    layoutFunction: (pageWidth, pageHeight, gap) => {
      const quarterWidth = (pageWidth - (gap * 5)) / 4;
      const thirdHeight = (pageHeight - (gap * 4)) / 3;
      
      return [
        // Top row - 2 panels, one double-wide
        {
          id: `panel-0`,
          x: gap,
          y: gap,
          width: quarterWidth * 2 + gap,
          height: thirdHeight,
        },
        {
          id: `panel-1`,
          x: gap * 3 + quarterWidth * 2,
          y: gap,
          width: quarterWidth,
          height: thirdHeight,
        },
        {
          id: `panel-2`,
          x: gap * 4 + quarterWidth * 3,
          y: gap,
          width: quarterWidth,
          height: thirdHeight,
        },
        // Middle row - 3 small panels
        {
          id: `panel-3`,
          x: gap,
          y: gap * 2 + thirdHeight,
          width: quarterWidth,
          height: thirdHeight,
        },
        {
          id: `panel-4`,
          x: gap * 2 + quarterWidth,
          y: gap * 2 + thirdHeight,
          width: quarterWidth,
          height: thirdHeight,
        },
        {
          id: `panel-5`,
          x: gap * 3 + quarterWidth * 2,
          y: gap * 2 + thirdHeight,
          width: quarterWidth * 2 + gap,
          height: thirdHeight,
        },
        // Bottom row - wide climax panel
        {
          id: `panel-6`,
          x: gap,
          y: gap * 3 + thirdHeight * 2,
          width: pageWidth - (gap * 2),
          height: thirdHeight,
        },
      ];
    }
  },
  {
    id: "minimalist",
    name: "Minimalist",
    description: "3 clean panels with generous spacing",
    columns: 1,
    rows: 3,
    panelCount: 3,
    layoutFunction: (pageWidth, pageHeight, gap) => {
      const largeGap = gap * 2;
      const panelHeight = (pageHeight - (largeGap * 4)) / 3;
      
      return [
        {
          id: `panel-0`,
          x: largeGap,
          y: largeGap,
          width: pageWidth - (largeGap * 2),
          height: panelHeight,
        },
        {
          id: `panel-1`,
          x: largeGap,
          y: largeGap * 2 + panelHeight,
          width: pageWidth - (largeGap * 2),
          height: panelHeight,
        },
        {
          id: `panel-2`,
          x: largeGap,
          y: largeGap * 3 + panelHeight * 2,
          width: pageWidth - (largeGap * 2),
          height: panelHeight,
        },
      ];
    }
  },
];