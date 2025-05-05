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
];