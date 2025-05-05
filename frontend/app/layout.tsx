// mangaui/frontend/app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Link from 'next/link';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Manga Creator Studio',
  description: 'Create manga from screenplay using Drawatoon',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav className="bg-indigo-600 text-white shadow-md">
          <div className="container mx-auto px-4 py-3">
            <div className="flex justify-between items-center">
              <Link href="/" className="text-xl font-bold">Manga Creator Studio</Link>
              
              <div className="space-x-6">
                <Link href="/" className="hover:underline">Manga Editor</Link>
                <Link href="/character-studio" className="hover:underline">Character Studio</Link>
                <Link href="/page-composer" className="hover:underline">Page Composer</Link>
              </div>
            </div>
          </div>
        </nav>
        
        <main className="min-h-screen bg-gray-100">
          {children}
        </main>
        
        <footer className="bg-gray-800 text-white py-4">
          <div className="container mx-auto px-4 text-center">
            <p>Manga Creator Studio - Powered by Drawatoon</p>
          </div>
        </footer>
      </body>
    </html>
  );
}