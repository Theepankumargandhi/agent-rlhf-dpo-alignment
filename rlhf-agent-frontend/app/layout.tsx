import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Link from 'next/link'
import { Bot } from 'lucide-react'
import { Toaster } from 'react-hot-toast'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'RLHF Agent',
  description: 'Customer support agent with human preference feedback loop',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-[#0f0f0f] text-white min-h-screen antialiased`}>
        {/* ── Navbar ─────────────────────────────────────────────── */}
        <header className="sticky top-0 z-50 border-b border-[#2a2a2a] bg-[#0f0f0f]/80 backdrop-blur-md">
          <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
            {/* Brand */}
            <Link href="/" className="flex items-center gap-2 group">
              <div className="w-7 h-7 rounded-lg bg-violet-600 flex items-center justify-center
                              group-hover:bg-violet-500 transition-colors">
                <Bot className="h-4 w-4 text-white" />
              </div>
              <span className="text-sm font-semibold text-white">RLHF Agent</span>
            </Link>

            {/* Nav links */}
            <nav className="flex items-center gap-1">
              <Link
                href="/"
                className="px-3 py-1.5 text-sm text-zinc-400 hover:text-white
                           hover:bg-white/5 rounded-md transition-colors"
              >
                Chat
              </Link>
              <Link
                href="/dashboard"
                className="px-3 py-1.5 text-sm text-zinc-400 hover:text-white
                           hover:bg-white/5 rounded-md transition-colors"
              >
                Dashboard
              </Link>
            </nav>
          </div>
        </header>

        {children}

        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: '#1a1a1a',
              color: '#fff',
              border: '1px solid #2a2a2a',
              fontSize: '13px',
              borderRadius: '10px',
            },
            success: { iconTheme: { primary: '#7c3aed', secondary: '#fff' } },
          }}
        />
      </body>
    </html>
  )
}
