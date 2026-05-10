import { ChatInterface } from '@/components/ChatInterface'

export default function HomePage() {
  return (
    // 100vh minus the 3.5rem (56px) navbar height
    <main className="h-[calc(100vh-3.5rem)]">
      <div className="max-w-3xl mx-auto h-full flex flex-col">
        <ChatInterface />
      </div>
    </main>
  )
}
