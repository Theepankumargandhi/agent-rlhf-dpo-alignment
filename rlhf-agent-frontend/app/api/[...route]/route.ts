import { NextRequest, NextResponse } from 'next/server'

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000'

async function proxy(
  request: NextRequest,
  { params }: { params: { route: string[] } },
): Promise<NextResponse> {
  const path = params.route.join('/')
  const search = request.nextUrl.searchParams.toString()
  const url = `${BACKEND}/api/${path}${search ? `?${search}` : ''}`

  const init: RequestInit = { method: request.method }

  if (request.method !== 'GET' && request.method !== 'HEAD') {
    init.body = await request.text()
    init.headers = { 'Content-Type': 'application/json' }
  }

  try {
    const upstream = await fetch(url, init)
    const body = await upstream.text()

    // Try to parse as JSON; fall back to plain text
    try {
      return NextResponse.json(JSON.parse(body), { status: upstream.status })
    } catch {
      return new NextResponse(body, {
        status: upstream.status,
        headers: { 'Content-Type': 'text/plain' },
      })
    }
  } catch (err) {
    return NextResponse.json(
      { detail: `Backend unreachable: ${(err as Error).message}` },
      { status: 502 },
    )
  }
}

export const GET = proxy
export const POST = proxy
export const PUT = proxy
export const DELETE = proxy
export const PATCH = proxy
