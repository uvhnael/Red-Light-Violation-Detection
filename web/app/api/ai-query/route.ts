// API route: AI Text-to-SQL assistant proxy
// Forwards request to Central Server's AI endpoint
import { NextRequest, NextResponse } from 'next/server';

const CENTRAL_AI_URL = (process.env.CENTRAL_SERVER_URL || 'http://localhost:8001') + '/api/ai/query';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const response = await fetch(CENTRAL_AI_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: body.question }),
      signal: AbortSignal.timeout(30_000),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });

  } catch (err: unknown) {
    console.error('[ai-query proxy] Error:', err);
    return NextResponse.json({
      error: 'Không thể kết nối Central Server. Vui lòng thử lại.',
    }, { status: 502 });
  }
}