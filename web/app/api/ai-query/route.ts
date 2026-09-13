// API route: AI Text-to-SQL assistant proxy
// Forward request + JWT của người dùng sang Central Server.
// Central yêu cầu vai trò đã đăng nhập cho /api/ai/** (SecurityConfig),
// nên bắt buộc phải chuyển tiếp header Authorization từ browser.
import { NextRequest, NextResponse } from 'next/server';

const CENTRAL_AI_URL =
  (process.env.CENTRAL_SERVER_URL || 'http://localhost:8002') + '/api/ai/query';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    // Chuyển tiếp JWT (nếu có) — không thêm token phía server để tránh
    // lộ biến môi trường; browser đã có session hợp lệ.
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    const auth = req.headers.get('authorization');
    if (auth) headers.Authorization = auth;

    const response = await fetch(CENTRAL_AI_URL, {
      method: 'POST',
      headers,
      body: JSON.stringify({ question: body.question }),
      // 2 lượt gọi Gemini (sinh SQL + tường thuật) + chạy SQL — 30s không đủ.
      signal: AbortSignal.timeout(90_000),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (err: unknown) {
    console.error('[ai-query proxy] Error:', err);
    return NextResponse.json(
      { error: 'Không thể kết nối Central Server. Vui lòng thử lại.' },
      { status: 502 }
    );
  }
}
