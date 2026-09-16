import { NextResponse } from "next/server";

function getBackendBaseUrl() {
  const raw =
    process.env.BACKEND_URL ||
    process.env.NEXT_PUBLIC_API_BASE_URL ||
    "http://127.0.0.1:8000";
  return raw.replace(/\/$/, "");
}

export async function POST(req: Request) {
  const backendBase = getBackendBaseUrl();
  const upstreamUrl = `${backendBase}/api/query`;

  let bodyText: string;
  try {
    bodyText = await req.text();
  } catch (e: any) {
    return NextResponse.json(
      { error: "invalid_request_body", detail: String(e?.message ?? e) },
      { status: 400 },
    );
  }

  try {
    const upstreamRes = await fetch(upstreamUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: bodyText,
    });

    const text = await upstreamRes.text();
    const contentType = upstreamRes.headers.get("content-type") ?? "application/json";

    return new NextResponse(text, {
      status: upstreamRes.status,
      headers: {
        "Content-Type": contentType,
      },
    });
  } catch (e: any) {
    // 代理失败（DNS、连接失败、超时等）
    return NextResponse.json(
      {
        error: "upstream_fetch_failed",
        upstreamUrl,
        detail: String(e?.message ?? e),
      },
      { status: 502 },
    );
  }
}


