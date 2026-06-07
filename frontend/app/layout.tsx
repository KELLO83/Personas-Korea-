import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Persona KG Console",
  description: "Korean Persona Knowledge Graph React dashboard",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem("persona-console-theme");document.documentElement.dataset.theme=t==="light"?"light":"dark";document.documentElement.style.colorScheme=t==="light"?"light":"dark"}catch(e){}`,
          }}
        />
        <link
          rel="stylesheet"
          as="style"
          crossOrigin="anonymous"
          href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
