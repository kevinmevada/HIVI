from pathlib import Path

p = Path(__file__).with_name("index.html")
text = p.read_text(encoding="utf-8")
start = text.index("  <script>")
end = text.index("</script>") + len("</script>")

open_t = "<" + "think" + ">"
close_t = "<" + "/" + "think" + ">"

script = f"""  <script>
    const chat = document.getElementById("chat");
    const form = document.getElementById("form");
    const input = document.getElementById("input");
    const sendBtn = document.getElementById("send");
    const statusEl = document.getElementById("status");
    const modelName = document.getElementById("modelName");
    const baseUrlEl = document.getElementById("baseUrl");
    let messages = [];
    let modelId = "local";
    const SYSTEM = "You are HIVI. Answer clearly and directly. Do not use chain-of-thought. Do not repeat sentences, lists, or paragraphs.";
    const OPEN = {open_t!r};
    const CLOSE = {close_t!r};

    function addBubble(role, text, extraClass) {{
      const el = document.createElement("div");
      el.className = "bubble " + (extraClass || role);
      el.textContent = text;
      chat.appendChild(el);
      chat.scrollTop = chat.scrollHeight;
      return el;
    }}

    function stripThink(text) {{
      if (!text) return "";
      let t = text;
      t = t.replace(new RegExp(OPEN + "[\\\\s\\\\S]*?" + CLOSE, "gi"), "");
      t = t.replace(new RegExp(OPEN + "[\\\\s\\\\S]*$", "i"), "");
      return t.trim();
    }}

    function stopRepetition(text) {{
      if (!text) return text;
      const m = text.match(/(.{{40,}}?)(\\s*\\1){{2,}}/s);
      if (m && m.index != null) {{
        return text.slice(0, m.index + m[1].length).trim();
      }}
      const lines = text.split(/\\n/);
      if (lines.length >= 6) {{
        const last = lines[lines.length - 1];
        let n = 0;
        for (let i = lines.length - 1; i >= 0; i--) {{
          if (lines[i] === last && last.trim()) n++;
          else break;
        }}
        if (n >= 4) return lines.slice(0, lines.length - n + 1).join("\\n").trim();
      }}
      return text;
    }}

    async function refreshModels() {{
      const base = baseUrlEl.value.replace(/\\/$/, "");
      try {{
        const r = await fetch(base + "/models");
        if (!r.ok) throw new Error("HTTP " + r.status);
        const data = await r.json();
        modelId = data?.data?.[0]?.id || "local";
        modelName.textContent = modelId;
        statusEl.textContent = "online · llama.cpp";
        statusEl.classList.remove("err");
      }} catch (e) {{
        modelName.textContent = "no model";
        statusEl.textContent = "offline — start llama-server on :8080";
        statusEl.classList.add("err");
      }}
    }}

    async function sendMessage(text) {{
      messages.push({{ role: "user", content: text }});
      addBubble("user", text);
      const assistantEl = addBubble("assistant", "…");
      sendBtn.disabled = true;
      const base = baseUrlEl.value.replace(/\\/$/, "");
      const payload = {{
        model: modelId,
        messages: [{{ role: "system", content: SYSTEM }}, ...messages],
        temperature: Number(document.getElementById("temp").value),
        max_tokens: Number(document.getElementById("maxTokens").value),
        stream: true,
        repeat_penalty: 1.18,
        frequency_penalty: 0.35,
        presence_penalty: 0.15,
      }};
      let full = "";
      try {{
        const r = await fetch(base + "/chat/completions", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload),
        }});
        if (!r.ok) throw new Error("HTTP " + r.status + " " + (await r.text()));
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        while (true) {{
          const {{ value, done }} = await reader.read();
          if (done) break;
          buf += decoder.decode(value, {{ stream: true }});
          const parts = buf.split("\\n");
          buf = parts.pop();
          for (const line of parts) {{
            const s = line.trim();
            if (!s.startsWith("data:")) continue;
            const data = s.slice(5).trim();
            if (data === "[DONE]") continue;
            try {{
              const obj = JSON.parse(data);
              const piece = obj?.choices?.[0]?.delta?.content || "";
              if (!piece) continue;
              full += piece;
              const visible = stopRepetition(stripThink(full));
              assistantEl.textContent = visible || "…";
              chat.scrollTop = chat.scrollHeight;
            }} catch (_) {{}}
          }}
        }}
        const visible = stopRepetition(stripThink(full)) || "(empty reply)";
        assistantEl.textContent = visible;
        messages.push({{ role: "assistant", content: visible }});
        statusEl.textContent = "online · llama.cpp";
      }} catch (e) {{
        assistantEl.textContent = "Error: " + e.message;
        assistantEl.classList.add("err");
        statusEl.textContent = String(e.message);
        statusEl.classList.add("err");
        messages.pop();
      }} finally {{
        sendBtn.disabled = false;
        input.focus();
      }}
    }}

    form.addEventListener("submit", (e) => {{
      e.preventDefault();
      const text = input.value.trim();
      if (!text || sendBtn.disabled) return;
      input.value = "";
      sendMessage(text);
    }});

    input.addEventListener("keydown", (e) => {{
      if (e.key === "Enter" && !e.shiftKey) {{
        e.preventDefault();
        form.requestSubmit();
      }}
    }});

    document.getElementById("clear").addEventListener("click", () => {{
      messages = [];
      chat.innerHTML = "";
      addBubble("system", "New chat · HIVI");
    }});

    addBubble("system", "HIVI chat · Qwen3.5-9B abliterated · thinking off · anti-repeat on");
    refreshModels();
    setInterval(refreshModels, 15000);
  </script>"""

p.write_text(text[:start] + script + text[end:], encoding="utf-8")
print("patched", p)
