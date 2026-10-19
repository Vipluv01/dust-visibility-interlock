"""Print report.html to PDF by driving headless Brave over the DevTools protocol.
Brave's own --print-to-pdf flag hangs without writing a file; CDP works."""
import asyncio, base64, http.client, json, os, sys, urllib.request

D = os.path.dirname(os.path.abspath(__file__))
URL = "file://" + os.path.join(D, "report.html")
OUT = os.path.join(os.path.dirname(D), "Dust-Visibility-Interlock.pdf")  # folder root


def new_tab():
    c = http.client.HTTPConnection("127.0.0.1", 9333, timeout=10)
    c.request("PUT", "/json/new?" + URL)          # newer Chrome requires PUT
    r = c.getresponse()
    if r.status != 200:
        c = http.client.HTTPConnection("127.0.0.1", 9333, timeout=10)
        c.request("GET", "/json/new?" + URL)
        r = c.getresponse()
    return json.loads(r.read())


async def main():
    import websockets
    tab = new_tab()
    ws_url = tab["webSocketDebuggerUrl"]
    async with websockets.connect(ws_url, max_size=None) as ws:
        n = 0

        async def call(method, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == n:
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    return msg["result"]

        await call("Page.enable")
        await call("Page.navigate", url=URL)
        await asyncio.sleep(4)                     # load + webfonts + layout
        res = await call("Page.printToPDF",
                         printBackground=True, preferCSSPageSize=True,
                         paperWidth=8.27, paperHeight=11.69, scale=1.0)
        data = base64.b64decode(res["data"])
        open(OUT, "wb").write(data)
        print(f"wrote {OUT} ({len(data)/1024:.0f} KB)")

asyncio.run(main())
