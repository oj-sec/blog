import re
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from feedgen.feed import FeedGenerator
from bs4 import BeautifulSoup
import uvicorn

DOMAIN = "https://yourdomain.com"
RSS_CACHE = {"content": b""}

def build_rss():
    fg = FeedGenerator()
    fg.id(f"{DOMAIN}/blog")
    fg.title("Blog Feed")
    fg.link(href=f"{DOMAIN}/blog", rel="alternate")
    fg.description("Technical updates and security research")

    blog_dir = Path("../client/build/blog")
    posts = []

    if blog_dir.exists():
        for path in blog_dir.iterdir():
            if path.is_dir() and re.match(r"^\d{8}$", path.name):
                index_file = path / "index.html"
                if not index_file.exists():
                    continue

                with open(index_file, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")
                    h2_tag = soup.find("h2")
                    title = h2_tag.get_text(strip=True) if h2_tag else path.name

                try:
                    pub_date = datetime.strptime(path.name, "%Y%m%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                posts.append({
                    "title": title,
                    "url": f"{DOMAIN}/blog/{path.name}",
                    "date": pub_date
                })

    posts.sort(key=lambda x: x["date"], reverse=True)

    for post in posts:
        fe = fg.add_entry()
        fe.id(post["url"])
        fe.title(post["title"])
        fe.link(href=post["url"])
        fe.pubDate(post["date"])

    return fg.rss_str(pretty=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    RSS_CACHE["content"] = build_rss()
    yield

app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

@app.get("/rss.xml")
async def get_rss():
    return Response(content=RSS_CACHE["content"], media_type="application/xml")

app.mount("/", StaticFiles(directory="../client/build", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", port=8000)