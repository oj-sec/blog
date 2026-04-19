from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from feedgen.feed import FeedGenerator
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import uvicorn

DOMAIN = "https://oj-sec.com"
RSS_CACHE = {"content": b""}

def build_rss():
    fg = FeedGenerator()
    fg.id(f"{DOMAIN}/blog")
    fg.title("Blog Feed")
    fg.link(href=f"{DOMAIN}/blog", rel="alternate")
    fg.description("Latest updates from the blog")

    blog_dir = Path("../client/build/blog")
    
    if blog_dir.exists():
        for path in blog_dir.iterdir():
            if path.is_dir() or (path.is_file() and path.suffix == ".html"):
                slug = path.stem if path.is_file() else path.name
                if slug == "index": continue
                
                url = f"{DOMAIN}/blog/{slug}"
                stats = path.stat()
                
                fe = fg.add_entry()
                fe.id(url)
                fe.title(slug.replace('-', ' ').title())
                fe.link(href=url)
                fe.pubDate(datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc))
    
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