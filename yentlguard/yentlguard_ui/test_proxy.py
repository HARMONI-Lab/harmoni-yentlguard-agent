import chainlit as cl
from chainlit.server import app as cl_app
from fastapi import Response
from google.cloud import storage as gcs

@cl_app.get("/report-proxy/{bucket_name}/{blob_name:path}")
async def proxy_report(bucket_name: str, blob_name: str):
    try:
        client = gcs.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        content = blob.download_as_string()
        return Response(content=content, media_type="text/html")
    except Exception as e:
        return Response(content=f"Error: {e}", status_code=500)

@cl.on_chat_start
async def on_chat_start():
    client = gcs.Client()
    bucket = client.bucket("yentlguard-analysis")
    blobs = list(bucket.list_blobs(prefix="reports/"))
    latest = sorted(blobs, key=lambda b: b.updated, reverse=True)[0]
    
    proxy_url = f"/report-proxy/yentlguard-analysis/{latest.name}"
    await cl.Message(content=f"Proxy URL: {proxy_url}").send()
    
    report_el = cl.CustomElement(
        name="ReportViewer",
        props={"html": "", "src": proxy_url, "title": "Test", "timestamp": "Now"},
        display="side",
    )
    
    import uuid
    await cl.ElementSidebar.set_title("REPORT TEST")
    await cl.ElementSidebar.set_elements([report_el], key=f"test-{uuid.uuid4().hex[:8]}")
