import chainlit as cl
from google.cloud import storage

@cl.on_chat_start
async def on_chat_start():
    try:
        client = storage.Client()
        bucket = client.bucket("yentlguard-analysis")
        
        # Get all reports and sort by updated time
        blobs = list(bucket.list_blobs(prefix="reports/"))
        if not blobs:
            await cl.Message(content="No reports found in bucket.").send()
            return
            
        latest_blob = sorted(blobs, key=lambda b: b.updated, reverse=True)[0]
        report_uri = f"https://storage.googleapis.com/yentlguard-analysis/{latest_blob.name}"
        
        await cl.Message(content=f"Loading latest report: {report_uri}").send()
        
        report_el = cl.CustomElement(
            name="ReportViewer",
            props={
                "html": "",
                "src": report_uri,
                "title": "Analysis Report",
                "timestamp": "Now",
            },
            display="side",
        )
        
        try:
            # Let's check if ElementSidebar exists
            await cl.ElementSidebar.set_title("ANALYSIS REPORT")
            await cl.ElementSidebar.set_elements([report_el], key="report-panel")
            await cl.Message(content="Successfully pushed to ElementSidebar.").send()
        except AttributeError:
            await report_el.send()
            await cl.Message(content="ElementSidebar API not found, sent inline via `report_el.send()`").send()
            
    except Exception as e:
        await cl.Message(content=f"Error loading report: {e}").send()
