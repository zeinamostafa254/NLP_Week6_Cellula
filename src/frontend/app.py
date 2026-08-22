"""
Streamlit Frontend — Evaluator-Generator AI Platform
=====================================================
- Sidebar: Upload files, input URLs, manage knowledge base
- Main area: Chat interface with answer quality metadata
"""

import streamlit as st
import requests

# ---- Configuration ----
API_BASE_URL = "http://localhost:8000"

# ---- Page Config ----
st.set_page_config(
    page_title="Evaluator-Generator AI Platform",
    page_icon="🤖",
    layout="wide",
)

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# Sidebar — Knowledge Base Management
# ============================================================

with st.sidebar:
    st.title("📚 Knowledge Base")
    st.markdown("---")

    # ---- File Upload ----
    st.subheader("📁 Upload Files")
    st.caption("Supported: PDF, DOCX, TXT, CSV, PPT/PPTX, Code, WAV/MP3/M4A, Images (PNG/JPG/BMP/TIFF)")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "doc", "txt", "csv", "ppt", "pptx",
              "py", "js", "ts", "java", "cpp", "c", "h", "cs", "go", "rb", "rs",
              "html", "css", "wav", "mp3", "m4a",
              "png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("📤 Process Files", use_container_width=True, disabled=not uploaded_files):
        for f in uploaded_files:
            with st.spinner(f"Processing {f.name}..."):
                try:
                    files = {"file": (f.name, f.getvalue(), f.type)}
                    resp = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
                    if resp.status_code == 200:
                        st.success(f"✅ {f.name}")
                    else:
                        st.error(f"❌ {f.name} — {resp.json().get('detail', 'Error')}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend offline. Start FastAPI first.")
                except Exception as e:
                    st.error(f"❌ {f.name} — {e}")

    st.markdown("---")

    # ---- URL Input ----
    st.subheader("🌐 Add URL / Wikipedia")
    url_input = st.text_input("URL or Wikipedia topic:", placeholder="https://example.com")
    is_wikipedia = st.checkbox("Wikipedia topic", value=False)

    if st.button("🔗 Process URL", use_container_width=True, disabled=not url_input):
        with st.spinner("Fetching..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/url",
                    json={"url": url_input, "is_wikipedia": is_wikipedia},
                    timeout=60,
                )
                if resp.status_code == 200:
                    st.success(f"✅ {resp.json()['message']}")
                else:
                    st.error(f"❌ {resp.json().get('detail', 'Error')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend offline.")
            except Exception as e:
                st.error(f"❌ {e}")

    st.markdown("---")

    # ---- Status ----
    st.subheader("📊 Status")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            pass
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            try:
                resp = requests.delete(f"{API_BASE_URL}/knowledge/clear", timeout=10)
                if resp.status_code == 200:
                    st.success("✅ Cleared.")
                    st.session_state.chat_history = []
            except:
                st.error("❌ Failed.")

    try:
        resp = requests.get(f"{API_BASE_URL}/knowledge/status", timeout=5)
        if resp.status_code == 200:
            stats = resp.json()
            if stats["status"] == "ready":
                st.info(f"📦 **{stats['total_chunks']}** chunks stored.")
            else:
                st.warning("📭 Empty. Upload files to get started.")
    except:
        st.warning("⚠️ Backend offline.")


# ============================================================
# Main — Chat Interface
# ============================================================

st.title("🤖 Evaluator-Generator AI Platform")
st.caption("Ask questions about your uploaded documents. Answers go through a quality feedback loop.")
st.markdown("---")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            meta = msg["metadata"]
            with st.expander("📋 Answer Details"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Iterations", meta.get("iterations", "?"))
                with c2:
                    st.metric("Status", "✅ Accepted" if meta.get("accepted") else "⚠️ Best Effort")
                with c3:
                    st.metric("Max", "4")
                st.markdown(f"**Evaluator Feedback:** {meta.get('feedback', 'N/A')}")

                # Show iteration history if available
                if "history" in meta and meta["history"]:
                    st.markdown("**Iteration History:**")
                    for step in meta["history"]:
                        icon = "✅" if step["accepted"] else "🔄"
                        st.markdown(f"{icon} **Iteration {step['iteration']}**: {step['answer'][:150]}...")

# Chat input
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving context & running feedback loop..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": user_question},
                    timeout=300,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.markdown(result["answer"])

                    with st.expander("📋 Answer Details"):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Iterations", result["iterations"])
                        with c2:
                            st.metric("Status", "✅ Accepted" if result["accepted"] else "⚠️ Best Effort")
                        with c3:
                            st.metric("Max", "4")
                        st.markdown(f"**Evaluator Feedback:** {result['feedback']}")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": {
                            "iterations": result["iterations"],
                            "accepted": result["accepted"],
                            "feedback": result["feedback"],
                            "history": result.get("history", []),
                        },
                    })
                else:
                    err = resp.json().get("detail", "Unknown error")
                    st.error(f"❌ {err}")
                    st.session_state.chat_history.append({"role": "assistant", "content": f"❌ {err}"})

            except requests.exceptions.ConnectionError:
                st.error("❌ Backend offline. Start FastAPI on port 8000.")
                st.session_state.chat_history.append({"role": "assistant", "content": "❌ Backend offline."})
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out.")
                st.session_state.chat_history.append({"role": "assistant", "content": "❌ Timeout."})
            except Exception as e:
                st.error(f"❌ {e}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"❌ {e}"})
