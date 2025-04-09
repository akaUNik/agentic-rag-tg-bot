"""
This script ingests data from PDF files or YouTube transcripts, processes it into chunks,
and stores it in a Chroma database for retrieval-augmented generation (RAG) tasks.

Usage:
    python ingest.py <PDF_FILE_OR_YOUTUBE_URL> [PERSIST_DIRECTORY]
"""

import sys
from dotenv import load_dotenv

# PDF loader
from langchain_community.document_loaders import PyPDFLoader

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Chroma + embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# For YouTube transcripts
from youtube_transcript_api import YouTubeTranscriptApi

###############################################################################
# 1. Helper to detect & parse YouTube
###############################################################################
def is_youtube_link(url: str) -> bool:
    """
    Checks if the given URL is a YouTube link.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is a YouTube link, False otherwise.
    """
    return "youtube.com" in url.lower() or "youtu.be" in url.lower()

def extract_video_id(url: str) -> str:
    """
    Extracts a YouTube video ID from a typical YouTube/YouTu.be link.
    """
    if "youtu.be/" in url:
        # Format like: https://youtu.be/<VIDEO_ID>?...
        return url.split("youtu.be/")[-1].split("?")[0]
    elif "v=" in url:
        # Format like: https://www.youtube.com/watch?v=<VIDEO_ID>&...
        return url.split("v=")[-1].split("&")[0]
    else:
        raise ValueError(f"Could not parse video ID from url: {url}")

def load_youtube_transcript(url: str) -> list[Document]:
    """
    Fetches the YouTube transcript using youtube-transcript-api
    and returns a single Document object containing the transcript text.
    """
    video_id = extract_video_id(url)
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "ru"])
    except Exception as e:
        raise ValueError(f"Failed to get YouTube transcript: {e}") from e

    # Combine all text segments
    transcript_text = " ".join(segment["text"] for segment in transcript_data)
    # Create a single Document
    return [Document(page_content=transcript_text, metadata={"source": url})]

###############################################################################
# 2. Ingest Function
###############################################################################
def main(input_path: str, persist_directory: str = "./chromadb"):
    """
    Loads a PDF or a YouTube transcript, splits it into chunks,
    and appends them to a local Chroma DB (creates if it doesn't exist).
    """

    load_dotenv()  # So we get OPENAI_API_KEY, etc.

    # 1) Load documents depending on input type
    if input_path.lower().endswith(".pdf"):
        # It's a PDF file
        loader = PyPDFLoader(input_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages from PDF: {input_path}")
    elif is_youtube_link(input_path):
        # It's a YouTube link
        docs = load_youtube_transcript(input_path)
        print(f"Loaded transcript from YouTube video: {input_path}")
    else:
        raise ValueError("Input must be a path to a .pdf or a YouTube link.")

    # 2) Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # 3) Create or open existing Chroma collection
    embeddings = OpenAIEmbeddings()  # needs OPENAI_API_KEY
    vectorstore = Chroma(
        collection_name="rag-chroma",
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    # 4) Add new chunks to the collection
    vectorstore.add_documents(chunks)
    print(f"Added {len(chunks)} new chunks to 'rag-chroma' collection in {persist_directory}.")

    # 5) Persist to disk (may be optional depending on your langchain_chroma version)
    try:
        vectorstore.persist()
        print(f"Chroma DB updated at: {persist_directory}")
    except AttributeError:
        # Some newer versions auto-persist, so no 'persist' method is needed
        print("Chroma changes saved (auto-persist).")

###############################################################################
# 3. CLI
###############################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <PDF_FILE_OR_YOUTUBE_URL> [PERSIST_DIRECTORY]")
        sys.exit(1)

    input_arg = sys.argv[1]
    if len(sys.argv) > 2:
        persist_dir = sys.argv[2]
        main(input_arg, persist_dir)
    else:
        main(input_arg)
