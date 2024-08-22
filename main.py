from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from io import BytesIO
from dotenv import load_dotenv, get_key
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from pydantic import BaseModel


app = FastAPI()

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoteRequest(BaseModel):
    transcript: str
    word_range: str
    selected_llm: str

class FlashCardRequest(BaseModel):
    transcript: str
    selected_llm: str
    flashcard_type: str
    flashcard_range: str


@app.post("/get_pdf_text")
async def get_pdf_text(pdf: UploadFile = File(...)):
    text = ""
    pdf_content = await pdf.read()
    pdf_file = BytesIO(pdf_content)
    pdf_reader = PdfReader(pdf_file)

    for page_num, page in enumerate(pdf_reader.pages, start=1):
        images = convert_from_bytes(pdf_content, first_page=page_num, last_page=page_num)
        for image in images:
            page_text = pytesseract.image_to_string(image)
            text += f"PAGE {page_num}: {page_text}\n"
    
    return {"text": text}


def get_llm(selected_llm: str = "gemini-pro", flashcard_type: str = ""):
    if selected_llm == "llama3-70b-8192":
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.3,
        )
    elif selected_llm == "gemini-pro":
        llm = GoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=get_key(dotenv_path=".env", key_to_get="GOOGLE_API_KEY"),
        )
    if flashcard_type == "":
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Create a note from this transcript, include all the main ideas in bullets, with supporting details in sub-bullets. Make sections headers using given page numbers and other important information. Output in markdown formatting. Do it in {word_range} words.",
                ),
                ("user", "{transcript}"),
            ]
        )
    
    elif flashcard_type == "Term --> Definition":
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are tasked with creating flashcards that will help students learn the important terms, proper nouns and concepts in this note. Only make flashcards directly related to the main idea of the note, include as much detail as possible in each flashcard, returning it in a CSV formate with | as the seperator like this: Term | Definition. make exactly from {flashcard_range} flashcards. only return the csv data without any other information.",
                ),
                ("user", "{transcript}"),
            ]
        )
    elif flashcard_type == "Question --> Answer":
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are tasked with creating a mock test that will help students learn and understand concepts in this note. Only make questions directly related to the main idea of the note, You should include all these question types: fill in the blank, essay questions and True or False. return the questions and answers in a CSV formate with | as the seperator like this: This is a question | This is the answer. make exactly from {flashcard_range} Questions, Make sure to not generate less or more than the given amount or you will be punished. only return the csv data without any other information.",
                ),
                ("user", "{transcript}"),
            ]
        )

    chain = prompt | llm
    return chain

@app.post("/get_note")
async def get_note(request: NoteRequest):
    llm = get_llm(request.selected_llm)
    output = llm.invoke({"transcript": request.transcript, "word_range": request.word_range})
    return {"note": output}

@app.post("/get_flashcards")
async def get_flashcards(request: FlashCardRequest):
    llm = get_llm(request.selected_llm, flashcard_type=request.flashcard_type)
    output = llm.invoke({"transcript": request.transcript, "flashcard_range": request.flashcard_range})
    return {"flashcards": output}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)