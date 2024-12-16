import os
import io
import re
import base64
from typing import Dict

import uvicorn
import fitz  # PyMuPDF
import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageEnhance
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Load environment variables
load_dotenv()

class PDFSummaryAssistant:
    def __init__(self, pdf_path: str):
        """Initialize the PDF Summary Assistant with a static PDF path."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
        
        self.doc = fitz.open(pdf_path)
        self.text_pages = self._extract_text_from_pdf()
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    def _extract_text_from_pdf(self) -> list:
        """Extract text from each page of the PDF."""
        return [page.get_text() for page in self.doc]

    def _highlight_text(self, pdf_path, output_png, target_page, search_phrases, similarity_threshold=0.03):
        """
        Highlight text in the PDF with advanced fuzzy matching and ultra-high quality rendering.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_png (str): Output path for the highlighted image
            target_page (int): Page number to highlight
            search_phrases (list): Phrases to search and highlight
            similarity_threshold (float): Minimum similarity threshold for matching
        
        Returns:
            str: Base64 encoded highlighted image
        """
        def normalize_text(text):
            """Normalize text by removing extra whitespace and converting to lowercase."""
            return re.sub(r'\s+', ' ', str(text)).strip().lower()

        def calculate_sentence_similarity(sentence, target_phrase):
            """
            Calculate similarity between a sentence and a target phrase.
            
            Args:
                sentence (str): Full sentence to check
                target_phrase (str): Phrase to match against
            
            Returns:
                float: Similarity ratio
            """
            # Normalize texts
            norm_sentence = normalize_text(sentence)
            norm_target = normalize_text(target_phrase)
            
            # Calculate similarity
            similarity = SequenceMatcher(None, norm_sentence, norm_target).ratio()
            
            print(f"Comparing: \n  Sentence: '{norm_sentence}'\n  Target:   '{norm_target}'\n  Similarity: {similarity}")
            
            return similarity

        # Open PDF and create image
        pdf_doc = fitz.open(pdf_path)
        page_index = target_page - 1

        if page_index < 0 or page_index >= len(pdf_doc):
            raise ValueError(f"Invalid page number. PDF has {len(pdf_doc)} pages.")

        page = pdf_doc[page_index]
        
        # Ultra-high resolution rendering
        zoom = 8.0  # Increased from 8.0 to 16.0 for even higher quality
        # Create transformation matrix for maximum quality
        mat = fitz.Matrix(zoom, zoom)
        
        # Configure pixmap for maximum quality and clarity
        pix = page.get_pixmap(
            matrix=mat,
            alpha=False,
            colorspace=fitz.csRGB,  # Ensure RGB colorspace
            annots=True,  # Include annotations
        )
        
        # Convert to PIL Image with maximum quality preservation
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img, 'RGBA')

        # Extract page text and sentences
        page_text = page.get_text("text")
        sentences = re.split(r'(?<=[.!?])\s+', page_text)

        highlighted = False
        highlight_count = 0

        print("\n--- Sentence Matching Process ---")
        print(f"Target Page: {target_page}")
        print(f"Search Phrases: {search_phrases}")
        print(f"Similarity Threshold: {similarity_threshold}\n")

        # Function to create a semi-transparent yellow highlight
        def create_highlight(bbox):
            nonlocal highlighted, highlight_count
            # Scale the bbox coordinates according to zoom factor
            scaled_bbox = [coord * zoom for coord in bbox]
            
            # Create a more visible highlight with slight transparency
            draw.rectangle(
                [scaled_bbox[0], scaled_bbox[1], scaled_bbox[2], scaled_bbox[3]], 
                fill=(255, 255, 0, 100)  # Slightly less transparent yellow
            )
            highlighted = True
            highlight_count += 1

        # First, try exact matches
        for phrase in search_phrases:
            matches = page.search_for(phrase)
            
            if matches:
                print(f"Exact match found for phrase: '{phrase}'")
                for bbox in matches:
                    create_highlight(bbox)

        # If no exact matches, do fuzzy sentence matching
        if not highlighted:
            for phrase in search_phrases:
                for sentence in sentences:
                    similarity = calculate_sentence_similarity(sentence, phrase)
                    
                    if similarity >= similarity_threshold:
                        print(f"Fuzzy match found! Phrase: '{phrase}', Sentence: '{sentence.strip()}', Similarity: {similarity}")
                        
                        # Search for sentence in the page to get its bounding boxes
                        sentence_matches = page.search_for(sentence.strip())
                        
                        if sentence_matches:
                            for bbox in sentence_matches:
                                create_highlight(bbox)

        print(f"\nHighlight Summary:")
        print(f"Total Highlights: {highlight_count}")
        print(f"Highlighted: {highlighted}")

        # Save with ultra-high quality
        # Ensure very high DPI and maximum quality
        img.save(
            output_png, 
            format='PNG',
            dpi=(1200, 1200),  # Very high DPI
            optimize=False  # Disable optimization to preserve quality
        )

        # Generate base64 encoded image
        buffered = io.BytesIO()
        img.save(
            buffered, 
            format="PNG",
            dpi=(1200, 1200),  # Consistent DPI
            optimize=False
        )
        pdf_doc.close()

        return base64.b64encode(buffered.getvalue()).decode()

    def query_pdf(self, question: str, model: str = "claude-3-5-haiku-20241022") -> Dict:
        """Query the PDF using Claude and return response with highlights."""
        try:
            context = "\n\n".join([f"Page {i+1} Content:\n{text}" for i, text in enumerate(self.text_pages)])

            prompt = f"""You are a precise and helpful assistant analyzing a PDF document. Your task is to answer questions about the document's content accurately and cite your sources clearly.

            Here is the document content:
            {context}

            Question: {question}

            Instructions:
            1. Analyze the document content carefully
            2. Provide a direct answer to the question
            3. Include EXACT quotes that support your answer, enclosed in quotation marks
            4. Always specify the exact page number where you found the information
            5. If the answer spans multiple pages, cite all relevant pages
            6. If you cannot find an exact answer, say so clearly

            Format your response exactly like this:
            Answer: [Your clear and concise answer]
            Page: [Page number]
            Supporting Quote: "[Exact text from the document]"

            If you need to cite multiple quotes, list them separately:
            Supporting Quote 1: "[First exact quote]" (Page X)
            Supporting Quote 2: "[Second exact quote]" (Page Y)

            Remember:
            - Use EXACT quotes from the text, don't paraphrase
            - Always include page numbers
            - Be precise and factual
            - If information is unclear or not found, say so explicitly"""

            response = self.client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            claude_response = response.content[0].text
            
            # Extract all page numbers and quotes
            page_quotes = {}
            # Look for page numbers in format "Page X" or "(Page X)"
            page_matches = re.finditer(r'(?:Page[:]?\s*(\d+)|\(Page\s*(\d+)\))', claude_response)
            
            for match in page_matches:
                page_num = int(match.group(1) or match.group(2))
                # Find the next quote after this page number
                quote_start = match.end()
                quote_match = re.search(r'"([^"]+)"', claude_response[quote_start:])
                if quote_match:
                    if page_num not in page_quotes:
                        page_quotes[page_num] = []
                    page_quotes[page_num].append(quote_match.group(1))

            # If no matches found, use default
            if not page_quotes:
                page_quotes = {1: [self.text_pages[0].split('\n')[0]]}

            # Generate highlighted images for each page
            highlighted_images = {}
            for page_number, quotes in page_quotes.items():
                output_png = f"highlighted_page_{page_number}.png"
                highlighted_image = self._highlight_text(
                    pdf_path=self.doc.name,
                    output_png=output_png,
                    target_page=page_number,
                    search_phrases=quotes
                )
                highlighted_images[str(page_number)] = highlighted_image

            return {
                "response": claude_response,
                "pages": {
                    str(page_num): {
                        "quotes": quotes,
                        "highlighted_image": highlighted_images[str(page_num)]
                    }
                    for page_num, quotes in page_quotes.items()
                }
            }
        except Exception as e:
            if hasattr(e, 'status_code') and e.status_code == 500:
                # Return a dummy response that mimics the original response structure
                default_page = 1
                default_text = self.text_pages[0].split('\n')[0] if self.text_pages else "No text available"
                dummy_response = "I apologize, but I'm unable to process your request at the moment. Here's a sample from the document."
                
                highlighted_image = self._highlight_text(
                    pdf_path=self.doc.name,
                    output_png="highlighted_page_1.png",
                    target_page=default_page,
                    search_phrases=[default_text]
                )
                
                return {
                    "response": dummy_response,
                    "pages": {
                        "1": {
                            "quotes": [default_text],
                            "highlighted_image": highlighted_image
                        }
                    }
                }
            raise  # Re-raise other exceptions

# FastAPI App
app = FastAPI(title="PDF Summary API with Claude 3.5")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize PDF Assistant
PDF_PATH = os.path.join(os.path.dirname(__file__), 'documents', 'AMZN-Q3-2024-Earnings-Release.pdf')
pdf_assistant = PDFSummaryAssistant(PDF_PATH)

class QueryRequest(BaseModel):
    question: str

@app.get("/pdf-info/")
async def get_pdf_info():
    return {"total_pages": len(pdf_assistant.text_pages), "document_path": PDF_PATH}

@app.post("/query-pdf/")
async def query_pdf(request: QueryRequest):
    try:
        return pdf_assistant.query_pdf(request.question)
    except Exception as e:
        # Return a dummy response instead of raising an error
        default_page = 1
        default_text = pdf_assistant.text_pages[0].split('\n')[0] if pdf_assistant.text_pages else "No text available"
        dummy_response = "I apologize, but I'm unable to process your request at the moment. Here's a sample from the document."
        
        highlighted_image = pdf_assistant._highlight_text(
            pdf_path=pdf_assistant.doc.name,
            output_png="highlighted_page_1.png",
            target_page=default_page,
            search_phrases=[default_text]
        )
        
        return {
            "response": dummy_response,
            "pages": {
                "1": {
                    "quotes": [default_text],
                    "highlighted_image": highlighted_image
                }
            }
        }

def main():
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
