import os
import json
import re
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import tiktoken
from tqdm import tqdm
import openai

# Try to import dotenv for loading .env files
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional, but recommended

# --- Khai báo hàm trích xuất văn bản từ PDF (không thay đổi) ---
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts text from a given PDF file using PyMuPDF.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
    return text

# --- Khai báo hàm làm sạch văn bản (không thay đổi) ---
def clean_text(text: str) -> str:
    """
    Cleans and preprocesses the extracted text with enhanced handling for various
    problematic characters and whitespace.
    """
    # 1. Loại bỏ các ký tự Unicode không in được
    text = re.sub(r'[\x00-\x1F\x7F\u00A0\u200B-\u200D\u202F\u205F\u3000\uFEFF]+', '', text)
    
    # 2. Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Trim
    text = text.strip()
    
    # 5. Loại bỏ dòng chỉ chứa số 
    text = re.sub(r'(^|\n)\s*\d+\s*($|\n)', '\n', text, flags=re.MULTILINE)
    
    # 6. Chuẩn hóa ngắt dòng
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 7. Trim lần 2
    text = text.strip()
    
    return text

# --- Khai báo hàm chia nhỏ văn bản (không thay đổi) ---
def chunk_text(text: str, filename: str, max_chunk_tokens: int = 1500, overlap_tokens: int = 200, min_chunk_tokens: int = 50) -> List[Dict[str, Any]]:
    """
    Splits text into chunks based on token count, with optional overlap,
    and performs a final cleaning pass on each chunk.
    """
    if not text.strip():
        return []

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    chunks = []
    current_token_idx = 0
    chunk_id = 0

    while current_token_idx < len(tokens):
        end_token_idx = min(current_token_idx + max_chunk_tokens, len(tokens))
        chunk_tokens = tokens[current_token_idx:end_token_idx]
        chunk_text_raw = enc.decode(chunk_tokens)
        
        # Làm sạch cuối cùng cho từng chunk
        cleaned_chunk_text = re.sub(r'^[^\w\s]*', '', chunk_text_raw).strip()
        cleaned_chunk_text = re.sub(r"^\W+", "", cleaned_chunk_text)
        cleaned_chunk_text = cleaned_chunk_text.strip()

        if len(enc.encode(cleaned_chunk_text)) >= min_chunk_tokens:
            chunks.append({
                "text": cleaned_chunk_text,
                "source": filename,
                "chunk_id": chunk_id
            })
            chunk_id += 1
        
        current_token_idx += max_chunk_tokens - overlap_tokens
        if current_token_idx < 0: 
            current_token_idx = 0 
        if current_token_idx >= len(tokens):
            break 
            
    return chunks

# --- HÀM MỚI: TẠO CÂU HỎI BẰNG OPENAI API (CÓ RETRY & HỖ TRỢ NHIỀU CÂU HỎI) ---
def generate_user_prompt(assistant_content: str, model: str = "gpt-3.5-turbo", max_retries: int = 3, num_questions: int = 1) -> List[str]:
    """
    Generates relevant user prompts based on the assistant's content using OpenAI API with retry logic.
    Returns a list of questions.
    """
    retries = 0
    while retries <= max_retries:
        try:
            system_instruction = (
                "You are a helpful assistant that generates concise and relevant questions based on provided text passages. "
                "The questions should directly relate to the information in the passage. "
                "The questions MUST be in the same language as the provided text (e.g. Vietnamese). "
                "Generate only the questions, one per line, without numbering or bullet points."
            )
            
            user_content = f"Generate {num_questions} distinct, concise questions in the same language as the text about the following text:\n\n{assistant_content}"

            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=60 * num_questions, # Tăng token cho nhiều câu hỏi
                temperature=0.7, # Tăng nhẹ để các câu hỏi đa dạng hơn
                top_p=0.9
            )
            content = response.choices[0].message.content.strip()
            
            # Tách các câu hỏi dựa trên dòng mới
            questions = [q.strip() for q in content.split('\n') if q.strip()]
            
            # Làm sạch từng câu hỏi
            cleaned_questions = []
            for q in questions:
                # Xóa số thứ tự ở đầu nếu có (ví dụ: "1. Câu hỏi...")
                q = re.sub(r'^\d+[\.\)\-]\s*', '', q)
                if not q.endswith('?'):
                    q = q.strip('." ').replace('\n', ' ') + '?'
                cleaned_questions.append(q)
            
            # Lọc bớt các câu hỏi rác nếu có
            final_questions = []
            for q in cleaned_questions:
                if q.lower().startswith("dựa trên văn bản này") or q.lower().startswith("what is the main point"):
                     final_questions.append("Nội dung chính của văn bản này là gì?")
                else:
                    final_questions.append(q)
            
            # Nếu không tìm thấy câu hỏi nào (do lỗi format), trả về list rỗng để xuống dưới fallback
            if not final_questions:
                return []

            return final_questions[:num_questions] # Đảm bảo không trả về quá số lượng yêu cầu

        except openai.APIError as e:
            print(f"  [Attempt {retries+1}/{max_retries+1}] OpenAI API Error: {e}")
        except Exception as e:
            print(f"  [Attempt {retries+1}/{max_retries+1}] Unexpected Error: {e}")
        
        retries += 1
        if retries <= max_retries:
            time.sleep(1) # Wait a bit before retrying

    return [] # Trả về rỗng nếu thất bại hết các lần thử

# --- SỬA ĐỔI HÀM: format_for_openai_finetuning ---
def format_for_openai_finetuning(chunks: List[Dict[str, Any]], system_prompt: Optional[str] = None, 
                                 generate_prompts: bool = False, prompt_generation_model: str = "gpt-3.5-turbo",
                                 questions_per_chunk: int = 1) -> List[Dict[str, Any]]:
    """
    Formats the text chunks into the JSONL structure.
    """
    formatted_data = []
    if generate_prompts:
        print(f"\nGenerating {questions_per_chunk} user prompts per chunk (this may take a while)...")
    
    # Use tqdm but if generating prompts is off, it will be instant
    iterator = tqdm(chunks, desc="Formatting/Generating") if generate_prompts else chunks

    for chunk in iterator:
        
        questions = []
        
        if generate_prompts:
            generated_qs = generate_user_prompt(chunk["text"], model=prompt_generation_model, num_questions=questions_per_chunk)
            if generated_qs:
                questions = generated_qs
            else:
                 questions = ["Hãy cho tôi biết về tài liệu này."] # Fallback nếu lỗi
        else:
             questions = ["Hãy cho tôi biết về tài liệu này."] # Default static

        # Tạo một entry cho MỖI câu hỏi (Nhiều câu hỏi -> Nhiều entry cùng chung 1 nội dung chunk)
        for q in questions:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": chunk["text"]})
            
            formatted_data.append({"messages": messages})
            
    return formatted_data

# --- Khai báo hàm lưu JSONL (không thay đổi) ---
def save_to_jsonl(data: List[Dict[str, Any]], output_file_path: Path):
    """
    Saves the formatted data to a JSONL file.
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for entry in tqdm(data, desc=f"Saving to {output_file_path.name}"):
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"Successfully saved {len(data)} entries to {output_file_path}")
    except Exception as e:
        print(f"Error saving to JSONL file {output_file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to OpenAI fine-tuning JSONL format with optional AI-generated prompts.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing PDF files to process.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output JSONL file.")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum tokens per chunk.")
    parser.add_argument("--overlap_tokens", type=int, default=100,
                        help="Tokens overlap.")
    parser.add_argument("--min_tokens", type=int, default=50,
                        help="Minimum token count.")
    parser.add_argument("--system_prompt", type=str,
                        default="Bạn là một trợ lý ảo hữu ích, cung cấp thông tin dựa trên tài liệu được cung cấp.",
                        help="System prompt.")
    parser.add_argument("--generate_prompts", action="store_true",
                        help="Use OpenAI API to generate user prompts.")
    parser.add_argument("--prompt_model", type=str, default="gpt-3.5-turbo",
                        help="Model for prompt generation.")

    parser.add_argument("--questions_per_chunk", type=int, default=3, # Mặc định là 3 câu hỏi
                        help="Number of AI-generated questions per chunk.")

    args = parser.parse_args()

    # --- SECURITY UPDATE: LOAD KEY FROM ENV ---
    if not os.getenv("OPENAI_API_KEY"):
        # Try finding a .env file nearby if not loaded yet
        env_path = Path('.') / '.env'
        if env_path.exists():
             # Basic manual read if python-dotenv not installed
            print("Found .env file, checking for OPENAI_API_KEY...")
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip().startswith('OPENAI_API_KEY='):
                            key = line.strip().split('=', 1)[1].strip().strip('"').strip("'")
                            os.environ["OPENAI_API_KEY"] = key
                            print("Loaded OPENAI_API_KEY from .env file.")
                            break
            except Exception as ex:
                 print(f"Error reading .env: {ex}")

    if not os.getenv("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY not found in environment variables.")
        print("Please create a '.env' file with: OPENAI_API_KEY=sk-...")
        print("Or set it in your terminal: $env:OPENAI_API_KEY='sk-...'")
        return

    openai.api_key = os.getenv("OPENAI_API_KEY")
    # ------------------------------------------

    input_path = Path(args.input_dir)
    output_path = Path(args.output_file)

    if not input_path.is_dir():
        print(f"Error: Input directory '{input_path}' not found.")
        return

    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{input_path}'.")
        return

    all_formatted_data = []
    print(f"Processing {len(pdf_files)} PDF files from '{input_path}'...")

    for pdf_file in tqdm(pdf_files, desc="Files"):
        print(f"\n--- {pdf_file.name} ---")
        extracted_text = extract_text_from_pdf(pdf_file)
        if extracted_text:
            cleaned_text = clean_text(extracted_text)
            chunks = chunk_text(cleaned_text, pdf_file.name,
                                max_chunk_tokens=args.max_tokens,
                                overlap_tokens=args.overlap_tokens,
                                min_chunk_tokens=args.min_tokens)
            
            if chunks:
                formatted_entries = format_for_openai_finetuning(
                    chunks, 
                    system_prompt=args.system_prompt,
                    generate_prompts=args.generate_prompts,
                    prompt_generation_model=args.prompt_model,
                    questions_per_chunk=args.questions_per_chunk
                )
                all_formatted_data.extend(formatted_entries)
                print(f"  -> {len(formatted_entries)} entries (approx {len(chunks)} chunks x {args.questions_per_chunk} questions)")
            else:
                print(f"  -> No valid chunks (check content/params).")
        else:
            print(f"  -> Empty/Unreadable.")

    if all_formatted_data:
        save_to_jsonl(all_formatted_data, output_path)
        print("\nDone!")
        print(f"Total: {len(all_formatted_data)}")
        print(f"File: {output_path}")
    else:
        print("\nNo entries generated.")

if __name__ == "__main__":
    main()