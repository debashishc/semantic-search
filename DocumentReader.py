import docx
import os
import pypdf

class DocumentReader:
    """
    A class for reading text from PDF, Word, and text files.
    
    Args:
        file_path (str): The path to the input file.
    
    Raises:
        ValueError: If the file format is not supported.
    """
    
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.file_extension = os.path.splitext(file_path)[-1].lower()
        
    def read(self) -> str:
        """
        Reads the input file and extracts the text.
        
        Returns:
            str: The extracted text.
        
        Raises:
            ValueError: If the file format is not supported.
        """
        if self.file_extension == ".pdf":
            return self._read_pdf()
        elif self.file_extension == ".docx":
            return self._read_docx()
        elif self.file_extension == ".txt":
            return self._read_text()
        else:
            raise ValueError("Unsupported file format")
            
    def _read_pdf(self) -> str:
        """
        Reads text from a PDF file.
        
        Returns:
            str: The extracted text.
        """
        with open(self.file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.strip()
    
    def _read_docx(self) -> str:
        """
        Reads text from a Word document.
        
        Returns:
            str: The extracted text.
        """
        doc = docx.Document(self.file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text.strip()
    
    def _read_text(self) -> str:
        """
        Reads text from a text file.
        
        Returns:
            str: The extracted text.
        """
        with open(self.file_path, 'r') as file:
            text = file.read()
            return text.strip()

if __name__ == '__main__':
    from pathlib import Path
    file_path = Path("~/Downloads/PS1.pdf").expanduser()
    reader = DocumentReader(file_path)
    text = reader.read()
    print(text)